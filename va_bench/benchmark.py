"""
Core benchmark loop for Vision Analysis.

Runs a single model through COCO val2017 with per-image timing,
prediction collection, and memory tracking.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .coco_eval import evaluate_coco
from .hardware import collect_all as collect_hw, get_runtime_device_name
from .models import ModelSpec, check_params, get_spec, load_model
from .output import assemble_result
from .timing import SyncTimer, compute_stats, device_sync, warmup


# COCO uses non-contiguous category IDs (1-90 with gaps).
# LibreYOLO models use 0-indexed class IDs (0-79).
# This mapping converts model class index -> COCO category ID.
COCO_80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def benchmark_model(
    model_key: str,
    coco_dir: str | Path,
    device: str = "auto",
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    verbose: bool = True,
) -> dict[str, Any]:
    """Benchmark a single model on COCO val2017.

    Args:
        model_key: Registry key (e.g. "yolov9t", "yolox-s").
        coco_dir: Path to COCO directory containing images/val2017/ and
            annotations/instances_val2017.json.
        device: Device string ("auto", "cuda", "mps", "cpu").
        conf: Confidence threshold for predictions.
        iou: IoU threshold for NMS.
        max_det: Maximum detections per image.
        verbose: Print progress.

    Returns:
        Dict matching the RawBenchmark schema for the website.
    """
    coco_dir = Path(coco_dir)
    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    img_dir = coco_dir / "images" / "val2017"

    if not img_dir.exists():
        img_dir = coco_dir / "val2017"
    if not ann_file.exists():
        raise FileNotFoundError(
            f"COCO annotations not found. Expected at {ann_file}. "
            f"Provide path to a directory containing annotations/ and images/val2017/ (or val2017/)."
        )

    # --- Load model ---
    spec = get_spec(model_key)
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {spec.display_name} ({spec.key})")
        print(f"{'=' * 70}")

    model, _ = load_model(model_key, device=device)
    actual_device = model.device
    imgsz = model._get_input_size()

    if verbose:
        print(f"  Device: {actual_device}")
        print(f"  Input size: {imgsz}")

    # --- Sanity check params ---
    param_warning = check_params(model, spec)
    if param_warning:
        warnings.warn(param_warning)
    measured_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    if verbose:
        print(f"  Parameters: {measured_params:.2f}M (paper: {spec.paper_params_m:.2f}M)")
        print(f"  GFLOPs: {spec.paper_flops_g:.2f} (from paper)")

    # --- Load COCO ---
    from pycocotools.coco import COCO

    if verbose:
        print(f"\nLoading COCO from {ann_file}...")
    coco_gt = COCO(str(ann_file))
    img_ids = sorted(coco_gt.getImgIds())
    if verbose:
        print(f"  {len(img_ids)} images")

    # --- Warmup ---
    if verbose:
        n_warmup = 10 if actual_device.type in ("cuda", "mps") else 3
        print(f"\nWarming up ({n_warmup} iterations)...")
    warmup(model, actual_device)
    if verbose:
        print("  Done")

    # --- Reset memory counters ---
    rss_before = _get_rss_mb()
    if actual_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual_device)

    # --- Main benchmark loop ---
    timer = SyncTimer(actual_device)
    pre_times: list[float] = []
    inf_times: list[float] = []
    post_times: list[float] = []
    total_times: list[float] = []
    predictions: list[dict] = []

    pbar = tqdm(img_ids, desc="Benchmarking", disable=not verbose)

    for img_id in pbar:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = str(img_dir / img_info["file_name"])

        # Load image as PIL
        pil_img = Image.open(img_path).convert("RGB")

        timer.reset()

        # Phase 1: Preprocess
        timer.mark()
        input_tensor, _orig_img, original_size, ratio = model._preprocess(
            pil_img, "rgb", input_size=imgsz,
        )
        input_tensor = input_tensor.to(actual_device)

        # Phase 2: Inference
        timer.mark()
        with torch.no_grad():
            output = model._forward(input_tensor)

        # Phase 3: Postprocess
        timer.mark()
        detections = model._postprocess(
            output, conf, iou, original_size, max_det=max_det, ratio=ratio,
        )
        timer.mark()

        # Collect timings
        phases = timer.phases_ms()
        pre_times.append(phases[0])
        inf_times.append(phases[1])
        post_times.append(phases[2])
        total_times.append(timer.total_ms())

        # Collect predictions in COCO format
        if detections["num_detections"] > 0:
            boxes = detections["boxes"]
            scores = detections["scores"]
            classes = detections["classes"]

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if isinstance(classes, torch.Tensor):
                classes = classes.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                cls_int = int(cls)
                cat_id = COCO_80_TO_91[cls_int] if cls_int < len(COCO_80_TO_91) else cls_int + 1

                predictions.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                })

        if verbose:
            pbar.set_postfix({"ms": f"{timer.total_ms():.1f}", "dets": len(predictions)})

    # --- Memory ---
    peak_vram_mb = 0.0
    if actual_device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(actual_device) / (1024 ** 2)
    rss_after = _get_rss_mb()
    peak_ram_mb = max(rss_after - rss_before, 0.0)

    # --- Timing stats ---
    pre_arr = np.array(pre_times)
    inf_arr = np.array(inf_times)
    post_arr = np.array(post_times)
    total_arr = np.array(total_times)

    total_stats = compute_stats(total_arr)
    fps_mean = 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / total_stats["p50"] if total_stats["p50"] > 0 else 0.0

    if verbose:
        print(f"\nTiming ({len(img_ids)} images):")
        print(f"  Preprocess:  {np.mean(pre_arr):.2f} ms/image")
        print(f"  Inference:   {np.mean(inf_arr):.2f} ms/image")
        print(f"  Postprocess: {np.mean(post_arr):.2f} ms/image")
        print(f"  Total:       {total_stats['mean']:.2f} ms/image (p50={total_stats['p50']:.2f})")
        print(f"  FPS:         {fps_mean:.1f} (p50: {fps_p50:.1f})")
        print(f"  Memory:      VRAM={peak_vram_mb:.0f}MB, RAM delta={peak_ram_mb:.0f}MB")

    # --- COCO evaluation ---
    if verbose:
        print(f"\nCOCO evaluation ({len(predictions)} detections)...")
    coco_metrics = evaluate_coco(coco_gt, predictions, image_ids=img_ids)

    if verbose:
        print(f"  mAP@50-95: {coco_metrics['mAP']:.4f}")
        print(f"  mAP@50:    {coco_metrics['mAP50']:.4f}")
        print(f"  mAP@75:    {coco_metrics['mAP75']:.4f}")
        print(f"  mAP_small: {coco_metrics['mAP_small']:.4f}")
        print(f"  mAP_med:   {coco_metrics['mAP_medium']:.4f}")
        print(f"  mAP_large: {coco_metrics['mAP_large']:.4f}")

    # --- Assemble result ---
    hw_sw = collect_hw()
    result = assemble_result(
        spec=spec,
        coco_metrics=coco_metrics,
        total_stats=total_stats,
        preprocess_ms=round(float(np.mean(pre_arr)), 3),
        inference_ms=round(float(np.mean(inf_arr)), 3),
        postprocess_ms=round(float(np.mean(post_arr)), 3),
        fps_mean=round(fps_mean, 2),
        fps_p50=round(fps_p50, 2),
        num_images=len(img_ids),
        measured_params_m=round(measured_params, 2),
        peak_vram_mb=round(peak_vram_mb, 1),
        peak_ram_mb=round(peak_ram_mb, 1),
        device_type=get_runtime_device_name(actual_device.type),
        hardware=hw_sw["hardware"],
        software=hw_sw["software"],
    )

    return result


def _get_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except ImportError:
        return 0.0
