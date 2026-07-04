"""
Core benchmark loop for Vision Analysis.

Runs a single model through COCO val2017 with per-image timing,
prediction collection, and memory tracking. Supports two backends:

* ``pytorch``  — LibreYOLO's native PyTorch path. Produces phase-split
  timing (preprocess / inference / postprocess) and VRAM stats on CUDA.
* ``onnx``     — LibreYOLO's ONNX Runtime backend. Treated as a black
  box: total wall time per image, no phase split, no VRAM stats.
* ``tensorrt`` — LibreYOLO's native TensorRT backend (FP16 engines).
  Treated as a black box exactly like ``onnx``: total wall time per
  image, no phase split, no VRAM stats.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .coco_eval import evaluate_coco
from .hardware import collect_all as collect_hw, get_runtime_device_name
from .models import (
    ModelSpec,
    check_params,
    count_onnx_params,
    get_spec,
    load_model,
    load_onnx,
    load_tensorrt,
    resolve_onnx_weights,
    resolve_tensorrt_weights,
)
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

SUPPORTED_LIBREYOLO_COMMIT = "24143e52339d71570ae3207ee60ada202a731200"
REQUIRED_PYTORCH_MODEL_API = (
    "_get_input_size",
    "_preprocess",
    "_forward",
    "_postprocess",
)


def benchmark_model(
    model_key: str,
    coco_dir: str | Path,
    fmt: str = "pytorch",
    weights_dir: str | Path | None = None,
    device: str = "auto",
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    limit: int | None = None,
    verbose: bool = True,
    precision: str = "fp16",
) -> dict[str, Any]:
    """Benchmark a single model on COCO val2017.

    Args:
        model_key: Registry key (e.g. "yolov9t", "yolox-s").
        coco_dir: Path to COCO directory containing images/val2017/ and
            annotations/instances_val2017.json.
        fmt: Backend format — "pytorch" (default), "onnx", or "tensorrt".
        weights_dir: Directory containing user-supplied ONNX files (.onnx)
            or TensorRT engines (.engine + .engine.json sidecar).
            Required when fmt="onnx" or fmt="tensorrt", ignored otherwise.
        device: Device string ("auto", "cuda", "mps", "cpu").
        conf: Confidence threshold for predictions.
        iou: IoU threshold for NMS.
        max_det: Maximum detections per image.
        limit: If set, evaluate only the first N val2017 images (dev/CPU
            subset). A subset run is NOT a valid full-val2017 submission.
        verbose: Print progress.

    Returns:
        Dict matching the RawBenchmark schema for the website.
    """
    spec = get_spec(model_key)
    if spec.source != "libreyolo":
        raise ValueError(
            f"Model '{model_key}' has source '{spec.source}'; benchmark_model only "
            f"handles libreyolo models. Use run_ultralytics_benchmark (the CLI "
            f"routes by source automatically)."
        )
    if fmt == "pytorch":
        return _benchmark_pytorch(
            model_key, coco_dir, device, conf, iou, max_det, limit, verbose,
        )
    if fmt == "onnx":
        if weights_dir is None:
            raise ValueError("weights_dir is required when fmt='onnx'")
        return _benchmark_onnx(
            model_key, coco_dir, weights_dir, device, conf, iou, max_det, limit, verbose,
        )
    if fmt == "tensorrt":
        if weights_dir is None:
            raise ValueError("weights_dir is required when fmt='tensorrt'")
        return _benchmark_tensorrt(
            model_key, coco_dir, weights_dir, device, conf, iou, max_det, limit, verbose, precision,
        )
    raise ValueError(
        f"Unknown format: {fmt!r}. Use 'pytorch', 'onnx', or 'tensorrt'."
    )


# =============================================================================
# Shared helpers
# =============================================================================

def _load_coco(coco_dir: Path, verbose: bool):
    """Resolve annotation + image paths and load the COCO index."""
    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    img_dir = coco_dir / "images" / "val2017"

    if not img_dir.exists():
        img_dir = coco_dir / "val2017"
    if not ann_file.exists():
        raise FileNotFoundError(
            f"COCO annotations not found. Expected at {ann_file}. "
            f"Provide a directory containing annotations/ and images/val2017/ "
            f"(or val2017/)."
        )

    from pycocotools.coco import COCO

    if verbose:
        print(f"\nLoading COCO from {ann_file}...")
    coco_gt = COCO(str(ann_file))
    img_ids = sorted(coco_gt.getImgIds())
    if verbose:
        print(f"  {len(img_ids)} images")
    return coco_gt, img_ids, img_dir


def _append_predictions(
    predictions: list[dict],
    boxes: Any,
    scores: Any,
    classes: Any,
    img_id: int,
) -> None:
    """Append per-image detections to a running COCO predictions list."""
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
        cat_id = (
            COCO_80_TO_91[cls_int] if cls_int < len(COCO_80_TO_91) else cls_int + 1
        )
        predictions.append({
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(score),
        })


def _print_timing_summary(
    verbose: bool,
    n_images: int,
    pre_arr: np.ndarray | None,
    inf_arr: np.ndarray | None,
    post_arr: np.ndarray | None,
    total_stats: dict[str, float],
    fps_mean: float,
    fps_p50: float,
    peak_vram_mb: float | None,
    peak_ram_mb: float,
) -> None:
    if not verbose:
        return
    print(f"\nTiming ({n_images} images):")
    if pre_arr is not None:
        print(f"  Preprocess:  {np.mean(pre_arr):.2f} ms/image")
        print(f"  Inference:   {np.mean(inf_arr):.2f} ms/image")
        print(f"  Postprocess: {np.mean(post_arr):.2f} ms/image")
    print(
        f"  Total:       {total_stats['mean']:.2f} ms/image "
        f"(p50={total_stats['p50']:.2f})"
    )
    print(f"  FPS:         {fps_mean:.1f} (p50: {fps_p50:.1f})")
    if peak_vram_mb is not None:
        print(f"  Memory:      VRAM={peak_vram_mb:.0f}MB, RAM delta={peak_ram_mb:.0f}MB")
    else:
        print(f"  Memory:      RAM delta={peak_ram_mb:.0f}MB (no VRAM stats for ONNX)")


def _get_rss_mb() -> float:
    """Get current process RSS in MB."""
    import psutil
    return psutil.Process().memory_info().rss / (1024 ** 2)


def _assert_supported_pytorch_model_api(model: Any) -> None:
    """Fail fast when the installed LibreYOLO build is missing harness APIs."""
    missing = [
        attr for attr in REQUIRED_PYTORCH_MODEL_API if not callable(getattr(model, attr, None))
    ]
    if not missing:
        return

    raise RuntimeError(
        "Installed libreyolo is incompatible with vision-analysis-benchmark. "
        f"Missing model API: {', '.join(missing)}. "
        f"Expected LibreYOLO commit {SUPPORTED_LIBREYOLO_COMMIT} "
        "or an equivalent compatible build."
    )


def _assert_onnx_cuda_provider_available() -> None:
    """Require ONNX Runtime CUDA support for explicit ONNX CUDA runs."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Requested ONNX CUDA benchmarking, but onnxruntime is not installed. "
            "Install the GPU runtime or rerun with --device cpu."
        ) from exc

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return

    raise RuntimeError(
        "Requested ONNX CUDA benchmarking, but ONNX Runtime has no CUDAExecutionProvider. "
        f"Available providers: {providers}. Install a CUDA-enabled onnxruntime build "
        "that matches the host driver/runtime, or rerun with --device cpu."
    )


# =============================================================================
# PyTorch path
# =============================================================================

def _apply_limit(img_ids: list[int], limit: int | None, verbose: bool) -> list[int]:
    """Slice img_ids to the first `limit` entries for dev/CPU subset runs."""
    if limit is None or limit >= len(img_ids):
        return img_ids
    if verbose:
        print(f"  --limit {limit}: evaluating first {limit}/{len(img_ids)} images "
              f"(SUBSET - not a full-val2017 submission)")
    return img_ids[:limit]


def _benchmark_pytorch(
    model_key: str,
    coco_dir: str | Path,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    limit: int | None,
    verbose: bool,
) -> dict[str, Any]:
    coco_dir = Path(coco_dir)

    spec = get_spec(model_key)
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {spec.display_name} ({spec.key}) [PyTorch]")
        print(f"{'=' * 70}")

    model, _ = load_model(model_key, device=device)
    _assert_supported_pytorch_model_api(model)
    actual_device = model.device
    imgsz = model._get_input_size()

    if verbose:
        print(f"  Device: {actual_device}")
        print(f"  Input size: {imgsz}")

    param_warning = check_params(model, spec)
    if param_warning:
        warnings.warn(param_warning)
    measured_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    if verbose:
        print(f"  Parameters: {measured_params:.2f}M (paper: {spec.paper_params_m:.2f}M)")
        print(f"  GFLOPs: {spec.paper_flops_g:.2f} (from paper)")

    coco_gt, img_ids, img_dir = _load_coco(coco_dir, verbose)
    img_ids = _apply_limit(img_ids, limit, verbose)

    if verbose:
        n_warmup = 10 if actual_device.type in ("cuda", "mps") else 3
        print(f"\nWarming up ({n_warmup} iterations)...")
    warmup(model, actual_device)
    if verbose:
        print("  Done")

    rss_before = _get_rss_mb()
    if actual_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual_device)

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
        pil_img = Image.open(img_path).convert("RGB")

        timer.reset()

        timer.mark()
        input_tensor, _orig_img, original_size, ratio = model._preprocess(
            pil_img, "rgb", input_size=imgsz,
        )
        input_tensor = input_tensor.to(actual_device)

        timer.mark()
        with torch.no_grad():
            output = model._forward(input_tensor)

        timer.mark()
        detections = model._postprocess(
            output, conf, iou, original_size, max_det=max_det, ratio=ratio,
        )
        timer.mark()

        phases = timer.phases_ms()
        pre_times.append(phases[0])
        inf_times.append(phases[1])
        post_times.append(phases[2])
        total_times.append(timer.total_ms())

        if detections["num_detections"] > 0:
            _append_predictions(
                predictions,
                detections["boxes"],
                detections["scores"],
                detections["classes"],
                img_id,
            )

        if verbose:
            pbar.set_postfix({"ms": f"{timer.total_ms():.1f}", "dets": len(predictions)})

    peak_vram_mb = 0.0
    if actual_device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(actual_device) / (1024 ** 2)
    rss_after = _get_rss_mb()
    peak_ram_mb = max(rss_after - rss_before, 0.0)

    pre_arr = np.array(pre_times)
    inf_arr = np.array(inf_times)
    post_arr = np.array(post_times)
    total_arr = np.array(total_times)

    total_stats = compute_stats(total_arr)
    fps_mean = 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / total_stats["p50"] if total_stats["p50"] > 0 else 0.0

    _print_timing_summary(
        verbose, len(img_ids), pre_arr, inf_arr, post_arr, total_stats,
        fps_mean, fps_p50, peak_vram_mb, peak_ram_mb,
    )

    if verbose:
        print(f"\nCOCO evaluation ({len(predictions)} detections)...")
    coco_metrics = evaluate_coco(coco_gt, predictions, image_ids=img_ids)

    if verbose:
        _print_accuracy(coco_metrics)

    hw_sw = collect_hw()
    return assemble_result(
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
        provider=actual_device.type,
        hardware=hw_sw["hardware"],
        software=hw_sw["software"],
        actual_input_size=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        fmt="pytorch",
    )


# =============================================================================
# ONNX path
# =============================================================================

def _benchmark_onnx(
    model_key: str,
    coco_dir: str | Path,
    weights_dir: str | Path,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    limit: int | None,
    verbose: bool,
) -> dict[str, Any]:
    coco_dir = Path(coco_dir)

    spec = get_spec(model_key)
    onnx_path = resolve_onnx_weights(spec, weights_dir)

    if device == "cuda":
        _assert_onnx_cuda_provider_available()

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {spec.display_name} ({spec.key}) [ONNX]")
        print(f"{'=' * 70}")
        print(f"  Weights: {onnx_path}")

    backend, _ = load_onnx(model_key, weights_dir, device=device)
    # BaseBackend stores device as a string ("cuda" or "cpu").
    backend_device = backend.device
    if device == "cuda" and backend_device != "cuda":
        raise RuntimeError(
            "Requested ONNX CUDA benchmarking, but LibreYOLO resolved the backend to "
            f"device={backend_device!r}. Check the ONNX Runtime GPU install and providers."
        )
    imgsz = backend.imgsz

    if verbose:
        print(f"  Device: {backend_device}")
        print(f"  Input size: {imgsz}")

    measured_params = count_onnx_params(onnx_path)
    if verbose:
        print(f"  Parameters: {measured_params:.2f}M (paper: {spec.paper_params_m:.2f}M)")
        print(f"  GFLOPs: {spec.paper_flops_g:.2f} (from paper)")

    coco_gt, img_ids, img_dir = _load_coco(coco_dir, verbose)
    img_ids = _apply_limit(img_ids, limit, verbose)

    n_warmup = 10 if backend_device == "cuda" else 3
    if verbose:
        print(f"\nWarming up ({n_warmup} iterations)...")
    _onnx_warmup(backend, n_warmup)
    if verbose:
        print("  Done")

    rss_before = _get_rss_mb()

    total_times: list[float] = []
    predictions: list[dict] = []

    import time
    pbar = tqdm(img_ids, desc="Benchmarking", disable=not verbose)
    for img_id in pbar:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = str(img_dir / img_info["file_name"])
        pil_img = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        result = backend.predict(
            pil_img, conf=conf, iou=iou, max_det=max_det, color_format="rgb",
        )
        t1 = time.perf_counter()
        total_times.append((t1 - t0) * 1000.0)

        if len(result.boxes.xyxy) > 0:
            _append_predictions(
                predictions,
                result.boxes.xyxy,
                result.boxes.conf,
                result.boxes.cls,
                img_id,
            )

        if verbose:
            pbar.set_postfix({
                "ms": f"{(t1 - t0) * 1000.0:.1f}",
                "dets": len(predictions),
            })

    rss_after = _get_rss_mb()
    peak_ram_mb = max(rss_after - rss_before, 0.0)

    total_arr = np.array(total_times)
    total_stats = compute_stats(total_arr)
    fps_mean = 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / total_stats["p50"] if total_stats["p50"] > 0 else 0.0

    _print_timing_summary(
        verbose, len(img_ids), None, None, None, total_stats,
        fps_mean, fps_p50, None, peak_ram_mb,
    )

    if verbose:
        print(f"\nCOCO evaluation ({len(predictions)} detections)...")
    coco_metrics = evaluate_coco(coco_gt, predictions, image_ids=img_ids)

    if verbose:
        _print_accuracy(coco_metrics)

    hw_sw = collect_hw()
    device_type = "gpu" if backend_device == "cuda" else "cpu"
    return assemble_result(
        spec=spec,
        coco_metrics=coco_metrics,
        total_stats=total_stats,
        preprocess_ms=None,
        inference_ms=None,
        postprocess_ms=None,
        fps_mean=round(fps_mean, 2),
        fps_p50=round(fps_p50, 2),
        num_images=len(img_ids),
        measured_params_m=round(measured_params, 2),
        peak_vram_mb=None,
        peak_ram_mb=round(peak_ram_mb, 1),
        device_type=device_type,
        provider=backend_device,
        hardware=hw_sw["hardware"],
        software=hw_sw["software"],
        actual_input_size=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        fmt="onnx",
    )


def _onnx_warmup(backend: Any, n_iters: int) -> None:
    """Run dummy forward passes through an ONNX backend to stabilize caches."""
    imgsz = backend.imgsz
    dummy = Image.fromarray(
        np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    )
    for _ in range(n_iters):
        try:
            backend.predict(dummy, conf=0.5, iou=0.5, color_format="rgb")
        except Exception:
            break


# =============================================================================
# TensorRT path
# =============================================================================

def _benchmark_tensorrt(
    model_key: str,
    coco_dir: str | Path,
    weights_dir: str | Path,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    limit: int | None,
    verbose: bool,
    precision: str = "fp16",
) -> dict[str, Any]:
    coco_dir = Path(coco_dir)

    spec = get_spec(model_key)
    engine_path = resolve_tensorrt_weights(spec, weights_dir)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {spec.display_name} ({spec.key}) [TensorRT]")
        print(f"{'=' * 70}")
        print(f"  Weights: {engine_path}")

    backend, _ = load_tensorrt(model_key, weights_dir, device=device)
    # TensorRTBackend stores device as a string ("cuda").
    backend_device = backend.device
    if backend_device != "cuda":
        raise RuntimeError(
            "TensorRT benchmarking requires a CUDA device, but the backend resolved "
            f"to device={backend_device!r}. Check the tensorrt + CUDA install."
        )
    imgsz = backend.imgsz

    if verbose:
        print(f"  Device: {backend_device}")
        print(f"  Input size: {imgsz}")

    measured_params = 0.0
    if verbose:
        print(f"  Parameters: n/a (TensorRT engine; paper: {spec.paper_params_m:.2f}M)")
        print(f"  GFLOPs: {spec.paper_flops_g:.2f} (from paper)")

    coco_gt, img_ids, img_dir = _load_coco(coco_dir, verbose)
    img_ids = _apply_limit(img_ids, limit, verbose)

    n_warmup = 10
    if verbose:
        print(f"\nWarming up ({n_warmup} iterations)...")
    _tensorrt_warmup(backend, n_warmup)
    if verbose:
        print("  Done")

    rss_before = _get_rss_mb()

    total_times: list[float] = []
    predictions: list[dict] = []

    import time
    pbar = tqdm(img_ids, desc="Benchmarking", disable=not verbose)
    for img_id in pbar:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = str(img_dir / img_info["file_name"])
        pil_img = Image.open(img_path).convert("RGB")

        t0 = time.perf_counter()
        result = backend(
            pil_img, conf=conf, iou=iou, max_det=max_det, color_format="rgb",
        )
        t1 = time.perf_counter()
        total_times.append((t1 - t0) * 1000.0)

        if len(result.boxes.xyxy) > 0:
            _append_predictions(
                predictions,
                result.boxes.xyxy,
                result.boxes.conf,
                result.boxes.cls,
                img_id,
            )

        if verbose:
            pbar.set_postfix({
                "ms": f"{(t1 - t0) * 1000.0:.1f}",
                "dets": len(predictions),
            })

    rss_after = _get_rss_mb()
    peak_ram_mb = max(rss_after - rss_before, 0.0)

    total_arr = np.array(total_times)
    total_stats = compute_stats(total_arr)
    fps_mean = 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / total_stats["p50"] if total_stats["p50"] > 0 else 0.0

    _print_timing_summary(
        verbose, len(img_ids), None, None, None, total_stats,
        fps_mean, fps_p50, None, peak_ram_mb,
    )

    if verbose:
        print(f"\nCOCO evaluation ({len(predictions)} detections)...")
    coco_metrics = evaluate_coco(coco_gt, predictions, image_ids=img_ids)

    if verbose:
        _print_accuracy(coco_metrics)

    hw_sw = collect_hw()
    return assemble_result(
        spec=spec,
        coco_metrics=coco_metrics,
        total_stats=total_stats,
        preprocess_ms=None,
        inference_ms=None,
        postprocess_ms=None,
        fps_mean=round(fps_mean, 2),
        fps_p50=round(fps_p50, 2),
        num_images=len(img_ids),
        measured_params_m=round(measured_params, 2),
        peak_vram_mb=None,
        peak_ram_mb=round(peak_ram_mb, 1),
        device_type="gpu",
        provider="cuda",
        hardware=hw_sw["hardware"],
        software=hw_sw["software"],
        actual_input_size=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        fmt="tensorrt",
        precision=precision,
    )


def _tensorrt_warmup(backend: Any, n_iters: int) -> None:
    """Run dummy forward passes through a TensorRT backend to stabilize caches."""
    imgsz = backend.imgsz
    dummy = Image.fromarray(
        np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    )
    for _ in range(n_iters):
        try:
            backend(dummy, conf=0.5, iou=0.5, color_format="rgb")
        except Exception:
            break


# =============================================================================
# Ultralytics driver path (subprocess, separate venv)
# =============================================================================
#
# The harness never imports the ultralytics package. A standalone driver
# script (drivers/ultralytics/uly_driver.py) runs in its own venv and
# communicates via JSON files — see PLAN_ultralytics_source.md for the
# contract. Phase timings (speed_ms) come from ultralytics' own per-result
# instrumentation; wall_ms is the driver's device-synced wall clock and
# feeds the harness total_ms stats and FPS.

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ULY_DRIVER = _REPO_ROOT / "drivers" / "ultralytics" / "uly_driver.py"
DEFAULT_ULY_WEIGHTS_DIR = _REPO_ROOT / "drivers" / "ultralytics" / "weights"

_ULY_REQUIRED_RESULT_KEYS = (
    "driver_version",
    "ultralytics_version",
    "torch_version",
    "telemetry",
    "model",
    "per_image",
)


def _resolve_uly_device(device: str) -> str:
    """Map the harness device arg to the driver's device string."""
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    if device.startswith("cuda:") or device == "cpu":
        return device
    raise ValueError(
        f"Unsupported device for the ultralytics driver: {device!r}. "
        f"Use 'auto', 'cuda', 'cuda:N', or 'cpu'."
    )


def _resolve_uly_weights(spec: ModelSpec, weights_dir: str | Path | None) -> Path:
    """Return the pre-downloaded official .pt for this spec.

    Looks in ``weights_dir`` if given, else in the default driver weights
    dir (drivers/ultralytics/weights/). We never let the driver download
    weights at benchmark time — campaigns run offline.
    """
    base = Path(weights_dir) if weights_dir is not None else DEFAULT_ULY_WEIGHTS_DIR
    weights_path = base / spec.weight_file
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Ultralytics weights not found for {spec.key}: expected {weights_path}. "
            f"Pre-download the official {spec.weight_file} (GitHub releases) into "
            f"that directory, or pass --weights-dir."
        )
    return weights_path.resolve()


def _run_uly_driver(
    uly_python: Path,
    driver: Path,
    run_config: dict[str, Any],
    spec: ModelSpec,
    verbose: bool,
) -> dict[str, Any]:
    """Write the run config, invoke the driver, parse and return its result."""
    with tempfile.TemporaryDirectory(prefix="va_uly_") as tmp:
        cfg_path = Path(tmp) / "run_config.json"
        out_path = Path(tmp) / "driver_result.json"
        cfg_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        cmd = [
            str(uly_python), str(driver),
            "--config", str(cfg_path),
            "--output", str(out_path),
        ]
        if verbose:
            print("\nInvoking ultralytics driver:")
            print(f"  {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        stderr_tail = (proc.stderr or "").strip()[-4000:]

        if proc.returncode != 0:
            raise RuntimeError(
                f"Ultralytics driver failed for {spec.key} "
                f"(exit code {proc.returncode}).\n"
                f"Command: {' '.join(cmd)}\n"
                f"--- driver stderr ---\n{stderr_tail or '<empty>'}"
            )
        if not out_path.exists():
            raise RuntimeError(
                f"Ultralytics driver for {spec.key} exited 0 but wrote no result "
                f"JSON at {out_path}.\n--- driver stderr ---\n{stderr_tail or '<empty>'}"
            )
        try:
            driver_result = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ultralytics driver for {spec.key} emitted invalid JSON: {exc}.\n"
                f"--- driver stderr ---\n{stderr_tail or '<empty>'}"
            ) from exc

    missing = [k for k in _ULY_REQUIRED_RESULT_KEYS if k not in driver_result]
    if missing:
        raise RuntimeError(
            f"Ultralytics driver result for {spec.key} is missing required keys: "
            f"{', '.join(missing)}. Driver/harness contract mismatch "
            f"(see PLAN_ultralytics_source.md)."
        )

    telemetry = driver_result.get("telemetry") or {}
    if telemetry.get("settings_sync") is not False:
        raise RuntimeError(
            f"Ultralytics driver for {spec.key} did not confirm telemetry is "
            f"disabled (telemetry.settings_sync must be false, got "
            f"{telemetry.get('settings_sync')!r}). Refusing to accept the run."
        )

    return driver_result


def run_ultralytics_benchmark(
    model_key: str,
    coco_dir: str | Path,
    uly_python: str | Path,
    weights_dir: str | Path | None = None,
    device: str = "auto",
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    limit: int | None = None,
    verbose: bool = True,
    driver_script: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark an ultralytics-source model on COCO val2017 via the driver.

    Args:
        model_key: Registry key with source "ultralytics" (e.g. "uly-yolo11n").
        coco_dir: Path to the COCO directory (same layout as benchmark_model).
        uly_python: Python executable of the driver venv (.venv-ultralytics).
        weights_dir: Directory with the pre-downloaded official .pt files.
            Defaults to drivers/ultralytics/weights/.
        device: "auto", "cuda", "cuda:N", or "cpu".
        conf / iou / max_det: Eval protocol (defaults: 0.001 / 0.7 / 300).
        limit: Evaluate only the first N images (subset run, not submittable).
        verbose: Print progress (driver stdout streams through when set).
        driver_script: Override the driver script path (used by tests to
            substitute a fake driver).

    Returns:
        Dict matching the RawBenchmark schema for the website, with
        model.source = "ultralytics" and implementation/software recording
        the driver-reported ultralytics version.
    """
    coco_dir = Path(coco_dir)
    spec = get_spec(model_key)
    if spec.source != "ultralytics":
        raise ValueError(
            f"Model '{model_key}' has source '{spec.source}'; "
            f"run_ultralytics_benchmark only handles source 'ultralytics'."
        )

    uly_python = Path(uly_python)
    if not uly_python.exists():
        raise FileNotFoundError(
            f"Ultralytics driver python not found: {uly_python}. "
            f"Point --uly-python (or VA_ULY_PYTHON) at the .venv-ultralytics "
            f"python executable."
        )
    driver = Path(driver_script) if driver_script is not None else DEFAULT_ULY_DRIVER
    if not driver.exists():
        raise FileNotFoundError(f"Ultralytics driver script not found: {driver}")

    weights_path = _resolve_uly_weights(spec, weights_dir)
    device_str = _resolve_uly_device(device)
    warmup_iters = 10 if device_str.startswith("cuda") else 3

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking: {spec.display_name} ({spec.key}) [Ultralytics driver]")
        print(f"{'=' * 70}")
        print(f"  Weights: {weights_path}")
        print(f"  Device:  {device_str}")
        print(f"  Input size: {spec.input_size}")

    coco_gt, img_ids, img_dir = _load_coco(coco_dir, verbose)
    img_ids = _apply_limit(img_ids, limit, verbose)

    image_paths: list[str] = []
    expected_names: list[str] = []
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"COCO image missing on disk: {img_path}")
        image_paths.append(str(img_path.resolve()))
        expected_names.append(img_info["file_name"])

    run_config = {
        "weights": str(weights_path),
        "images": image_paths,
        "imgsz": spec.input_size,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "device": device_str,
        "warmup_iters": warmup_iters,
        "half": False,
    }

    driver_result = _run_uly_driver(uly_python, driver, run_config, spec, verbose)

    per_image = driver_result["per_image"]
    if len(per_image) != len(img_ids):
        raise RuntimeError(
            f"Ultralytics driver returned {len(per_image)} per_image entries "
            f"for {len(img_ids)} requested images ({spec.key})."
        )

    pre_times: list[float] = []
    inf_times: list[float] = []
    post_times: list[float] = []
    total_times: list[float] = []
    predictions: list[dict] = []

    for img_id, expected_name, entry in zip(img_ids, expected_names, per_image):
        got_name = Path(str(entry["image"])).name
        if got_name != Path(expected_name).name:
            raise RuntimeError(
                f"Ultralytics driver result out of order: expected "
                f"{expected_name}, got {got_name} ({spec.key})."
            )

        speed = entry["speed_ms"]
        pre_times.append(float(speed["preprocess"]))
        inf_times.append(float(speed["inference"]))
        post_times.append(float(speed["postprocess"]))
        total_times.append(float(entry["wall_ms"]))

        detections = entry.get("detections") or []
        if detections:
            boxes = np.array(
                [
                    [
                        d["bbox_xywh"][0],
                        d["bbox_xywh"][1],
                        d["bbox_xywh"][0] + d["bbox_xywh"][2],
                        d["bbox_xywh"][1] + d["bbox_xywh"][3],
                    ]
                    for d in detections
                ],
                dtype=np.float64,
            )
            scores = np.array([d["score"] for d in detections], dtype=np.float64)
            classes = np.array([d["class_index"] for d in detections], dtype=np.int64)
            _append_predictions(predictions, boxes, scores, classes, img_id)

    pre_arr = np.array(pre_times)
    inf_arr = np.array(inf_times)
    post_arr = np.array(post_times)
    total_arr = np.array(total_times)

    total_stats = compute_stats(total_arr)
    fps_mean = 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / total_stats["p50"] if total_stats["p50"] > 0 else 0.0

    peak_vram_raw = driver_result.get("peak_vram_mb")
    peak_vram_mb = round(float(peak_vram_raw), 1) if peak_vram_raw is not None else None
    peak_ram_mb = round(float(driver_result.get("peak_ram_mb") or 0.0), 1)

    _print_timing_summary(
        verbose, len(img_ids), pre_arr, inf_arr, post_arr, total_stats,
        fps_mean, fps_p50, peak_vram_mb, peak_ram_mb,
    )

    if verbose:
        print(f"\nCOCO evaluation ({len(predictions)} detections)...")
    coco_metrics = evaluate_coco(coco_gt, predictions, image_ids=img_ids)

    if verbose:
        _print_accuracy(coco_metrics)

    uly_version = str(driver_result["ultralytics_version"])
    measured_params = float(driver_result["model"].get("params_millions") or 0.0)

    hw_sw = collect_hw()
    is_cuda = device_str.startswith("cuda")
    return assemble_result(
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
        peak_vram_mb=peak_vram_mb,
        peak_ram_mb=peak_ram_mb,
        device_type="gpu" if is_cuda else "cpu",
        provider="cuda" if is_cuda else "cpu",
        hardware=hw_sw["hardware"],
        software=hw_sw["software"],
        actual_input_size=spec.input_size,
        conf=conf,
        iou=iou,
        max_det=max_det,
        fmt="pytorch",
        precision="fp32",
        source="ultralytics",
        impl_provider="ultralytics",
        impl_version=uly_version,
        extra_software={
            "ultralytics": uly_version,
            "ultralytics_torch": str(driver_result["torch_version"]),
        },
    )


def _print_accuracy(coco_metrics: dict[str, float]) -> None:
    print(f"  mAP@50-95: {coco_metrics['mAP']:.4f}")
    print(f"  mAP@50:    {coco_metrics['mAP50']:.4f}")
    print(f"  mAP@75:    {coco_metrics['mAP75']:.4f}")
    print(f"  mAP_small: {coco_metrics['mAP_small']:.4f}")
    print(f"  mAP_med:   {coco_metrics['mAP_medium']:.4f}")
    print(f"  mAP_large: {coco_metrics['mAP_large']:.4f}")
