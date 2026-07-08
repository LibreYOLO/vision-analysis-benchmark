"""
RF100-VL benchmark support for Vision Analysis.

Roboflow100-VL (https://rf100-vl.org, NeurIPS 2025 D&B) is 100 object
detection datasets across 7 domains, distributed in COCO JSON format under
Apache 2.0. Unlike COCO val2017, the headline regime is *fine-tuned*: a model
is trained per dataset, evaluated on that dataset's test split, and the
published score is the mean AP50 / AP50:95 across all datasets.

This module implements the evaluation half of that protocol:

* ``download_datasets``  — fetch RF100-VL (or the RF20-VL subset) via the
  ``rf100vl`` pip package into a local directory, one sub-folder per dataset.
* ``benchmark_model_rf100vl`` — run one registered model over every dataset's
  chosen split, score each dataset independently with pycocotools, and emit a
  single submission JSON whose accuracy block is the across-dataset mean plus
  a per-dataset breakdown under ``rf100vl``.

Fine-tuned checkpoints are supplied per dataset via ``--weights-root``:

    <weights_root>/<dataset_name>/<spec.weight_file>          (pytorch)
    <weights_root>/<dataset_name>/<spec weight stem>.onnx     (onnx)

Running COCO-pretrained weights across RF100-VL is meaningless for closed-
vocabulary detectors (the class spaces differ), so it is refused unless
``allow_pretrained=True`` — that escape hatch exists for smoke tests and for
future open-vocabulary models whose class space is prompt-defined.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .coco_eval import evaluate_coco
from .hardware import collect_all as collect_hw, get_runtime_device_name
from .models import ModelSpec, get_spec
from .output import assemble_result
from .timing import compute_stats

DATASET_ID = "rf100_vl"
ANNOTATION_FILENAME = "_annotations.coco.json"

# Metrics averaged across datasets (keys of coco_eval.evaluate_coco output).
_METRIC_KEYS = (
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
)


# =============================================================================
# Dataset acquisition / discovery
# =============================================================================

def download_datasets(
    data_dir: str | Path,
    subset: str = "rf100vl",
    verbose: bool = True,
) -> Path:
    """Download RF100-VL datasets with the ``rf100vl`` package.

    Requires ``pip install rf100vl`` and the ``ROBOFLOW_API_KEY`` environment
    variable (free Roboflow Universe key). Datasets land in one sub-folder
    each, in COCO JSON format.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import rf100vl
    except ImportError as exc:
        raise RuntimeError(
            "The 'rf100vl' package is required to download RF100-VL. "
            "Install with: pip install rf100vl"
        ) from exc

    downloaders = {
        "rf100vl": "download_rf100vl",
        "rf20vl": "download_rf20vl_full",
        "rf100vl-fsod": "download_rf100vl_fsod",
        "rf20vl-fsod": "download_rf20vl_fsod",
    }
    if subset not in downloaders:
        raise ValueError(f"Unknown subset {subset!r}. Options: {sorted(downloaders)}")

    fn = getattr(rf100vl, downloaders[subset], None)
    if fn is None:
        raise RuntimeError(
            f"Installed rf100vl package has no {downloaders[subset]}(); "
            "upgrade with: pip install -U rf100vl"
        )

    if verbose:
        print(f"Downloading {subset} to {data_dir} (skips already-present datasets)...")
    fn(path=str(data_dir))
    return data_dir


def discover_datasets(data_dir: str | Path, split: str = "test") -> list[Path]:
    """Return sorted dataset directories containing ``<split>/_annotations.coco.json``."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"RF100-VL data dir not found: {data_dir}. "
            "Download first with: va-bench rf100vl --download"
        )
    found = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / split / ANNOTATION_FILENAME).exists()
    )
    if not found:
        raise FileNotFoundError(
            f"No datasets with {split}/{ANNOTATION_FILENAME} under {data_dir}."
        )
    return found


def load_dataset_split(dataset_dir: Path, split: str, verbose: bool = False):
    """Load one RF100-VL dataset split as (coco_gt, img_ids, img_dir).

    Roboflow COCO exports keep images and ``_annotations.coco.json`` in the
    same split directory.
    """
    from pycocotools.coco import COCO
    import contextlib
    import io

    ann_file = dataset_dir / split / ANNOTATION_FILENAME
    img_dir = dataset_dir / split

    if verbose:
        coco_gt = COCO(str(ann_file))
    else:
        # pycocotools prints unconditionally; keep per-dataset output quiet.
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(ann_file))
    img_ids = sorted(coco_gt.getImgIds())
    return coco_gt, img_ids, img_dir


def category_mapping(coco_gt: Any) -> list[int]:
    """Model class index -> ground-truth category id, by ascending category id.

    LibreYOLO training on a Roboflow export enumerates classes in ascending
    category-id order, so index i corresponds to the i-th smallest category id.
    (Roboflow COCO exports usually reserve id 0 for a placeholder
    supercategory; when present it has no annotations and simply occupies
    index 0 on both sides, keeping the mapping aligned.)
    """
    return sorted(coco_gt.getCatIds())


# =============================================================================
# Per-dataset weight resolution
# =============================================================================

def resolve_finetuned_weights(
    spec: ModelSpec,
    weights_root: str | Path,
    dataset_name: str,
    fmt: str,
) -> Path | None:
    """Path to this model's fine-tuned checkpoint for one dataset, or None."""
    stem = Path(spec.weight_file).stem
    dataset_dir = Path(weights_root) / dataset_name
    if fmt == "pytorch":
        candidates = [dataset_dir / spec.weight_file, dataset_dir / f"{stem}.pt"]
    elif fmt == "onnx":
        candidates = [dataset_dir / f"{stem}.onnx"]
    else:
        raise ValueError(f"Unsupported RF100-VL format: {fmt!r} (use pytorch or onnx)")
    for c in candidates:
        if c.exists():
            return c
    return None


# =============================================================================
# Core loop
# =============================================================================

def _predictions_for_split(
    predict: Callable[[Image.Image], tuple[Any, Any, Any]],
    coco_gt: Any,
    img_ids: list[int],
    img_dir: Path,
    cat_ids: list[int],
    desc: str,
    verbose: bool,
) -> tuple[list[dict], list[float]]:
    """Run ``predict`` over a split; return COCO predictions + per-image ms."""
    predictions: list[dict] = []
    total_times: list[float] = []

    pbar = tqdm(img_ids, desc=desc, disable=not verbose, leave=False)
    for img_id in pbar:
        img_info = coco_gt.loadImgs(img_id)[0]
        pil_img = Image.open(img_dir / img_info["file_name"]).convert("RGB")

        t0 = time.perf_counter()
        boxes, scores, classes = predict(pil_img)
        total_times.append((time.perf_counter() - t0) * 1000.0)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = (float(v) for v in box)
            cls_int = int(cls)
            cat_id = cat_ids[cls_int] if cls_int < len(cat_ids) else cls_int
            predictions.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    return predictions, total_times


def _make_pytorch_predict(model: Any, conf: float, iou: float, max_det: int):
    import torch

    imgsz = model._get_input_size()

    def predict(pil_img: Image.Image):
        input_tensor, _orig, original_size, ratio = model._preprocess(
            pil_img, "rgb", input_size=imgsz,
        )
        input_tensor = input_tensor.to(model.device)
        with torch.no_grad():
            output = model._forward(input_tensor)
        det = model._postprocess(
            output, conf, iou, original_size, max_det=max_det, ratio=ratio,
        )
        if det["num_detections"] == 0:
            return [], [], []
        def _np(v):
            return v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        return _np(det["boxes"]), _np(det["scores"]), _np(det["classes"])

    return predict, imgsz


def _make_onnx_predict(backend: Any, conf: float, iou: float, max_det: int):
    def predict(pil_img: Image.Image):
        result = backend.predict(
            pil_img, conf=conf, iou=iou, max_det=max_det, color_format="rgb",
        )
        b = result.boxes
        if len(b.xyxy) == 0:
            return [], [], []
        return b.xyxy, b.conf, b.cls
    return predict, backend.imgsz


def _load_for_dataset(
    spec: ModelSpec,
    fmt: str,
    weights_path: Path | None,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
):
    """Load the model/backend for one dataset; return (predict_fn, imgsz, params_m, device_str)."""
    from libreyolo import LibreYOLO

    if fmt == "pytorch":
        model = LibreYOLO(
            model_path=str(weights_path) if weights_path else spec.weight_file,
            size=spec.constructor_size,
            device=device,
        )
        predict, imgsz = _make_pytorch_predict(model, conf, iou, max_det)
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
        return predict, imgsz, params_m, str(model.device)

    if fmt == "onnx":
        if weights_path is None:
            raise ValueError("RF100-VL onnx runs need per-dataset .onnx weights")
        backend = LibreYOLO(model_path=str(weights_path), device=device)
        predict, imgsz = _make_onnx_predict(backend, conf, iou, max_det)
        return predict, imgsz, 0.0, str(backend.device)

    raise ValueError(f"Unsupported RF100-VL format: {fmt!r} (use pytorch or onnx)")


def _aggregate_metrics(per_dataset: list[dict[str, float]]) -> dict[str, float]:
    """Unweighted mean of each COCO metric across datasets (RF100-VL protocol)."""
    if not per_dataset:
        raise ValueError("No datasets were evaluated; cannot aggregate.")
    return {
        key: float(np.mean([m[key] for m in per_dataset]))
        for key in _METRIC_KEYS
    }


def benchmark_model_rf100vl(
    model_key: str,
    data_dir: str | Path,
    fmt: str = "pytorch",
    weights_root: str | Path | None = None,
    device: str = "auto",
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 300,
    split: str = "test",
    limit: int | None = None,
    limit_datasets: int | None = None,
    allow_pretrained: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate one model across all RF100-VL datasets and aggregate.

    Args:
        model_key: Registry key (e.g. "yolov9t").
        data_dir: Directory of RF100-VL datasets (one sub-folder each).
        fmt: "pytorch" or "onnx".
        weights_root: Root of per-dataset fine-tuned checkpoints
            (``<weights_root>/<dataset>/<weight file>``). Datasets without a
            checkpoint are skipped and listed in the result.
        split: Which split to score ("test" per the RF100-VL protocol).
        limit: Max images per dataset (smoke runs only; not submittable).
        limit_datasets: Evaluate only the first N datasets (smoke runs only).
        allow_pretrained: Permit registry COCO weights when weights_root is
            not given. Scores are meaningless for closed-vocab models; exists
            for smoke tests and open-vocabulary models.

    Returns:
        Submission dict: schema va.submission.v1 with dataset id "rf100_vl",
        accuracy = across-dataset means, and an ``rf100vl`` breakdown block.
    """
    spec = get_spec(model_key)
    if weights_root is None and not allow_pretrained:
        raise ValueError(
            "RF100-VL is a fine-tuned benchmark: pass weights_root with one "
            "checkpoint per dataset, or set allow_pretrained=True to force "
            "COCO-pretrained weights (smoke tests / open-vocab only)."
        )

    dataset_dirs = discover_datasets(data_dir, split=split)
    if limit_datasets is not None:
        dataset_dirs = dataset_dirs[:limit_datasets]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"RF100-VL: {spec.display_name} ({spec.key}) [{fmt}] "
              f"— {len(dataset_dirs)} datasets, split={split}")
        print(f"{'=' * 70}")

    per_dataset_metrics: list[dict[str, float]] = []
    per_dataset_records: list[dict[str, Any]] = []
    skipped: list[str] = []
    all_times: list[float] = []
    total_images = 0
    imgsz = spec.input_size
    params_m = 0.0
    provider = "cpu"

    for i, dataset_dir in enumerate(dataset_dirs):
        name = dataset_dir.name

        weights_path = None
        if weights_root is not None:
            weights_path = resolve_finetuned_weights(spec, weights_root, name, fmt)
            if weights_path is None:
                skipped.append(name)
                if verbose:
                    print(f"  [{i + 1}/{len(dataset_dirs)}] {name}: no checkpoint, skipped")
                continue

        coco_gt, img_ids, img_dir = load_dataset_split(dataset_dir, split)
        if limit is not None:
            img_ids = img_ids[:limit]
        cat_ids = category_mapping(coco_gt)

        predict, imgsz, params_m, provider = _load_for_dataset(
            spec, fmt, weights_path, device, conf, iou, max_det,
        )

        predictions, times = _predictions_for_split(
            predict, coco_gt, img_ids, img_dir, cat_ids,
            desc=f"[{i + 1}/{len(dataset_dirs)}] {name}", verbose=verbose,
        )
        metrics = evaluate_coco(coco_gt, predictions, image_ids=img_ids)

        per_dataset_metrics.append(metrics)
        per_dataset_records.append({
            "dataset": name,
            "num_images": len(img_ids),
            "num_classes": len(cat_ids),
            "weights": str(weights_path) if weights_path else spec.weight_file,
            "mAP_50": metrics["mAP50"],
            "mAP_50_95": metrics["mAP"],
        })
        all_times.extend(times)
        total_images += len(img_ids)

        if verbose:
            print(f"  [{i + 1}/{len(dataset_dirs)}] {name}: "
                  f"AP50={metrics['mAP50']:.4f} AP50:95={metrics['mAP']:.4f} "
                  f"({len(img_ids)} images)")

    mean_metrics = _aggregate_metrics(per_dataset_metrics)
    total_stats = compute_stats(np.array(all_times))
    fps_mean = 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / total_stats["p50"] if total_stats["p50"] > 0 else 0.0

    if verbose:
        print(f"\nRF100-VL mean over {len(per_dataset_metrics)} datasets:")
        print(f"  AP50:    {mean_metrics['mAP50']:.4f}")
        print(f"  AP50:95: {mean_metrics['mAP']:.4f}")
        if skipped:
            print(f"  Skipped (no checkpoint): {len(skipped)}")

    regime = "fine-tuned" if weights_root is not None else "pretrained-forced"
    if regime == "pretrained-forced":
        warnings.warn(
            "RF100-VL scored with COCO-pretrained weights: numbers are NOT "
            "comparable to published fine-tuned results and must not be "
            "submitted."
        )

    hw_sw = collect_hw()
    device_str = provider if provider in ("cuda", "cpu", "mps") else provider.split(":")[0]
    result = assemble_result(
        spec=spec,
        coco_metrics=mean_metrics,
        total_stats=total_stats,
        preprocess_ms=None,
        inference_ms=None,
        postprocess_ms=None,
        fps_mean=round(fps_mean, 2),
        fps_p50=round(fps_p50, 2),
        num_images=total_images,
        measured_params_m=round(params_m, 2),
        peak_vram_mb=None,
        peak_ram_mb=0.0,
        device_type=get_runtime_device_name(device_str),
        provider=device_str,
        hardware=hw_sw["hardware"],
        software=hw_sw["software"],
        actual_input_size=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        fmt=fmt,
    )

    # Re-stamp the COCO defaults from assemble_result with RF100-VL identity.
    result["submission_id"] = f"{spec.key}-{DATASET_ID}-" + result["submission_id"].split(f"{spec.key}-", 1)[1]
    result["dataset"] = {
        "id": DATASET_ID,
        "split": split,
        "num_images": total_images,
        "num_datasets": len(per_dataset_metrics),
    }
    result["eval"] = {
        "dataset": DATASET_ID,
        "split": split,
        "numImages": total_images,
    }
    result["rf100vl"] = {
        "regime": regime,
        "aggregation": "unweighted mean across datasets",
        "datasets": per_dataset_records,
        "skipped_datasets": skipped,
    }
    if limit is not None or limit_datasets is not None:
        result["rf100vl"]["subset_run"] = {
            "limit_images": limit,
            "limit_datasets": limit_datasets,
            "note": "SUBSET - not a valid RF100-VL submission",
        }
    return result
