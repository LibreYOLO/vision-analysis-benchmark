"""
Output assembly for Vision Analysis benchmarks.

Produces JSON matching the RawBenchmark schema from the website's transform.ts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .models import ModelSpec


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat_z(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def detect_hardware_id(hardware: dict[str, Any]) -> str:
    """Return a stable hardware slug from the benchmarked machine."""
    gpu = str(hardware.get("gpu", "")).lower()
    cpu = str(hardware.get("cpu", "")).lower()

    if "5080" in gpu:
        return "rtx5080"
    if "4090" in gpu:
        return "rtx4090"
    if "3090" in gpu:
        return "rtx3090"
    if "a100" in gpu:
        return "a100"
    if "dgx spark" in gpu or "gb10" in gpu:
        return "dgx_spark"
    if "jetson orin" in gpu or "orin" in gpu:
        return "jetson_orin"
    if "raspberry pi 5" in gpu or "raspberry pi 5" in cpu:
        return "rpi5"
    if "apple" in gpu or "m1" in gpu or "m2" in gpu or "m3" in gpu or "m4" in gpu:
        return "mac"
    if gpu and gpu != "cpu":
        return _slugify(gpu)
    if cpu:
        return _slugify(cpu)
    return "unknown"


def _accuracy_block(
    coco_metrics: dict[str, float],
    mask_metrics: dict[str, float] | None,
) -> dict[str, float]:
    """Build the submission accuracy block.

    The unprefixed keys always carry the task's headline metric: box mAP for
    detection, mask mAP for instance segmentation. Seg entries additionally
    record their box mAP under bbox_-prefixed keys, so nothing is discarded
    and every downstream consumer (leaderboard, pareto, badges) can keep
    reading the unprefixed keys without knowing about tasks.
    """
    headline = mask_metrics if mask_metrics is not None else coco_metrics
    block = {
        "mAP_50_95": headline["mAP"],
        "mAP_50": headline["mAP50"],
        "mAP_75": headline["mAP75"],
        "mAP_small": headline["mAP_small"],
        "mAP_medium": headline["mAP_medium"],
        "mAP_large": headline["mAP_large"],
        "AR1": headline["AR1"],
        "AR10": headline["AR10"],
        "AR100": headline["AR100"],
        "AR_small": headline["AR_small"],
        "AR_medium": headline["AR_medium"],
        "AR_large": headline["AR_large"],
    }
    if mask_metrics is not None:
        block.update({
            "bbox_mAP_50_95": coco_metrics["mAP"],
            "bbox_mAP_50": coco_metrics["mAP50"],
            "bbox_mAP_75": coco_metrics["mAP75"],
            "bbox_mAP_small": coco_metrics["mAP_small"],
            "bbox_mAP_medium": coco_metrics["mAP_medium"],
            "bbox_mAP_large": coco_metrics["mAP_large"],
        })
    return block


def assemble_result(
    spec: ModelSpec,
    coco_metrics: dict[str, float],
    total_stats: dict[str, float],
    preprocess_ms: float | None,
    inference_ms: float | None,
    postprocess_ms: float | None,
    fps_mean: float,
    fps_p50: float,
    num_images: int,
    measured_params_m: float,
    peak_vram_mb: float | None,
    peak_ram_mb: float,
    device_type: str,
    provider: str,
    hardware: dict[str, Any],
    software: dict[str, str],
    actual_input_size: int,
    conf: float,
    iou: float,
    max_det: int,
    fmt: str = "pytorch",
    mask_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the final result dict matching the website's RawBenchmark schema.

    For segment-task models pass ``mask_metrics`` (COCO segm eval); it becomes
    the headline accuracy and ``coco_metrics`` (box eval) is kept under
    bbox_-prefixed keys.
    """
    if (spec.task == "segment") != (mask_metrics is not None):
        raise ValueError(
            f"mask_metrics must be provided exactly when spec.task == 'segment' "
            f"(got task={spec.task!r}, mask_metrics={'set' if mask_metrics is not None else 'None'})"
        )
    gflops = spec.paper_flops_g if spec.paper_flops_g > 0 else 0.0
    params_m = measured_params_m if measured_params_m > 0 else spec.paper_params_m
    now = _utc_now()
    created_at = _isoformat_z(now)
    hardware_id = detect_hardware_id(hardware)

    return {
        "schema_version": "va.submission.v1",
        "submission_id": f"{spec.key}-{fmt}-{provider}-{hardware_id}-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": created_at,
        "benchmark": {
            "harness": "vision-analysis-benchmark",
            "harness_version": __version__,
            "libreyolo_version": software.get("libreyolo", "unknown"),
            "libreyolo_commit": software.get("libreyolo_commit", "unknown"),
        },
        "model": {
            "id": spec.key,
            "name": spec.display_name.lower().replace(" ", ""),
            "family": spec.family,
            "variant": spec.variant,
            "task": spec.task,
            "source": "libreyolo",
            "weights": spec.weight_file,
            "input_size": actual_input_size,
        },
        "dataset": {
            "id": "coco2017",
            "split": "val2017",
            "num_images": num_images,
        },
        "config": {
            "batch_size": 1,
            "input_size": actual_input_size,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
        },
        "hardware": {
            **hardware,
            "id": hardware_id,
        },
        "software": software,
        "accuracy": _accuracy_block(coco_metrics, mask_metrics),
        "timing": {
            "batch_size": 1,
            "num_images": num_images,
            "total_ms": {
                **total_stats,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
            },
        },
        "throughput": {
            "fps_mean": fps_mean,
            "fps_p50": fps_p50,
        },
        "model_stats": {
            "params_millions": params_m,
            "gflops": gflops,
        },
        "memory": {
            "peak_vram_mb": peak_vram_mb,
            "peak_ram_mb": peak_ram_mb,
        },
        "metadata": {
            "benchmark_date": created_at,
            "benchmark_version": __version__,
        },
        "eval": {
            "dataset": "coco",
            "split": "val2017",
            "numImages": num_images,
        },
        "implementation": {
            "provider": "libreyolo",
            "version": software.get("libreyolo", "unknown"),
        },
        "runtime": {
            "format": fmt,
            "precision": "fp32",
            "provider": provider,
            "device": device_type,
        },
    }


def save_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Save benchmark result to JSON file.

    Filename: {model_id}__{backend}__{provider}__{hardware_id}__{timestamp}.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = result["model"].get("id") or result["model"]["name"]
    fmt = result.get("runtime", {}).get("format", "pytorch")
    provider = result.get("runtime", {}).get("provider", "unknown")
    hardware_id = result.get("hardware", {}).get("id") or detect_hardware_id(result["hardware"])
    timestamp = result.get("created_at", "")
    timestamp_slug = (
        timestamp.replace("-", "").replace(":", "").replace(".", "").replace("+0000", "Z")
        .replace("+00:00", "Z").replace("T", "T").replace("Z", "Z")
    )
    timestamp_slug = timestamp_slug.rstrip("Z") + "Z" if timestamp_slug else "unknown"
    filename = f"{model_id}__{fmt}__{provider}__{hardware_id}__{timestamp_slug}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    return filepath
