"""
Output assembly for Vision Analysis benchmarks.

Produces JSON matching the RawBenchmark schema from the website's transform.ts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import ModelSpec


def assemble_result(
    spec: ModelSpec,
    coco_metrics: dict[str, float],
    total_stats: dict[str, float],
    preprocess_ms: float,
    inference_ms: float,
    postprocess_ms: float,
    fps_mean: float,
    fps_p50: float,
    num_images: int,
    measured_params_m: float,
    peak_vram_mb: float,
    peak_ram_mb: float,
    device_type: str,
    hardware: dict[str, Any],
    software: dict[str, str],
) -> dict[str, Any]:
    """Assemble the final result dict matching the website's RawBenchmark schema."""
    gflops = spec.paper_flops_g if spec.paper_flops_g > 0 else 0.0
    params_m = measured_params_m if measured_params_m > 0 else spec.paper_params_m

    return {
        "model": {
            "name": spec.display_name.lower().replace(" ", ""),
            "family": spec.family,
            "variant": spec.variant,
            "source": "libreyolo",
            "weights": spec.weight_file,
            "input_size": spec.input_size,
        },
        "hardware": hardware,
        "software": software,
        "accuracy": {
            "mAP_50_95": coco_metrics["mAP"],
            "mAP_50": coco_metrics["mAP50"],
            "mAP_75": coco_metrics["mAP75"],
            "mAP_small": coco_metrics["mAP_small"],
            "mAP_medium": coco_metrics["mAP_medium"],
            "mAP_large": coco_metrics["mAP_large"],
            "AR1": coco_metrics["AR1"],
            "AR10": coco_metrics["AR10"],
            "AR100": coco_metrics["AR100"],
            "AR_small": coco_metrics["AR_small"],
            "AR_medium": coco_metrics["AR_medium"],
            "AR_large": coco_metrics["AR_large"],
        },
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
            "benchmark_date": datetime.now().strftime("%Y-%m-%d"),
            "benchmark_version": "2.0",
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
            "format": "pytorch",
            "precision": "fp32",
            "device": device_type,
        },
    }


def save_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Save benchmark result to JSON file.

    Filename: {model_key}_{hardware_slug}.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = result["model"]["name"]
    hw = result["hardware"]["gpu"].lower()
    if "5080" in hw:
        hw_slug = "rtx5080"
    elif "raspberry" in hw or "rpi" in hw:
        hw_slug = "rpi5"
    elif "apple" in hw or "m1" in hw or "m2" in hw or "m3" in hw or "m4" in hw:
        hw_slug = "mac"
    else:
        hw_slug = hw.replace(" ", "_")[:20]

    filename = f"{model_name}_{hw_slug}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    return filepath
