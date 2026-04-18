"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from copy import deepcopy


def make_result(
    name: str,
    gpu: str,
    mAP_50_95: float,
    mAP_50: float,
    mAP_small: float,
    mean_ms: float,
    gflops: float,
    fmt: str = "pytorch",
) -> dict:
    """Build a single benchmark result dict matching the current output schema."""
    return deepcopy({
        "model": {
            "name": name,
            "family": "yolox",
            "variant": "s",
            "source": "libreyolo",
            "weights": f"{name}.pt",
            "input_size": 640,
        },
        "hardware": {
            "gpu": gpu,
            "gpu_memory_gb": 16.0,
            "driver_version": "test",
            "cuda_version": "12",
            "cpu": "test",
            "cpu_cores": 8,
            "ram_gb": 32,
        },
        "software": {
            "python": "3.14",
            "torch": "2.11",
            "libreyolo": "1.0.0",
            "onnxruntime": "not-installed",
        },
        "accuracy": {
            "mAP_50_95": mAP_50_95,
            "mAP_50": mAP_50,
            "mAP_75": 0.5,
            "mAP_small": mAP_small,
            "mAP_medium": 0.6,
            "mAP_large": 0.7,
            "AR1": 0.1,
            "AR10": 0.2,
            "AR100": 0.3,
            "AR_small": 0.1,
            "AR_medium": 0.2,
            "AR_large": 0.3,
        },
        "timing": {
            "batch_size": 1,
            "num_images": 5000,
            "total_ms": {
                "mean": mean_ms,
                "std": 1.0,
                "p50": mean_ms,
                "p95": mean_ms * 1.1,
                "p99": mean_ms * 1.2,
                "preprocess_ms": 1.0,
                "inference_ms": mean_ms - 2.0,
                "postprocess_ms": 1.0,
            },
        },
        "throughput": {
            "fps_mean": 1000.0 / mean_ms,
            "fps_p50": 1000.0 / mean_ms,
        },
        "model_stats": {
            "params_millions": 9.0,
            "gflops": gflops,
        },
        "memory": {
            "peak_vram_mb": 100.0,
            "peak_ram_mb": 500.0,
        },
        "metadata": {
            "benchmark_date": "2026-04-18",
            "benchmark_version": "2.0",
        },
        "eval": {
            "dataset": "coco",
            "split": "val2017",
            "numImages": 5000,
        },
        "implementation": {"provider": "libreyolo", "version": "1.0.0"},
        "runtime": {
            "format": fmt,
            "precision": "fp32",
            "device": "gpu",
        },
    })
