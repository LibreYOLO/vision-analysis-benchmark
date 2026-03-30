"""
Timing utilities for Vision Analysis benchmarks.

Provides device-aware synchronization and statistics computation.
Uses sync() + time.perf_counter() which works correctly on CUDA, MPS, and CPU.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def device_sync(device: torch.device) -> None:
    """Synchronize the device to ensure all queued operations are complete."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def warmup(model: Any, device: torch.device, n_iters: int | None = None) -> None:
    """Run dummy forward passes to stabilize GPU kernels.

    Uses 10 iterations for GPU (CUDA/MPS) and 3 for CPU.
    """
    if n_iters is None:
        n_iters = 10 if device.type in ("cuda", "mps") else 3

    imgsz = model._get_input_size()
    dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device=device)

    if hasattr(model, "_original_size"):
        model._original_size = (imgsz, imgsz)

    model.model.eval()
    with torch.no_grad():
        for _ in range(n_iters):
            try:
                model._forward(dummy)
            except Exception:
                break

    if hasattr(model, "_original_size"):
        model._original_size = None

    device_sync(device)


def compute_stats(timings_ms: np.ndarray) -> dict[str, float]:
    """Compute timing statistics from per-image measurements.

    Returns dict with mean, std, p50, p95, p99.
    """
    return {
        "mean": round(float(np.mean(timings_ms)), 3),
        "std": round(float(np.std(timings_ms)), 3),
        "p50": round(float(np.percentile(timings_ms, 50)), 3),
        "p95": round(float(np.percentile(timings_ms, 95)), 3),
        "p99": round(float(np.percentile(timings_ms, 99)), 3),
    }


class SyncTimer:
    """Context-manager style timer that syncs the device before reading the clock.

    Usage:
        timer = SyncTimer(device)
        timer.mark()          # sync + record t0
        # ... preprocess ...
        timer.mark()          # sync + record t1
        # ... inference ...
        timer.mark()          # sync + record t2
        # ... postprocess ...
        timer.mark()          # sync + record t3
        phases = timer.phases_ms()  # [pre, inf, post]
        total = timer.total_ms()
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._marks: list[float] = []

    def reset(self) -> None:
        self._marks.clear()

    def mark(self) -> None:
        """Sync device and record a timestamp."""
        device_sync(self.device)
        self._marks.append(time.perf_counter())

    def phases_ms(self) -> list[float]:
        """Return durations in ms between consecutive marks."""
        return [
            (self._marks[i + 1] - self._marks[i]) * 1000.0
            for i in range(len(self._marks) - 1)
        ]

    def total_ms(self) -> float:
        """Return total duration from first to last mark in ms."""
        if len(self._marks) < 2:
            return 0.0
        return (self._marks[-1] - self._marks[0]) * 1000.0
