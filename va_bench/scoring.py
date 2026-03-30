"""
VA v1 Score computation.

The VA v1 Score is a compound index (0-100) ranking object detection models
across 6 metrics. Each metric is min-max normalized across all qualifying
models, then averaged. A model must have benchmarks on both RTX 5080 and
Raspberry Pi 5 to receive a score.

Metrics:
    1. mAP@50          — detection quality
    2. mAP@50-95       — localization precision
    3. mAP_small        — small object detection (<32x32 px)
    4. FPS (RTX 5080)   — desktop GPU throughput
    5. FPS (RPi5)       — edge device throughput
    6. mAP/GFLOP        — compute efficiency
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCORE_METRICS = [
    "mAP_50",
    "mAP_50_95",
    "mAP_small",
    "fps_rtx5080",
    "fps_rpi5",
    "mAP_per_gflop",
]


def _detect_hardware(result: dict) -> str | None:
    """Detect hardware ID from a benchmark result."""
    gpu = result.get("hardware", {}).get("gpu", "").lower()
    if "5080" in gpu:
        return "rtx5080"
    if "raspberry" in gpu or "rpi" in gpu:
        return "rpi5"
    return None


def _extract_metrics(result: dict) -> dict[str, float]:
    """Extract the raw metric values from a single benchmark result."""
    acc = result.get("accuracy", {})
    timing = result.get("timing", {})
    total_ms = timing.get("total_ms", {})
    stats = result.get("model_stats", {})

    mAP_50_95_raw = acc.get("mAP_50_95", 0.0)
    # Convert to percentage if in decimal form
    mAP_50_95 = mAP_50_95_raw * 100 if mAP_50_95_raw < 1 else mAP_50_95_raw
    mAP_50_raw = acc.get("mAP_50", 0.0)
    mAP_50 = mAP_50_raw * 100 if mAP_50_raw < 1 else mAP_50_raw
    mAP_small_raw = acc.get("mAP_small", 0.0)
    mAP_small = mAP_small_raw * 100 if mAP_small_raw < 1 else mAP_small_raw

    mean_ms = total_ms.get("mean", 0.0) if isinstance(total_ms, dict) else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    gflops = stats.get("gflops", 0.0)
    mAP_per_gflop = mAP_50_95 / gflops if gflops > 0 else 0.0

    return {
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "mAP_small": mAP_small,
        "fps": fps,
        "mAP_per_gflop": mAP_per_gflop,
    }


def _get_model_id(result: dict) -> str:
    """Extract a consistent model ID from a benchmark result."""
    model = result.get("model", {})
    if isinstance(model, dict):
        return model.get("name", "unknown").lower()
    return str(model).lower()


def load_results(results_dir: str | Path) -> list[dict]:
    """Load all benchmark JSON files from a directory."""
    results_dir = Path(results_dir)
    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping {f.name}: {e}")
    return results


def compute_va_v1_scores(
    results: list[dict],
) -> dict[str, Any]:
    """Compute VA v1 Scores from a collection of benchmark results.

    Returns a dict with:
        - va_v1_scores: {model_id: {composite, components}}
        - normalization: {metric: {min, max}}
        - skipped: [model_ids without both hardware]
    """
    # Group by model, then by hardware
    by_model: dict[str, dict[str, dict]] = {}
    for r in results:
        model_id = _get_model_id(r)
        hw = _detect_hardware(r)
        if hw is None:
            continue
        if model_id not in by_model:
            by_model[model_id] = {}
        by_model[model_id][hw] = r

    # Build per-model raw metric vectors (only models with both RTX 5080 and RPi5)
    qualifying: dict[str, dict[str, float]] = {}
    skipped: list[str] = []

    for model_id, hw_results in by_model.items():
        if "rtx5080" not in hw_results or "rpi5" not in hw_results:
            skipped.append(model_id)
            continue

        gpu_metrics = _extract_metrics(hw_results["rtx5080"])
        rpi5_metrics = _extract_metrics(hw_results["rpi5"])

        qualifying[model_id] = {
            "mAP_50": gpu_metrics["mAP_50"],
            "mAP_50_95": gpu_metrics["mAP_50_95"],
            "mAP_small": gpu_metrics["mAP_small"],
            "fps_rtx5080": gpu_metrics["fps"],
            "fps_rpi5": rpi5_metrics["fps"],
            "mAP_per_gflop": gpu_metrics["mAP_per_gflop"],
        }

    if not qualifying:
        return {"va_v1_scores": {}, "normalization": {}, "skipped": skipped}

    # Min-max normalization per metric
    norm_ranges: dict[str, dict[str, float]] = {}
    for metric in SCORE_METRICS:
        values = [q[metric] for q in qualifying.values()]
        mn, mx = min(values), max(values)
        norm_ranges[metric] = {"min": round(mn, 4), "max": round(mx, 4)}

    # Compute normalized scores
    va_scores: dict[str, dict[str, Any]] = {}
    for model_id, raw in qualifying.items():
        components: dict[str, dict[str, float]] = {}
        normalized_sum = 0.0

        for metric in SCORE_METRICS:
            mn = norm_ranges[metric]["min"]
            mx = norm_ranges[metric]["max"]
            raw_val = raw[metric]
            norm_val = (raw_val - mn) / (mx - mn) if mx > mn else 0.5
            components[metric] = {
                "raw": round(raw_val, 4),
                "normalized": round(norm_val, 4),
            }
            normalized_sum += norm_val

        composite = round(normalized_sum / len(SCORE_METRICS) * 100, 0)
        va_scores[model_id] = {
            "composite": int(composite),
            "components": components,
        }

    # Sort by composite score descending
    va_scores = dict(sorted(va_scores.items(), key=lambda x: x[1]["composite"], reverse=True))

    return {
        "va_v1_scores": va_scores,
        "normalization": norm_ranges,
        "skipped": skipped,
    }


def save_scores(scores: dict[str, Any], output_path: str | Path) -> Path:
    """Save VA v1 Scores to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    return output_path
