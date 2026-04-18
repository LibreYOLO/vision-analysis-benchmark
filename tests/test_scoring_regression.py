"""Regression test for VA v1 scoring against current-shape output JSON.

Guards against output.py's schema drifting in a way that silently breaks
scoring.py — which stays dormant while the `run` path gets all the love.
"""

from __future__ import annotations

from va_bench.scoring import compute_va_v1_scores

from .conftest import make_result


def test_three_models_two_hardware_produces_valid_scores():
    """Three models on both RTX 5080 and RPi5 should all qualify and get scored."""
    results = [
        make_result("yolox-s",  "NVIDIA GeForce RTX 5080", 0.41, 0.60, 0.25,   5.0,  13.5),
        make_result("yolox-s",  "Raspberry Pi 5",          0.41, 0.60, 0.25, 200.0,  13.5),
        make_result("yolov9t",  "NVIDIA GeForce RTX 5080", 0.38, 0.55, 0.20,   3.0,   4.0),
        make_result("yolov9t",  "Raspberry Pi 5",          0.38, 0.55, 0.20, 120.0,   4.0),
        make_result("rfdetr-l", "NVIDIA GeForce RTX 5080", 0.55, 0.70, 0.35,  12.0, 340.0),
        make_result("rfdetr-l", "Raspberry Pi 5",          0.55, 0.70, 0.35, 300.0, 340.0),
    ]

    out = compute_va_v1_scores(results)

    assert set(out["va_v1_scores"]) == {"yolox-s", "yolov9t", "rfdetr-l"}
    assert out["skipped"] == []
    for model_id, data in out["va_v1_scores"].items():
        assert 0 <= data["composite"] <= 100, model_id
        assert set(data["components"]) == {
            "mAP_50", "mAP_50_95", "mAP_small",
            "fps_rtx5080", "fps_rpi5", "mAP_per_gflop",
        }


def test_model_missing_second_hardware_is_skipped():
    results = [
        make_result("yolox-s",  "NVIDIA GeForce RTX 5080", 0.41, 0.60, 0.25, 5.0, 13.5),
        make_result("yolox-s",  "Raspberry Pi 5",          0.41, 0.60, 0.25, 200.0, 13.5),
        make_result("yolov9t",  "NVIDIA GeForce RTX 5080", 0.38, 0.55, 0.20, 3.0, 4.0),
    ]
    out = compute_va_v1_scores(results)
    assert "yolov9t" in out["skipped"]


def test_onnx_shaped_result_is_still_scored():
    """ONNX results (no phase split, null VRAM) must not break scoring."""
    onnx_result = make_result(
        "yolox-s", "NVIDIA GeForce RTX 5080", 0.41, 0.60, 0.25, 5.0, 13.5, fmt="onnx",
    )
    onnx_result["timing"]["total_ms"]["preprocess_ms"] = None
    onnx_result["timing"]["total_ms"]["inference_ms"] = None
    onnx_result["timing"]["total_ms"]["postprocess_ms"] = None
    onnx_result["memory"]["peak_vram_mb"] = None

    results = [
        onnx_result,
        make_result("yolox-s",  "Raspberry Pi 5",          0.41, 0.60, 0.25, 200.0, 13.5),
        make_result("yolov9t",  "NVIDIA GeForce RTX 5080", 0.38, 0.55, 0.20, 3.0, 4.0),
        make_result("yolov9t",  "Raspberry Pi 5",          0.38, 0.55, 0.20, 120.0, 4.0),
    ]
    out = compute_va_v1_scores(results)
    assert "yolox-s" in out["va_v1_scores"]
    assert "yolov9t" in out["va_v1_scores"]
