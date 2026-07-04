"""Unit tests for the model registry and weights resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from va_bench.models import (
    MODEL_REGISTRY,
    get_spec,
    list_models,
    resolve_onnx_weights,
)


def test_registry_has_expected_models():
    keys = list_models()
    assert len(keys) == 77
    for fam in (
        "deim-",
        "deimv2-",
        "dfine-",
        "ec-",
        "picodet-",
        "rfdetr-",
        "rtdetr-",
        "rtdetrv2-",
        "rtdetrv4-",
        "rtmdet-",
        "uly-",
        "yolonas-",
        "yolov9",
        "yolov9e2e-",
        "yolox-",
    ):
        assert any(k.startswith(fam) for k in keys)
    # damoyolo was swapped out for yolonas (c2d7c3d)
    assert not any(k.startswith("damoyolo-") for k in keys)


def test_registry_ultralytics_source():
    keys = list_models()
    uly_keys = [k for k in keys if k.startswith("uly-")]
    assert sorted(uly_keys) == [
        f"uly-{fam}{size}"
        for fam in ("yolo11", "yolov8")
        for size in ("l", "m", "n", "s", "x")
    ]
    for key in keys:
        spec = MODEL_REGISTRY[key]
        if key.startswith("uly-"):
            assert spec.source == "ultralytics"
            # official ultralytics weight files, not LibreYOLO ones
            assert spec.weight_file.startswith("yolo")
            assert spec.input_size == 640
        else:
            assert spec.source == "libreyolo"


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec("nonexistent-model")


def test_resolve_onnx_missing_file(tmp_path):
    spec = get_spec("yolox-nano")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_onnx_weights(spec, tmp_path)
    assert "LibreYOLOXn.onnx" in str(exc.value)


def test_resolve_onnx_happy_path(tmp_path):
    spec = get_spec("yolox-nano")
    onnx_file = tmp_path / "LibreYOLOXn.onnx"
    onnx_file.write_bytes(b"not-a-real-onnx")
    result = resolve_onnx_weights(spec, tmp_path)
    assert result == onnx_file
