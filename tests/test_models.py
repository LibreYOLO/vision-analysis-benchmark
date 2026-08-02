"""Unit tests for the model registry and weights resolution."""

from __future__ import annotations

from collections import Counter

import pytest

from va_bench.models import (
    MODEL_REGISTRY,
    get_spec,
    list_models,
    resolve_onnx_weights,
)


def test_registry_has_expected_models():
    keys = list_models()
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
        "yolov7",
        "yolov9",
        "yolov9e2e-",
        "yolonas-",
        "yolox-",
    ):
        assert any(k.startswith(fam) for k in keys)
    assert {"yolonas-s", "yolonas-m", "yolonas-l"} <= set(keys)
    assert not any(k.startswith("damoyolo-") for k in keys)
    assert len(keys) == 68
    assert Counter(spec.family for spec in MODEL_REGISTRY.values()) == {
        "deim": 5,
        "deimv2": 8,
        "dfine": 5,
        "ec": 4,
        "picodet": 3,
        "rfdetr": 4,
        "rtdetr": 7,
        "rtdetrv2": 5,
        "rtdetrv4": 4,
        "rtmdet": 5,
        "yolonas": 3,
        "yolov7": 1,
        "yolov9": 4,
        "yolov9-e2e": 4,
        "yolox": 6,
    }


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
