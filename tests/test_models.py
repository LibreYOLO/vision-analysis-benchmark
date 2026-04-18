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
    assert len(keys) == 14
    for fam in ("yolox-", "yolov9", "rfdetr-"):
        assert any(k.startswith(fam) for k in keys)


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
