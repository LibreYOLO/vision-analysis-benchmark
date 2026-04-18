"""Unit tests for output assembly and filename disambiguation."""

from __future__ import annotations

import json

from va_bench.output import save_result

from .conftest import make_result


def test_pytorch_filename_has_no_format_suffix(tmp_path):
    result = make_result("yolox-s", "NVIDIA GeForce RTX 5080", 0.4, 0.6, 0.2, 5.0, 13.5)
    path = save_result(result, tmp_path)
    assert path.name == "yolox-s_rtx5080.json"


def test_onnx_filename_has_format_suffix(tmp_path):
    result = make_result(
        "yolox-s", "NVIDIA GeForce RTX 5080", 0.4, 0.6, 0.2, 5.0, 13.5, fmt="onnx",
    )
    path = save_result(result, tmp_path)
    assert path.name == "yolox-s_rtx5080_onnx.json"


def test_saved_file_is_valid_json(tmp_path):
    result = make_result("yolox-s", "Raspberry Pi 5", 0.4, 0.6, 0.2, 100.0, 13.5)
    path = save_result(result, tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["model"]["name"] == "yolox-s"
    assert loaded["runtime"]["format"] == "pytorch"
