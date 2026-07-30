"""Unit tests for output assembly and filename disambiguation."""

from __future__ import annotations

import json

from va_bench.models import get_spec
from va_bench.output import assemble_result, save_result

from .conftest import make_result

_COCO_METRICS = {
    "mAP": 0.4, "mAP50": 0.6, "mAP75": 0.45,
    "mAP_small": 0.2, "mAP_medium": 0.4, "mAP_large": 0.6,
    "AR1": 0.3, "AR10": 0.5, "AR100": 0.6,
    "AR_small": 0.2, "AR_medium": 0.4, "AR_large": 0.6,
}
_TOTAL_STATS = {"mean": 10.0, "std": 1.0, "p50": 10.0, "p95": 11.0, "p99": 12.0}


def _assemble(repro=None):
    return assemble_result(
        spec=get_spec("yolox-s"),
        coco_metrics=_COCO_METRICS,
        total_stats=_TOTAL_STATS,
        preprocess_ms=1.0, inference_ms=8.0, postprocess_ms=1.0,
        fps_mean=100.0, fps_p50=100.0, num_images=500,
        measured_params_m=9.0, peak_vram_mb=100.0, peak_ram_mb=200.0,
        device_type="gpu", provider="cuda",
        hardware={"gpu": "x", "cpu": "y", "ram_gb": 8, "id": "x"},
        software={"libreyolo": "1.0.0"},
        actual_input_size=640, conf=0.001, iou=0.6, max_det=300,
        repro=repro,
    )


def test_pytorch_filename_has_no_format_suffix(tmp_path):
    result = make_result("yolox-s", "NVIDIA GeForce RTX 5080", 0.4, 0.6, 0.2, 5.0, 13.5)
    path = save_result(result, tmp_path)
    assert path.name == "yolox-s__pytorch__cuda__rtx5080__20260418T101512Z.json"


def test_onnx_filename_has_format_suffix(tmp_path):
    result = make_result(
        "yolox-s", "NVIDIA GeForce RTX 5080", 0.4, 0.6, 0.2, 5.0, 13.5, fmt="onnx",
    )
    path = save_result(result, tmp_path)
    assert path.name == "yolox-s__onnx__cuda__rtx5080__20260418T101512Z.json"


def test_saved_file_is_valid_json(tmp_path):
    result = make_result("yolox-s", "Raspberry Pi 5", 0.4, 0.6, 0.2, 100.0, 13.5)
    path = save_result(result, tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == "va.submission.v1"
    assert loaded["model"]["id"] == "yolox-s"
    assert loaded["model"]["name"] == "yolox-s"
    assert loaded["runtime"]["format"] == "pytorch"


def test_assemble_result_threads_repro_block():
    repro = {"harness_version": "2.0.0", "dataset": {"image_id_sha256": "abc"}}
    result = _assemble(repro=repro)
    assert result["repro"] == repro


def test_assemble_result_repro_defaults_to_empty():
    result = _assemble(repro=None)
    assert result["repro"] == {}


def test_save_result_preserves_repro(tmp_path):
    result = _assemble(repro={"command": "va-bench run --models yolox-s"})
    loaded = json.loads(save_result(result, tmp_path).read_text())
    assert loaded["repro"]["command"] == "va-bench run --models yolox-s"
