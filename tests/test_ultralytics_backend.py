"""Unit tests for the ultralytics subprocess-driver benchmark path.

Uses tests/fake_uly_driver.py — a stdlib-only stand-in for
drivers/ultralytics/uly_driver.py that honors the same CLI/JSON contract
(PLAN_ultralytics_source.md). No ultralytics install is needed.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from va_bench import benchmark
from va_bench.benchmark import benchmark_model, run_ultralytics_benchmark
from va_bench.models import MODEL_REGISTRY, get_spec

from .conftest import make_result

FAKE_DRIVER = Path(__file__).parent / "fake_uly_driver.py"

# Two-image COCO fixture. Image ids are deliberately non-contiguous and
# unsorted on disk order; _load_coco sorts ids -> [42, 73].
COCO_IMAGES = [
    {"id": 73, "file_name": "000000000073.jpg", "width": 640, "height": 480},
    {"id": 42, "file_name": "000000000042.jpg", "width": 640, "height": 480},
]
COCO_ANNOTATIONS = [
    {"id": 1, "image_id": 42, "category_id": 1, "bbox": [10, 20, 30, 40],
     "area": 1200.0, "iscrowd": 0},
    {"id": 2, "image_id": 73, "category_id": 18, "bbox": [5, 5, 10, 10],
     "area": 100.0, "iscrowd": 0},
]
COCO_CATEGORIES = [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}]

FAKE_HW = {
    "hardware": {
        "gpu": "NVIDIA GeForce RTX 5080",
        "gpu_memory_gb": 16.0,
        "driver_version": "test",
        "cuda_version": "12",
        "cpu": "test-cpu",
        "cpu_cores": 8,
        "ram_gb": 32,
    },
    "software": {
        "python": "3.12",
        "torch": "2.11",
        "libreyolo": "1.3.0",
        "libreyolo_commit": "deadbeef",
        "onnxruntime": "not-installed",
    },
}


@pytest.fixture
def coco_dir(tmp_path):
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    img_dir = root / "images" / "val2017"
    img_dir.mkdir(parents=True)
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps({
        "images": COCO_IMAGES,
        "annotations": COCO_ANNOTATIONS,
        "categories": COCO_CATEGORIES,
    }))
    for img in COCO_IMAGES:
        (img_dir / img["file_name"]).write_bytes(b"not-a-real-jpeg")
    return root


@pytest.fixture
def weights_dir(tmp_path):
    wdir = tmp_path / "uly_weights"
    wdir.mkdir()
    (wdir / "yolo11n.pt").write_bytes(b"not-a-real-checkpoint")
    return wdir


@pytest.fixture
def fake_hw(monkeypatch):
    monkeypatch.setattr(benchmark, "collect_hw", lambda: deepcopy(FAKE_HW))


def _run(coco_dir, weights_dir, **kwargs):
    return run_ultralytics_benchmark(
        "uly-yolo11n",
        coco_dir,
        uly_python=sys.executable,
        weights_dir=weights_dir,
        device="cpu",
        verbose=False,
        driver_script=FAKE_DRIVER,
        **kwargs,
    )


# =============================================================================
# Registry
# =============================================================================

EXPECTED_ULY_SPECS = {
    # key: (family, weight_file, paper_params_m, paper_flops_g)
    "uly-yolo11n": ("yolo11", "yolo11n.pt", 2.6, 6.5),
    "uly-yolo11s": ("yolo11", "yolo11s.pt", 9.4, 21.5),
    "uly-yolo11m": ("yolo11", "yolo11m.pt", 20.1, 68.0),
    "uly-yolo11l": ("yolo11", "yolo11l.pt", 25.3, 86.9),
    "uly-yolo11x": ("yolo11", "yolo11x.pt", 56.9, 194.9),
    "uly-yolov8n": ("yolov8", "yolov8n.pt", 3.2, 8.7),
    "uly-yolov8s": ("yolov8", "yolov8s.pt", 11.2, 28.6),
    "uly-yolov8m": ("yolov8", "yolov8m.pt", 25.9, 78.9),
    "uly-yolov8l": ("yolov8", "yolov8l.pt", 43.7, 165.2),
    "uly-yolov8x": ("yolov8", "yolov8x.pt", 68.2, 257.8),
}


def test_registry_has_ultralytics_rows():
    for key, (family, weight_file, params_m, flops_g) in EXPECTED_ULY_SPECS.items():
        spec = get_spec(key)
        assert spec.source == "ultralytics"
        assert spec.family == family
        assert spec.weight_file == weight_file
        assert spec.paper_params_m == params_m
        assert spec.paper_flops_g == flops_g
        assert spec.input_size == 640


def test_libreyolo_rows_keep_default_source():
    assert get_spec("yolov9t").source == "libreyolo"
    assert all(
        s.source in ("libreyolo", "ultralytics") for s in MODEL_REGISTRY.values()
    )


def test_benchmark_model_rejects_ultralytics_source(tmp_path):
    with pytest.raises(ValueError, match="source 'ultralytics'"):
        benchmark_model("uly-yolo11n", coco_dir=tmp_path)


# =============================================================================
# Driver invocation: config contents
# =============================================================================

def test_run_config_written_with_eval_protocol(
    coco_dir, weights_dir, fake_hw, monkeypatch, tmp_path,
):
    dump_path = tmp_path / "config_dump.json"
    monkeypatch.setenv("FAKE_ULY_CONFIG_DUMP", str(dump_path))

    _run(coco_dir, weights_dir)

    config = json.loads(dump_path.read_text())
    assert set(config.keys()) == {
        "weights", "images", "imgsz", "conf", "iou", "max_det",
        "device", "warmup_iters", "half",
    }
    assert config["conf"] == 0.001
    assert config["iou"] == 0.7
    assert config["max_det"] == 300
    assert config["imgsz"] == 640
    assert config["device"] == "cpu"
    assert config["warmup_iters"] == 3  # harness decides: 3 on cpu
    assert config["half"] is False

    assert Path(config["weights"]).is_absolute()
    assert Path(config["weights"]).name == "yolo11n.pt"

    # Images: absolute paths, ordered by sorted COCO image id (42 then 73).
    names = [Path(p).name for p in config["images"]]
    assert names == ["000000000042.jpg", "000000000073.jpg"]
    assert all(Path(p).is_absolute() for p in config["images"])


def test_missing_weights_is_a_clear_error(coco_dir, tmp_path, fake_hw):
    empty = tmp_path / "empty_weights"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="yolo11n.pt"):
        _run(coco_dir, empty)


# =============================================================================
# Detection mapping (contiguous 0-79 -> COCO 91 category ids)
# =============================================================================

def test_detections_mapped_to_coco91_ids(coco_dir, weights_dir, fake_hw, monkeypatch):
    captured = {}

    def spy_evaluate(coco_gt, predictions, image_ids=None):
        captured["predictions"] = predictions
        captured["image_ids"] = image_ids
        return {
            "mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0,
            "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
            "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
            "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0,
        }

    monkeypatch.setattr(benchmark, "evaluate_coco", spy_evaluate)

    _run(coco_dir, weights_dir)

    preds = captured["predictions"]
    assert captured["image_ids"] == [42, 73]
    # Fake driver emits 2 detections per image (class_index 0 and 16).
    assert len(preds) == 4
    assert {p["image_id"] for p in preds} == {42, 73}
    # class_index 0 -> COCO category 1 (person); 16 -> 18 (dog).
    assert sorted({p["category_id"] for p in preds}) == [1, 18]

    first = next(p for p in preds if p["image_id"] == 42 and p["category_id"] == 1)
    # bbox_xywh from the driver survives the xywh -> xyxy -> xywh round trip.
    assert first["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert first["score"] == pytest.approx(0.9)


# =============================================================================
# Timing: total from wall_ms, phases from speed_ms
# =============================================================================

def test_timing_stats_computed_from_wall_ms(coco_dir, weights_dir, fake_hw):
    result = _run(coco_dir, weights_dir)

    total = result["timing"]["total_ms"]
    # Fake driver wall_ms per image: [10.0, 20.0].
    assert total["mean"] == 15.0
    assert total["p50"] == 15.0
    assert total["std"] == 5.0
    # Phase means come from the driver's speed_ms dicts.
    assert total["preprocess_ms"] == 1.5   # mean(1.0, 2.0)
    assert total["inference_ms"] == 5.5    # mean(5.0, 6.0)
    assert total["postprocess_ms"] == 0.5

    assert result["throughput"]["fps_mean"] == pytest.approx(66.67)
    assert result["throughput"]["fps_p50"] == pytest.approx(66.67)


# =============================================================================
# Result JSON: source / provider / version / schema shape
# =============================================================================

def test_result_carries_ultralytics_provenance(coco_dir, weights_dir, fake_hw):
    result = _run(coco_dir, weights_dir)

    assert result["schema_version"] == "va.submission.v1"
    # model.id is the site-canonical competitor id, not the uly-* CLI key,
    # so the numbers attach to the reference row the website already has.
    assert result["model"]["id"] == "yolo11n"
    assert result["model"]["source"] == "ultralytics"
    assert result["model"]["family"] == "yolo11"
    assert result["model"]["weights"] == "yolo11n.pt"
    assert result["implementation"] == {
        "provider": "ultralytics",
        "version": "8.4.60",
    }
    assert result["software"]["ultralytics"] == "8.4.60"
    assert result["software"]["ultralytics_torch"] == "2.7.0-fake"
    # Harness-venv software entries are still present alongside.
    assert result["software"]["python"] == "3.12"

    assert result["runtime"] == {
        "format": "pytorch",
        "precision": "fp32",
        "provider": "cpu",
        "device": "cpu",
    }
    assert result["config"] == {
        "batch_size": 1,
        "input_size": 640,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
    }
    # No LibreYOLO is loaded for a competitor run, so its provenance is blanked
    # (the website validates libreyolo_commit against an allowlist).
    assert result["benchmark"]["libreyolo_version"] == "unknown"
    assert result["benchmark"]["libreyolo_commit"] == "unknown"
    assert result["model_stats"]["params_millions"] == 2.62
    assert result["memory"]["peak_vram_mb"] == 812.0
    assert result["memory"]["peak_ram_mb"] == 1234.0
    assert result["dataset"]["num_images"] == 2


def test_result_matches_existing_schema_shape(coco_dir, weights_dir, fake_hw, tmp_path):
    result = _run(coco_dir, weights_dir)
    reference = make_result("yolox-s", "NVIDIA GeForce RTX 5080", 0.4, 0.6, 0.2, 5.0, 13.5)

    assert set(result.keys()) == set(reference.keys())
    for section in ("model", "accuracy", "timing", "throughput", "model_stats",
                    "memory", "eval", "implementation", "runtime", "dataset",
                    "config", "benchmark"):
        assert set(result[section].keys()) == set(reference[section].keys()), section
    assert set(result["timing"]["total_ms"].keys()) == set(
        reference["timing"]["total_ms"].keys()
    )

    from va_bench.output import save_result

    path = save_result(result, tmp_path)
    assert path.name.startswith("yolo11n__pytorch__cpu__")
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == "va.submission.v1"


def test_assemble_result_defaults_unchanged_for_libreyolo(fake_hw):
    """Existing callers that don't pass the new args keep today's output."""
    from va_bench.output import assemble_result

    result = assemble_result(
        spec=get_spec("yolov9t"),
        coco_metrics={
            "mAP": 0.1, "mAP50": 0.2, "mAP75": 0.1,
            "mAP_small": 0.1, "mAP_medium": 0.1, "mAP_large": 0.1,
            "AR1": 0.1, "AR10": 0.1, "AR100": 0.1,
            "AR_small": 0.1, "AR_medium": 0.1, "AR_large": 0.1,
        },
        total_stats={"mean": 5.0, "std": 1.0, "p50": 5.0, "p95": 6.0, "p99": 7.0},
        preprocess_ms=1.0,
        inference_ms=3.0,
        postprocess_ms=1.0,
        fps_mean=200.0,
        fps_p50=200.0,
        num_images=5000,
        measured_params_m=2.0,
        peak_vram_mb=100.0,
        peak_ram_mb=500.0,
        device_type="gpu",
        provider="cuda",
        hardware=deepcopy(FAKE_HW["hardware"]),
        software=deepcopy(FAKE_HW["software"]),
        actual_input_size=640,
        conf=0.001,
        iou=0.6,
        max_det=300,
    )

    assert result["model"]["source"] == "libreyolo"
    assert result["implementation"] == {"provider": "libreyolo", "version": "1.3.0"}
    assert "ultralytics" not in result["software"]


# =============================================================================
# Driver failure modes
# =============================================================================

def test_driver_nonzero_exit_raises_with_stderr(
    coco_dir, weights_dir, fake_hw, monkeypatch,
):
    monkeypatch.setenv("FAKE_ULY_FAIL", "1")
    with pytest.raises(RuntimeError) as exc:
        _run(coco_dir, weights_dir)
    message = str(exc.value)
    assert "exit code 3" in message
    assert "CUDA out of imagination" in message


def test_driver_invalid_json_raises_clean_error(
    coco_dir, weights_dir, fake_hw, monkeypatch,
):
    monkeypatch.setenv("FAKE_ULY_BAD_JSON", "1")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        _run(coco_dir, weights_dir)


def test_driver_with_telemetry_on_is_rejected(
    coco_dir, weights_dir, fake_hw, monkeypatch,
):
    monkeypatch.setenv("FAKE_ULY_TELEMETRY_ON", "1")
    with pytest.raises(RuntimeError, match="telemetry"):
        _run(coco_dir, weights_dir)


def test_missing_driver_python_is_a_clear_error(coco_dir, weights_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="driver python"):
        run_ultralytics_benchmark(
            "uly-yolo11n",
            coco_dir,
            uly_python=tmp_path / "no-such-venv" / "python.exe",
            weights_dir=weights_dir,
            device="cpu",
            verbose=False,
            driver_script=FAKE_DRIVER,
        )


# =============================================================================
# CLI routing
# =============================================================================

def test_cli_run_requires_uly_python(monkeypatch, capsys):
    from va_bench import cli

    monkeypatch.delenv("VA_ULY_PYTHON", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "va-bench", "run", "--models", "uly-yolo11n", "--coco-dir", "X",
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "--uly-python" in out
    assert "VA_ULY_PYTHON" in out


def test_cli_run_rejects_non_pytorch_format_for_uly(monkeypatch, capsys, tmp_path):
    from va_bench import cli

    monkeypatch.setenv("VA_ULY_PYTHON", sys.executable)
    monkeypatch.setattr(sys, "argv", [
        "va-bench", "run", "--models", "uly-yolo11n", "--coco-dir", "X",
        "--format", "onnx", "--weights-dir", str(tmp_path),
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert "--format pytorch" in capsys.readouterr().out


def test_cli_routes_by_source_and_defaults_iou(monkeypatch, tmp_path, capsys):
    from va_bench import cli

    calls = {}

    def fake_uly(**kwargs):
        calls["uly"] = kwargs
        return make_result("uly-yolo11n", "NVIDIA GeForce RTX 5080",
                           0.4, 0.6, 0.2, 5.0, 6.5)

    def fake_libre(**kwargs):
        calls["libre"] = kwargs
        return make_result("yolov9t", "NVIDIA GeForce RTX 5080",
                           0.4, 0.6, 0.2, 5.0, 4.0)

    monkeypatch.setattr(benchmark, "run_ultralytics_benchmark", fake_uly)
    monkeypatch.setattr(benchmark, "benchmark_model", fake_libre)
    monkeypatch.setenv("VA_ULY_PYTHON", sys.executable)
    monkeypatch.setattr(sys, "argv", [
        "va-bench", "run", "--models", "uly-yolo11n", "yolov9t",
        "--coco-dir", "X", "--output-dir", str(tmp_path / "out"),
    ])

    cli.main()

    assert calls["uly"]["model_key"] == "uly-yolo11n"
    assert calls["uly"]["iou"] == 0.7           # ultralytics default
    assert calls["uly"]["uly_python"] == sys.executable
    assert calls["libre"]["model_key"] == "yolov9t"
    assert calls["libre"]["iou"] == 0.6         # libreyolo default preserved
    # Both results were saved.
    saved = list((tmp_path / "out").glob("*.json"))
    assert len(saved) == 2


def test_cli_explicit_iou_overrides_both_sources(monkeypatch, tmp_path):
    from va_bench import cli

    calls = {}

    def fake_uly(**kwargs):
        calls["uly"] = kwargs
        return make_result("uly-yolo11n", "NVIDIA GeForce RTX 5080",
                           0.4, 0.6, 0.2, 5.0, 6.5)

    monkeypatch.setattr(benchmark, "run_ultralytics_benchmark", fake_uly)
    monkeypatch.setenv("VA_ULY_PYTHON", sys.executable)
    monkeypatch.setattr(sys, "argv", [
        "va-bench", "run", "--models", "uly-yolo11n",
        "--coco-dir", "X", "--output-dir", str(tmp_path / "out"),
        "--iou", "0.65",
    ])

    cli.main()

    assert calls["uly"]["iou"] == 0.65


def test_cli_list_shows_source_column(monkeypatch, capsys):
    from va_bench import cli

    monkeypatch.setattr(sys, "argv", ["va-bench", "list"])
    cli.main()
    out = capsys.readouterr().out
    assert "Source" in out
    assert "ultralytics" in out
    assert "libreyolo" in out
    assert "uly-yolo11n" in out
