"""Failure recovery in the training orchestrator.

Two recovery modes, both learned from the first yolov9t campaign
(2026-08-01, 8 failures on an 8x16GB box at 2 lanes/GPU):

* CUDA OOM in a packed lane -> park the dataset and retry it with a whole
  GPU after the packed queue drains. Every observed OOM was a co-tenancy
  problem (the dense-dataset IoU matrix needs the card, not a smaller
  batch), and jobs-per-gpu is not part of the run signature, so the retry
  is protocol-identical. Only an OOM with the card to itself is final.
* CUDA-graph capture race (pin-memory thread allocating during capture)
  -> retry once, eagerly, in the same lane. Graph replay is bit-identical
  on loss and not part of the run signature; the deviation is recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from va_bench import rf100vl_train
from va_bench.rf100vl_data import VERSION_LOCK_SCHEMA, atomic_write_json


def _make_dataset(root, name="aerial-cows", num_images=3, per_image=1):
    """Tiny COCO-format dataset (same shape as test_rf100vl_train's helper)."""
    dataset = root / name
    for split in ("train", "valid", "test"):
        split_dir = dataset / split
        split_dir.mkdir(parents=True)
        annotations = []
        annotation_id = 1
        for image_id in range(1, num_images + 1):
            for _ in range(per_image):
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 7,
                        "bbox": [0, 0, 1, 1],
                    }
                )
                annotation_id += 1
        payload = {
            "images": [
                {"id": i, "file_name": f"{i}.jpg", "width": 8, "height": 8}
                for i in range(1, num_images + 1)
            ],
            "annotations": annotations,
            "categories": [{"id": 7, "name": "zebra"}, {"id": 2, "name": "ant"}],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return dataset


FAKE_CAPABILITIES = {
    "validated": True,
    "version": "test",
    "amp_dtype": "bfloat16",
    "prediction_max_det": 500,
    "eval_max_det": 500,
    "default_eval_max_det": 100,
    "cuda_graph": True,
    "cache": True,
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_capture_race_classification():
    # The exact shape torch surfaces: wrapper text names the original error.
    race = RuntimeError(
        "Caught AcceleratorError in pin memory thread for device 0.\n"
        "Original Traceback (most recent call last): ..."
    )
    assert rf100vl_train._is_capture_race(race)
    assert rf100vl_train._is_capture_race(
        RuntimeError("CUDA error: cudaErrorStreamCaptureUnsupported")
    )
    # OOM stays OOM (checked first by the worker), ordinary errors stay failed.
    oom = RuntimeError("CUDA out of memory. Tried to allocate 52.00 MiB")
    assert rf100vl_train._is_cuda_oom(oom)
    assert not rf100vl_train._is_capture_race(oom)
    assert not rf100vl_train._is_capture_race(ValueError("bad annotation"))


def test_worker_exit_codes_distinguish_failure_classes(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path / "data")
    recipe_path = rf100vl_train.recipe_path_for_family("yolov9")

    def worker_config(exc):
        class RaisingLibreYOLO:
            def __init__(self, **kwargs):
                pass

            def train(self, **kwargs):
                raise exc

        monkeypatch.setitem(
            sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=RaisingLibreYOLO)
        )
        config = {
            "model_key": "yolov9t",
            "dataset_name": "aerial-cows",
            "dataset_dir": str(dataset),
            "dataset_version": 4,
            "versions_sha256": "a" * 64,
            "recipe_path": str(recipe_path),
            "run_dir": str(tmp_path / "runs" / "aerial-cows" / "primary"),
            "target_checkpoint": str(tmp_path / "weights" / "aerial-cows" / "w.pt"),
            "worker_result_path": str(tmp_path / "state" / "worker.json"),
            "run_signature": "b" * 64,
            "batch_plan": {
                "run_variant": "primary",
                "selection_reason": None,
                "physical_batch": 16,
                "effective_batch": 16,
                "gradient_accumulation_steps": 1,
                "num_train_images": 3,
                "expected_batches_per_epoch_minimum": 1,
            },
            "resume": False,
            "smoke_epochs": 1,
            "restart_reason": None,
        }
        path = tmp_path / "worker-config.json"
        atomic_write_json(path, config)
        return path

    monkeypatch.setattr(
        rf100vl_train,
        "require_libreyolo_protocol_capabilities",
        lambda: dict(FAKE_CAPABILITIES),
    )

    result_path = tmp_path / "state" / "worker.json"

    exit_code = rf100vl_train.run_dataset_worker(
        worker_config(RuntimeError("CUDA out of memory. Tried to allocate 52.00 MiB"))
    )
    assert exit_code == 86
    assert json.loads(result_path.read_text(encoding="utf-8"))["state"] == "oom"

    exit_code = rf100vl_train.run_dataset_worker(
        worker_config(
            RuntimeError("Caught AcceleratorError in pin memory thread for device 0.")
        )
    )
    assert exit_code == 87
    assert json.loads(result_path.read_text(encoding="utf-8"))["state"] == "capture_race"

    exit_code = rf100vl_train.run_dataset_worker(worker_config(ValueError("boom")))
    assert exit_code == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["state"] == "failed"


def test_worker_disable_cuda_graph_overrides_recipe(tmp_path, monkeypatch):
    """disable_cuda_graph must reach model.train() and be recorded in stats."""
    dataset = _make_dataset(tmp_path / "data")
    recipe_path = rf100vl_train.recipe_path_for_family("yolov9")
    seen_kwargs: dict = {}

    class FakeLibreYOLO:
        def __init__(self, **kwargs):
            pass

        def train(self, **kwargs):
            seen_kwargs.update(kwargs)
            best = Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"checkpoint")
            return {
                "best_checkpoint": str(best),
                "best_epoch": 1,
                "best_mAP50": 0.5,
                "best_mAP50_95": 0.4,
            }

    monkeypatch.setitem(
        sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=FakeLibreYOLO)
    )
    monkeypatch.setattr(
        rf100vl_train,
        "require_libreyolo_protocol_capabilities",
        lambda: dict(FAKE_CAPABILITIES),
    )
    # Recipe asks for graphs and the (fake) install supports them, so the
    # disable flag is meaningful rather than a no-op.
    monkeypatch.setattr(rf100vl_train, "_libreyolo_supports_cuda_graph", lambda: True)
    monkeypatch.setattr(rf100vl_train, "_libreyolo_supports_cache", lambda: True)

    target = tmp_path / "weights" / "aerial-cows" / "LibreYOLO9t.pt"
    config = {
        "model_key": "yolov9t",
        "dataset_name": "aerial-cows",
        "dataset_dir": str(dataset),
        "dataset_version": 4,
        "versions_sha256": "a" * 64,
        "recipe_path": str(recipe_path),
        "run_dir": str(tmp_path / "runs" / "aerial-cows" / "primary"),
        "target_checkpoint": str(target),
        "worker_result_path": str(tmp_path / "state" / "worker.json"),
        "run_signature": "b" * 64,
        "batch_plan": {
            "run_variant": "primary",
            "selection_reason": None,
            "physical_batch": 16,
            "effective_batch": 16,
            "gradient_accumulation_steps": 1,
            "num_train_images": 3,
            "expected_batches_per_epoch_minimum": 1,
        },
        "resume": False,
        "smoke_epochs": 1,
        "restart_reason": "cuda_graph_capture_race",
        "disable_cuda_graph": True,
    }
    config_path = tmp_path / "worker-config.json"
    atomic_write_json(config_path, config)

    assert rf100vl_train.run_dataset_worker(config_path) == 0
    assert seen_kwargs["cuda_graph"] is False
    stats = json.loads((target.parent / "stats.json").read_text(encoding="utf-8"))
    assert stats["cuda_graph_disabled"] is True
    assert stats["restart_reason"] == "cuda_graph_capture_race"


# ---------------------------------------------------------------------------
# Orchestrator recovery flows
# ---------------------------------------------------------------------------


def _campaign_root(tmp_path, names):
    data_dir = tmp_path / "rf100vl"
    data_dir.mkdir()
    for index, name in enumerate(names):
        # Distinct image counts keep order_longest_first deterministic.
        _make_dataset(data_dir, name=name, num_images=3 + len(names) - index)
    atomic_write_json(
        data_dir / "versions.json",
        {
            "schema_version": VERSION_LOCK_SCHEMA,
            "subset": "rf100vl",
            "source": {"commit": "abc"},
            "datasets": {name: {"project_name": name, "version_id": 1} for name in names},
            "downloaded": list(names),
            "selection_complete": True,
        },
    )
    return data_dir


class _ScriptedChildren:
    """Fake _launch_child: replays a per-dataset script of outcomes."""

    def __init__(self, script):
        self.script = {name: list(outcomes) for name, outcomes in script.items()}
        self.calls: list[dict] = []

    def __call__(self, worker_config, *, gpu, log_path, timeout_seconds, on_started=None):
        config = json.loads(Path(worker_config).read_text(encoding="utf-8"))
        name = config["dataset_name"]
        outcome = self.script[name].pop(0)
        self.calls.append(
            {
                "dataset": name,
                "gpu": gpu,
                "restart_reason": config.get("restart_reason"),
                "disable_cuda_graph": bool(config.get("disable_cuda_graph")),
            }
        )
        result_path = Path(config["worker_result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if outcome == "done":
            payload = {
                "state": "done",
                "stats_path": str(result_path.with_name("stats.json")),
                "target_checkpoint": config["target_checkpoint"],
            }
            exit_code = 0
        elif outcome == "oom":
            payload = {"state": "oom", "message": "CUDA out of memory."}
            exit_code = 86
        elif outcome == "capture_race":
            payload = {
                "state": "capture_race",
                "message": "Caught AcceleratorError in pin memory thread for device 0.",
            }
            exit_code = 87
        else:  # pragma: no cover - script typo guard
            raise AssertionError(outcome)
        atomic_write_json(result_path, payload)
        return exit_code, False


def _orchestrate(tmp_path, monkeypatch, script, names, **kwargs):
    data_dir = _campaign_root(tmp_path, names)
    launcher = _ScriptedChildren(script)
    monkeypatch.setattr(
        rf100vl_train,
        "require_libreyolo_protocol_capabilities",
        lambda: dict(FAKE_CAPABILITIES),
    )
    monkeypatch.setattr(rf100vl_train, "_launch_child", launcher)
    summary = rf100vl_train.orchestrate_training(
        model_key="yolov9t",
        data_dir=data_dir,
        weights_root=tmp_path / "weights",
        state_root=tmp_path / "state",
        runs_root=tmp_path / "runs",
        **kwargs,
    )
    return summary, launcher


def test_packed_oom_is_retried_solo_after_queue_drain(tmp_path, monkeypatch):
    names = ["alpha", "densy", "omega"]
    summary, launcher = _orchestrate(
        tmp_path,
        monkeypatch,
        {"alpha": ["done"], "densy": ["oom", "done"], "omega": ["done"]},
        names,
        gpus=["0", "1"],
        jobs_per_gpu=2,
    )

    assert sorted(summary["completed"]) == names
    assert summary["failed"] == []
    assert summary["oom_solo_retried"] == ["densy"]
    assert summary["protocol_conformant"] is True

    densy_calls = [c for c in launcher.calls if c["dataset"] == "densy"]
    assert len(densy_calls) == 2
    assert densy_calls[1]["restart_reason"] == "solo_gpu_after_oom"
    # The solo retry runs only after every packed-phase attempt finished.
    assert launcher.calls[-1]["dataset"] == "densy"

    status = json.loads(
        (tmp_path / "state" / "densy.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "done"


def test_solo_oom_is_final(tmp_path, monkeypatch):
    names = ["alpha", "densy"]
    summary, launcher = _orchestrate(
        tmp_path,
        monkeypatch,
        {"alpha": ["done"], "densy": ["oom", "oom"]},
        names,
        gpus=["0"],
        jobs_per_gpu=2,
    )

    assert summary["completed"] == ["alpha"]
    assert summary["failed"] == ["densy"]
    assert summary["oom_solo_retried"] == ["densy"]
    assert summary["protocol_conformant"] is False
    assert len([c for c in launcher.calls if c["dataset"] == "densy"]) == 2

    status = json.loads(
        (tmp_path / "state" / "densy.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"


def test_capture_race_retries_eagerly_in_lane(tmp_path, monkeypatch):
    names = ["racy", "alpha"]
    summary, launcher = _orchestrate(
        tmp_path,
        monkeypatch,
        {"racy": ["capture_race", "done"], "alpha": ["done"]},
        names,
        gpus=["0"],
        jobs_per_gpu=1,
    )

    assert sorted(summary["completed"]) == sorted(names)
    assert summary["failed"] == []
    assert summary["protocol_conformant"] is True

    racy_calls = [c for c in launcher.calls if c["dataset"] == "racy"]
    assert len(racy_calls) == 2
    first, second = racy_calls
    assert first["disable_cuda_graph"] is False
    assert second["disable_cuda_graph"] is True
    assert second["restart_reason"] == "cuda_graph_capture_race"
    # In-lane: the retry happens immediately, on the same GPU.
    assert first["gpu"] == second["gpu"]
    assert launcher.calls.index(second) == launcher.calls.index(first) + 1

    status = json.loads((tmp_path / "state" / "racy.json").read_text(encoding="utf-8"))
    assert status["state"] == "done"
    assert status["cuda_graph_disabled"] is True


def test_capture_race_with_graphs_already_disabled_is_final(tmp_path, monkeypatch):
    """The eager retry must not loop: a second race (impossible by
    construction, but) lands in failed rather than retrying forever."""
    names = ["racy"]
    summary, launcher = _orchestrate(
        tmp_path,
        monkeypatch,
        {"racy": ["capture_race", "capture_race"]},
        names,
        gpus=["0"],
        jobs_per_gpu=1,
    )

    assert summary["completed"] == []
    assert summary["failed"] == ["racy"]
    assert len(launcher.calls) == 2
