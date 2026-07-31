"""Contracts for the RF100-VL training driver."""

from __future__ import annotations

import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
import torch
import yaml

from va_bench import rf100vl_train
from va_bench.models import get_spec, list_families
from va_bench.rf100vl_data import atomic_write_json


def _make_dataset(root, name="aerial-cows", num_images=3, per_image=1):
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
                {
                    "id": image_id,
                    "file_name": f"{image_id}.jpg",
                    "width": 8,
                    "height": 8,
                }
                for image_id in range(1, num_images + 1)
            ],
            "annotations": annotations,
            "categories": [
                {"id": 7, "name": "zebra"},
                {"id": 2, "name": "ant"},
            ],
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    return dataset


def test_generate_data_yaml_uses_all_splits_and_sorted_category_ids(tmp_path):
    dataset = _make_dataset(tmp_path)
    output, facts = rf100vl_train.generate_data_yaml(dataset, tmp_path / "data.yaml")
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert payload["train"] == "train"
    assert payload["val"] == "valid"
    assert payload["test"] == "test"
    assert payload["annotations"] == {
        "train": "train/_annotations.coco.json",
        "val": "valid/_annotations.coco.json",
        "test": "test/_annotations.coco.json",
    }
    assert payload["names"] == {0: "ant", 1: "zebra"}
    assert facts["category_ids"] == [2, 7]


def test_every_campaign_family_has_a_protocol_recipe():
    families = set(list_families()) - {"yolov9-e2e"}
    for family in families:
        recipe = rf100vl_train.load_recipe(
            rf100vl_train.recipe_path_for_family(family),
            family=family,
        )
        assert recipe["protocol"]["precision"] == "fp32"


def test_worker_process_seed_covers_python_numpy_and_torch():
    samples = []
    for ambient_seed in (1, 999):
        random.seed(ambient_seed)
        np.random.seed(ambient_seed)
        torch.manual_seed(ambient_seed)
        rf100vl_train.seed_worker_process(17)
        samples.append(
            (
                random.random(),
                float(np.random.random()),
                torch.rand(4),
            )
        )

    assert samples[0][0] == samples[1][0]
    assert samples[0][1] == samples[1][1]
    assert torch.equal(samples[0][2], samples[1][2])


def test_dense_rfdetr_selects_fallback_and_micro_dataset_keeps_one_batch():
    spec = get_spec("rfdetr-s")
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family("rfdetr"),
        family="rfdetr",
    )
    facts = {
        "max_annotations_per_image": 255,
        "num_train_images": 1,
    }
    plan = rf100vl_train.select_batch_plan(recipe, spec, facts)
    assert plan["run_variant"] == "fallback"
    assert plan["physical_batch"] == 2
    assert plan["effective_batch"] == 16
    assert plan["expected_batches_per_epoch_minimum"] == 1


def test_rfdetr_l_oom_fallback_reduces_batch():
    spec = get_spec("rfdetr-l")
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family("rfdetr"),
        family="rfdetr",
    )
    facts = {
        "max_annotations_per_image": 1,
        "num_train_images": 20,
    }

    primary = rf100vl_train.select_batch_plan(recipe, spec, facts)
    fallback = rf100vl_train.select_batch_plan(
        recipe,
        spec,
        facts,
        force_fallback=True,
    )

    assert primary["physical_batch"] == 2
    assert fallback["physical_batch"] == 1
    assert fallback["gradient_accumulation_steps"] == 16


def test_fp32_recipe_disables_amp_and_freezes_selection_contract(tmp_path):
    spec = get_spec("yolov9s")
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family("yolov9"),
        family="yolov9",
    )
    batch = rf100vl_train.select_batch_plan(
        recipe,
        spec,
        {"max_annotations_per_image": 1, "num_train_images": 20},
    )
    kwargs = rf100vl_train.build_train_kwargs(
        recipe,
        spec,
        batch,
        data_yaml=tmp_path / "data.yaml",
        run_dir=tmp_path / "run",
        resume=False,
    )
    assert kwargs["amp"] is False
    assert kwargs["epochs"] == 100
    assert kwargs["nbs"] == 16
    assert kwargs["eval_interval"] == 1
    assert kwargs["patience"] == 0
    assert kwargs["ema"] is True
    assert kwargs["max_det"] == 500
    assert kwargs["eval_max_det"] == 500


def test_capability_guard_rejects_pre_protocol_libreyolo(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class OldTrainConfig:
        amp_dtype: str = "float16"
        max_det: int = 300

    @dataclass
    class OldValidationConfig:
        data: str
        amp_dtype: str = "float16"
        max_det: int = 300

    class OldValidator:
        pass

    monkeypatch.setattr(
        rf100vl_train,
        "_load_libreyolo_protocol_types",
        lambda: (OldTrainConfig, OldValidationConfig, OldValidator),
    )

    with pytest.raises(RuntimeError, match="cannot enforce"):
        rf100vl_train.require_libreyolo_protocol_capabilities()


def test_reconcile_demotes_stale_running_atomically(tmp_path):
    path = tmp_path / "aerial-cows.json"
    atomic_write_json(
        path,
        {
            "schema_version": rf100vl_train.STATUS_SCHEMA,
            "state": "running",
            "dataset": "aerial-cows",
        },
    )
    active = rf100vl_train.reconcile_statuses(tmp_path, ["aerial-cows"])
    status = json.loads(path.read_text(encoding="utf-8"))
    assert active == set()
    assert status["state"] == "pending"
    assert status["reconcile_reason"]


def test_reconcile_preserves_live_child(tmp_path, monkeypatch):
    path = tmp_path / "aerial-cows.json"
    atomic_write_json(
        path,
        {
            "schema_version": rf100vl_train.STATUS_SCHEMA,
            "state": "running",
            "dataset": "aerial-cows",
            "pid": 123,
            "pid_create_time": 456.0,
        },
    )
    monkeypatch.setattr(
        rf100vl_train,
        "_status_process_is_live",
        lambda status: True,
    )

    active = rf100vl_train.reconcile_statuses(tmp_path, ["aerial-cows"])

    assert active == {"aerial-cows"}
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "running"


def test_status_process_identity_rejects_pid_reuse():
    process = psutil.Process(os.getpid())
    live = {
        "pid": process.pid,
        "pid_create_time": process.create_time(),
    }

    assert rf100vl_train._status_process_is_live(live)
    assert not rf100vl_train._status_process_is_live(
        {**live, "pid_create_time": process.create_time() - 10}
    )


def test_reconcile_promotes_completed_orphan_result(tmp_path):
    result_path = tmp_path / "worker-result.json"
    atomic_write_json(
        result_path,
        {
            "state": "done",
            "stats_path": "stats.json",
            "target_checkpoint": "best.pt",
            "finished_at": "done-time",
        },
    )
    path = tmp_path / "aerial-cows.json"
    atomic_write_json(
        path,
        {
            "schema_version": rf100vl_train.STATUS_SCHEMA,
            "state": "running",
            "dataset": "aerial-cows",
            "worker_result_path": str(result_path),
        },
    )

    active = rf100vl_train.reconcile_statuses(tmp_path, ["aerial-cows"])
    status = json.loads(path.read_text(encoding="utf-8"))

    assert active == set()
    assert status["state"] == "done"
    assert status["stats_path"] == "stats.json"


def test_done_skip_requires_current_recipe_version_and_annotations(tmp_path):
    dataset = _make_dataset(tmp_path / "data")
    target = tmp_path / "weights" / "aerial-cows" / "LibreYOLO9s.pt"
    stats_path = target.parent / "stats.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"checkpoint")
    facts = rf100vl_train.inspect_dataset(dataset)
    recipe_sha = "a" * 64
    versions_sha = "b" * 64
    signature = "c" * 64
    status = {
        "schema_version": rf100vl_train.STATUS_SCHEMA,
        "state": "done",
        "run_signature": signature,
        "recipe_sha256": recipe_sha,
        "versions_sha256": versions_sha,
    }
    stats = {
        "schema_version": rf100vl_train.STATS_SCHEMA,
        "protocol_version": rf100vl_train.PROTOCOL_VERSION,
        "protocol_conformant": True,
            "libreyolo_capabilities": {
                "validated": True,
                "eval_max_det": 500,
                "default_eval_max_det": 100,
            },
        "dataset": "aerial-cows",
        "dataset_version": 4,
        "model_key": "yolov9s",
        "epochs_requested": 100,
        "run_signature": signature,
        "recipe": {"sha256": recipe_sha},
        "versions_sha256": versions_sha,
        "data": {
            "train_annotations_sha256": facts["train_annotations_sha256"],
        },
    }
    atomic_write_json(stats_path, stats)

    kwargs = {
        "status": status,
        "target_checkpoint": target,
        "stats_path": stats_path,
        "dataset_dir": dataset,
        "dataset_name": "aerial-cows",
        "dataset_version_id": 4,
        "model_key": "yolov9s",
        "recipe_sha256": recipe_sha,
        "versions_sha256": versions_sha,
        "smoke_epochs": None,
    }
    assert rf100vl_train._completed_run_matches(**kwargs)

    kwargs["dataset_version_id"] = 5
    assert not rf100vl_train._completed_run_matches(**kwargs)


def test_worker_generates_stats_and_copies_best_checkpoint(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path / "data")
    recipe_path = rf100vl_train.recipe_path_for_family("yolov9")
    run_dir = tmp_path / "runs" / "aerial-cows" / "primary"
    target = tmp_path / "weights" / "aerial-cows" / "LibreYOLO9s.pt"
    worker_result = tmp_path / "state" / "worker.json"

    class FakeLibreYOLO:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self, **kwargs):
            best = rf100vl_train.Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"checkpoint")
            return {
                "best_checkpoint": str(best),
                "best_epoch": 1,
                "best_mAP50": 0.8,
                "best_mAP50_95": 0.6,
            }

    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=FakeLibreYOLO),
    )
    monkeypatch.setattr(
        rf100vl_train,
        "require_libreyolo_protocol_capabilities",
        lambda: {
            "validated": True,
            "version": "test",
            "amp_dtype": "bfloat16",
            "prediction_max_det": 500,
            "eval_max_det": 500,
            "default_eval_max_det": 100,
        },
    )
    config = {
        "model_key": "yolov9s",
        "dataset_name": "aerial-cows",
        "dataset_dir": str(dataset),
        "dataset_version": 4,
        "versions_sha256": "a" * 64,
        "recipe_path": str(recipe_path),
        "run_dir": str(run_dir),
        "target_checkpoint": str(target),
        "worker_result_path": str(worker_result),
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
    config_path = tmp_path / "worker-config.json"
    atomic_write_json(config_path, config)

    assert rf100vl_train.run_dataset_worker(config_path) == 0
    assert target.read_bytes() == b"checkpoint"
    stats = json.loads((target.parent / "stats.json").read_text(encoding="utf-8"))
    assert stats["protocol_conformant"] is False
    assert stats["best_epoch"] == 1
    assert stats["valid_mAP50_95"] == pytest.approx(0.6)
    assert stats["dataset_version"] == 4
    assert len(stats["recipe"]["sha256"]) == 64


def test_smoke_leftover_detection_and_quarantine(tmp_path):
    from va_bench.rf100vl_train import (
        _quarantine_run_dir,
        _smoke_leftover_config,
        _worker_config_path,
        atomic_write_json,
    )

    state_root = tmp_path / "state"
    run_dir = tmp_path / "runs" / "ds" / "primary"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"ckpt")

    # No jobs record at all: not provably smoke.
    assert _smoke_leftover_config(state_root, "ds", run_dir) is None

    # A real (non-smoke) previous launch: not provably smoke.
    config_path = _worker_config_path(state_root, "ds")
    atomic_write_json(
        config_path, {"smoke_epochs": None, "run_dir": str(run_dir)}
    )
    assert _smoke_leftover_config(state_root, "ds", run_dir) is None

    # A smoke launch against a DIFFERENT run dir: not this leftover.
    atomic_write_json(
        config_path, {"smoke_epochs": 2, "run_dir": str(tmp_path / "other")}
    )
    assert _smoke_leftover_config(state_root, "ds", run_dir) is None

    # A smoke launch against this run dir: provably smoke.
    atomic_write_json(config_path, {"smoke_epochs": 2, "run_dir": str(run_dir)})
    assert _smoke_leftover_config(state_root, "ds", run_dir) is not None

    quarantined = _quarantine_run_dir(run_dir)
    assert not run_dir.exists()
    assert (quarantined / "weights" / "last.pt").is_file()
    assert quarantined.name.startswith("primary-smoke-")


def test_child_registry_terminates_live_children_and_blocks_new_ones():
    import subprocess
    import sys

    from va_bench.rf100vl_train import _ChildProcesses

    children = _ChildProcesses()
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        children.add(sleeper)
        assert sleeper.poll() is None
        children.request_stop()
        assert sleeper.poll() is not None, "request_stop must terminate live children"
        assert children.stopping.is_set()

        # A child that starts after the stop is killed immediately rather than
        # being allowed to run on unsupervised.
        late = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            children.add(late)
            assert late.poll() is not None
        finally:
            if late.poll() is None:
                late.kill()
                late.wait()
    finally:
        if sleeper.poll() is None:
            sleeper.kill()
            sleeper.wait()
