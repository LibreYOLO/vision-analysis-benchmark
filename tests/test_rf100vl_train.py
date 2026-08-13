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


# Families whose campaign precision is deliberately not fp32, and why. A new
# family cannot quietly pick its own precision: it has to be added here, which
# is the moment to justify it against the upstream reference.
NON_FP32_FAMILIES = {
    # EdgeCrafter trains with use_amp + GradScaler upstream
    # (ecdetseg/configs/ecdet/ecdet.yml), and LibreYOLO's ECConfig defaults to
    # amp with the inherited float16 dtype. fp32 would match neither.
    "ec": "fp16",
    # rf-detr trains under autocast with the dtype hardcoded to torch.bfloat16
    # (rfdetr/engine.py::get_autocast_args at tag 1.2.0, the release current
    # when Roboflow published their RF100-VL numbers) and ModelConfig.amp
    # defaults to True. fp32 would be a deviation, not the conservative choice.
    "rfdetr": "bfloat16",
}


def test_every_campaign_family_has_a_protocol_recipe():
    for family in set(list_families()):
        recipe = rf100vl_train.load_recipe(
            rf100vl_train.recipe_path_for_family(family),
            family=family,
        )
        precision = recipe["protocol"]["precision"]
        assert precision in rf100vl_train.PRECISIONS
        assert precision == NON_FP32_FAMILIES.get(family, "fp32"), (
            f"{family} uses {precision}; add it to NON_FP32_FAMILIES with a "
            "reason if that is intended"
        )


def test_yolov9_e2e_recipe_keeps_training_capture_off():
    """Training capture is deliberately unsupported for the e2e dual-assignment
    head (YOLO9Trainer.cuda_graph_train_spec requires the plain DDetect head),
    so the recipe must not ask for it until the library supports it and a
    parity gate covers it."""
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family("yolov9-e2e"),
        family="yolov9-e2e",
    )
    assert recipe["protocol"]["cuda_graph"] is False


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
    assert plan["physical_batch"] == 4
    assert plan["effective_batch"] == 16
    assert plan["expected_batches_per_epoch_minimum"] == 1


def test_rfdetr_recipe_matches_roboflows_own_rf100vl_settings():
    """RF-DETR's authors state their RF100-VL numbers came from the rf-detr
    defaults with one override: batch 16 and grad accum 1, not the library's
    default batch 4 / accum 4. Autocast in rf-detr 1.2.0 is hardcoded to
    bfloat16, so fp32 is a deviation too. Both are easy to reintroduce by
    copying another family's recipe, and neither is visible in a result table,
    so pin them here.
    """
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family("rfdetr"),
        family="rfdetr",
    )
    protocol = recipe["protocol"]
    assert protocol["physical_batch"] == 16, "Roboflow ran batch 16, grad accum 1"
    assert protocol["precision"] == "bfloat16", "rf-detr autocasts to bfloat16"
    # RF-DETR's transform sets wants_unresized_image, so libreyolo/data/cache.py
    # gives it the PRE-resize cache point: full-resolution decoded pixels, which
    # skips JPEG decode but not the resize. That trades an order of magnitude of
    # disk per image for decode only, and filling 100 datasets' worth of it is
    # what deadlocked this box once already. Families that take the post-resize
    # point (yolox, yolov9, yolonas) do cache, and should.
    assert not protocol.get("cache", False)
    assert recipe["train"]["multi_scale"] is True
    # Sizes n/s/m inherit the batch; only l steps down, and only for VRAM.
    for size in ("n", "s", "m"):
        assert "physical_batch" not in recipe["sizes"][size]


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


@pytest.mark.parametrize(
    ("precision", "expect_amp", "expect_dtype"),
    [
        ("fp32", False, "bfloat16"),
        ("bfloat16", True, "bfloat16"),
        ("fp16", True, "float16"),
    ],
)
def test_recipe_precision_selects_the_autocast_dtype(
    precision, expect_amp, expect_dtype, tmp_path
):
    """The autocast dtype must follow the recipe.

    Regression: amp was derived as ``precision == "bfloat16"`` and the dtype
    was the literal ``"bfloat16"``, so an fp16 recipe trained in fp32 while
    claiming fp16 in its stats and submission.
    """
    spec = get_spec("ec-s")
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family("ec"),
        family="ec",
    )
    recipe["protocol"]["precision"] = precision
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
    assert kwargs["amp"] is expect_amp
    assert kwargs["amp_dtype"] == expect_dtype


def test_ec_recipe_matches_the_upstream_reference_precision():
    """EdgeCrafter trains with use_amp + GradScaler upstream, and LibreYOLO's
    ECConfig defaults to amp with the inherited float16 dtype. The campaign
    recipe follows both rather than forcing a precision neither uses."""
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family("ec"),
        family="ec",
    )
    assert recipe["protocol"]["precision"] == "fp16"


@pytest.mark.parametrize("model_key", ["ec-s", "rtmdet-t", "picodet-s"])
def test_experimental_trainer_families_get_the_opt_in(model_key, tmp_path):
    """ec/rtmdet/picodet trainers refuse to start without allow_experimental;
    the harness opts in because issue #674 scopes them as campaign families."""
    spec = get_spec(model_key)
    recipe = rf100vl_train.load_recipe(
        rf100vl_train.recipe_path_for_family(spec.family),
        family=spec.family,
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
    assert kwargs["allow_experimental"] is True


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


def test_finished_dataset_reclaims_cache_and_resume_checkpoint(tmp_path):
    """A finished dataset keeps what is read again and drops what is not.

    The post-resize cache and last.pt are the two largest consumers on a
    campaign box. A campaign has already filled a 250 GB disk and deadlocked
    every worker because nothing removed them.
    """
    dataset = tmp_path / "some-dataset"
    (dataset / "train").mkdir(parents=True)
    image = dataset / "train" / "a.jpg"
    image.write_bytes(b"jpeg")
    cached = dataset / "train" / "a.jpg.npy"
    cached.write_bytes(b"x" * 4096)
    (dataset / "train" / "_annotations.coco.json").write_text("{}")

    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"y" * 2048)
    (weights / "best.pt").write_bytes(b"z" * 1024)

    freed = rf100vl_train.reclaim_finished_dataset(dataset, tmp_path / "run")

    assert freed == 4096 + 2048
    assert not cached.exists()
    assert not (weights / "last.pt").exists()
    # what the uploader ships and what the images are must survive
    assert (weights / "best.pt").read_bytes() == b"z" * 1024
    assert image.read_bytes() == b"jpeg"
    assert (dataset / "train" / "_annotations.coco.json").exists()


def test_keep_cache_opts_out_of_reclaiming_the_cache(tmp_path):
    dataset = tmp_path / "some-dataset"
    (dataset / "train").mkdir(parents=True)
    cached = dataset / "train" / "a.jpg.npy"
    cached.write_bytes(b"x" * 4096)
    weights = tmp_path / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"y" * 2048)

    freed = rf100vl_train.reclaim_finished_dataset(
        dataset, tmp_path / "run", keep_cache=True
    )

    assert cached.exists()
    assert freed == 2048


def test_every_recipe_disables_periodic_snapshots():
    """LibreYOLO's TrainConfig defaults to save_period=10, so families whose
    trainer honours it write a full checkpoint every 10 epochs: ~1000 files and
    150 GB across one campaign, on a 250 GB box. Nothing reads them. Resume
    uses last.pt, selection uses best.pt, and the uploader ships best.pt. A
    campaign already hit 97% disk on these alone."""
    for family in set(list_families()):
        recipe = rf100vl_train.load_recipe(
            rf100vl_train.recipe_path_for_family(family),
            family=family,
        )
        assert recipe["train"].get("save_period") == 0, (
            f"{family} does not disable save_period; periodic snapshots will "
            "fill the campaign box"
        )


def test_submission_recipe_sha_reads_only_this_model(tmp_path):
    """Submissions land in one shared directory that accumulates every earlier
    campaign. Reading the newest of all of them compared a neighbour's recipe
    against this run and refused the upload."""
    import json as _json
    from va_bench import artifacts

    def write(name, sha):
        (tmp_path / name).write_text(
            _json.dumps({"rf100vl": {"recipe_sha256": sha}}), encoding="utf-8"
        )

    mine = "a" * 64
    theirs = "b" * 64
    write("ec-s__pytorch__cuda__x__20260101T000000Z.json", mine)
    # sorts after ours, and is what the old code would have picked
    write("yolox-m__pytorch__cuda__x__20260909T000000Z.json", theirs)

    assert artifacts._submission_recipe_sha(tmp_path, "ec-s") == mine
    assert artifacts._submission_recipe_sha(tmp_path, "yolox-m") == theirs
