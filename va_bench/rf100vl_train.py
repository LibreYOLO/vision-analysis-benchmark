"""Resumable, one-dataset-per-process RF100-VL training orchestration."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from .models import ModelSpec, get_spec
from .provenance import file_sha256, harness_git_info
from .rf100vl_data import (
    ANNOTATION_FILENAME,
    atomic_write_json,
    canonical_json_sha256,
    dataset_version,
    load_json,
    load_version_lock,
    version_lock_sha256,
)

RECIPE_SCHEMA = "rf100vl.recipe.v1"
STATUS_SCHEMA = "rf100vl.train-status.v1"
STATS_SCHEMA = "rf100vl.train-stats.v1"
FAILURES_SCHEMA = "rf100vl.failures.v1"
PROTOCOL_VERSION = "rf100vl.libreyolo.v1"
EXPECTED_EPOCHS = 100
EXPECTED_EFFECTIVE_BATCH = 16
EXPECTED_SELECTION_METRIC = "valid_mAP50_95"


def _load_libreyolo_protocol_types():
    from libreyolo.training.config import TrainConfig
    from libreyolo.validation.config import ValidationConfig
    from libreyolo.validation.detection_validator import DetectionValidator

    return TrainConfig, ValidationConfig, DetectionValidator


def require_libreyolo_protocol_capabilities() -> dict[str, Any]:
    """Fail before training if LibreYOLO cannot enforce the fixed protocol."""
    from dataclasses import fields

    try:
        train_config_cls, validation_config_cls, validator_cls = (
            _load_libreyolo_protocol_types()
        )
        train_fields = {field.name for field in fields(train_config_cls)}
        validation_fields = {field.name for field in fields(validation_config_cls)}
    except Exception as exc:
        raise RuntimeError(
            "Installed LibreYOLO lacks the RF100-VL protocol API; install the "
            "LibreYOLO revision that provides eval_max_det and amp_dtype."
        ) from exc

    required_train = {"amp_dtype", "max_det", "eval_max_det"}
    required_validation = {"amp_dtype", "max_det", "eval_max_det"}
    missing_train = sorted(required_train - train_fields)
    missing_validation = sorted(required_validation - validation_fields)
    if missing_train or missing_validation or not hasattr(validator_cls, "_coco_max_det"):
        raise RuntimeError(
            "Installed LibreYOLO cannot enforce the RF100-VL protocol "
            f"(missing TrainConfig={missing_train}, "
            f"ValidationConfig={missing_validation}, "
            f"evaluator_plumbing={hasattr(validator_cls, '_coco_max_det')})."
        )

    train_config = train_config_cls(
        amp_dtype="bfloat16",
        max_det=500,
        eval_max_det=500,
    )
    default_validation = validation_config_cls(
        data="__rf100vl_capability_probe__.yaml",
        max_det=300,
        eval_max_det=None,
    )
    protocol_validation = validation_config_cls(
        data="__rf100vl_capability_probe__.yaml",
        max_det=500,
        eval_max_det=500,
        amp_dtype="bfloat16",
    )
    default_validator = object.__new__(validator_cls)
    default_validator.config = default_validation
    protocol_validator = object.__new__(validator_cls)
    protocol_validator.config = protocol_validation
    if (
        getattr(train_config, "eval_max_det", None) != 500
        or default_validator._coco_max_det() != 100
        or protocol_validator._coco_max_det() != 500
    ):
        raise RuntimeError(
            "Installed LibreYOLO exposes RF100-VL options but does not preserve "
            "default AP@100 and opt-in AP@500 semantics."
        )

    try:
        version = importlib.metadata.version("libreyolo")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "validated": True,
        "version": version,
        "amp_dtype": getattr(train_config, "amp_dtype", None),
        "prediction_max_det": getattr(train_config, "max_det", None),
        "eval_max_det": getattr(train_config, "eval_max_det", None),
        "default_eval_max_det": default_validator._coco_max_det(),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_worker_process(seed: int) -> None:
    """Seed the child before model construction and class-head adaptation."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recipe_path_for_family(family: str) -> Path:
    return Path(__file__).with_name("recipes") / "rf100vl" / f"{family}.json"


def load_recipe(
    path: str | Path,
    *,
    family: str | None = None,
    protocol_required: bool = True,
) -> dict[str, Any]:
    """Load a versioned recipe and enforce the cross-family protocol skeleton."""
    path = Path(path)
    recipe = load_json(path)
    if recipe.get("schema_version") != RECIPE_SCHEMA:
        raise ValueError(f"Unsupported RF100-VL recipe schema in {path}")
    if family is not None and recipe.get("family") != family:
        raise ValueError(f"Recipe {path} is for family {recipe.get('family')!r}, not {family!r}")
    protocol = recipe.get("protocol")
    train = recipe.get("train")
    if not isinstance(protocol, dict) or not isinstance(train, dict):
        raise ValueError(f"Recipe {path} needs object-valued protocol and train blocks")

    if protocol_required:
        expected = {
            "epochs": EXPECTED_EPOCHS,
            "effective_batch": EXPECTED_EFFECTIVE_BATCH,
            "selection_metric": EXPECTED_SELECTION_METRIC,
            "eval_interval": 1,
            "patience": 0,
            "ema": True,
            "seed": 0,
        }
        mismatches = {
            key: (protocol.get(key), value)
            for key, value in expected.items()
            if protocol.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Recipe {path} violates the RF100-VL fixed skeleton: {mismatches}")
        if protocol.get("precision") not in {"fp32", "bfloat16"}:
            raise ValueError(
                f"Recipe {path} precision must be fp32 or bfloat16; "
                "fp16 autocast is not protocol-conformant"
            )

    physical_batch = protocol.get("physical_batch")
    if (
        isinstance(physical_batch, bool)
        or not isinstance(physical_batch, int)
        or physical_batch < 1
    ):
        raise ValueError(f"Recipe {path} has invalid protocol.physical_batch")
    if EXPECTED_EFFECTIVE_BATCH % physical_batch != 0:
        raise ValueError(
            f"Recipe {path} physical batch {physical_batch} does not divide "
            f"effective batch {EXPECTED_EFFECTIVE_BATCH}"
        )
    sizes = recipe.get("sizes", {})
    if not isinstance(sizes, dict):
        raise ValueError(f"Recipe {path} sizes must be an object")
    for variant, override in sizes.items():
        if not isinstance(override, dict):
            raise ValueError(f"Recipe {path} size override {variant!r} must be an object")
        variant_batch = int(override.get("physical_batch", physical_batch))
        if variant_batch < 1 or EXPECTED_EFFECTIVE_BATCH % variant_batch != 0:
            raise ValueError(
                f"Recipe {path} size {variant!r} has incompatible physical batch {variant_batch}"
            )
        if "fallback_physical_batch" in override:
            variant_fallback = int(override["fallback_physical_batch"])
            if (
                variant_fallback < 1
                or EXPECTED_EFFECTIVE_BATCH % variant_fallback != 0
                or variant_fallback >= variant_batch
            ):
                raise ValueError(
                    f"Recipe {path} size {variant!r} has ineffective fallback "
                    f"batch {variant_fallback}"
                )
    fallback = recipe.get("dense_oom_fallback")
    if isinstance(fallback, dict):
        fallback_batch = int(fallback.get("physical_batch", 0))
        if fallback_batch < 1 or EXPECTED_EFFECTIVE_BATCH % fallback_batch != 0:
            raise ValueError(
                f"Recipe {path} has incompatible dense fallback batch {fallback_batch}"
            )
    if recipe.get("family") == "ec":
        if str(train.get("optimizer", "")).lower() != "adamw":
            raise ValueError("The EC RF100-VL recipe must use AdamW")
        if float(train.get("mosaic_prob", -1.0)) != 0.0:
            raise ValueError("The EC RF100-VL recipe must disable mosaic")
    return recipe


def _annotation_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / split / ANNOTATION_FILENAME


def inspect_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Inspect train annotations for class order, size, and density guards."""
    dataset_dir = Path(dataset_dir)
    train_path = _annotation_path(dataset_dir, "train")
    annotation = load_json(train_path)
    categories = annotation.get("categories")
    images = annotation.get("images")
    annotations = annotation.get("annotations")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"RF100-VL train annotations have no categories: {train_path}")
    if not isinstance(images, list) or not images:
        raise ValueError(f"RF100-VL train annotations have no images: {train_path}")
    if not isinstance(annotations, list):
        raise ValueError(f"RF100-VL train annotations have no annotation list: {train_path}")

    ordered = sorted(categories, key=lambda category: int(category["id"]))
    names = [str(category["name"]) for category in ordered]
    if len(set(names)) != len(names):
        raise ValueError(f"RF100-VL category names are not unique: {train_path}")

    per_image = Counter(int(item["image_id"]) for item in annotations)
    return {
        "num_train_images": len(images),
        "num_classes": len(ordered),
        "category_ids": [int(category["id"]) for category in ordered],
        "names": names,
        "num_annotations": len(annotations),
        "max_annotations_per_image": max(per_image.values(), default=0),
        "train_annotations_sha256": file_sha256(train_path),
    }


def generate_data_yaml(
    dataset_dir: str | Path,
    output_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Generate the exact LibreYOLO data YAML for one RF100-VL dataset."""
    dataset_dir = Path(dataset_dir).resolve()
    output_path = Path(output_path)
    for split in ("train", "valid", "test"):
        annotation_path = _annotation_path(dataset_dir, split)
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"RF100-VL dataset is missing {split}/{ANNOTATION_FILENAME}: {dataset_dir}"
            )

    facts = inspect_dataset(dataset_dir)
    config = {
        "path": str(dataset_dir),
        "train": "train",
        "val": "valid",
        "test": "test",
        "annotations": {
            "train": f"train/{ANNOTATION_FILENAME}",
            "val": f"valid/{ANNOTATION_FILENAME}",
            "test": f"test/{ANNOTATION_FILENAME}",
        },
        "nc": facts["num_classes"],
        "names": {index: name for index, name in enumerate(facts["names"])},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return output_path, facts


def _size_overrides(recipe: dict[str, Any], spec: ModelSpec) -> dict[str, Any]:
    sizes = recipe.get("sizes", {})
    if not isinstance(sizes, dict):
        raise ValueError("Recipe sizes must be an object")
    override = sizes.get(spec.variant, {})
    if not isinstance(override, dict):
        raise ValueError(f"Recipe size override for {spec.variant!r} must be an object")
    return dict(override)


def select_batch_plan(
    recipe: dict[str, Any],
    spec: ModelSpec,
    dataset_facts: dict[str, Any],
    *,
    force_fallback: bool = False,
) -> dict[str, Any]:
    """Choose the immutable per-dataset physical/effective batch plan."""
    protocol = recipe["protocol"]
    size_override = _size_overrides(recipe, spec)
    primary_batch = int(size_override.get("physical_batch", protocol["physical_batch"]))
    batch = primary_batch
    variant = "primary"
    reason = None

    fallback = recipe.get("dense_oom_fallback")
    if spec.family == "rfdetr" and isinstance(fallback, dict):
        threshold = int(fallback["max_annotations_per_image_threshold"])
        fallback_batch = int(
            size_override.get(
                "fallback_physical_batch",
                fallback["physical_batch"],
            )
        )
        dense = int(dataset_facts["max_annotations_per_image"]) >= threshold
        if dense or force_fallback:
            batch = fallback_batch
            variant = "fallback"
            reason = "dense_dataset_start" if dense and not force_fallback else "mid_run_cuda_oom"

    effective_batch = int(protocol["effective_batch"])
    if effective_batch % batch != 0:
        raise ValueError(
            f"Physical batch {batch} does not divide effective batch {effective_batch}"
        )
    num_images = int(dataset_facts["num_train_images"])
    expected_batches = 1 if num_images < batch else num_images // batch
    if expected_batches < 1:
        raise AssertionError(f"Micro-dataset guard failed: {num_images} images, batch {batch}")
    return {
        "run_variant": variant,
        "selection_reason": reason,
        "physical_batch": batch,
        "effective_batch": effective_batch,
        "gradient_accumulation_steps": effective_batch // batch,
        "num_train_images": num_images,
        "expected_batches_per_epoch_minimum": expected_batches,
    }


def build_train_kwargs(
    recipe: dict[str, Any],
    spec: ModelSpec,
    batch_plan: dict[str, Any],
    *,
    data_yaml: Path,
    run_dir: Path,
    resume: bool,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    """Normalize family knobs under the fixed RF100-VL training skeleton."""
    protocol = recipe["protocol"]
    kwargs = dict(recipe["train"])
    kwargs.update(_size_overrides(recipe, spec))
    kwargs.pop("physical_batch", None)
    kwargs.pop("fallback_physical_batch", None)
    precision = str(protocol["precision"])
    kwargs.update(
        {
            "data": str(data_yaml),
            "epochs": int(smoke_epochs or protocol["epochs"]),
            "batch": int(batch_plan["physical_batch"]),
            "nbs": int(batch_plan["effective_batch"]),
            "imgsz": int(kwargs.get("imgsz", spec.input_size)),
            "device": "0",
            "seed": int(protocol["seed"]),
            "project": str(run_dir.parent),
            "name": run_dir.name,
            "exist_ok": True,
            "resume": resume,
            "amp": precision == "bfloat16",
            "amp_dtype": "bfloat16",
            "patience": 0,
            "eval_interval": 1,
            "ema": True,
            "max_det": 500,
            "eval_max_det": 500,
        }
    )
    if spec.family == "ec":
        kwargs["allow_experimental"] = True
        kwargs["optimizer"] = "adamw"
        kwargs["mosaic_prob"] = 0.0
    return kwargs


def _run_signature(
    *,
    model_key: str,
    recipe_sha256: str,
    versions_sha256: str,
    dataset_name: str,
    dataset_version_id: int,
    dataset_facts: dict[str, Any],
    batch_plan: dict[str, Any],
    train_kwargs: dict[str, Any],
) -> str:
    return canonical_json_sha256(
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": model_key,
            "recipe_sha256": recipe_sha256,
            "versions_sha256": versions_sha256,
            "dataset": dataset_name,
            "dataset_version": dataset_version_id,
            "train_annotations_sha256": dataset_facts["train_annotations_sha256"],
            "run_variant": batch_plan["run_variant"],
            "physical_batch": batch_plan["physical_batch"],
            "effective_batch": batch_plan["effective_batch"],
            "epochs": train_kwargs["epochs"],
            "imgsz": train_kwargs["imgsz"],
            "precision": {
                "amp": train_kwargs["amp"],
                "amp_dtype": train_kwargs["amp_dtype"],
            },
            "evaluation": {
                "max_det": train_kwargs["max_det"],
                "eval_max_det": train_kwargs["eval_max_det"],
            },
        }
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in message and ("out of memory" in message or "memory allocation" in message)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def run_dataset_worker(config_path: str | Path) -> int:
    """Train one dataset inside an isolated child process."""
    config = load_json(config_path)
    result_path = Path(config["worker_result_path"])
    started_at = utc_now()
    start_time = time.perf_counter()
    try:
        from PIL import Image, ImageFile

        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        libreyolo_capabilities = require_libreyolo_protocol_capabilities()
        from libreyolo import LibreYOLO

        spec = get_spec(config["model_key"])
        dataset_dir = Path(config["dataset_dir"])
        run_dir = Path(config["run_dir"])
        data_yaml, dataset_facts = generate_data_yaml(
            dataset_dir,
            run_dir / "data.yaml",
        )
        recipe_path = Path(config["recipe_path"])
        recipe = load_recipe(recipe_path, family=spec.family)
        seed_worker_process(int(recipe["protocol"]["seed"]))
        batch_plan = dict(config["batch_plan"])
        resume = bool(config["resume"])
        train_kwargs = build_train_kwargs(
            recipe,
            spec,
            batch_plan,
            data_yaml=data_yaml,
            run_dir=run_dir,
            resume=resume,
            smoke_epochs=config.get("smoke_epochs"),
        )

        checkpoint = run_dir / "weights" / "last.pt"
        model_path = str(checkpoint) if resume else spec.weight_file
        model = LibreYOLO(
            model_path=model_path,
            size=spec.constructor_size,
            device="cuda:0",
        )
        train_result = model.train(**train_kwargs)
        best_checkpoint_value = train_result.get("best_checkpoint")
        best_checkpoint = (
            Path(best_checkpoint_value)
            if best_checkpoint_value
            else run_dir / "weights" / "best.pt"
        )
        if not best_checkpoint.is_file():
            raise FileNotFoundError(
                f"LibreYOLO training returned no best checkpoint: {best_checkpoint}"
            )

        target_checkpoint = Path(config["target_checkpoint"])
        _atomic_copy(best_checkpoint, target_checkpoint)
        wall_seconds = time.perf_counter() - start_time
        recipe_sha256 = file_sha256(recipe_path)
        protocol_conformant = (
            config.get("smoke_epochs") is None
            and int(train_kwargs["epochs"]) == EXPECTED_EPOCHS
            and libreyolo_capabilities.get("validated") is True
        )
        stats = {
            "schema_version": STATS_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_conformant": protocol_conformant,
            "libreyolo_capabilities": libreyolo_capabilities,
            "dataset": config["dataset_name"],
            "dataset_version": config["dataset_version"],
            "model_key": spec.key,
            "family": spec.family,
            "variant": spec.variant,
            "seed": int(recipe["protocol"]["seed"]),
            "precision": recipe["protocol"]["precision"],
            "recipe": {
                "file": str(recipe_path),
                "sha256": recipe_sha256,
            },
            "versions_sha256": config["versions_sha256"],
            "run_signature": config["run_signature"],
            "run_variant": batch_plan["run_variant"],
            "restart_reason": config.get("restart_reason"),
            "batch": batch_plan,
            "data": {
                **dataset_facts,
                "yaml": str(data_yaml),
                "yaml_sha256": file_sha256(data_yaml),
            },
            "epochs_requested": int(train_kwargs["epochs"]),
            "best_epoch": int(train_result.get("best_epoch", 0)),
            "valid_mAP50": float(train_result.get("best_mAP50", 0.0)),
            "valid_mAP50_95": float(train_result.get("best_mAP50_95", 0.0)),
            "wall_seconds": wall_seconds,
            "resumed": resume,
            "started_at": started_at,
            "finished_at": utc_now(),
            "best_checkpoint": str(target_checkpoint),
            "source_best_checkpoint": str(best_checkpoint),
            "harness": harness_git_info(),
        }
        stats_path = target_checkpoint.parent / "stats.json"
        atomic_write_json(stats_path, stats)
        atomic_write_json(
            result_path,
            {
                "state": "done",
                "stats_path": str(stats_path),
                "target_checkpoint": str(target_checkpoint),
                "wall_seconds": wall_seconds,
            },
        )
        return 0
    except BaseException as exc:
        state = "oom" if _is_cuda_oom(exc) else "failed"
        atomic_write_json(
            result_path,
            {
                "state": state,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-80:],
                "started_at": started_at,
                "finished_at": utc_now(),
                "wall_seconds": time.perf_counter() - start_time,
            },
        )
        return 86 if state == "oom" else 1


def _status_path(state_root: Path, dataset_name: str) -> Path:
    return state_root / f"{dataset_name}.json"


def _read_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _status_process_is_live(status: dict[str, Any]) -> bool:
    """Return whether a running status still identifies the same child."""
    import psutil

    pid = status.get("pid")
    recorded_create_time = status.get("pid_create_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid < 1
        or not isinstance(recorded_create_time, (int, float))
    ):
        return False
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        if abs(process.create_time() - float(recorded_create_time)) > 0.01:
            return False
        expected_config = status.get("worker_config")
        if expected_config:
            try:
                command = " ".join(process.cmdline())
            except (psutil.AccessDenied, psutil.ZombieProcess):
                command = ""
            if command and str(expected_config) not in command:
                return False
        return True
    except (psutil.Error, OSError):
        return False


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def train_image_count(data_dir: Path, name: str) -> int:
    """Cheap size proxy: count image files under ``<dataset>/train``.

    A directory listing rather than a COCO parse, because this runs for every
    dataset before any training starts and the annotation files are large.
    """
    train_dir = Path(data_dir) / name / "train"
    try:
        return sum(
            1 for path in train_dir.iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES
        )
    except OSError:
        return 0


def order_longest_first(names: list[str], data_dir: Path) -> list[str]:
    """Longest Processing Time first: start the big datasets before the small.

    With an alphabetical queue, a dataset that takes hours can be pulled last
    and run alone on one GPU while seven idle, so the campaign's finish time is
    set by one job rather than by the work. Sorting longest-first is the
    classic LPT heuristic, and its makespan is provably within
    (4/3 - 1/(3m)) of optimal, versus a worst case of 2x for an arbitrary
    order.

    Per-epoch cost is roughly ``fixed + per_image * images`` (fitted on a real
    campaign at 9.1s + 13.9ms per image), and epochs are fixed by the protocol,
    so ordering by image count descending IS ordering by predicted duration: no
    timing model is needed to get the ordering right, only a monotone proxy.

    This changes scheduling only. Each dataset trains exactly as before, so
    results are unaffected; ties break by name to keep the order deterministic.
    """
    sizes = {name: train_image_count(data_dir, name) for name in names}
    return sorted(names, key=lambda name: (-sizes[name], name))


def reconcile_statuses(
    state_root: str | Path,
    dataset_names: list[str],
) -> set[str]:
    """Reconcile dead children while preserving any still-live training job."""
    state_root = Path(state_root)
    active: set[str] = set()
    for name in dataset_names:
        path = _status_path(state_root, name)
        status = _read_status(path)
        if status is not None and status.get("state") == "running":
            if _status_process_is_live(status):
                active.add(name)
                continue
            result_path = status.get("worker_result_path")
            if result_path and Path(result_path).is_file():
                try:
                    result = load_json(result_path)
                except (OSError, TypeError, ValueError):
                    result = {}
                if result.get("state") == "done":
                    status.update(
                        {
                            "state": "done",
                            "finished_at": result.get("finished_at", utc_now()),
                            "stats_path": result.get("stats_path"),
                            "target_checkpoint": result.get("target_checkpoint"),
                            "reconciled_at": utc_now(),
                            "reconcile_reason": "child completed after orchestrator exit",
                        }
                    )
                    status.pop("pid", None)
                    status.pop("pid_create_time", None)
                    atomic_write_json(path, status)
                    continue
            status["state"] = "pending"
            status["reconciled_at"] = utc_now()
            status["reconcile_reason"] = "stale running state from previous orchestrator"
            status.pop("pid", None)
            status.pop("pid_create_time", None)
            atomic_write_json(path, status)
    return active


def _completed_run_matches(
    *,
    status: dict[str, Any] | None,
    target_checkpoint: Path,
    stats_path: Path,
    dataset_dir: Path,
    dataset_name: str,
    dataset_version_id: int,
    model_key: str,
    recipe_sha256: str,
    versions_sha256: str,
    smoke_epochs: int | None,
) -> bool:
    """Return whether completed artifacts match the current immutable inputs."""
    if (
        status is None
        or status.get("state") != "done"
        or not target_checkpoint.is_file()
        or not stats_path.is_file()
    ):
        return False
    try:
        stats = load_json(stats_path)
        facts = inspect_dataset(dataset_dir)
    except (OSError, KeyError, TypeError, ValueError):
        return False

    data = stats.get("data")
    recipe = stats.get("recipe")
    expected_epochs = int(smoke_epochs or EXPECTED_EPOCHS)
    expected_protocol_conformant = smoke_epochs is None
    return (
        stats.get("schema_version") == STATS_SCHEMA
        and stats.get("protocol_version") == PROTOCOL_VERSION
        and stats.get("protocol_conformant") is expected_protocol_conformant
        and isinstance(stats.get("libreyolo_capabilities"), dict)
        and stats["libreyolo_capabilities"].get("validated") is True
        and stats["libreyolo_capabilities"].get("eval_max_det") == 500
        and stats["libreyolo_capabilities"].get("default_eval_max_det") == 100
        and stats.get("dataset") == dataset_name
        and stats.get("dataset_version") == dataset_version_id
        and stats.get("model_key") == model_key
        and stats.get("epochs_requested") == expected_epochs
        and isinstance(recipe, dict)
        and recipe.get("sha256") == recipe_sha256
        and stats.get("versions_sha256") == versions_sha256
        and isinstance(data, dict)
        and data.get("train_annotations_sha256") == facts["train_annotations_sha256"]
        and status.get("run_signature") == stats.get("run_signature")
        and status.get("recipe_sha256") == recipe_sha256
        and status.get("versions_sha256") == versions_sha256
    )


def _worker_config_path(state_root: Path, dataset_name: str) -> Path:
    return state_root / "jobs" / f"{dataset_name}.json"


def _worker_result_path(
    state_root: Path,
    dataset_name: str,
    signature: str,
) -> Path:
    return state_root / "worker-results" / f"{dataset_name}.{signature[:16]}.json"


def _terminate_child(process: subprocess.Popen) -> None:
    """Stop one trainer child, escalating to kill if it ignores the signal."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        pass


class _ChildProcesses:
    """Registry of live trainer children so Ctrl-C can stop a campaign.

    Without this, an interrupt reaches the orchestrator's main thread while
    every worker thread sits in ``process.wait()``; the thread pool then joins
    those workers on the way out, so the campaign keeps training and the
    terminal looks hung until the per-dataset timeout expires. An operator who
    cannot stop a run is not in control of it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()
        self.stopping = threading.Event()

    def add(self, process: subprocess.Popen) -> None:
        with self._lock:
            if self.stopping.is_set():
                # The interrupt landed while this child was starting.
                _terminate_child(process)
                return
            self._processes.add(process)

    def discard(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)

    def request_stop(self) -> None:
        self.stopping.set()
        with self._lock:
            live = list(self._processes)
        for process in live:
            _terminate_child(process)


def _heartbeat_line(state_root: Path, names: list[str]) -> str:
    """One campaign progress line, built purely from the on-disk status files."""
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    running: list[str] = []
    for name in names:
        status = _read_status(_status_path(state_root, name)) or {}
        state = str(status.get("state", "pending"))
        counts[state if state in counts else "pending"] += 1
        if state != "running":
            continue
        detail = f"{name}[gpu{status.get('gpu', '?')}]"
        live: dict[str, Any] = {}
        run_dir = status.get("run_dir")
        if run_dir:
            try:
                loaded = load_json(Path(run_dir) / "status.json")
                if isinstance(loaded, dict):
                    live = loaded
            except Exception:
                pass  # trainer may be mid-write; skip the detail this round
        if live:
            detail += (
                f" {live.get('completed_epochs', '?')}/{live.get('total_epochs', '?')}"
            )
            best = live.get("best_metric")
            if isinstance(best, (int, float)):
                detail += f" best {best:.3f}"
            eta = live.get("eta_seconds")
            if isinstance(eta, (int, float)) and eta > 0:
                detail += f" eta {max(1, round(eta / 60))}m"
        running.append(detail)
    line = (
        f"[{utc_now()}] progress: {counts['done']} done, {counts['running']} running, "
        f"{counts['pending']} pending, {counts['failed']} failed"
    )
    if running:
        line += " | " + "; ".join(running)
    return line


def _launch_child(
    worker_config: Path,
    *,
    gpu: str,
    log_path: Path,
    timeout_seconds: float,
    on_started: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[int | None, bool]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] launching on CUDA_VISIBLE_DEVICES={gpu}\n")
        log.flush()
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "va_bench.rf100vl_train",
                    "--worker-config",
                    str(worker_config),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if on_started is not None:
                try:
                    on_started(process)
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
            return process.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            log.write(f"\n[{utc_now()}] dataset exceeded timeout\n")
            return None, True


def _failure_record(
    *,
    dataset_name: str,
    model_key: str,
    gpu: str,
    phase: str,
    worker_result: dict[str, Any],
    exit_code: int | None,
    timed_out: bool,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "model_key": model_key,
        "gpu": gpu,
        "phase": phase,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "state": "timeout" if timed_out else worker_result.get("state", "failed"),
        "exception_type": worker_result.get("exception_type"),
        "message": worker_result.get("message"),
        "traceback_tail": worker_result.get("traceback_tail", []),
        "started_at": worker_result.get("started_at"),
        "finished_at": worker_result.get("finished_at", utc_now()),
    }


def _append_failure(
    failures_path: Path,
    record: dict[str, Any],
    *,
    lock: threading.Lock,
) -> None:
    with lock:
        if failures_path.exists():
            value = load_json(failures_path)
        else:
            value = {
                "schema_version": FAILURES_SCHEMA,
                "failures": [],
            }
        value["failures"].append(record)
        atomic_write_json(failures_path, value)


def _smoke_leftover_config(
    state_root: Path,
    dataset_name: str,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Return the last worker config iff it proves the leftover run is a smoke run.

    The persisted ``jobs/<dataset>.json`` describes the most recent launch of
    this dataset. Only when it explicitly recorded ``smoke_epochs`` AND targeted
    the same run directory is the leftover ``last.pt`` provably a smoke
    artifact. Anything else (a real interrupted run, an edited recipe, an
    unknown directory) must keep the hard refusal: auto-deleting real training
    is worse than asking a human.
    """
    try:
        config = load_json(_worker_config_path(state_root, dataset_name))
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    if config.get("smoke_epochs") is None:
        return None
    if config.get("run_dir") != str(run_dir):
        return None
    return config


def _quarantine_run_dir(run_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantined = run_dir.with_name(f"{run_dir.name}-smoke-{stamp}")
    index = 0
    while quarantined.exists():
        index += 1
        quarantined = run_dir.with_name(f"{run_dir.name}-smoke-{stamp}-{index}")
    run_dir.rename(quarantined)
    return quarantined


def _run_attempt(
    *,
    dataset_name: str,
    model_key: str,
    data_dir: Path,
    weights_root: Path,
    runs_root: Path,
    state_root: Path,
    recipe_path: Path,
    recipe: dict[str, Any],
    version_lock: dict[str, Any],
    versions_sha256: str,
    gpu: str,
    timeout_seconds: float,
    smoke_epochs: int | None,
    force_fallback: bool,
    restart_reason: str | None,
    children: _ChildProcesses | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    spec = get_spec(model_key)
    dataset_dir = data_dir / dataset_name
    facts = inspect_dataset(dataset_dir)
    batch_plan = select_batch_plan(
        recipe,
        spec,
        facts,
        force_fallback=force_fallback,
    )
    run_dir = runs_root / dataset_name / batch_plan["run_variant"]
    data_yaml = run_dir / "data.yaml"
    train_kwargs = build_train_kwargs(
        recipe,
        spec,
        batch_plan,
        data_yaml=data_yaml,
        run_dir=run_dir,
        resume=False,
        smoke_epochs=smoke_epochs,
    )
    recipe_sha256 = file_sha256(recipe_path)
    dataset_version_id = dataset_version(version_lock, dataset_name)
    if dataset_version_id is None:
        raise ValueError(f"No locked dataset version for {dataset_name!r}")
    signature = _run_signature(
        model_key=model_key,
        recipe_sha256=recipe_sha256,
        versions_sha256=versions_sha256,
        dataset_name=dataset_name,
        dataset_version_id=dataset_version_id,
        dataset_facts=facts,
        batch_plan=batch_plan,
        train_kwargs=train_kwargs,
    )

    status_path = _status_path(state_root, dataset_name)
    previous = _read_status(status_path)
    last_checkpoint = run_dir / "weights" / "last.pt"
    resume = last_checkpoint.is_file()
    if resume and (previous is None or previous.get("run_signature") != signature):
        smoke_config = _smoke_leftover_config(state_root, dataset_name, run_dir)
        if smoke_config is not None and smoke_epochs is None:
            quarantined = _quarantine_run_dir(run_dir)
            print(
                f"[{utc_now()}] {dataset_name}: leftover smoke run "
                f"(smoke_epochs={smoke_config.get('smoke_epochs')}) quarantined "
                f"to {quarantined.name}; starting the protocol run fresh",
                flush=True,
            )
            resume = False
        else:
            raise RuntimeError(
                f"Refusing to resume {dataset_name!r}: last.pt exists but the "
                "recorded physical batch/accumulation/recipe signature differs"
            )

    target_checkpoint = weights_root / dataset_name / spec.weight_file
    worker_result_path = _worker_result_path(
        state_root,
        dataset_name,
        signature,
    )
    # A forced rerun can reuse the same signature. Remove only that previous
    # process-result envelope so a child that dies before writing cannot be
    # mistaken for the earlier successful attempt.
    worker_result_path.unlink(missing_ok=True)
    worker_config = {
        "model_key": model_key,
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "dataset_version": dataset_version_id,
        "versions_sha256": versions_sha256,
        "recipe_path": str(recipe_path),
        "run_dir": str(run_dir),
        "target_checkpoint": str(target_checkpoint),
        "worker_result_path": str(worker_result_path),
        "run_signature": signature,
        "batch_plan": batch_plan,
        "resume": resume,
        "smoke_epochs": smoke_epochs,
        "restart_reason": restart_reason,
    }
    worker_config_path = _worker_config_path(state_root, dataset_name)
    atomic_write_json(worker_config_path, worker_config)
    status = {
        "schema_version": STATUS_SCHEMA,
        "state": "running",
        "dataset": dataset_name,
        "model_key": model_key,
        "gpu": gpu,
        "run_variant": batch_plan["run_variant"],
        "batch": batch_plan,
        "run_signature": signature,
        "recipe_sha256": recipe_sha256,
        "versions_sha256": versions_sha256,
        "resume": resume,
        "run_dir": str(run_dir),
        "target_checkpoint": str(target_checkpoint),
        "worker_config": str(worker_config_path),
        "worker_result_path": str(worker_result_path),
        "started_at": utc_now(),
        "restart_reason": restart_reason,
    }
    atomic_write_json(status_path, status)

    launched: subprocess.Popen | None = None

    def record_process(process: subprocess.Popen) -> None:
        import psutil

        nonlocal status, launched
        launched = process
        if children is not None:
            children.add(process)
        status = dict(status)
        status.update(
            {
                "pid": process.pid,
                "pid_create_time": psutil.Process(process.pid).create_time(),
                "launched_at": utc_now(),
            }
        )
        atomic_write_json(status_path, status)

    try:
        exit_code, timed_out = _launch_child(
            worker_config_path,
            gpu=gpu,
            log_path=state_root / "logs" / f"{dataset_name}.log",
            timeout_seconds=timeout_seconds,
            on_started=record_process,
        )
    finally:
        if children is not None and launched is not None:
            children.discard(launched)
    if timed_out:
        worker_result = {
            "state": "timeout",
            "message": f"exceeded {timeout_seconds:.0f} seconds",
            "started_at": status["started_at"],
            "finished_at": utc_now(),
        }
    elif worker_result_path.exists():
        worker_result = load_json(worker_result_path)
    else:
        worker_result = {
            "state": "failed",
            "message": "child exited without writing a worker result",
            "finished_at": utc_now(),
        }
    worker_result["exit_code"] = exit_code
    worker_result["timed_out"] = timed_out
    return worker_result, status, status_path


def orchestrate_training(
    *,
    model_key: str,
    data_dir: str | Path,
    weights_root: str | Path,
    recipe_path: str | Path | None = None,
    gpus: list[str] | None = None,
    datasets: list[str] | None = None,
    limit_datasets: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    timeout_hours: float = 6.0,
    jobs_per_gpu: int = 1,
    on_dataset_complete: Callable[[str], None] | None = None,
    runs_root: str | Path | None = None,
    state_root: str | Path | None = None,
    smoke_epochs: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run a name-addressed dataset queue with ``jobs_per_gpu`` lanes per GPU."""
    # (see order_longest_first for why the queue is not alphabetical)
    # Heartbeat interval for the periodic progress line on stdout. The
    # orchestrator is otherwise silent between launch and summary, which on a
    # multi-hour campaign reads as a hang.
    heartbeat_seconds = 60.0
    libreyolo_capabilities = require_libreyolo_protocol_capabilities()
    spec = get_spec(model_key)
    data_dir = Path(data_dir).resolve()
    weights_root = Path(weights_root).resolve()
    recipe_path = Path(recipe_path or recipe_path_for_family(spec.family)).resolve()
    recipe = load_recipe(recipe_path, family=spec.family)
    version_lock = load_version_lock(data_dir, required=True)
    versions_sha256 = version_lock_sha256(version_lock)
    locked_names = sorted(version_lock["datasets"])

    if datasets is not None:
        requested = set(datasets)
        unknown = requested - set(locked_names)
        if unknown:
            raise ValueError(
                "Requested datasets are absent from versions.json: " + ", ".join(sorted(unknown))
            )
        names = [name for name in locked_names if name in requested]
    else:
        names = locked_names
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    names = [name for index, name in enumerate(names) if index % num_shards == shard_index]
    if limit_datasets is not None:
        if limit_datasets < 1:
            raise ValueError("limit_datasets must be >= 1")
        names = names[:limit_datasets]
    # Scheduling order only; which datasets run and how they train is
    # untouched. Applied after sharding and limiting so both stay addressed by
    # name and a resumed run picks the same set.
    names = order_longest_first(names, data_dir)
    if not names:
        raise ValueError("No RF100-VL datasets selected")

    gpus = [str(value) for value in (gpus or ["0"])]
    if not gpus:
        raise ValueError("At least one GPU id is required")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU ids must be unique (use --jobs-per-gpu to pack a card)")
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1")
    if timeout_hours <= 0:
        raise ValueError("timeout_hours must be positive")
    if smoke_epochs is not None and smoke_epochs < 1:
        raise ValueError("smoke_epochs must be >= 1")

    runs_root = Path(runs_root or weights_root / ".runs" / model_key).resolve()
    state_root = Path(state_root or weights_root / ".state" / model_key).resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    failures_path = state_root / "failures.json"
    rerun_path = state_root / "rerun.json"
    failure_lock = threading.Lock()

    active_running = reconcile_statuses(state_root, names)
    work: queue.Queue[str] = queue.Queue()
    skipped_done: list[str] = []
    recipe_sha256 = file_sha256(recipe_path)
    for name in names:
        if name in active_running:
            continue
        status = _read_status(_status_path(state_root, name))
        target = weights_root / name / spec.weight_file
        stats = weights_root / name / "stats.json"
        version_id = dataset_version(version_lock, name)
        if version_id is None:
            raise ValueError(f"No locked dataset version for {name!r}")
        if not force and _completed_run_matches(
            status=status,
            target_checkpoint=target,
            stats_path=stats,
            dataset_dir=data_dir / name,
            dataset_name=name,
            dataset_version_id=version_id,
            model_key=model_key,
            recipe_sha256=recipe_sha256,
            versions_sha256=versions_sha256,
            smoke_epochs=smoke_epochs,
        ):
            skipped_done.append(name)
            continue
        pending = dict(status or {})
        pending.update(
            {
                "schema_version": STATUS_SCHEMA,
                "state": "pending",
                "dataset": name,
                "model_key": model_key,
                "queued_at": utc_now(),
            }
        )
        atomic_write_json(_status_path(state_root, name), pending)
        work.put(name)

    completed: list[str] = []
    failed: list[str] = []
    interrupted: list[str] = []
    outcome_lock = threading.Lock()
    children = _ChildProcesses()

    def consume(gpu: str) -> None:
        while not children.stopping.is_set():
            try:
                name = work.get_nowait()
            except queue.Empty:
                return
            try:
                old_status = _read_status(_status_path(state_root, name))
                force_fallback = bool(old_status and old_status.get("run_variant") == "fallback")
                worker_result, status, status_path = _run_attempt(
                    children=children,
                    dataset_name=name,
                    model_key=model_key,
                    data_dir=data_dir,
                    weights_root=weights_root,
                    runs_root=runs_root,
                    state_root=state_root,
                    recipe_path=recipe_path,
                    recipe=recipe,
                    version_lock=version_lock,
                    versions_sha256=versions_sha256,
                    gpu=gpu,
                    timeout_seconds=timeout_hours * 3600,
                    smoke_epochs=smoke_epochs,
                    force_fallback=force_fallback,
                    restart_reason=(
                        old_status.get("restart_reason") if old_status is not None else None
                    ),
                )
                if children.stopping.is_set() and worker_result.get("state") != "done":
                    # We killed this child on purpose. Leave the dataset
                    # resumable rather than recording an operator's Ctrl-C as
                    # a training failure; the signature is unchanged, so a
                    # re-run picks up from the epoch-boundary last.pt.
                    stopped_status = dict(status)
                    stopped_status.update(
                        {
                            "state": "pending",
                            "interrupted_at": utc_now(),
                        }
                    )
                    stopped_status.pop("pid", None)
                    stopped_status.pop("pid_create_time", None)
                    atomic_write_json(status_path, stopped_status)
                    with outcome_lock:
                        interrupted.append(name)
                    return
                if (
                    worker_result.get("state") == "oom"
                    and spec.family == "rfdetr"
                    and status.get("run_variant") == "primary"
                ):
                    fallback_pending = dict(status)
                    fallback_pending.update(
                        {
                            "state": "pending",
                            "run_variant": "fallback",
                            "restart_reason": "mid_run_cuda_oom",
                            "updated_at": utc_now(),
                        }
                    )
                    atomic_write_json(status_path, fallback_pending)
                    worker_result, status, status_path = _run_attempt(
                        dataset_name=name,
                        model_key=model_key,
                        data_dir=data_dir,
                        weights_root=weights_root,
                        runs_root=runs_root,
                        state_root=state_root,
                        recipe_path=recipe_path,
                        recipe=recipe,
                        version_lock=version_lock,
                        versions_sha256=versions_sha256,
                        gpu=gpu,
                        timeout_seconds=timeout_hours * 3600,
                        smoke_epochs=smoke_epochs,
                        force_fallback=True,
                        restart_reason="mid_run_cuda_oom",
                    )

                if worker_result.get("state") == "done":
                    final_status = dict(status)
                    final_status.update(
                        {
                            "state": "done",
                            "finished_at": utc_now(),
                            "stats_path": worker_result.get("stats_path"),
                            "target_checkpoint": worker_result.get("target_checkpoint"),
                        }
                    )
                    final_status.pop("pid", None)
                    final_status.pop("pid_create_time", None)
                    atomic_write_json(status_path, final_status)
                    with outcome_lock:
                        completed.append(name)
                    # Tell the syncer AFTER the status file is on disk, so a
                    # sync triggered by this always sees the finished state
                    # rather than racing the write it was triggered by.
                    if on_dataset_complete is not None:
                        try:
                            on_dataset_complete(name)
                        except Exception:
                            pass  # uploading must never fail a finished dataset
                else:
                    record = _failure_record(
                        dataset_name=name,
                        model_key=model_key,
                        gpu=gpu,
                        phase="train",
                        worker_result=worker_result,
                        exit_code=worker_result.get("exit_code"),
                        timed_out=bool(worker_result.get("timed_out")),
                    )
                    _append_failure(failures_path, record, lock=failure_lock)
                    final_status = dict(status)
                    final_status.update(
                        {
                            "state": "failed",
                            "finished_at": utc_now(),
                            "failure": record,
                        }
                    )
                    final_status.pop("pid", None)
                    final_status.pop("pid_create_time", None)
                    atomic_write_json(status_path, final_status)
                    with outcome_lock:
                        failed.append(name)
            except BaseException as exc:
                record = _failure_record(
                    dataset_name=name,
                    model_key=model_key,
                    gpu=gpu,
                    phase="orchestrator",
                    worker_result={
                        "state": "failed",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback_tail": traceback.format_exc().splitlines()[-80:],
                        "finished_at": utc_now(),
                    },
                    exit_code=None,
                    timed_out=False,
                )
                _append_failure(failures_path, record, lock=failure_lock)
                status = _read_status(_status_path(state_root, name)) or {
                    "schema_version": STATUS_SCHEMA,
                    "dataset": name,
                    "model_key": model_key,
                }
                status.update(
                    {
                        "state": "failed",
                        "finished_at": utc_now(),
                        "failure": record,
                    }
                )
                atomic_write_json(_status_path(state_root, name), status)
                with outcome_lock:
                    failed.append(name)
            finally:
                work.task_done()

    stop_heartbeat = threading.Event()

    def emit_heartbeat() -> None:
        while not stop_heartbeat.wait(heartbeat_seconds):
            try:
                print(_heartbeat_line(state_root, names), flush=True)
            except Exception:
                pass  # a broken progress line must never touch the campaign

    heartbeat_thread = threading.Thread(
        target=emit_heartbeat, name="rf100vl-heartbeat", daemon=True
    )
    heartbeat_thread.start()
    # One lane per concurrent training. Packing several onto a card is the
    # only way to raise utilization without touching the protocol: each lane
    # runs an ordinary independent training at the recipe's physical batch, so
    # every per-run computation is byte-for-byte what it would have been alone.
    # Raising the batch size instead would change the thing under test, and
    # gradient accumulation is not numerically equivalent for this family
    # (BatchNorm statistics and a batch-global loss normalizer both break it).
    lanes = [gpu for gpu in gpus for _ in range(jobs_per_gpu)]
    executor = ThreadPoolExecutor(max_workers=len(lanes))
    futures: list[Any] = []
    try:
        futures = [executor.submit(consume, gpu) for gpu in lanes]
        for future in futures:
            future.result()
    except KeyboardInterrupt:
        print(
            f"\n[{utc_now()}] interrupt received: stopping trainers. Datasets in "
            "flight stay resumable; re-run the same command to continue.",
            flush=True,
        )
        children.request_stop()
        for future in futures:
            try:
                future.result(timeout=120)
            except BaseException:
                pass
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=5.0)
        executor.shutdown(wait=True)

    rerun = {
        "schema_version": "rf100vl.rerun.v1",
        "model_key": model_key,
        "datasets": sorted(failed),
        "generated_at": utc_now(),
    }
    atomic_write_json(rerun_path, rerun)
    summary = {
        "schema_version": "rf100vl.train-summary.v1",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_conformant": (
            smoke_epochs is None
            and not failed
            and not interrupted
            and not active_running
            and libreyolo_capabilities.get("validated") is True
        ),
        "libreyolo_capabilities": libreyolo_capabilities,
        "model_key": model_key,
        "recipe": {
            "file": str(recipe_path),
            "sha256": file_sha256(recipe_path),
        },
        "versions": {
            "file": str(data_dir / "versions.json"),
            "sha256": versions_sha256,
        },
        "selected": names,
        "completed": sorted(completed),
        "skipped_done": sorted(skipped_done),
        "failed": sorted(failed),
        "interrupted": sorted(interrupted),
        "active_running": sorted(active_running),
        "failures_file": str(failures_path),
        "rerun_file": str(rerun_path),
        "finished_at": utc_now(),
    }
    atomic_write_json(state_root / "summary.json", summary)
    return summary


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-config", required=True)
    return parser


def main() -> None:
    """Internal child-process entry point."""
    args = _worker_parser().parse_args()
    raise SystemExit(run_dataset_worker(args.worker_config))


if __name__ == "__main__":
    main()
