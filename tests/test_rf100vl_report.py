"""Tests for the report renderer and the campaign preflight."""

from __future__ import annotations

import json
from pathlib import Path

from va_bench.rf100vl_preflight import has_failure, render, run_preflight
from va_bench.rf100vl_report import (
    build_leaderboard,
    build_report,
    load_submissions,
)


def _submission(model_id: str, map5095: float, n_datasets: int = 12) -> dict:
    records = [
        {
            "dataset": f"ds-{index:02d}",
            "mAP_50_95": map5095 + index * 0.001,
            "mAP_50": map5095 + 0.2,
            "wall_seconds": 30.0,
            "num_images": 10,
            "num_classes": 3,
            "dataset_version": 1,
        }
        for index in range(n_datasets)
    ]
    return {
        "submission_id": f"{model_id}-rf100-vl-abc",
        "model": {"id": model_id, "name": model_id},
        "accuracy": {"mAP_50_95": map5095, "mAP_50": map5095 + 0.2},
        "rf100vl": {
            "regime": "fine-tuned",
            "valid_submission": False,
            "invalid_reasons": ["an image or dataset smoke-test limit was applied"],
            "protocol": {"version": "rf100vl.libreyolo.v1", "max_det": 500,
                         "expected_datasets": 100},
            "recipe_sha256": "deadbeef" * 8,
            "dataset_versions_sha256": "cafebabe" * 8,
            "datasets": records,
            "skipped_datasets": ["ds-skipped"],
        },
    }


def test_report_contains_headline_completion_and_reasons() -> None:
    text = build_report(_submission("yolov9t", 0.5))
    assert "RF100-VL report: yolov9t" in text
    assert "mAP50-95 0.500" in text
    assert "ok 12/100" in text
    assert "valid submission: NO" in text
    assert "smoke-test limit" in text
    assert "Weakest datasets" in text
    assert "ds-skipped" in text


def test_report_train_cost_from_stats(tmp_path: Path) -> None:
    submission = _submission("yolov9t", 0.5, n_datasets=3)
    for record in submission["rf100vl"]["datasets"]:
        stats_dir = tmp_path / record["dataset"]
        stats_dir.mkdir(parents=True)
        (stats_dir / "stats.json").write_text(
            json.dumps({"wall_seconds": 600.0, "best_epoch": 42}), encoding="utf-8"
        )
    text = build_report(submission, weights_root=tmp_path)
    assert "median wall per dataset: 10 min" in text
    assert "median best epoch: 42" in text


def test_leaderboard_sorts_and_keeps_latest_per_model() -> None:
    older = _submission("yolov9t", 0.40)
    older["submission_id"] = "yolov9t-rf100-vl-aaa"
    newer = _submission("yolov9t", 0.50)
    newer["submission_id"] = "yolov9t-rf100-vl-zzz"
    other = _submission("rfdetr_s", 0.55)
    text = build_leaderboard([older, other, newer])
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| ") and not line.startswith("| # ")
    ]
    assert "rfdetr_s" in rows[0] and "0.550" in rows[0]
    assert "yolov9t" in rows[1] and "0.500" in rows[1]
    assert "0.400" not in text


def test_load_submissions_filters_non_rf100vl(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps(_submission("m", 0.1)), "utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"other": True}), "utf-8")
    (tmp_path / "c.json").write_text("{broken", "utf-8")
    assert len(load_submissions(tmp_path)) == 1


def _fake_data_dir(tmp_path: Path, names: list[str], complete: bool = True) -> Path:
    data_dir = tmp_path / "rf100-vl"
    data_dir.mkdir()
    lock = {
        "schema_version": "rf100vl.versions.v1",
        "subset": "rf100vl",
        "source": "test",
        "selection_complete": True,
        "downloaded": list(names),
        "datasets": {
            name: {"project_name": name, "version_id": 1} for name in names
        },
    }
    (data_dir / "versions.json").write_text(json.dumps(lock), encoding="utf-8")
    for name in names:
        for split in ("train", "valid", "test"):
            split_dir = data_dir / name / split
            split_dir.mkdir(parents=True)
            if complete or split != "test":
                (split_dir / "_annotations.coco.json").write_text("{}", "utf-8")
    return data_dir


_SKIP_ENV = ("libreyolo", "recipe", "gpu")  # need a real install / GPU


def test_preflight_passes_on_complete_data(tmp_path: Path) -> None:
    data_dir = _fake_data_dir(tmp_path, ["alpha", "beta"])
    checks = run_preflight(
        "yolov9t", data_dir, tmp_path / "weights", skip=_SKIP_ENV
    )
    assert not has_failure(checks)
    assert "ready" in render(checks)


def test_preflight_fails_on_missing_split_and_missing_dataset(tmp_path: Path) -> None:
    data_dir = _fake_data_dir(tmp_path, ["alpha", "beta"], complete=False)
    checks = run_preflight(
        "yolov9t", data_dir, tmp_path / "weights", skip=_SKIP_ENV
    )
    by_name = {check.name: check for check in checks}
    assert not by_name["data"].ok or not by_name["splits"].ok
    # now remove a locked dataset dir entirely
    import shutil as _shutil

    _shutil.rmtree(data_dir / "beta")
    checks = run_preflight(
        "yolov9t", data_dir, tmp_path / "weights", skip=_SKIP_ENV
    )
    by_name = {check.name: check for check in checks}
    assert not by_name["data"].ok
    assert "beta" in by_name["data"].detail
    assert has_failure(checks)


def _stats(**overrides):
    stats = {
        "schema_version": "rf100vl.train-stats.v1",
        "dataset": "ds-00",
        "dataset_version": 1,
        "model_key": "yolov9t",
        "seed": 0,
        "epochs_requested": 100,
        "versions_sha256": None,
        "precision": "fp32",
        "protocol_conformant": True,
        "protocol_version": "rf100vl.libreyolo.v1",
        "recipe": {"file": "yolov9.json", "sha256": "a" * 64},
        "libreyolo_capabilities": {
            "validated": True,
            "eval_max_det": 500,
            "default_eval_max_det": 100,
        },
    }
    stats.update(overrides)
    return stats


def test_metadata_mismatch_reason_names_the_offending_field(tmp_path: Path) -> None:
    """A smoke run must say which field disagrees, not a generic sentence."""
    from va_bench.rf100vl import _recipe_repro

    (tmp_path / "ds-00").mkdir()
    (tmp_path / "ds-00" / "stats.json").write_text(
        json.dumps(_stats(epochs_requested=2, protocol_conformant=False)), "utf-8"
    )
    _, reasons = _recipe_repro(None, tmp_path, ["ds-00"], model_key="yolov9t")
    mismatch = [r for r in reasons if "does not match this campaign" in r]
    assert len(mismatch) == 1
    assert "epochs_requested" in mismatch[0]
    assert "model_key" not in mismatch[0]  # only the field that actually differs


def test_metadata_mismatch_reason_distinguishes_a_wrong_checkpoint(tmp_path: Path) -> None:
    from va_bench.rf100vl import _recipe_repro

    (tmp_path / "ds-00").mkdir()
    (tmp_path / "ds-00" / "stats.json").write_text(
        json.dumps(_stats(model_key="yolov9s", seed=7)), "utf-8"
    )
    _, reasons = _recipe_repro(None, tmp_path, ["ds-00"], model_key="yolov9t")
    mismatch = [r for r in reasons if "does not match this campaign" in r]
    assert len(mismatch) == 1
    assert "model_key" in mismatch[0] and "seed" in mismatch[0]


def test_conformant_stats_produce_no_mismatch_reason(tmp_path: Path) -> None:
    from va_bench.rf100vl import _recipe_repro

    (tmp_path / "ds-00").mkdir()
    (tmp_path / "ds-00" / "stats.json").write_text(json.dumps(_stats()), "utf-8")
    _, reasons = _recipe_repro(None, tmp_path, ["ds-00"], model_key="yolov9t")
    assert reasons == []
