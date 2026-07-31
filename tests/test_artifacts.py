"""Tests for campaign artifact sync and offline re-scoring."""

from __future__ import annotations

import gzip
import json

import pytest

from va_bench.artifacts import (
    collect_artifacts,
    load_predictions,
    rescore_from_predictions,
    select_prediction_dumps,
)


def _write_dump(directory, fingerprint, detections):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fingerprint}.predictions.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            {"schema_version": "rf100vl.predictions.v1",
             "fingerprint": fingerprint, "detections": detections},
            handle,
        )
    return path


def _write_recorded(directory, fingerprint, m_ap):
    path = directory / f"{fingerprint}.json"
    path.write_text(json.dumps({"result": {"metrics": {"mAP": m_ap}}}), encoding="utf-8")
    return path


def _make_campaign(tmp_path, tier_files=True):
    weights = tmp_path / "weights"
    state = weights / ".state" / "yolov9t"
    (state / "logs").mkdir(parents=True)
    for name in ("summary.json", "rerun.json", "failures.json"):
        (state / name).write_text("{}", encoding="utf-8")
    (state / "logs" / "ds1.log").write_text("log", encoding="utf-8")

    run = weights / ".runs" / "yolov9t" / "ds1" / "primary"
    (run / "weights").mkdir(parents=True)
    if tier_files:
        for name in ("data.yaml", "metrics.jsonl", "train_config.yaml", "train.log"):
            (run / name).write_text("x", encoding="utf-8")
    (run / "weights" / "best.pt").write_bytes(b"best")
    (run / "weights" / "last.pt").write_bytes(b"last")

    (weights / "ds1").mkdir(parents=True)
    (weights / "ds1" / "stats.json").write_text("{}", encoding="utf-8")
    return weights


def test_collect_results_tier_excludes_weights(tmp_path):
    weights = _make_campaign(tmp_path)
    items = collect_artifacts(
        model_key="yolov9t", run_id="r1", weights_root=weights, tier="results"
    )
    repo_paths = {repo for _, repo in items}
    assert "yolov9t/r1/stats/ds1.json" in repo_paths
    assert "yolov9t/r1/state/summary.json" in repo_paths
    assert "yolov9t/r1/state/logs/ds1.log" in repo_paths
    assert "yolov9t/r1/runs/ds1/primary/train_config.yaml" in repo_paths
    # The whole point of the default tier: no weights.
    assert not [p for p in repo_paths if p.endswith(".pt")]


def test_collect_tiers_add_weights_progressively(tmp_path):
    weights = _make_campaign(tmp_path)

    def paths(tier):
        return {repo for _, repo in collect_artifacts(
            model_key="yolov9t", run_id="r1", weights_root=weights, tier=tier)}

    checkpoints = paths("checkpoints")
    everything = paths("all")
    assert any(p.endswith("weights/best.pt") for p in checkpoints)
    assert not any(p.endswith("weights/last.pt") for p in checkpoints)
    assert any(p.endswith("weights/last.pt") for p in everything)
    assert paths("results") < checkpoints < everything


def test_collect_rejects_unknown_tier(tmp_path):
    with pytest.raises(ValueError, match="tier must be one of"):
        collect_artifacts(model_key="m", run_id="r", weights_root=tmp_path, tier="nope")


def test_select_dumps_refuses_to_double_count_a_dataset(tmp_path):
    """Two runs of one dataset must not be averaged as if they were two."""
    dataset_dir = tmp_path / "eval" / "ds1"
    _write_dump(dataset_dir, "aaaa1111", [])
    _write_dump(dataset_dir, "bbbb2222", [])
    with pytest.raises(ValueError, match="several prediction dumps"):
        select_prediction_dumps(tmp_path / "eval")

    chosen = select_prediction_dumps(tmp_path / "eval", fingerprint_prefix="aaaa")
    assert len(chosen) == 1
    assert chosen[0].name.startswith("aaaa")


def test_predictions_round_trip(tmp_path):
    detections = [{"image_id": 1, "category_id": 0, "bbox": [1.0, 2.0, 3.0, 4.0],
                   "score": 0.5}]
    path = _write_dump(tmp_path / "eval" / "ds1", "ffff0000", detections)
    fingerprint, loaded = load_predictions(path)
    assert fingerprint == "ffff0000"
    assert loaded == detections


def _gt_fixture(tmp_path, dataset="ds1"):
    gt_dir = tmp_path / "data" / dataset / "test"
    gt_dir.mkdir(parents=True)
    (gt_dir / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                         "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "thing", "supercategory": "none"}],
    }), encoding="utf-8")
    return tmp_path / "data"


def test_rescore_flags_a_subset_and_reports_mean(tmp_path):
    data_dir = _gt_fixture(tmp_path)
    _write_dump(tmp_path / "eval" / "ds1", "aaaa1111", [])

    def fake_evaluator(coco_gt, detections, image_ids=None, max_det=None):
        assert max_det == 500
        return {"mAP": 0.25, "mAP50": 0.5}

    report = rescore_from_predictions(
        eval_root=tmp_path / "eval", data_dir=data_dir, evaluator=fake_evaluator
    )
    assert report["num_datasets"] == 1
    assert report["is_full_benchmark"] is False
    assert report["mean_mAP_50_95"] == pytest.approx(0.25)
    assert report["mismatches"] == []


def test_rescore_surfaces_disagreement_with_recorded_metrics(tmp_path):
    """A silent divergence between box and rescore is the dangerous case."""
    data_dir = _gt_fixture(tmp_path)
    dataset_dir = tmp_path / "eval" / "ds1"
    _write_dump(dataset_dir, "aaaa1111", [])
    _write_recorded(dataset_dir, "aaaa1111", 0.90)

    report = rescore_from_predictions(
        eval_root=tmp_path / "eval", data_dir=data_dir,
        evaluator=lambda *a, **k: {"mAP": 0.25, "mAP50": 0.5},
    )
    assert len(report["mismatches"]) == 1
    assert report["mismatches"][0]["recorded"] == pytest.approx(0.90)
    assert report["mismatches"][0]["rescored"] == pytest.approx(0.25)


def test_rescore_agrees_when_metrics_match(tmp_path):
    data_dir = _gt_fixture(tmp_path)
    dataset_dir = tmp_path / "eval" / "ds1"
    _write_dump(dataset_dir, "aaaa1111", [])
    _write_recorded(dataset_dir, "aaaa1111", 0.25)

    report = rescore_from_predictions(
        eval_root=tmp_path / "eval", data_dir=data_dir,
        evaluator=lambda *a, **k: {"mAP": 0.25, "mAP50": 0.5},
    )
    assert report["mismatches"] == []


def test_rescore_skips_datasets_without_ground_truth(tmp_path):
    data_dir = _gt_fixture(tmp_path)
    _write_dump(tmp_path / "eval" / "ds1", "aaaa1111", [])
    _write_dump(tmp_path / "eval" / "unknown-dataset", "cccc3333", [])

    report = rescore_from_predictions(
        eval_root=tmp_path / "eval", data_dir=data_dir,
        evaluator=lambda *a, **k: {"mAP": 0.25, "mAP50": 0.5},
    )
    assert report["num_datasets"] == 1
    assert report["datasets"][0]["dataset"] == "ds1"


def test_manifest_records_code_identity_and_hashes(tmp_path):
    """Results without an exact commit are anecdotes, not evidence."""
    import json as _json

    from va_bench.artifacts import STATE_FILES, build_manifest, write_manifest

    state = tmp_path / "state"
    state.mkdir()
    (state / "summary.json").write_text(
        _json.dumps({"completed": ["a"], "failed": [], "protocol_conformant": True}),
        encoding="utf-8",
    )
    (state / "a.json").write_text(
        _json.dumps({"recipe_sha256": "rrr", "versions_sha256": "vvv"}), encoding="utf-8"
    )
    recipe = tmp_path / "yolov9.json"
    recipe.write_text(_json.dumps({"protocol": {"epochs": 100}}), encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "versions.json").write_text("{}", encoding="utf-8")

    manifest = build_manifest(
        model_key="yolov9t",
        run_id="r1",
        state_dir=state,
        data_dir=data_dir,
        recipe_path=recipe,
    )
    names = {entry["package"] for entry in manifest["packages"]}
    assert names == {"libreyolo", "va-bench"}
    assert manifest["recipe"]["sha256"]
    assert manifest["recipe"]["protocol"] == {"epochs": 100}
    assert manifest["dataset_versions"]["sha256"]
    assert manifest["campaign"]["protocol_conformant"] is True
    assert manifest["observed_hashes"] == {"recipe_sha256": "rrr", "versions_sha256": "vvv"}
    assert manifest["created_at"]

    # and it must actually be uploaded, not just written
    written = write_manifest(state, manifest)
    assert written.name in STATE_FILES


def test_default_run_id_is_stable_and_code_specific():
    """Reusing a run id silently mixes campaigns, so it must not be hand-typed."""
    from va_bench.artifacts import default_run_id

    base = {
        "model_key": "yolov9t",
        "created_at": "2026-07-31T20:00:00+00:00",
        "packages": [{"package": "libreyolo", "commit": "aaa"}, {"package": "va-bench", "commit": "bbb"}],
        "recipe": {"sha256": "ccc"},
    }
    first = default_run_id(base)
    assert first.startswith("20260731-yolov9t-")
    assert default_run_id(dict(base)) == first          # same code -> same id

    moved = dict(base, packages=[{"package": "libreyolo", "commit": "zzz"}])
    assert default_run_id(moved) != first               # new code -> new id


def test_manifest_counts_datasets_from_status_not_the_last_invocation(tmp_path):
    """A resumed campaign reported 0 completed while seven were done."""
    import json as _json

    from va_bench.artifacts import build_manifest

    state = tmp_path / "state"
    state.mkdir()
    # summary.json only knows about THIS invocation
    (state / "summary.json").write_text(
        _json.dumps({"completed": [], "interrupted": ["c"]}), encoding="utf-8"
    )
    for name, value in (("a", "done"), ("b", "done"), ("c", "pending")):
        (state / f"{name}.json").write_text(
            _json.dumps(
                {"schema_version": "rf100vl.train-status.v1", "dataset": name, "state": value}
            ),
            encoding="utf-8",
        )

    manifest = build_manifest(model_key="yolov9t", run_id="r", state_dir=state)
    assert manifest["dataset_states"] == {"done": 2, "pending": 1}
    assert manifest["datasets_total"] == 3
