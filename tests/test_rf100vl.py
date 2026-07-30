"""Tests for RF100-VL support: discovery, mapping, aggregation, full loop."""

import json

import pytest
from PIL import Image

from va_bench import rf100vl
from va_bench.rf100vl import (
    ANNOTATION_FILENAME,
    _aggregate_metrics,
    benchmark_model_rf100vl,
    category_mapping,
    discover_datasets,
    load_dataset_split,
    resolve_finetuned_weights,
)
from va_bench.rf100vl_data import VERSION_LOCK_SCHEMA, atomic_write_json

# ---------------------------------------------------------------------------
# Synthetic RF100-VL layout: <root>/<dataset>/test/{_annotations.coco.json, *.jpg}
# ---------------------------------------------------------------------------

GT_BOX = [4.0, 4.0, 16.0, 16.0]  # xywh


def make_dataset(root, name, num_images=2):
    split_dir = root / name / "test"
    split_dir.mkdir(parents=True)

    images, annotations = [], []
    for i in range(num_images):
        fname = f"img_{i}.jpg"
        Image.new("RGB", (32, 32), (i * 40 % 255, 10, 10)).save(split_dir / fname)
        images.append({"id": i + 1, "file_name": fname, "width": 32, "height": 32})
        annotations.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": list(GT_BOX),
                "area": GT_BOX[2] * GT_BOX[3],
                "iscrowd": 0,
            }
        )

    ann = {
        "images": images,
        "annotations": annotations,
        "categories": [
            # Roboflow exports often reserve id 0 for a placeholder supercategory.
            {"id": 0, "name": name, "supercategory": "none"},
            {"id": 1, "name": "thing", "supercategory": name},
        ],
    }
    with open(split_dir / ANNOTATION_FILENAME, "w") as f:
        json.dump(ann, f)
    return root / name


@pytest.fixture
def rf_root(tmp_path):
    root = tmp_path / "rf100-vl"
    make_dataset(root, "aerial-cows")
    make_dataset(root, "bacteria")
    return root


def test_discover_datasets_sorted(rf_root):
    dirs = discover_datasets(rf_root, split="test")
    assert [d.name for d in dirs] == ["aerial-cows", "bacteria"]


def test_discover_datasets_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_datasets(tmp_path / "nope")


def test_discover_datasets_wrong_split(rf_root):
    with pytest.raises(FileNotFoundError):
        discover_datasets(rf_root, split="valid")


def test_load_and_category_mapping(rf_root):
    coco_gt, img_ids, img_dir = load_dataset_split(rf_root / "bacteria", "test")
    assert len(img_ids) == 2
    assert (img_dir / "img_0.jpg").exists()
    # Placeholder id 0 occupies index 0; real class "thing" is index 1 -> id 1.
    assert category_mapping(coco_gt) == [0, 1]


def test_aggregate_metrics_is_unweighted_mean():
    a = {k: 0.2 for k in rf100vl._METRIC_KEYS}
    b = {k: 0.6 for k in rf100vl._METRIC_KEYS}
    agg = _aggregate_metrics([a, b])
    assert agg["mAP"] == pytest.approx(0.4)
    assert agg["AR100"] == pytest.approx(0.4)


def test_aggregate_metrics_empty_raises():
    with pytest.raises(ValueError):
        _aggregate_metrics([])


def test_resolve_finetuned_weights(tmp_path):
    spec = _fake_spec()
    ckpt = tmp_path / "aerial-cows" / "LibreFake.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    assert resolve_finetuned_weights(spec, tmp_path, "aerial-cows", "pytorch") == ckpt
    assert resolve_finetuned_weights(spec, tmp_path, "bacteria", "pytorch") is None
    assert resolve_finetuned_weights(spec, tmp_path, "aerial-cows", "onnx") is None


def _fake_spec():
    from va_bench.models import ModelSpec

    return ModelSpec(
        key="fake-n",
        display_name="Fake N",
        family="fake",
        variant="n",
        weight_file="LibreFake.pt",
        constructor_size="n",
        input_size=640,
        paper_params_m=1.0,
        paper_flops_g=1.0,
    )


def _stub_loader(perfect=True):
    """Stand-in for _load_for_dataset: predicts the GT box (class index 1)."""

    def loader(spec, fmt, weights_path, device, conf, iou, max_det):
        x, y, w, h = GT_BOX

        def predict(pil_img):
            if not perfect:
                return [], [], []
            return [[x, y, x + w, y + h]], [0.9], [1]

        return predict, 640, 1.23, "cpu"

    return loader


def test_full_loop_perfect_predictions(rf_root, monkeypatch):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", _stub_loader(perfect=True))

    result = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        verbose=False,
    )

    assert result["dataset"]["id"] == "rf100_vl"
    assert result["dataset"]["num_datasets"] == 2
    assert result["dataset"]["num_images"] == 4
    assert result["eval"]["dataset"] == "rf100_vl"
    assert result["submission_id"].startswith("fake-n-rf100_vl-")
    assert result["accuracy"]["mAP_50"] == pytest.approx(1.0)
    assert result["accuracy"]["mAP_50_95"] == pytest.approx(1.0)
    assert result["accuracy"]["AR_max_det"] == pytest.approx(1.0)
    assert result["config"]["iou"] == pytest.approx(0.65)
    assert result["config"]["max_det"] == 500
    assert result["eval"]["maxDets"] == [1, 10, 100, 500]
    assert result["rf100vl"]["regime"] == "pretrained-forced"
    assert result["rf100vl"]["valid_submission"] is False
    assert result["rf100vl"]["invalid_reasons"]
    assert [d["dataset"] for d in result["rf100vl"]["datasets"]] == [
        "aerial-cows",
        "bacteria",
    ]
    assert result["rf100vl"]["skipped_datasets"] == []
    assert len(result["repro"]["dataset"]["image_id_sha256"]) == 64
    assert len(result["repro"]["weights"]["sha256"]) == 64


def test_full_loop_skips_datasets_without_checkpoint(rf_root, tmp_path, monkeypatch):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", _stub_loader(perfect=True))

    weights_root = tmp_path / "weights"
    (weights_root / "bacteria").mkdir(parents=True)
    (weights_root / "bacteria" / "LibreFake.pt").write_bytes(b"x")

    result = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        weights_root=weights_root,
        verbose=False,
    )

    assert result["rf100vl"]["regime"] == "fine-tuned"
    assert result["rf100vl"]["skipped_datasets"] == ["aerial-cows"]
    assert [d["dataset"] for d in result["rf100vl"]["datasets"]] == ["bacteria"]
    assert result["dataset"]["num_datasets"] == 1
    assert result["rf100vl"]["valid_submission"] is False
    assert any("no checkpoint" in reason for reason in result["rf100vl"]["invalid_reasons"])


def test_requires_weights_or_explicit_pretrained(rf_root, monkeypatch):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    with pytest.raises(ValueError, match="fine-tuned benchmark"):
        benchmark_model_rf100vl("fake-n", rf_root, verbose=False)


def test_subset_run_is_flagged(rf_root, monkeypatch):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", _stub_loader(perfect=True))

    result = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        limit=1,
        limit_datasets=1,
        verbose=False,
    )
    assert result["dataset"]["num_datasets"] == 1
    assert result["dataset"]["num_images"] == 1
    assert "subset_run" in result["rf100vl"]


def test_non_protocol_iou_and_cap_are_not_submittable(rf_root, monkeypatch):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", _stub_loader(perfect=True))

    result = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        iou=0.6,
        max_det=300,
        verbose=False,
    )

    reasons = result["rf100vl"]["invalid_reasons"]
    assert result["rf100vl"]["valid_submission"] is False
    assert any("NMS IoU" in reason for reason in reasons)
    assert any("max_det" in reason for reason in reasons)


@pytest.mark.parametrize(
    ("argument", "value"),
    (("limit", 0), ("limit_datasets", 0)),
)
def test_nonpositive_smoke_limits_are_rejected(rf_root, monkeypatch, argument, value):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())

    with pytest.raises(ValueError, match=argument):
        benchmark_model_rf100vl(
            "fake-n",
            rf_root,
            allow_pretrained=True,
            verbose=False,
            **{argument: value},
        )


def test_per_dataset_results_resume_and_record_version_recipe_hash(
    rf_root,
    tmp_path,
    monkeypatch,
):
    calls = []
    loader = _stub_loader(perfect=True)

    def counting_loader(*args, **kwargs):
        calls.append(args[0].key)
        return loader(*args, **kwargs)

    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", counting_loader)
    atomic_write_json(
        rf_root / "versions.json",
        {
            "schema_version": VERSION_LOCK_SCHEMA,
            "subset": "rf100vl",
            "source": {"commit": "abc"},
            "datasets": {
                "aerial-cows": {"project_name": "aerial-cows", "version_id": 3},
                "bacteria": {"project_name": "bacteria", "version_id": 7},
            },
            "downloaded": ["aerial-cows", "bacteria"],
            "selection_complete": True,
        },
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"family":"fake"}\n', encoding="utf-8")
    cache_dir = tmp_path / "per-dataset"

    first = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        recipe_path=recipe,
        per_dataset_dir=cache_dir,
        verbose=False,
    )
    second = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        recipe_path=recipe,
        per_dataset_dir=cache_dir,
        verbose=False,
    )

    assert calls == ["fake-n", "fake-n"]
    assert len(list(cache_dir.rglob("*.json"))) == 2
    assert all(item["resumed_from_cache"] for item in second["rf100vl"]["datasets"])
    assert len(first["repro"]["dataset"]["versions"]["sha256"]) == 64
    assert first["repro"]["dataset"]["datasets"][0]["version_id"] == 3
    assert first["repro"]["recipe"]["sha256"] == rf100vl.file_sha256(recipe)


def test_smoke_cache_does_not_overwrite_full_campaign_resume(
    rf_root,
    tmp_path,
    monkeypatch,
):
    calls = []
    loader = _stub_loader(perfect=True)

    def counting_loader(*args, **kwargs):
        calls.append(args[0].key)
        return loader(*args, **kwargs)

    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", counting_loader)
    cache_dir = tmp_path / "per-dataset"

    benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        per_dataset_dir=cache_dir,
        verbose=False,
    )
    benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        per_dataset_dir=cache_dir,
        limit=1,
        verbose=False,
    )
    resumed = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        allow_pretrained=True,
        per_dataset_dir=cache_dir,
        verbose=False,
    )

    assert calls == ["fake-n"] * 4
    assert len(list(cache_dir.rglob("*.json"))) == 4
    assert all(item["resumed_from_cache"] for item in resumed["rf100vl"]["datasets"])


def test_non_protocol_training_stats_invalidate_finetuned_result(
    rf_root,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", _stub_loader(perfect=True))
    weights_root = tmp_path / "weights"
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"family":"fake"}\n', encoding="utf-8")
    recipe_sha = rf100vl.file_sha256(recipe)
    for name in ("aerial-cows", "bacteria"):
        target = weights_root / name
        target.mkdir(parents=True)
        (target / "LibreFake.pt").write_bytes(b"x")
        atomic_write_json(
            target / "stats.json",
            {
                "protocol_version": rf100vl.PROTOCOL_VERSION,
                "protocol_conformant": False,
                "recipe": {
                    "file": str(recipe),
                    "sha256": recipe_sha,
                },
            },
        )

    result = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        weights_root=weights_root,
        recipe_path=recipe,
        verbose=False,
    )
    assert any(
        "non-protocol training runs" in reason for reason in result["rf100vl"]["invalid_reasons"]
    )


def test_training_stats_require_verified_libreyolo_capabilities(tmp_path):
    weights_root = tmp_path / "weights"
    target = weights_root / "aerial-cows"
    target.mkdir(parents=True)
    lock = {
        "schema_version": VERSION_LOCK_SCHEMA,
        "subset": "rf100vl",
        "source": {"commit": "abc"},
        "datasets": {
            "aerial-cows": {
                "project_name": "aerial-cows",
                "version_id": 3,
            }
        },
        "downloaded": ["aerial-cows"],
        "selection_complete": True,
    }
    stats = {
        "schema_version": rf100vl.TRAIN_STATS_SCHEMA,
        "protocol_version": rf100vl.PROTOCOL_VERSION,
        "protocol_conformant": True,
        "dataset": "aerial-cows",
        "dataset_version": 3,
        "model_key": "fake-n",
        "seed": rf100vl.PROTOCOL_SEED,
        "precision": "fp32",
        "epochs_requested": rf100vl.PROTOCOL_EPOCHS,
        "versions_sha256": rf100vl.version_lock_sha256(lock),
        "recipe": {
            "file": "recipe.json",
            "sha256": "a" * 64,
        },
    }
    atomic_write_json(target / "stats.json", stats)

    _, reasons = rf100vl._recipe_repro(
        None,
        weights_root,
        ["aerial-cows"],
        version_lock=lock,
        model_key="fake-n",
    )
    assert any("training metadata" in reason for reason in reasons)

    stats["libreyolo_capabilities"] = {
        "validated": True,
        "eval_max_det": 500,
        "default_eval_max_det": 100,
    }
    atomic_write_json(target / "stats.json", stats)
    _, reasons = rf100vl._recipe_repro(
        None,
        weights_root,
        ["aerial-cows"],
        version_lock=lock,
        model_key="fake-n",
    )
    assert not any("training metadata" in reason for reason in reasons)


def test_training_stats_must_match_dataset_version_lock(
    rf_root,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(rf100vl, "get_spec", lambda key: _fake_spec())
    monkeypatch.setattr(rf100vl, "_load_for_dataset", _stub_loader(perfect=True))
    weights_root = tmp_path / "weights"
    recipe = tmp_path / "recipe.json"
    recipe.write_text('{"family":"fake"}\n', encoding="utf-8")
    recipe_sha = rf100vl.file_sha256(recipe)
    lock = {
        "schema_version": VERSION_LOCK_SCHEMA,
        "subset": "rf100vl",
        "source": {"commit": "abc"},
        "datasets": {
            "aerial-cows": {"project_name": "aerial-cows", "version_id": 3},
            "bacteria": {"project_name": "bacteria", "version_id": 7},
        },
        "downloaded": ["aerial-cows", "bacteria"],
        "selection_complete": True,
    }
    atomic_write_json(rf_root / "versions.json", lock)
    versions_sha = rf100vl.version_lock_sha256(lock)
    for name, version_id in (("aerial-cows", 3), ("bacteria", 999)):
        target = weights_root / name
        target.mkdir(parents=True)
        (target / "LibreFake.pt").write_bytes(b"x")
        atomic_write_json(
            target / "stats.json",
            {
                "schema_version": rf100vl.TRAIN_STATS_SCHEMA,
                "protocol_version": rf100vl.PROTOCOL_VERSION,
                "protocol_conformant": True,
                "dataset": name,
                "dataset_version": version_id,
                "model_key": "fake-n",
                "seed": rf100vl.PROTOCOL_SEED,
                "precision": "fp32",
                "epochs_requested": rf100vl.PROTOCOL_EPOCHS,
                "versions_sha256": versions_sha,
                "recipe": {
                    "file": str(recipe),
                    "sha256": recipe_sha,
                },
            },
        )

    result = benchmark_model_rf100vl(
        "fake-n",
        rf_root,
        weights_root=weights_root,
        recipe_path=recipe,
        verbose=False,
    )

    assert any("training metadata" in reason for reason in result["rf100vl"]["invalid_reasons"])
