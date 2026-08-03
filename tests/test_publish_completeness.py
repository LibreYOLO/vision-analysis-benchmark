"""sync-artifacts must not be able to half-publish a benchmark run.

Every case here is a real 2026-08 failure. All three printed a green log and a
working HF URL while shipping an artifact that could not be reproduced.
"""

import hashlib
import json

import pytest

from va_bench.artifacts import IncompletePublish, collect_artifacts

RECIPE = json.dumps({"protocol": {"epochs": 100}}, indent=2)
RECIPE_SHA = hashlib.sha256(RECIPE.encode()).hexdigest()


def _campaign(tmp_path, *, layout="runs", datasets=("ball", "bees")):
    """Build a weights root in either the .runs or the flat layout."""
    root = tmp_path / "weights"
    for name in datasets:
        (root / name).mkdir(parents=True)
        (root / name / "stats.json").write_text(json.dumps({"valid_mAP50_95": 0.5}))
        if layout == "runs":
            wdir = root / ".runs" / "yolox-s" / name / "primary" / "weights"
            wdir.mkdir(parents=True)
            (wdir / "best.pt").write_bytes(b"weights")
        else:  # the canonical <root>/<dataset>/<weight file> layout
            (root / name / "LibreYOLOXs.pt").write_bytes(b"weights")

    data = tmp_path / "data"
    data.mkdir()
    (data / "versions.json").write_text("{}")

    subs = tmp_path / "subs"
    subs.mkdir()
    (subs / "yolox-s__run.json").write_text(
        json.dumps({"rf100vl": {"recipe_sha256": RECIPE_SHA}})
    )

    recipe = tmp_path / "recipe.json"
    # write_bytes, not write_text: on Windows the latter rewrites \n as \r\n
    # and the file no longer hashes to RECIPE_SHA.
    recipe.write_bytes(RECIPE.encode())
    return root, data, subs, recipe


def _collect(root, data, subs, recipe, *, tier="checkpoints", report=None):
    return collect_artifacts(
        model_key="yolox-s",
        run_id="r1",
        weights_root=root,
        data_dir=data,
        submissions_dir=subs,
        recipe_path=recipe,
        tier=tier,
        report=report,
    )


class TestFlatWeightsLayout:
    """The rescue set that uploaded zero checkpoints and said 'skipped 0'."""

    def test_flat_layout_checkpoints_are_collected(self, tmp_path):
        root, data, subs, recipe = _campaign(tmp_path, layout="flat")
        items = _collect(root, data, subs, recipe)
        weights = [r for _, r in items if r.endswith(".pt")]
        assert len(weights) == 2, weights
        assert any("weights/ball/LibreYOLOXs.pt" in r for r in weights)

    def test_runs_layout_still_works(self, tmp_path):
        root, data, subs, recipe = _campaign(tmp_path, layout="runs")
        items = _collect(root, data, subs, recipe)
        assert len([r for _, r in items if r.endswith("best.pt")]) == 2


class TestRefusesToHalfPublish:
    def test_trained_dataset_without_a_checkpoint_raises(self, tmp_path):
        root, data, subs, recipe = _campaign(tmp_path, layout="flat")
        # A dataset that trained but whose checkpoint never got written.
        (root / "orphan").mkdir()
        (root / "orphan" / "stats.json").write_text("{}")
        with pytest.raises(IncompletePublish, match="checkpoint"):
            _collect(root, data, subs, recipe)

    def test_missing_recipe_raises(self, tmp_path):
        root, data, subs, _ = _campaign(tmp_path, layout="flat")
        with pytest.raises(IncompletePublish, match="recipe"):
            _collect(root, data, subs, None)

    def test_results_tier_only_reports(self, tmp_path):
        """Ordinary syncs must stay non-brittle: warn, never raise."""
        root, data, subs, _ = _campaign(tmp_path, layout="flat")
        seen = []
        items = _collect(root, data, subs, None, tier="results", report=seen.append)
        assert items
        assert seen == [] or all(isinstance(m, str) for m in seen)


class TestRecipeMustMatchTheHashItClaims:
    """Shipping the wrong recipe is worse than shipping none."""

    def test_mismatched_recipe_raises(self, tmp_path):
        root, data, subs, _ = _campaign(tmp_path, layout="flat")
        wrong = tmp_path / "packaged.json"
        wrong.write_text(json.dumps({"protocol": {"epochs": 50}}))
        with pytest.raises(IncompletePublish, match="Recipe mismatch"):
            _collect(root, data, subs, wrong)

    def test_matching_recipe_is_accepted(self, tmp_path):
        root, data, subs, recipe = _campaign(tmp_path, layout="flat")
        items = _collect(root, data, subs, recipe)
        assert any(r.endswith("provenance/recipe.json") for _, r in items)
