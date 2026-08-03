"""Guards for the failure modes that silently shipped wrong RF100-VL numbers.

All three were found in the 2026-08 yolox campaigns. None of them announced
itself: each produced a plausible-looking result that was wrong.
"""

import json

from va_bench.rf100vl import (
    SELECTION_TEST_DIVERGENCE_ATOL,
    _aggregate_metrics,
    _selection_vs_test_divergence,
)


class TestSentinelExclusion:
    """pycocotools -1 means 'not applicable', not 'scored zero'."""

    def test_minus_one_is_excluded_from_the_mean(self):
        per_dataset = [
            {k: 0.5 for k in _metric_keys()},
            {**{k: 0.3 for k in _metric_keys()}, "mAP_small": -1.0},
        ]
        out = _aggregate_metrics(per_dataset)
        # Only the dataset that HAS small objects contributes.
        assert out["mAP_small"] == 0.5
        # Metrics with no sentinel are unaffected.
        assert abs(out["mAP"] - 0.4) < 1e-9

    def test_published_negative_map_small_cannot_recur(self):
        """A negative mean AP is never a valid score."""
        per_dataset = [
            {**{k: 0.4 for k in _metric_keys()}, "mAP_small": -1.0},
            {**{k: 0.4 for k in _metric_keys()}, "mAP_small": 0.02},
        ]
        assert _aggregate_metrics(per_dataset)["mAP_small"] >= 0.0

    def test_metric_defined_nowhere_stays_minus_one(self):
        per_dataset = [
            {**{k: 0.4 for k in _metric_keys()}, "mAP_small": -1.0},
            {**{k: 0.4 for k in _metric_keys()}, "mAP_small": -1.0},
        ]
        assert _aggregate_metrics(per_dataset)["mAP_small"] == -1.0


class TestSelectionVsTestDivergence:
    def test_flags_the_yolox_nano_failure(self, tmp_path):
        """valid 0.5663 vs test 0.1620 is the real eps-bug signature."""
        _write_stats(tmp_path, "ball", valid=0.5663)
        flagged = _selection_vs_test_divergence(
            [{"dataset": "ball", "mAP_50_95": 0.1620}], tmp_path
        )
        assert len(flagged) == 1
        assert flagged[0]["dataset"] == "ball"
        assert flagged[0]["delta"] > SELECTION_TEST_DIVERGENCE_ATOL

    def test_healthy_run_is_not_flagged(self, tmp_path):
        """yolox-tiny on the same dataset agreed; it must stay quiet."""
        _write_stats(tmp_path, "ball", valid=0.6198)
        assert (
            _selection_vs_test_divergence(
                [{"dataset": "ball", "mAP_50_95": 0.6091}], tmp_path
            )
            == []
        )

    def test_missing_stats_is_not_an_error(self, tmp_path):
        assert (
            _selection_vs_test_divergence(
                [{"dataset": "absent", "mAP_50_95": 0.5}], tmp_path
            )
            == []
        )

    def test_no_weights_root_is_not_an_error(self):
        assert _selection_vs_test_divergence([{"dataset": "x", "mAP_50_95": 0.5}], None) == []


def _metric_keys():
    from va_bench.rf100vl import _METRIC_KEYS

    return _METRIC_KEYS


def _write_stats(root, name, *, valid):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "stats.json").write_text(json.dumps({"valid_mAP50_95": valid}))
