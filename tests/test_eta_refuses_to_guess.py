"""The ETA must say "not yet" rather than extrapolate startup cost.

Both 2026-08 campaigns produced badly wrong headline ETAs from the same
artifact: a run whose only evidence is its opening epochs, where wall time
carries the image-cache fill, allocator warmup and cuDNN autotuning. Annualising
that across 100 epochs gave "est. remaining 64.9h" for ec-s with zero completed
epochs, and 16.3h for yolox-tiny.
"""

import pytest

from va_bench.rf100vl_eta import MIN_EPOCHS_OBSERVED, estimate_eta


def _running(name, epochs_done, wall):
    return {
        "dataset": name,
        "state": "running",
        "epochs_done": epochs_done,
        "wall_seconds": wall,
        "epochs_total": 100,
    }


def _pending(name):
    return {"dataset": name, "state": "pending", "epochs_total": 100}


IMAGES = {"a": 4000, "b": 4000, "c": 4000, "p1": 3000, "p2": 3000}


class TestRefusesDuringWarmup:
    def test_no_eta_when_every_lane_is_in_its_first_epochs(self):
        """The exact ec-s situation: cache filling, nothing past epoch 1."""
        records = [_running("a", 1, 3000.0), _running("b", 1, 3100.0), _pending("p1")]
        out = estimate_eta(records, train_images=IMAGES, lanes=2)
        assert out["available"] is False
        assert out["p50_seconds"] is None
        assert "cache" in out["reason"].lower()

    def test_no_eta_with_zero_epochs_done(self):
        records = [_running("a", 0, 900.0), _pending("p1")]
        out = estimate_eta(records, train_images=IMAGES, lanes=2)
        assert out["available"] is False

    def test_becomes_available_once_a_lane_clears_warmup(self):
        records = [
            _running("a", MIN_EPOCHS_OBSERVED + 2, 700.0),
            _running("b", 1, 3000.0),
            _pending("p1"),
        ]
        out = estimate_eta(records, train_images=IMAGES, lanes=2)
        assert out["available"] is True
        assert out["p50_seconds"] is not None

    def test_available_when_a_dataset_has_finished(self):
        records = [
            {"dataset": "a", "state": "done", "wall_seconds": 5000.0, "epochs_total": 100},
            _running("b", 1, 3000.0),
            _pending("p1"),
        ]
        out = estimate_eta(records, train_images=IMAGES, lanes=2)
        assert out["available"] is True


class TestWarmupLanesDoNotInflate:
    def test_a_warming_lane_does_not_drive_the_estimate(self):
        """A lane at epoch 1 with a huge wall must not annualise its own rate.

        Same evidence twice, except one run has an extra lane still filling its
        cache. That lane must not multiply the answer.
        """
        settled = _running("a", MIN_EPOCHS_OBSERVED + 5, 600.0)
        clean = estimate_eta([settled, _pending("p1")], train_images=IMAGES, lanes=2)
        # 'b' is 1 epoch in but has burned an enormous wall filling cache.
        noisy = estimate_eta(
            [settled, _running("b", 1, 9000.0), _pending("p1")],
            train_images=IMAGES,
            lanes=2,
        )
        assert clean["available"] and noisy["available"]
        # Extrapolating b's rate would give ~9000*99 seconds on that lane alone.
        assert noisy["p50_seconds"] < 99 * 9000.0
