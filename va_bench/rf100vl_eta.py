"""Campaign ETA that does not lie about the tail.

The obvious estimator, mean wall time of finished datasets times the number
remaining over the number of lanes, is wrong three times over.

1. **Survivorship bias.** Short datasets finish first, so early in a campaign
   the finished set is the fast tail of the distribution and the mean is an
   underestimate that gets worse the earlier you ask. Measured on a real
   8 lane campaign: two finished datasets implied 6.7 hours while the in-flight
   evidence implied about 14.

2. **It ignores what we already know about the work.** Every pending dataset
   has a known training image count, and epochs are fixed by the protocol at
   100. Runtime is close to `epochs * (fixed_per_epoch + per_image * images)`.
   Fitting that from observed runs turns each pending dataset from an unknown
   into a prediction, and the fit improves continuously as the campaign runs.

3. **Dividing total work by lanes is a lower bound, not a finish time.** It is
   the answer for perfectly divisible work. Real datasets are indivisible and
   scheduled in queue order, so one long job started late sets the makespan.
   That needs simulating, not dividing.

Everything here is a pure function of observations so it can be tested without
a GPU, and every path reports which method produced the number, because an ETA
whose provenance you cannot see is just a confident-looking guess.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Iterable, Sequence

# Below this many usable observations, a size-based fit is noise, so fall back
# to a plain median of observed per-epoch times.
MIN_OBSERVATIONS_FOR_FIT = 3

# Epochs a running dataset must have finished before its per-epoch time is
# treated as evidence. Below this the number is mostly warmup.
MIN_EPOCHS_OBSERVED = 5


def _theil_sen(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Median of pairwise slopes: robust to the handful of weird datasets.

    Least squares would let one dataset of enormous images drag the slope for
    all 100. Theil-Sen tolerates up to ~29% outliers before it breaks down,
    which matches what a benchmark suite of scraped datasets actually looks
    like.
    """
    slopes = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            if x1 != x2:
                slopes.append((y2 - y1) / (x2 - x1))
    if not slopes:
        return None
    slope = statistics.median(slopes)
    intercept = statistics.median(y - slope * x for x, y in points)
    return slope, intercept


def observations(
    records: Iterable[dict[str, Any]],
    train_images: dict[str, int],
) -> list[tuple[float, float]]:
    """(train_images, seconds_per_epoch) from finished AND running datasets.

    Running datasets are included deliberately. Waiting for completion is what
    creates the survivorship bias in the first place; a dataset 40 epochs into
    100 is a perfectly good measurement of its own per-epoch cost, and early in
    a campaign the in-flight runs are most of the evidence there is.
    """
    points: list[tuple[float, float]] = []
    for record in records:
        name = record.get("dataset")
        images = train_images.get(str(name))
        if not images:
            continue
        # A run's first epochs are not representative: they carry dataset
        # caching, allocator warmup and cuDNN autotuning, so a dataset three
        # epochs in looks far slower than it will average out to. Including
        # those points inflates both the fit and its apparent spread.
        if record.get("state") == "running":
            done = record.get("epochs_done")
            if not isinstance(done, int) or done < MIN_EPOCHS_OBSERVED:
                continue
        seconds = _epoch_seconds(record)
        if seconds is not None and seconds > 0:
            points.append((float(images), seconds))
    return points


def _epoch_seconds(record: dict[str, Any]) -> float | None:
    mean_epoch = record.get("mean_epoch_seconds")
    if isinstance(mean_epoch, (int, float)) and mean_epoch > 0:
        return float(mean_epoch)
    done = record.get("epochs_done")
    wall = record.get("wall_seconds")
    if isinstance(done, int) and done > 0 and isinstance(wall, (int, float)) and wall > 0:
        return float(wall) / done
    if record.get("state") == "done":
        total = record.get("epochs_total")
        if isinstance(total, int) and total > 0 and isinstance(wall, (int, float)):
            return float(wall) / total
    return None


def fit_cost_model(points: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Per-epoch seconds as a function of dataset size."""
    if len(points) >= MIN_OBSERVATIONS_FOR_FIT and len({x for x, _ in points}) >= 2:
        fitted = _theil_sen(points)
        if fitted is not None:
            slope, intercept = fitted
            # A negative slope means bigger datasets train faster, which is not
            # a thing; it means the signal is noise. Degrade rather than
            # extrapolate nonsense onto 90 unseen datasets.
            if slope > 0:
                residuals = [
                    y / max(1e-9, intercept + slope * x) for x, y in points
                ]
                return {
                    "method": "theil-sen on (train_images, epoch_seconds)",
                    "per_image_seconds": slope,
                    "fixed_epoch_seconds": max(0.0, intercept),
                    "observations": len(points),
                    "residual_ratio_p90": _quantile(residuals, 0.9),
                    "residual_ratios": residuals,
                }
    if points:
        median = statistics.median(y for _, y in points)
        ratios = [y / median for _, y in points] if median else []
        return {
            "method": "median epoch seconds (too few or too noisy for a size fit)",
            "per_image_seconds": 0.0,
            "fixed_epoch_seconds": median,
            "observations": len(points),
            "residual_ratio_p90": _quantile(ratios, 0.9),
            "residual_ratios": ratios,
        }
    return {
        "method": "no observations yet",
        "per_image_seconds": 0.0,
        "fixed_epoch_seconds": 0.0,
        "observations": 0,
        "residual_ratio_p90": 1.0,
        "residual_ratios": [],
    }


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 1.0
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def predict_seconds(
    model: dict[str, Any], images: int | None, epochs: int, default: float
) -> float:
    if not model.get("observations"):
        return default
    per_epoch = model["fixed_epoch_seconds"] + model["per_image_seconds"] * float(
        images or 0
    )
    if per_epoch <= 0:
        return default
    return per_epoch * epochs


def simulate_makespan(
    lane_free_at: Sequence[float], durations: Sequence[float]
) -> float:
    """Finish time when each queued job goes to the lane that frees first.

    This is the orchestrator's actual behaviour: workers pull the next dataset
    off a shared queue as they become free. Dividing total work by lane count
    would ignore that a single long job pulled late still has to run to
    completion after everything else is done.
    """
    lanes = list(lane_free_at) or [0.0]
    for duration in durations:
        earliest = min(range(len(lanes)), key=lambda i: lanes[i])
        lanes[earliest] += duration
    return max(lanes) if lanes else 0.0


MONTE_CARLO_TRIALS = 300
MONTE_CARLO_SEED = 0


def _monte_carlo_p90(
    lane_free_at: Sequence[float],
    durations: Sequence[float],
    residual_ratios: Sequence[float],
) -> float:
    """A p90 that does not assume every dataset is wrong in the same direction.

    Inflating every predicted duration by the p90 residual is the obvious thing
    and it is far too pessimistic: it models 88 datasets all landing at their
    individual 90th percentile simultaneously. Errors that are roughly
    independent cancel in aggregate, so the honest approach is to resample the
    OBSERVED residual ratios independently per dataset and read the 90th
    percentile off the resulting makespan distribution. No distributional
    assumption is made beyond "future errors look like past errors", and with
    fewer than two observed residuals it degrades to the point estimate rather
    than inventing a spread.
    """
    ratios = [r for r in residual_ratios if r > 0]
    if len(ratios) < 2 or not durations:
        return simulate_makespan(lane_free_at, durations)
    rng = random.Random(MONTE_CARLO_SEED)
    outcomes = []
    for _ in range(MONTE_CARLO_TRIALS):
        sampled = [value * rng.choice(ratios) for value in durations]
        lanes = [value * rng.choice(ratios) for value in lane_free_at]
        outcomes.append(simulate_makespan(lanes, sampled))
    return _quantile(outcomes, 0.9)


def estimate_eta(
    records: Sequence[dict[str, Any]],
    *,
    train_images: dict[str, int] | None = None,
    lanes: int,
    epochs_total: int = 100,
    queue_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Seconds until the last dataset finishes, with a p90 band.

    ``records`` are the dashboard's per-dataset records. ``train_images`` maps
    dataset name to training image count; without it this degrades to a
    size-blind median, which is still better than mean-of-finished because it
    uses in-flight runs.
    """
    train_images = train_images or {}
    running = [r for r in records if r.get("state") == "running"]
    pending = [r for r in records if r.get("state") == "pending"]
    finished = [r for r in records if r.get("state") == "done"]

    points = observations(running + finished, train_images)
    model = fit_cost_model(points)

    finished_walls = [
        float(r["wall_seconds"])
        for r in finished
        if isinstance(r.get("wall_seconds"), (int, float))
    ]
    default_total = (
        statistics.median(finished_walls)
        if finished_walls
        else (model["fixed_epoch_seconds"] * epochs_total if model["observations"] else 0.0)
    )

    # Refuse to guess while every lane is still in its opening epochs. A run's
    # first epochs carry the image-cache fill, allocator warmup and cuDNN
    # autotuning, so `wall / epochs_done` there is not a per-epoch cost, it is a
    # startup cost being annualised. `observations()` already excludes those
    # points from the fit; extrapolating them per-lane below produced the same
    # contamination by another route. Measured 2026-08: a cache-filling ec-s
    # campaign with zero completed epochs reported "est. remaining 64.9h", and
    # a yolox-tiny campaign reported 16.3h from the same artifact. A number
    # that wrong is worse than no number, because it gets used for money
    # decisions.
    usable = [
        r
        for r in running
        if isinstance(r.get("epochs_done"), int)
        and r["epochs_done"] >= MIN_EPOCHS_OBSERVED
    ]
    # Only when lanes are ACTUALLY running and all of them are still warming.
    # A campaign that has not started yet has no misleading evidence, just no
    # evidence, and keeps the long-standing "no observations yet" answer.
    if running and not finished and not usable:
        return {
            "p50_seconds": None,
            "p90_seconds": None,
            "available": False,
            "reason": (
                "no dataset has finished, and no running dataset has passed "
                f"epoch {MIN_EPOCHS_OBSERVED}. Early epochs carry the image-cache "
                "fill and warmup, so any estimate now would be extrapolating "
                "startup cost across the whole campaign."
            ),
            "lanes": lanes,
            "pending": len(pending),
            "running": len(running),
            "done": 0,
            "model": model,
            "predicted_total_seconds": {},
        }

    # Remaining time on each lane currently busy.
    lane_free_at: list[float] = []
    for record in running:
        images = train_images.get(str(record.get("dataset")))
        total = predict_seconds(model, images, epochs_total, default_total)
        done = record.get("epochs_done") or 0
        elapsed = record.get("wall_seconds") or 0.0
        # Only trust a lane's own per-epoch rate once it is past the warmup
        # epochs; otherwise fall back to the fitted model, which was built from
        # vetted points.
        per_epoch = (
            _epoch_seconds(record)
            if isinstance(done, int) and done >= MIN_EPOCHS_OBSERVED
            else None
        )
        if per_epoch:
            remaining = max(0.0, (epochs_total - float(done)) * per_epoch)
        else:
            remaining = max(0.0, total - float(elapsed))
        lane_free_at.append(remaining)
    while len(lane_free_at) < lanes:
        lane_free_at.append(0.0)

    order = list(queue_order) if queue_order else sorted(
        str(r.get("dataset")) for r in pending
    )
    known = {str(r.get("dataset")) for r in pending}
    durations = [
        predict_seconds(model, train_images.get(name), epochs_total, default_total)
        for name in order
        if name in known
    ]

    p50 = simulate_makespan(lane_free_at, durations)
    p90 = _monte_carlo_p90(lane_free_at, durations, model.get("residual_ratios") or [])
    return {
        "p50_seconds": p50,
        "p90_seconds": p90,
        "available": True,
        "reason": None,
        "lanes": lanes,
        "pending": len(pending),
        "running": len(running),
        "done": len(finished),
        "model": model,
        "predicted_total_seconds": dict(
            zip([n for n in order if n in known], durations)
        ),
    }
