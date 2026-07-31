"""Tests for GPU telemetry capture/summary and the campaign ETA estimator."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from va_bench.gpu_trace import (
    decode_throttle,
    render_efficiency_report,
    run_dirs_from_state,
    split_by_dataset,
    summarise_trace,
    summarise_values,
    write_dataset_traces,
)
from va_bench.rf100vl_eta import (
    estimate_eta,
    fit_cost_model,
    observations,
    simulate_makespan,
)


def _record(ts: float, gpu: int, dataset: str | None, util_max: float, power: float = 100.0):
    return {
        "ts": ts,
        "gpu": gpu,
        "dataset": dataset,
        "util": {"max": util_max, "p95": util_max, "mean": util_max / 2, "min": 0.0, "n": 5},
        "power_w": {"max": power, "p95": power, "mean": power / 2, "min": 0.0, "n": 50},
        "mem_used_mb": 6400,
        "temp_c": 60,
        "sm_clock_mhz": 2500,
        "throttle": [],
    }


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------


def test_summarise_values_keeps_the_peak_not_just_the_mean() -> None:
    """A spike must survive bucketing; that is the whole point of the format."""
    summary = summarise_values([0, 0, 0, 0, 99])
    assert summary["max"] == 99
    assert summary["mean"] < 25
    assert summary["n"] == 5


def test_split_by_dataset_drops_unattributed_gaps() -> None:
    """Time between jobs belongs to no run and must not inflate one's idle."""
    records = [
        _record(1.0, 0, "alpha", 50),
        _record(2.0, 0, None, 0),  # gap between jobs
        _record(3.0, 0, "beta", 50),
    ]
    grouped = split_by_dataset(records)
    assert set(grouped) == {"alpha", "beta"}
    assert len(grouped["alpha"]) == 1


def test_summarise_trace_reports_idle_and_headroom() -> None:
    records = [_record(float(i), 0, "alpha", 0 if i < 6 else 80) for i in range(10)]
    meta = {"poll_seconds": 1.0, "gpus": {"0": {"power_cap_w": 450.0, "mem_total_mb": 24564}}}
    summary = summarise_trace(records, meta, dollars_per_hour=3.6)

    assert summary["idle_fraction"] == 0.6  # 6 of 10 buckets below threshold
    assert summary["idle_seconds"] == 6.0
    assert summary["util_percent"]["peak"] == 80
    assert summary["mem_headroom_mb"] == 24564 - 6400
    assert summary["power_fraction_of_cap"] < 1.0
    assert summary["dollars"] > 0


def test_decode_throttle_names_only_real_reasons() -> None:
    assert decode_throttle(0) == []
    assert "sw_power_cap" in decode_throttle(0x4)
    assert "hw_thermal" in decode_throttle(0x40)


def test_write_dataset_traces_emits_one_compressed_trace_per_run(tmp_path: Path) -> None:
    gpu_dir = tmp_path / "gpu"
    gpu_dir.mkdir()
    (gpu_dir / "meta.json").write_text(
        json.dumps({"poll_seconds": 1.0, "gpus": {"0": {"power_cap_w": 450.0}}}),
        encoding="utf-8",
    )
    with (gpu_dir / "gpu0.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(5):
            handle.write(json.dumps(_record(float(i), 0, "alpha", 50)) + "\n")
        handle.write('{"torn line, killed samp')  # a killed sampler leaves this

    run_dir = tmp_path / "runs" / "alpha"
    run_dir.mkdir(parents=True)
    summaries = write_dataset_traces(gpu_dir, {"alpha": run_dir})

    assert set(summaries) == {"alpha"}
    with gzip.open(run_dir / "gpu_trace.jsonl.gz", "rt", encoding="utf-8") as handle:
        assert len(handle.read().strip().splitlines()) == 5
    assert json.loads((run_dir / "gpu_summary.json").read_text())["dataset"] == "alpha"
    assert "alpha" in render_efficiency_report(summaries)


def test_run_dirs_from_state_reads_status_files(tmp_path: Path) -> None:
    (tmp_path / "alpha.json").write_text(
        json.dumps({"dataset": "alpha", "run_dir": "/runs/alpha"}), encoding="utf-8"
    )
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert run_dirs_from_state(tmp_path) == {"alpha": "/runs/alpha"}


# --------------------------------------------------------------------------
# ETA
# --------------------------------------------------------------------------


def test_makespan_is_not_total_work_over_lanes() -> None:
    """One long job pulled last sets the finish time; dividing hides that."""
    durations = [10.0, 10.0, 10.0, 100.0]
    assert sum(durations) / 2 == 65.0
    assert simulate_makespan([0.0, 0.0], durations) == 110.0


def test_fit_recovers_a_known_size_relationship() -> None:
    points = [(n, 5.0 + 0.02 * n) for n in (100, 500, 1000, 5000)]
    model = fit_cost_model(points)
    assert model["method"].startswith("theil-sen")
    assert abs(model["per_image_seconds"] - 0.02) < 1e-6
    assert abs(model["fixed_epoch_seconds"] - 5.0) < 1e-6


def test_fit_degrades_instead_of_extrapolating_noise() -> None:
    """A negative slope is noise, not a discovery; do not project it onto 90 runs."""
    points = [(100.0, 50.0), (500.0, 10.0), (1000.0, 5.0)]
    model = fit_cost_model(points)
    assert model["per_image_seconds"] == 0.0
    assert "median" in model["method"]


def test_observations_ignore_warmup_epochs_of_running_runs() -> None:
    records = [
        {"dataset": "young", "state": "running", "epochs_done": 2, "mean_epoch_seconds": 99.0},
        {"dataset": "mature", "state": "running", "epochs_done": 40, "mean_epoch_seconds": 10.0},
    ]
    points = observations(records, {"young": 100, "mature": 100})
    assert points == [(100.0, 10.0)]


def test_eta_beats_mean_of_finished_when_the_tail_is_big() -> None:
    """The survivorship trap: finished runs are the fast ones, by construction."""
    train_images = {"small1": 100, "small2": 300, "small3": 500, "huge": 10_000}
    records = [
        {"dataset": "small1", "state": "done", "wall_seconds": 300.0, "epochs_total": 100,
         "mean_epoch_seconds": 3.0},
        {"dataset": "small2", "state": "done", "wall_seconds": 500.0, "epochs_total": 100,
         "mean_epoch_seconds": 5.0},
        {"dataset": "small3", "state": "done", "wall_seconds": 700.0, "epochs_total": 100,
         "mean_epoch_seconds": 7.0},
        {"dataset": "huge", "state": "pending"},
    ]
    estimate = estimate_eta(records, train_images=train_images, lanes=1, epochs_total=100)

    # What mean-of-finished would have said: the average of three fast runs.
    naive = 500.0
    assert estimate["p50_seconds"] > naive * 5
    assert estimate["model"]["per_image_seconds"] > 0


def test_p90_band_does_not_assume_every_dataset_fails_together() -> None:
    """Independent errors cancel; a p90 that multiplies everything is wrong."""
    train_images = {f"d{i}": 100 * (i + 1) for i in range(40)}
    # Deliberately NOT collinear: a perfect fit leaves zero residual spread and
    # there would be no band to test.
    records = [
        {"dataset": "a", "state": "done", "wall_seconds": 300.0, "epochs_total": 100,
         "mean_epoch_seconds": 3.0},
        {"dataset": "b", "state": "done", "wall_seconds": 1200.0, "epochs_total": 100,
         "mean_epoch_seconds": 12.0},
        {"dataset": "c", "state": "done", "wall_seconds": 1500.0, "epochs_total": 100,
         "mean_epoch_seconds": 15.0},
    ]
    train_images.update({"a": 100, "b": 500, "c": 900})
    records += [{"dataset": name, "state": "pending"} for name in list(train_images)[:20]]

    estimate = estimate_eta(records, train_images=train_images, lanes=4, epochs_total=100)
    worst_ratio = max(estimate["model"]["residual_ratios"])

    assert estimate["p90_seconds"] >= estimate["p50_seconds"]
    assert estimate["p90_seconds"] < estimate["p50_seconds"] * worst_ratio


def test_eta_reports_its_own_method_when_it_knows_nothing() -> None:
    estimate = estimate_eta(
        [{"dataset": "a", "state": "pending"}], train_images={}, lanes=8
    )
    assert estimate["model"]["method"] == "no observations yet"
    assert estimate["p50_seconds"] == 0.0
