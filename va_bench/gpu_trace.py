"""GPU telemetry for a campaign: capture, per-dataset split, post-mortem.

Two properties make the difference between telemetry you can act on and
telemetry that quietly misleads you.

**Polling aliases peaks away.** Reading `nvidia-smi` once a second tells you
what was happening at one instant per second and nothing about the other 999
milliseconds. NVML keeps its own ring buffer of utilization and power samples
at driver resolution, and `nvmlDeviceGetSamples` returns every sample since
the previous call, including the ones taken while this process was asleep. So
a cheap 1 Hz poll still recovers sub-second peaks. Measured on an 8x4090
campaign: each 1 second bucket carried about 5 utilization and 50 power
samples the driver had already captured for us.

**Averaging destroys peaks.** Every bucket therefore keeps max, p95, mean and
min. Anything that downsamples later must drop resolution by keeping maxima,
never by averaging maxima together.

One more trap, surfaced rather than hidden: NVML "utilization" is the fraction
of TIME during which at least one kernel was resident, not the fraction of the
GPU doing work. A small model can read 90% while using a sliver of the die.
Power as a fraction of the enforced cap is the better honesty check, so it is
captured alongside and reported next to it.

Attribution is by GPU index, which is exact while the orchestrator pins one
dataset per GPU. If that ever stops being true, these numbers describe the
CARD rather than the job, and `attribution` in the summary says so.
"""

from __future__ import annotations

import gzip
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Iterable

TRACE_SCHEMA = "rf100vl.gputrace.v1"
SUMMARY_SCHEMA = "rf100vl.gpusummary.v1"

POLL_SECONDS = 1.0
REMAP_SECONDS = 10.0
IDLE_UTIL_PERCENT = 5.0

TRACE_FILENAME = "gpu_trace.jsonl.gz"
SUMMARY_FILENAME = "gpu_summary.json"

# Bit positions from NVML's clocks-event-reasons mask. Only the ones that mean
# "your run went slower than the hardware could have gone" are decoded.
_THROTTLE_FLAGS = {
    "sw_power_cap": 0x0000000000000004,
    "hw_slowdown": 0x0000000000000008,
    "clocks_setting": 0x0000000000000010,
    "sw_thermal": 0x0000000000000020,
    "hw_thermal": 0x0000000000000040,
    "hw_power_brake": 0x0000000000000080,
}


def decode_throttle(mask: int) -> list[str]:
    return sorted(name for name, bit in _THROTTLE_FLAGS.items() if mask & bit)


def summarise_values(values: Iterable[float]) -> dict[str, float] | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "max": ordered[-1],
        "p95": ordered[index],
        "mean": round(statistics.fmean(ordered), 2),
        "min": ordered[0],
        "n": len(ordered),
    }


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def _gpu_to_dataset(state_dir: Path) -> dict[int, str]:
    """Which dataset is on which card, from the status files already written."""
    mapping: dict[int, str] = {}
    if not state_dir.is_dir():
        return mapping
    for path in state_dir.glob("*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("state") == "running" and status.get("gpu") is not None:
            try:
                mapping[int(status["gpu"])] = str(status.get("dataset", "?"))
            except (TypeError, ValueError):
                continue
    return mapping


class GpuSampler:
    """Samples every visible GPU into <out_dir>/gpu<N>.jsonl until stopped.

    Runs as a daemon thread inside the orchestrator. Read-only against NVML, so
    it cannot perturb training; if NVML is unavailable it degrades to doing
    nothing rather than taking the campaign down with it.
    """

    def __init__(
        self,
        out_dir: str | Path,
        state_dir: str | Path,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.state_dir = Path(state_dir)
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        try:
            import pynvml  # noqa: F401, PLC0415
        except Exception as exc:
            self.error = f"pynvml unavailable, no GPU telemetry: {exc}"
            return False
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:  # telemetry must never kill a paid campaign
            self.error = str(exc)

    def _loop(self) -> None:
        import pynvml as N  # noqa: PLC0415

        N.nvmlInit()
        count = N.nvmlDeviceGetCount()
        handles = [N.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        self.out_dir.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA,
            "poll_seconds": self.poll_seconds,
            "idle_util_percent": IDLE_UTIL_PERCENT,
            "gpus": {},
        }
        for index, handle in enumerate(handles):
            entry: dict[str, Any] = {}
            for key, call in (
                ("name", lambda h: N.nvmlDeviceGetName(h)),
                ("power_cap_w", lambda h: N.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0),
                ("mem_total_mb", lambda h: N.nvmlDeviceGetMemoryInfo(h).total // 2**20),
            ):
                try:
                    entry[key] = call(handle)
                except Exception:
                    entry[key] = None
            meta["gpus"][str(index)] = entry
        (self.out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        files = {
            i: (self.out_dir / f"gpu{i}.jsonl").open("a", encoding="utf-8")
            for i in range(count)
        }
        last_util = {i: 0 for i in range(count)}
        last_power = {i: 0 for i in range(count)}
        mapping: dict[int, str] = {}
        last_remap = 0.0

        try:
            while not self.stop_event.is_set():
                now = time.time()
                if now - last_remap >= REMAP_SECONDS:
                    mapping = _gpu_to_dataset(self.state_dir)
                    last_remap = now
                for index, handle in enumerate(handles):
                    record, last_util[index], last_power[index] = _sample_one(
                        N, handle, index, mapping.get(index),
                        last_util[index], last_power[index], now,
                    )
                    files[index].write(json.dumps(record) + "\n")
                    files[index].flush()
                self.stop_event.wait(
                    max(0.0, self.poll_seconds - (time.time() - now))
                )
        finally:
            for handle in files.values():
                handle.close()


def _drain(N: Any, handle: Any, sample_type: Any, last_seen: int) -> tuple[list[float], int]:
    """Every driver-buffered sample newer than last_seen."""
    try:
        _kind, samples = N.nvmlDeviceGetSamples(handle, sample_type, last_seen)
    except Exception:
        return [], last_seen
    values: list[float] = []
    newest = last_seen
    for sample in samples:
        stamp = int(sample.timeStamp)
        if stamp <= last_seen:
            continue
        newest = max(newest, stamp)
        values.append(float(sample.sampleValue.uiVal))
    return values, newest


def _sample_one(
    N: Any,
    handle: Any,
    index: int,
    dataset: str | None,
    last_util: int,
    last_power: int,
    now: float,
) -> tuple[dict[str, Any], int, int]:
    util_values, last_util = _drain(N, handle, N.NVML_GPU_UTILIZATION_SAMPLES, last_util)
    power_values, last_power = _drain(N, handle, N.NVML_TOTAL_POWER_SAMPLES, last_power)

    def _safe(call: Any) -> Any:
        try:
            return call()
        except Exception:
            return None

    mask = 0
    for name in (
        "nvmlDeviceGetCurrentClocksEventReasons",
        "nvmlDeviceGetCurrentClocksThrottleReasons",
    ):
        function = getattr(N, name, None)
        if function is not None:
            mask = _safe(lambda: int(function(handle))) or 0
            break

    record = {
        "ts": round(now, 3),
        "gpu": index,
        "dataset": dataset,
        "util": summarise_values(util_values),
        "power_w": summarise_values([value / 1000.0 for value in power_values]),
        "mem_used_mb": _safe(lambda: N.nvmlDeviceGetMemoryInfo(handle).used // 2**20),
        "temp_c": _safe(lambda: N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU)),
        "sm_clock_mhz": _safe(lambda: N.nvmlDeviceGetClockInfo(handle, N.NVML_CLOCK_SM)),
        "throttle": decode_throttle(mask),
    }
    return record, last_util, last_power


# --------------------------------------------------------------------------
# split and summarise
# --------------------------------------------------------------------------


def read_trace_dir(gpu_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """All records from every gpu*.jsonl, plus meta.json if present."""
    gpu_dir = Path(gpu_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(gpu_dir.glob("gpu*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line from a killed sampler
    meta_path = gpu_dir / "meta.json"
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    return records, meta


def split_by_dataset(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group by the dataset stamped on each record, dropping unattributed ones.

    Records with no dataset are the gaps between jobs. They belong to no run
    and are deliberately not folded into a neighbour, which would inflate that
    run's idle time with time it did not own.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dataset = record.get("dataset")
        if not dataset:
            continue
        grouped.setdefault(str(dataset), []).append(record)
    for values in grouped.values():
        values.sort(key=lambda item: item.get("ts", 0.0))
    return grouped


def summarise_trace(
    records: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    dollars_per_hour: float | None = None,
) -> dict[str, Any]:
    """Post-mortem for one dataset: where the time and the money went."""
    if not records:
        raise ValueError("no records to summarise")
    meta = meta or {}
    poll = float(meta.get("poll_seconds") or POLL_SECONDS)
    gpus = sorted({record.get("gpu") for record in records if record.get("gpu") is not None})
    gpu_meta = (meta.get("gpus") or {}).get(str(gpus[0])) if gpus else None
    power_cap = (gpu_meta or {}).get("power_cap_w")
    mem_total = (gpu_meta or {}).get("mem_total_mb")

    util_max = [r["util"]["max"] for r in records if r.get("util")]
    util_p95 = [r["util"]["p95"] for r in records if r.get("util")]
    util_mean = [r["util"]["mean"] for r in records if r.get("util")]
    power_max = [r["power_w"]["max"] for r in records if r.get("power_w")]
    power_mean = [r["power_w"]["mean"] for r in records if r.get("power_w")]
    mem = [r["mem_used_mb"] for r in records if r.get("mem_used_mb") is not None]
    temps = [r["temp_c"] for r in records if r.get("temp_c") is not None]

    idle_buckets = sum(1 for value in util_max if value < IDLE_UTIL_PERCENT)
    throttle_counts: dict[str, int] = {}
    for record in records:
        for reason in record.get("throttle") or []:
            throttle_counts[reason] = throttle_counts.get(reason, 0) + 1

    stamps = [record["ts"] for record in records if record.get("ts") is not None]
    span = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0
    # Gaps mean the sampler was descheduled; say so rather than pretending the
    # covered seconds are the whole story.
    gaps = 0
    for earlier, later in zip(stamps, stamps[1:]):
        if later - earlier > poll * 3:
            gaps += 1

    covered_seconds = len(records) * poll
    gpu_hours = covered_seconds / 3600.0
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "gpus": gpus,
        "attribution": "per-gpu" if len(gpus) == 1 else "per-gpu (multiple cards seen)",
        "samples": len(records),
        "poll_seconds": poll,
        "wall_seconds": round(span, 1),
        "covered_seconds": round(covered_seconds, 1),
        "sampler_gaps": gaps,
        "gpu_hours": round(gpu_hours, 4),
        "util_percent": {
            "peak": max(util_max) if util_max else None,
            "p95_of_buckets": round(statistics.fmean(util_p95), 2) if util_p95 else None,
            "mean": round(statistics.fmean(util_mean), 2) if util_mean else None,
        },
        "power_w": {
            "peak": round(max(power_max), 1) if power_max else None,
            "mean": round(statistics.fmean(power_mean), 1) if power_mean else None,
            "cap": power_cap,
        },
        "mem_mb": {"peak": max(mem) if mem else None, "total": mem_total},
        "temp_c": {"peak": max(temps) if temps else None},
        "idle_seconds": round(idle_buckets * poll, 1),
        "idle_fraction": round(idle_buckets / len(records), 4),
        "throttle_seconds": {k: round(v * poll, 1) for k, v in sorted(throttle_counts.items())},
    }
    if power_cap:
        mean_power = summary["power_w"]["mean"]
        summary["power_fraction_of_cap"] = (
            round(mean_power / power_cap, 4) if mean_power else None
        )
    if mem_total and summary["mem_mb"]["peak"]:
        summary["mem_headroom_mb"] = mem_total - summary["mem_mb"]["peak"]
    if dollars_per_hour:
        summary["dollars"] = round(gpu_hours * dollars_per_hour, 4)
    return summary


def run_dirs_from_state(state_dir: str | Path) -> dict[str, str]:
    """dataset -> run_dir, read from the status files the orchestrator writes."""
    state_dir = Path(state_dir)
    mapping: dict[str, str] = {}
    if not state_dir.is_dir():
        return mapping
    for path in state_dir.glob("*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset = status.get("dataset")
        run_dir = status.get("run_dir")
        if dataset and run_dir:
            mapping[str(dataset)] = str(run_dir)
    return mapping


def write_dataset_traces(
    gpu_dir: str | Path,
    run_dirs: dict[str, str | Path],
    dollars_per_hour: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Split the per-GPU streams into one compressed trace per training run.

    Measured on a real campaign: 280 bytes per record raw, 29 gzipped, so a
    25 minute run costs about 44 KB and a 100 dataset campaign about 4.4 MB.
    Cheaper than one checkpoint, which is why nothing is thrown away.
    """
    records, meta = read_trace_dir(gpu_dir)
    grouped = split_by_dataset(records)
    summaries: dict[str, dict[str, Any]] = {}
    for dataset, dataset_records in grouped.items():
        target = run_dirs.get(dataset)
        if target is None:
            continue
        target = Path(target)
        if not target.is_dir():
            continue
        payload = "\n".join(json.dumps(record) for record in dataset_records) + "\n"
        with gzip.open(target / TRACE_FILENAME, "wt", encoding="utf-8") as handle:
            handle.write(payload)
        summary = summarise_trace(dataset_records, meta, dollars_per_hour)
        summary["dataset"] = dataset
        (target / SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        summaries[dataset] = summary
    return summaries


def render_efficiency_report(summaries: dict[str, dict[str, Any]]) -> str:
    """Post-mortem table, ordered so the worst waste is at the top."""
    if not summaries:
        return "No GPU telemetry found. Was the sampler running?"
    rows = sorted(
        summaries.values(),
        key=lambda item: item.get("idle_fraction") or 0.0,
        reverse=True,
    )
    total_hours = sum(row.get("gpu_hours") or 0.0 for row in rows)
    idle_hours = sum(
        (row.get("gpu_hours") or 0.0) * (row.get("idle_fraction") or 0.0) for row in rows
    )
    means = [
        row["util_percent"]["mean"]
        for row in rows
        if (row.get("util_percent") or {}).get("mean") is not None
    ]
    lines = [
        "# RF100-VL GPU efficiency",
        "",
        f"- datasets with telemetry: {len(rows)}",
        f"- total GPU-hours: {total_hours:.2f}",
        f"- GPU-hours below {IDLE_UTIL_PERCENT:.0f}% utilization: "
        f"{idle_hours:.2f} ({idle_hours / total_hours * 100:.1f}%)"
        if total_hours
        else "- GPU-hours idle: n/a",
    ]
    if means:
        lines.append(f"- mean utilization across datasets: {statistics.fmean(means):.1f}%")
    dollars = [row["dollars"] for row in rows if row.get("dollars") is not None]
    if dollars:
        lines.append(f"- attributed spend: ${sum(dollars):.2f}")
    lines += [
        "",
        "Utilization is the fraction of TIME a kernel was resident, not the",
        "fraction of the die at work. Read it next to power/cap.",
        "",
        "| dataset | GPU-h | util mean | util peak | power/cap | idle | peak mem | throttled |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        util = row.get("util_percent") or {}
        power_fraction = row.get("power_fraction_of_cap")
        throttled = sum((row.get("throttle_seconds") or {}).values())
        lines.append(
            "| {dataset} | {hours:.2f} | {mean} | {peak} | {power} | {idle:.0%} | {mem} | {throttled} |".format(
                dataset=row.get("dataset", "?"),
                hours=row.get("gpu_hours") or 0.0,
                mean=f"{util.get('mean'):.1f}%" if util.get("mean") is not None else "?",
                peak=f"{util.get('peak'):.0f}%" if util.get("peak") is not None else "?",
                power=f"{power_fraction:.0%}" if power_fraction is not None else "?",
                idle=row.get("idle_fraction") or 0.0,
                mem=f"{(row.get('mem_mb') or {}).get('peak', '?')} MB",
                throttled=f"{throttled:.0f}s" if throttled else "no",
            )
        )
    return "\n".join(lines) + "\n"
