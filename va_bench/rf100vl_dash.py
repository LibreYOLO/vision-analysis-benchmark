"""Live web dashboard for RF100-VL training campaigns.

Serves a single-page view of everything the orchestrator and the LibreYOLO
trainers already write to disk: per-dataset queue state, what runs on every
GPU, per-epoch loss and mAP curves, ETAs, and worker log tails. Read-only,
stdlib-only, and safe to point at a live campaign: it only ever opens the
same atomic status files the orchestrator writes.

Binds to 127.0.0.1 by default. On a rented box, keep that default and open
an SSH tunnel (``ssh -L 8877:127.0.0.1:8877 root@box``) instead of exposing
the port.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .rf100vl_eta import estimate_eta

STATUS_SCHEMA = "rf100vl.train-status.v1"

# Non-dataset JSON files that share the model state directory.
_RESERVED_STEMS = {"failures", "rerun", "summary"}

_CURVE_KEYS = {
    "loss": "train/loss",
    "map5095": "metrics/mAP50-95",
    "map50": "metrics/mAP50",
    "lr": "lr/group0",
    "sec": "time",
}
_MAX_CURVE_POINTS = 400


def _read_json(path: Path) -> dict[str, Any] | None:
    """Best-effort JSON read; a live campaign may be mid-write."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _parse_ts(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _utc_now() -> float:
    return datetime.now(timezone.utc).timestamp()


def _model_state_dirs(state_root: Path) -> dict[str, Path]:
    """Map model key -> model state dir.

    Accepts either the parent ``.state`` directory (one subdir per model) or
    a single model's state dir directly.
    """
    for status_file in state_root.glob("*.json"):
        status = _read_json(status_file)
        if status and status.get("schema_version") == STATUS_SCHEMA:
            model_key = str(status.get("model_key") or state_root.name)
            return {model_key: state_root}
    found: dict[str, Path] = {}
    if state_root.is_dir():
        for child in sorted(state_root.iterdir()):
            if child.is_dir() and any(child.glob("*.json")):
                found[child.name] = child
    return found


def _dataset_statuses(model_dir: Path) -> list[dict[str, Any]]:
    records = []
    for status_file in sorted(model_dir.glob("*.json")):
        if status_file.stem in _RESERVED_STEMS:
            continue
        status = _read_json(status_file)
        if not status or status.get("schema_version") != STATUS_SCHEMA:
            continue
        records.append(status)
    return records


def _live_run_status(status: dict[str, Any]) -> dict[str, Any] | None:
    run_dir = status.get("run_dir")
    if not run_dir:
        return None
    return _read_json(Path(run_dir) / "status.json")


def _dataset_record(status: dict[str, Any]) -> dict[str, Any]:
    name = str(status.get("dataset", "?"))
    state = str(status.get("state", "pending"))
    launched = _parse_ts(status.get("launched_at"))
    finished = _parse_ts(status.get("finished_at"))
    record: dict[str, Any] = {
        "dataset": name,
        "state": state,
        "gpu": status.get("gpu"),
        "run_variant": status.get("run_variant"),
        "restart_reason": status.get("restart_reason"),
        "launched_at": status.get("launched_at"),
        "finished_at": status.get("finished_at"),
        "run_dir": status.get("run_dir"),
    }
    if launched is not None:
        end = finished if finished is not None else _utc_now()
        record["wall_seconds"] = max(0.0, end - launched)
    failure = status.get("failure")
    if isinstance(failure, dict):
        record["failure_message"] = failure.get("message")
    live = _live_run_status(status) if state in ("running", "done") else None
    if live:
        record["epochs_done"] = live.get("completed_epochs")
        record["epochs_total"] = live.get("total_epochs")
        record["best_metric"] = live.get("best_metric")
        record["best_epoch"] = live.get("best_epoch")
        record["current_metric"] = live.get("current_metric")
        record["train_loss"] = live.get("train_loss")
        record["eta_seconds"] = live.get("eta_seconds")
        record["mean_epoch_seconds"] = live.get("mean_epoch_seconds")
        record["live_updated_at"] = live.get("updated_at")
        updated = _parse_ts(live.get("updated_at"))
        if updated is not None:
            record["live_stale_seconds"] = max(0.0, _utc_now() - updated)
    return record


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_SIZE_CACHE: dict[str, dict[str, int]] = {}


def read_gpu(state_root: Path, model: str, seconds: int = 600) -> dict[str, Any]:
    """Recent GPU telemetry per card, straight off the sampler's JSONL.

    Reads only the tail: a campaign writes one line per card per second, so the
    whole file is large and the last few minutes is what a live page wants.
    """
    model_dirs = _model_state_dirs(state_root)
    model_dir = model_dirs.get(model) or next(iter(model_dirs.values()), None)
    if model_dir is None:
        return {"gpus": []}
    gpu_dir = model_dir / "gpu"
    meta = _read_json(gpu_dir / "meta.json") or {}
    poll = float(meta.get("poll_seconds") or 1.0)
    want = max(10, int(seconds / max(poll, 0.001)))

    cards = []
    for path in sorted(gpu_dir.glob("gpu*.jsonl")):
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                # ~300 bytes per record; grab a generous slice and keep the tail.
                handle.seek(max(0, size - want * 400))
                lines = handle.read().decode("utf-8", "replace").splitlines()
        except OSError:
            continue
        records = []
        for line in lines[-want:]:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records:
            continue
        last = records[-1]
        gpu_meta = (meta.get("gpus") or {}).get(str(last.get("gpu")), {})
        cap = gpu_meta.get("power_cap_w")
        cards.append(
            {
                "gpu": last.get("gpu"),
                "name": gpu_meta.get("name"),
                "datasets": last.get("datasets") or ([last["dataset"]] if last.get("dataset") else []),
                "shared": bool(last.get("shared")),
                "power_cap_w": cap,
                "mem_total_mb": gpu_meta.get("mem_total_mb"),
                "mem_used_mb": last.get("mem_used_mb"),
                "temp_c": last.get("temp_c"),
                "throttle": last.get("throttle") or [],
                # Peaks, not means: a smoothed line is exactly what hides a spike.
                "util_max": [(r.get("util") or {}).get("max") for r in records],
                "util_mean": [(r.get("util") or {}).get("mean") for r in records],
                "power_mean": [(r.get("power_w") or {}).get("mean") for r in records],
                "ts": [r.get("ts") for r in records],
            }
        )
    return {"poll_seconds": poll, "gpus": cards}


def _train_image_counts(model_dir: Path, data_dir: Path | None = None) -> dict[str, int]:
    """dataset -> training image count.

    Two sources, in order of authority. The orchestrator records
    ``batch_plan.num_train_images`` when it launches a dataset, which is
    exactly what the trainer saw. That only covers datasets already launched,
    so the ETA would be size-blind for everything still queued, which is most
    of a campaign for most of its life. Counting files under ``<dataset>/train``
    fills in the rest and is cached, since it does not change during a run.
    """
    counts: dict[str, int] = {}
    if data_dir is not None:
        key = str(data_dir)
        if key not in _SIZE_CACHE:
            scanned: dict[str, int] = {}
            if data_dir.is_dir():
                for dataset_dir in data_dir.iterdir():
                    train_dir = dataset_dir / "train"
                    if not train_dir.is_dir():
                        continue
                    scanned[dataset_dir.name] = sum(
                        1
                        for path in train_dir.iterdir()
                        if path.suffix.lower() in _IMAGE_SUFFIXES
                    )
            _SIZE_CACHE[key] = scanned
        counts.update(_SIZE_CACHE[key])

    jobs_dir = model_dir / "jobs"
    if jobs_dir.is_dir():
        for path in jobs_dir.glob("*.json"):
            config = _read_json(path)
            if not config:
                continue
            images = (config.get("batch_plan") or {}).get("num_train_images")
            name = config.get("dataset_name") or path.stem
            if isinstance(images, int) and images > 0:
                counts[str(name)] = images
    return counts


def _model_snapshot(
    model_key: str, model_dir: Path, data_dir: Path | None = None
) -> dict[str, Any]:
    datasets = [_dataset_record(status) for status in _dataset_statuses(model_dir)]
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for record in datasets:
        counts[record["state"] if record["state"] in counts else "pending"] += 1

    running = [r for r in datasets if r["state"] == "running"]
    gpus_active = {str(r.get("gpu")) for r in running if r.get("gpu") is not None}
    lanes = max(1, len(gpus_active))

    epochs_total = next(
        (
            r["epochs_total"]
            for r in datasets
            if isinstance(r.get("epochs_total"), int) and r["epochs_total"] > 0
        ),
        100,
    )
    estimate = estimate_eta(
        datasets,
        train_images=_train_image_counts(model_dir, data_dir),
        lanes=lanes,
        epochs_total=epochs_total,
    )
    eta_seconds = estimate["p50_seconds"] or None

    summary = _read_json(model_dir / "summary.json") or {}
    capabilities = summary.get("libreyolo_capabilities") or {}
    return {
        "model": model_key,
        "counts": counts,
        "total": len(datasets),
        "eta_seconds": eta_seconds,
        "eta_is_estimate": True,
        "eta": estimate,
        "libreyolo_version": capabilities.get("version"),
        "last_summary_protocol_conformant": summary.get("protocol_conformant"),
        "datasets": datasets,
    }


_DISTRIBUTION_ALIASES = {
    "va-bench": ("vision-analysis-benchmark", "va-bench", "va_bench"),
    "libreyolo": ("libreyolo",),
}


def _package_provenance(name: str) -> dict[str, Any]:
    """Version AND exact commit for an installed package.

    A campaign box pip-installs from git, so there is no .git directory to
    interrogate and ``harness_git_info`` finds nothing. pip does record the
    resolved commit in ``direct_url.json`` (PEP 610) when installing from a
    VCS URL, which is the only reliable way to answer "what code is actually
    running on this box" after the fact.
    """
    from importlib.metadata import PackageNotFoundError, distribution  # noqa: PLC0415

    info: dict[str, Any] = {"package": name}
    # The import name and the distribution name differ for this harness
    # (``va_bench`` ships as ``vision-analysis-benchmark``), so try the aliases
    # rather than reporting "not installed" for a package that plainly is.
    candidates = _DISTRIBUTION_ALIASES.get(name, (name,))
    dist = None
    for candidate in candidates:
        try:
            dist = distribution(candidate)
            break
        except PackageNotFoundError:
            continue
    if dist is None:
        info["error"] = "not installed"
        return info
    info["version"] = dist.version
    raw = dist.read_text("direct_url.json")
    if raw:
        try:
            direct = json.loads(raw)
        except json.JSONDecodeError:
            direct = {}
        vcs = direct.get("vcs_info") or {}
        if vcs.get("commit_id"):
            info["commit"] = vcs["commit_id"]
            info["commit_short"] = str(vcs["commit_id"])[:12]
        if direct.get("url"):
            info["url"] = direct["url"]
        if vcs.get("requested_revision"):
            info["branch"] = vcs["requested_revision"]
    return info


def _provenance(state_root: Path) -> dict[str, Any]:
    """What code and what inputs produced the run being watched."""
    import platform  # noqa: PLC0415

    provenance: dict[str, Any] = {
        "host": platform.node(),
        "packages": [_package_provenance("va-bench"), _package_provenance("libreyolo")],
    }
    for _model_key, model_dir in _model_state_dirs(state_root).items():
        summary = _read_json(model_dir / "summary.json") or {}
        for key in ("recipe_sha256", "versions_sha256", "protocol"):
            if summary.get(key) is not None:
                provenance.setdefault(key, summary[key])
        # Status files carry the hashes even before any summary exists.
        for status in _dataset_statuses(model_dir)[:1]:
            for key in ("recipe_sha256", "versions_sha256"):
                if status.get(key) and key not in provenance:
                    provenance[key] = status[key]
    return provenance


def scan_state(state_root: Path, data_dir: Path | None = None) -> dict[str, Any]:
    models = [
        _model_snapshot(model_key, model_dir, data_dir)
        for model_key, model_dir in _model_state_dirs(state_root).items()
    ]
    return {
        "schema_version": "rf100vl.dash-state.v1",
        "state_root": str(state_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": _provenance(state_root),
        "models": models,
    }


def read_curves(state_root: Path, model: str, dataset: str) -> dict[str, Any]:
    model_dirs = _model_state_dirs(state_root)
    model_dir = model_dirs.get(model)
    if model_dir is None:
        return {"error": f"unknown model {model!r}"}
    status = _read_json(model_dir / f"{dataset}.json")
    if not status or status.get("schema_version") != STATUS_SCHEMA:
        return {"error": f"unknown dataset {dataset!r}"}
    run_dir = status.get("run_dir")
    if not run_dir:
        return {"error": "no run directory yet"}
    metrics_path = Path(run_dir) / "metrics.jsonl"
    epochs: list[int] = []
    series: dict[str, list[float | None]] = {key: [] for key in _CURVE_KEYS}
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a live trainer may be mid-write on the last line
                epochs.append(int(row.get("epoch", len(epochs) + 1)))
                for key, column in _CURVE_KEYS.items():
                    value = row.get(column)
                    series[key].append(
                        float(value) if isinstance(value, (int, float)) else None
                    )
    except OSError:
        return {"error": "metrics.jsonl not readable yet"}
    if len(epochs) > _MAX_CURVE_POINTS:
        stride = -(-len(epochs) // _MAX_CURVE_POINTS)
        keep = list(range(0, len(epochs), stride))
        if keep[-1] != len(epochs) - 1:
            keep.append(len(epochs) - 1)
        epochs = [epochs[i] for i in keep]
        series = {key: [values[i] for i in keep] for key, values in series.items()}
    live = _live_run_status(status) or {}
    return {
        "model": model,
        "dataset": dataset,
        "epochs": epochs,
        "series": series,
        "best_metric": live.get("best_metric"),
        "best_epoch": live.get("best_epoch"),
        "total_epochs": live.get("total_epochs"),
    }


def tail_log(state_root: Path, model: str, dataset: str, lines: int) -> dict[str, Any]:
    model_dirs = _model_state_dirs(state_root)
    model_dir = model_dirs.get(model)
    if model_dir is None:
        return {"error": f"unknown model {model!r}"}
    log_path = model_dir / "logs" / f"{dataset}.log"
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as handle:
            handle.seek(max(0, size - 128 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return {"lines": [], "path": str(log_path), "error": "log not readable yet"}
    return {
        "lines": text.splitlines()[-max(1, lines):],
        "path": str(log_path),
        "size": size,
    }


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>RF100-VL campaign</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3; --dim:#8b949e;
  --run:#3b82f6; --done:#22c55e; --fail:#ef4444; --pend:#484f58; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--fg);
  font:14px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; padding:16px; }
h1 { font-size:17px; margin-bottom:2px; }
.dim { color:var(--dim); }
.small { font-size:12px; }
.model { margin-top:14px; border:1px solid var(--line); border-radius:8px;
  background:var(--panel); padding:12px; }
.mhead { display:flex; flex-wrap:wrap; gap:14px; align-items:baseline; }
.mhead b { font-size:15px; }
.chip { padding:1px 8px; border-radius:10px; font-size:12px; }
.c-done { background:#12351f; color:var(--done); }
.c-run { background:#132a4d; color:#79b8ff; }
.c-fail { background:#3d1418; color:#ff8f98; }
.c-pend { background:#21262d; color:var(--dim); }
.lanes { margin-top:10px; display:grid; gap:6px; }
.lane { display:flex; gap:10px; align-items:center; background:#0d1420;
  border:1px solid #1f3a5f; border-radius:6px; padding:6px 10px; cursor:pointer; }
.lane .gpu { color:#79b8ff; min-width:52px; font-weight:600; }
.bar { flex:1; height:8px; background:#21262d; border-radius:4px; overflow:hidden;
  min-width:80px; }
.bar i { display:block; height:100%; background:var(--run); }
.grid { margin-top:10px; display:grid; gap:5px;
  grid-template-columns:repeat(auto-fill,minmax(168px,1fr)); }
.cell { border-radius:5px; padding:5px 8px; font-size:12px; cursor:pointer;
  border:1px solid var(--line); overflow:hidden; white-space:nowrap;
  text-overflow:ellipsis; }
.cell small { display:block; color:inherit; opacity:.75; font-size:11px; }
.s-pending { background:#161b22; color:var(--dim); }
.s-running { background:#0f2a52; color:#cfe3ff; border-color:#2a5b9f;
  animation:pulse 2.2s infinite; }
.s-done { background:#0f2c1a; color:#b3f0c8; border-color:#1e5c35; }
.s-failed { background:#3a1216; color:#ffc2c7; border-color:#8f2730; }
@keyframes pulse { 50% { border-color:#79b8ff; } }
#drawer { position:fixed; top:0; right:-620px; width:min(620px,95vw); height:100vh;
  background:var(--panel); border-left:1px solid var(--line); padding:14px;
  overflow-y:auto; transition:right .15s ease; z-index:5; }
#drawer.open { right:0; }
#drawer h2 { font-size:15px; word-break:break-all; }
#drawer .x { float:right; cursor:pointer; color:var(--dim); font-size:18px; }
canvas { width:100%; height:170px; background:#0d1117; border:1px solid var(--line);
  border-radius:6px; margin-top:8px; }
pre { margin-top:8px; background:#0d1117; border:1px solid var(--line);
  border-radius:6px; padding:8px; font-size:11px; max-height:300px;
  overflow:auto; white-space:pre-wrap; word-break:break-all; }
.kv { margin-top:6px; display:grid; grid-template-columns:auto 1fr; gap:2px 12px;
  font-size:12px; }
.kv span:nth-child(odd) { color:var(--dim); }
#err { color:#ff8f98; margin-top:8px; }
</style></head><body>
<h1>RF100-VL campaign <span id="clock" class="dim small"></span></h1>
<div class="dim small" id="root"></div>
<div class="small" id="prov"></div>
<div id="err"></div>
<div id="gpus"></div>
<div id="models"></div>
<div id="drawer">
  <span class="x" onclick="closeDrawer()">&#10005;</span>
  <h2 id="dTitle"></h2>
  <div class="kv" id="dInfo"></div>
  <canvas id="cLoss" width="1160" height="340"></canvas>
  <canvas id="cMap" width="1160" height="340"></canvas>
  <pre id="dLog" class="dim">...</pre>
</div>
<script>
"use strict";
let sel = null;   // {model, dataset}
let timer = null;

function fmtEta(s) {
  if (s == null) return "";
  s = Math.max(0, Math.round(s));
  if (s < 90) return s + "s";
  if (s < 5400) return Math.round(s / 60) + "m";
  return (s / 3600).toFixed(1) + "h";
}
function fmtMetric(v) { return v == null ? "-" : Number(v).toFixed(3); }
function esc(t) { return String(t).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function laneHtml(r, model) {
  const pct = (r.epochs_total && r.epochs_done != null)
    ? Math.round(100 * r.epochs_done / r.epochs_total) : 0;
  const stale = (r.live_stale_seconds != null && r.live_stale_seconds > 180)
    ? ` <span style="color:#f0883e">stale ${fmtEta(r.live_stale_seconds)}</span>` : "";
  return `<div class="lane" onclick="openDrawer('${esc(model)}','${esc(r.dataset)}')">
    <span class="gpu">gpu ${esc(r.gpu ?? "?")}</span>
    <span>${esc(r.dataset)}</span>
    <span class="dim">${r.epochs_done ?? "?"} / ${r.epochs_total ?? "?"}</span>
    <div class="bar"><i style="width:${pct}%"></i></div>
    <span>best ${fmtMetric(r.best_metric)}</span>
    <span class="dim">eta ${fmtEta(r.eta_seconds) || "?"}${stale}</span></div>`;
}

function cellHtml(r, model) {
  let sub = "";
  if (r.state === "running")
    sub = `${r.epochs_done ?? "?"}/${r.epochs_total ?? "?"} best ${fmtMetric(r.best_metric)}`;
  else if (r.state === "done")
    sub = `best ${fmtMetric(r.best_metric)} @ ep ${r.best_epoch ?? "?"}`;
  else if (r.state === "failed")
    sub = esc((r.failure_message || "failed").slice(0, 60));
  else sub = "queued";
  const variant = r.run_variant === "fallback" ? " &#9888;" : "";
  return `<div class="cell s-${esc(r.state)}" title="${esc(r.dataset)}"
    onclick="openDrawer('${esc(model)}','${esc(r.dataset)}')">${esc(r.dataset)}${variant}
    <small>${sub}</small></div>`;
}

function renderProvenance(state) {
  // Which code produced this, and is the page even looking at a live run.
  // A dashboard pointed at a stale state root looks identical to a live one
  // unless it says so out loud.
  const p = state.provenance || {};
  const bits = [];
  (p.packages || []).forEach(function (pkg) {
    if (pkg.error) { bits.push(pkg.package + " " + pkg.error); return; }
    let s = pkg.package + " " + (pkg.version || "?");
    if (pkg.commit_short) s += " @" + pkg.commit_short;
    if (pkg.branch) s += " (" + pkg.branch + ")";
    bits.push(s);
  });
  if (p.host) bits.push("host " + p.host);
  if (p.recipe_sha256) bits.push("recipe " + String(p.recipe_sha256).slice(0, 12));
  if (p.versions_sha256) bits.push("data " + String(p.versions_sha256).slice(0, 12));

  let stale = 0;
  (state.models || []).forEach(function (m) {
    (m.datasets || []).forEach(function (d) {
      if (d.state === "running" && typeof d.live_stale_seconds === "number") {
        stale = Math.max(stale, d.live_stale_seconds);
      }
    });
  });
  const anyRunning = (state.models || []).some(function (m) { return m.counts.running > 0; });
  let warn = "";
  if (anyRunning && stale > 300) {
    warn = ' <b style="color:#ff8f98">STALE: no worker update for ' +
           Math.round(stale / 60) + ' min. Is this the live campaign?</b>';
  } else if (!anyRunning) {
    warn = ' <b style="color:#ffcc66">nothing running</b>';
  }
  document.getElementById("prov").innerHTML =
    '<span class="dim">' + bits.join(" &middot; ") + "</span>" + warn;
}

function fmtGpu(c) {
  const cap = c.power_cap_w || 0;
  const pw = c.power_mean.filter(function (v) { return v != null; });
  const um = c.util_max.filter(function (v) { return v != null; });
  const meanPw = pw.length ? pw.reduce(function (a, b) { return a + b; }, 0) / pw.length : 0;
  const meanUt = c.util_mean.filter(function (v) { return v != null; });
  const avgUt = meanUt.length ? meanUt.reduce(function (a, b) { return a + b; }, 0) / meanUt.length : 0;
  const peak = um.length ? Math.max.apply(null, um) : 0;
  const memPct = c.mem_total_mb ? (100 * c.mem_used_mb / c.mem_total_mb) : 0;
  const pwPct = cap ? (100 * meanPw / cap) : 0;

  // Sparkline of PEAKS. Plotting means would smooth away the spikes that are
  // the only reason to look at this at all.
  let spark = "";
  const pts = c.util_max.filter(function (v) { return v != null; }).slice(-120);
  if (pts.length) {
    const w = 240, h = 26;
    const step = w / Math.max(1, pts.length - 1);
    spark = '<svg width="' + w + '" height="' + h + '" style="vertical-align:middle">' +
      '<polyline fill="none" stroke="#7fd1ff" stroke-width="1.2" points="' +
      pts.map(function (v, i) { return (i * step).toFixed(1) + "," + (h - (v / 100) * h).toFixed(1); }).join(" ") +
      '" /></svg>';
  }
  const who = c.datasets.length === 0 ? '<span class="dim">idle</span>'
    : c.datasets.length === 1 ? c.datasets[0]
    : '<span title="' + c.datasets.join(", ") + '">' + c.datasets.length +
      ' jobs <span class="dim">(shared)</span></span>';
  const hot = (c.throttle && c.throttle.length)
    ? ' <b style="color:#ff8f98">' + c.throttle.join(",") + '</b>' : "";
  return '<tr><td>GPU ' + c.gpu + '</td><td>' + who + '</td>' +
    '<td>' + spark + '</td>' +
    '<td>' + avgUt.toFixed(0) + '% <span class="dim">mean</span></td>' +
    '<td>' + peak.toFixed(0) + '% <span class="dim">peak</span></td>' +
    '<td>' + meanPw.toFixed(0) + 'W <span class="dim">/' + cap.toFixed(0) + ' (' + pwPct.toFixed(0) + '%)</span></td>' +
    '<td>' + (c.mem_used_mb / 1024).toFixed(1) + 'G <span class="dim">/' +
      ((c.mem_total_mb || 0) / 1024).toFixed(0) + 'G (' + memPct.toFixed(0) + '%)</span></td>' +
    '<td>' + (c.temp_c != null ? c.temp_c + "C" : "?") + hot + '</td></tr>';
}

function renderGpus(data) {
  const el = document.getElementById("gpus");
  if (!data || !data.gpus || !data.gpus.length) {
    el.innerHTML = '<div class="dim small">No GPU telemetry. Campaign started with --no-gpu-trace, or pynvml is missing.</div>';
    return;
  }
  const anyShared = data.gpus.some(function (c) { return c.shared; });
  el.innerHTML = '<h2>GPUs <span class="dim small">utilization is time-with-a-kernel-resident, not die occupancy; read it next to power/cap</span></h2>' +
    (anyShared ? '<div class="dim small">Some cards run several jobs: those rows describe the CARD, not one dataset.</div>' : "") +
    '<table>' + data.gpus.map(fmtGpu).join("") + "</table>";
}

function refreshGpus() {
  fetch("/api/gpu").then(function (r) { return r.json(); }).then(renderGpus).catch(function () {});
}

function render(state) {
  document.getElementById("root").textContent = state.state_root;
  renderProvenance(state);
  refreshGpus();
  document.getElementById("clock").textContent =
    "updated " + new Date().toLocaleTimeString();
  const parts = [];
  for (const m of state.models) {
    const c = m.counts;
    const running = m.datasets.filter(r => r.state === "running");
    parts.push(`<div class="model"><div class="mhead"><b>${esc(m.model)}</b>
      <span class="chip c-done">${c.done} done</span>
      <span class="chip c-run">${c.running} running</span>
      <span class="chip c-pend">${c.pending} pending</span>
      <span class="chip c-fail">${c.failed} failed</span>
      <span class="dim small">of ${m.total}${m.eta_seconds != null
        ? " &middot; est. remaining " + fmtEta(m.eta_seconds) : ""}${m.libreyolo_version
        ? " &middot; libreyolo " + esc(m.libreyolo_version) : ""}</span></div>
      ${running.length ? '<div class="lanes">'
        + running.map(r => laneHtml(r, m.model)).join("") + "</div>" : ""}
      <div class="grid">${m.datasets.map(r => cellHtml(r, m.model)).join("")}</div>
      </div>`);
  }
  document.getElementById("models").innerHTML =
    parts.join("") || '<div class="dim" style="margin-top:14px">no campaign state found</div>';
}

function drawChart(canvas, epochs, seriesList, opts) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, L = 56, R = 12, T = 26, B = 30;
  ctx.clearRect(0, 0, W, H);
  ctx.font = "20px ui-monospace,monospace";
  ctx.fillStyle = "#8b949e";
  ctx.fillText(opts.title, L, 18);
  const values = seriesList.flatMap(s => s.data).filter(v => v != null);
  if (!values.length || epochs.length < 2) return;
  let lo = Math.min(...values), hi = Math.max(...values);
  if (opts.zero) lo = Math.min(lo, 0);
  if (hi === lo) hi = lo + 1;
  const x = i => L + (W - L - R) * (epochs[i] - epochs[0]) /
    Math.max(1, epochs[epochs.length - 1] - epochs[0]);
  const y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  ctx.strokeStyle = "#30363d";
  ctx.strokeRect(L, T, W - L - R, H - T - B);
  ctx.fillText(hi.toFixed(3), 2, T + 16);
  ctx.fillText(lo.toFixed(3), 2, H - B);
  ctx.fillText("ep " + epochs[0], L, H - 6);
  ctx.fillText("ep " + epochs[epochs.length - 1], W - 90, H - 6);
  for (const s of seriesList) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2.5; ctx.beginPath();
    let started = false;
    for (let i = 0; i < epochs.length; i++) {
      const v = s.data[i];
      if (v == null) continue;
      if (!started) { ctx.moveTo(x(i), y(v)); started = true; }
      else ctx.lineTo(x(i), y(v));
    }
    ctx.stroke();
  }
  if (opts.bestEpoch != null) {
    const i = epochs.indexOf(opts.bestEpoch);
    if (i >= 0) {
      ctx.strokeStyle = "#f0883e"; ctx.setLineDash([6, 6]); ctx.beginPath();
      ctx.moveTo(x(i), T); ctx.lineTo(x(i), H - B); ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}

async function refreshDrawer() {
  if (!sel) return;
  const q = `model=${encodeURIComponent(sel.model)}&dataset=${encodeURIComponent(sel.dataset)}`;
  try {
    const [curves, log] = await Promise.all([
      fetch("/api/curves?" + q).then(r => r.json()),
      fetch("/api/log?" + q + "&n=150").then(r => r.json()),
    ]);
    if (!curves.error) {
      drawChart(document.getElementById("cLoss"), curves.epochs,
        [{ data: curves.series.loss, color: "#ef4444" }],
        { title: "train/loss", zero: false });
      drawChart(document.getElementById("cMap"), curves.epochs,
        [{ data: curves.series.map5095, color: "#22c55e" },
         { data: curves.series.map50, color: "#2f6f43" }],
        { title: `valid mAP50-95 (dark: mAP50) - best ${fmtMetric(curves.best_metric)}`
          + ` @ ep ${curves.best_epoch ?? "?"}`, zero: true,
          bestEpoch: curves.best_epoch });
    }
    const pre = document.getElementById("dLog");
    pre.textContent = (log.lines || []).join("\\n") || log.error || "(empty log)";
    pre.scrollTop = pre.scrollHeight;
  } catch (e) { /* transient; next poll wins */ }
}

function openDrawer(model, dataset) {
  sel = { model, dataset };
  document.getElementById("dTitle").textContent = model + " / " + dataset;
  document.getElementById("drawer").classList.add("open");
  refreshDrawer();
}
function closeDrawer() {
  sel = null;
  document.getElementById("drawer").classList.remove("open");
}
document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });

async function tick() {
  try {
    const state = await fetch("/api/state").then(r => r.json());
    document.getElementById("err").textContent = "";
    render(state);
  } catch (e) {
    document.getElementById("err").textContent =
      "dashboard server unreachable - is the campaign box still up?";
  }
  if (sel) refreshDrawer();
}
tick();
timer = setInterval(tick, 3000);
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    state_root: Path  # set by serve()
    data_dir: Path | None = None  # set by serve()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002 - quiet server
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/":
                self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(scan_state(self.state_root, self.data_dir))
            elif parsed.path == "/api/curves":
                self._send_json(
                    read_curves(
                        self.state_root, query.get("model", ""), query.get("dataset", "")
                    )
                )
            elif parsed.path == "/api/gpu":
                self._send_json(read_gpu(self.state_root, query.get("model", "")))
            elif parsed.path == "/api/log":
                lines = min(2000, max(1, int(query.get("n", "150"))))
                self._send_json(
                    tail_log(
                        self.state_root,
                        query.get("model", ""),
                        query.get("dataset", ""),
                        lines,
                    )
                )
            else:
                self._send_json({"error": "not found"}, code=404)
        except BrokenPipeError:
            pass
        except Exception as exc:  # a broken poll must never kill the server
            try:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, code=500)
            except OSError:
                pass


def serve(
    state_root: Path,
    host: str = "127.0.0.1",
    port: int = 8877,
    open_browser: bool = False,
    data_dir: Path | None = None,
) -> None:
    state_root = Path(state_root).resolve()
    if not state_root.is_dir():
        raise SystemExit(f"state root does not exist: {state_root}")
    handler = type(
        "BoundHandler",
        (_Handler,),
        {"state_root": state_root, "data_dir": Path(data_dir) if data_dir else None},
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"RF100-VL dashboard: {url}  (state root: {state_root})")
    if host not in ("127.0.0.1", "localhost"):
        print("WARNING: bound beyond localhost; anyone who can reach this port can read logs.")
    else:
        print("Remote box? Tunnel with: ssh -L "
              f"{port}:127.0.0.1:{port} <user>@<box>  then open {url}")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard stopped")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RF100-VL live campaign dashboard")
    parser.add_argument(
        "--state-root",
        required=True,
        help="Campaign .state dir (parent of per-model dirs) or one model state dir",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="RF100-VL data root. Lets the ETA size datasets that have not "
        "started yet, which is most of them for most of a campaign.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--open", action="store_true", help="Open the browser")
    args = parser.parse_args(argv)
    serve(
        Path(args.state_root),
        host=args.host,
        port=args.port,
        open_browser=args.open,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )


if __name__ == "__main__":
    main()
