"""Human-readable reports from RF100-VL submission JSONs.

Submissions stay raw (per-dataset records, no derived aggregates beyond the
overall mean). This module renders them for humans: a per-model report with
domain means, completion, train cost, and the weak tail, plus a multi-model
leaderboard table. Markdown out, paste-ready for issues and model cards.

Rendering is a pure function of the submission JSON (plus optional per-dataset
train stats), so a report can be rebuilt at any time from published artifacts.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .rf100vl_data import load_domain_manifest


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def _dataset_domains() -> dict[str, str]:
    try:
        manifest = load_domain_manifest()
    except Exception:
        return {}
    return {
        name: (str(entry.get("domain", "unknown")) if isinstance(entry, dict) else str(entry))
        for name, entry in manifest.get("datasets", {}).items()
    }


def load_train_stats(weights_root: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    """Read per-dataset training stats.json files under the weights root."""
    stats: dict[str, dict[str, Any]] = {}
    for name in names:
        path = weights_root / name / "stats.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(loaded, dict):
            stats[name] = loaded
    return stats


def _completion_line(rf: dict[str, Any]) -> str:
    expected = int(rf.get("protocol", {}).get("expected_datasets", 100))
    ok = len(rf.get("datasets", []))
    return f"{ok}/{expected}"


def _domain_rows(records: list[dict[str, Any]]) -> list[tuple[str, int, float, float]]:
    domains = _dataset_domains()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        domain = domains.get(str(record.get("dataset")), "unknown")
        grouped.setdefault(domain, []).append(record)
    rows = []
    for domain in sorted(grouped):
        group = grouped[domain]
        rows.append(
            (
                domain,
                len(group),
                statistics.fmean(float(r.get("mAP_50_95", 0.0)) for r in group),
                statistics.fmean(float(r.get("mAP_50", 0.0)) for r in group),
            )
        )
    return rows


def build_report(
    submission: dict[str, Any],
    weights_root: Path | None = None,
) -> str:
    """Render one submission as a markdown report."""
    model = submission.get("model", {})
    accuracy = submission.get("accuracy", {})
    rf = submission.get("rf100vl", {})
    records: list[dict[str, Any]] = list(rf.get("datasets", []))
    names = [str(r.get("dataset")) for r in records]
    train_stats = (
        load_train_stats(Path(weights_root), names) if weights_root is not None else {}
    )

    lines: list[str] = []
    title = model.get("name") or model.get("id") or "model"
    lines.append(f"# RF100-VL report: {title}")
    lines.append("")
    valid = bool(rf.get("valid_submission"))
    lines.append(
        f"**mAP50-95 {_fmt(accuracy.get('mAP_50_95'))}** | "
        f"mAP50 {_fmt(accuracy.get('mAP_50'))} | "
        f"ok {_completion_line(rf)} | "
        f"regime {rf.get('regime', '?')} | "
        f"valid submission: {'YES' if valid else 'NO'}"
    )
    lines.append("")

    if not valid:
        lines.append("## Not submittable because")
        lines.append("")
        for reason in rf.get("invalid_reasons", []) or ["(no reasons recorded)"]:
            lines.append(f"- {reason}")
        lines.append("")

    if records:
        lines.append("## Domains")
        lines.append("")
        lines.append("| domain | datasets | mAP50-95 | mAP50 |")
        lines.append("|---|---|---|---|")
        for domain, count, map5095, map50 in _domain_rows(records):
            lines.append(
                f"| {domain} | {count} | {_fmt(map5095)} | {_fmt(map50)} |"
            )
        lines.append("")

    skipped = rf.get("skipped_datasets") or []
    if skipped:
        preview = ", ".join(sorted(str(s) for s in skipped)[:8])
        suffix = ", ..." if len(skipped) > 8 else ""
        lines.append(f"Skipped (no checkpoint): {len(skipped)} - {preview}{suffix}")
        lines.append("")

    if train_stats:
        walls = [
            float(s["wall_seconds"])
            for s in train_stats.values()
            if isinstance(s.get("wall_seconds"), (int, float))
        ]
        best_epochs = [
            int(s["best_epoch"])
            for s in train_stats.values()
            if isinstance(s.get("best_epoch"), int)
        ]
        if walls:
            lines.append("## Train cost")
            lines.append("")
            lines.append(
                f"- median wall per dataset: {statistics.median(walls) / 60:.0f} min"
            )
            lines.append(f"- total: {sum(walls) / 3600:.1f} GPU-hours")
            if best_epochs:
                lines.append(
                    f"- median best epoch: {int(statistics.median(best_epochs))}"
                )
            lines.append("")

    scored = [r for r in records if isinstance(r.get("mAP_50_95"), (int, float))]
    if len(scored) >= 10:
        lines.append("## Weakest datasets")
        lines.append("")
        for record in sorted(scored, key=lambda r: float(r["mAP_50_95"]))[:5]:
            lines.append(
                f"- {record.get('dataset')}: {_fmt(record.get('mAP_50_95'))}"
            )
        lines.append("")

    lines.append(
        f"Protocol {rf.get('protocol', {}).get('version', '?')}, "
        f"maxDets {rf.get('protocol', {}).get('max_det', '?')}, "
        f"recipe sha {str(rf.get('recipe_sha256') or '?')[:12]}, "
        f"dataset versions sha {str(rf.get('dataset_versions_sha256') or '?')[:12]}."
    )
    return "\n".join(lines) + "\n"


def build_leaderboard(
    submissions: list[dict[str, Any]],
    weights_roots: dict[str, Path] | None = None,
) -> str:
    """Render several submissions as one leaderboard table (newest per model)."""
    latest: dict[str, dict[str, Any]] = {}
    for submission in submissions:
        model_id = str(submission.get("model", {}).get("id", "?"))
        key = str(submission.get("submission_id", ""))
        held = latest.get(model_id)
        if held is None or key > str(held.get("submission_id", "")):
            latest[model_id] = submission

    rows = []
    for model_id, submission in latest.items():
        accuracy = submission.get("accuracy", {})
        rf = submission.get("rf100vl", {})
        median_min = None
        root = (weights_roots or {}).get(model_id)
        if root is not None:
            names = [str(r.get("dataset")) for r in rf.get("datasets", [])]
            walls = [
                float(s["wall_seconds"])
                for s in load_train_stats(root, names).values()
                if isinstance(s.get("wall_seconds"), (int, float))
            ]
            if walls:
                median_min = statistics.median(walls) / 60
        rows.append(
            (
                float(accuracy.get("mAP_50_95") or 0.0),
                model_id,
                _completion_line(rf),
                accuracy.get("mAP_50"),
                accuracy.get("mAP_50_95"),
                median_min,
                bool(rf.get("valid_submission")),
            )
        )
    rows.sort(reverse=True)

    lines = [
        "| # | model | ok | mAP50 | mAP50-95 | median train | valid |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, (_, model_id, ok, map50, map5095, median_min, valid) in enumerate(
        rows, start=1
    ):
        train = f"~{median_min:.0f} min" if median_min is not None else "-"
        lines.append(
            f"| {rank} | {model_id} | {ok} | {_fmt(map50)} | **{_fmt(map5095)}** "
            f"| {train} | {'yes' if valid else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


def load_submissions(path: Path) -> list[dict[str, Any]]:
    """Load one submission file or every submission JSON in a directory."""
    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted(path.glob("*.json"))
    submissions = []
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(loaded, dict) and "rf100vl" in loaded:
            submissions.append(loaded)
    return submissions
