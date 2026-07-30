"""Campaign artifact handling: sync off-box, and re-score without a box.

Interruptible boxes are destroyed and take container disk with them, so anything
not synced before that is gone. And the raw detections are the durable
scientific artifact: with them, an evaluation fix costs a CPU afternoon rather
than re-running a paid campaign.

Both halves are here rather than in a script so they are importable and covered
by tests.
"""

from __future__ import annotations

import gzip
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

ARTIFACT_TIERS = ("results", "checkpoints", "all")
RESCORE_SCHEMA = "rf100vl.rescore.v1"
RESCORE_TOLERANCE = 1e-6

# Per-dataset run files worth keeping: the config that produced the run, the
# per-epoch metrics, and the log. Deliberately not the weights, which are two
# orders of magnitude larger and rarely reloaded.
RUN_FILES = (
    "data.yaml",
    "metrics.jsonl",
    "results.csv",
    "train_config.yaml",
    "train.log",
    "summary.json",
    "status.json",
)
STATE_FILES = ("summary.json", "rerun.json", "failures.json")


def collect_artifacts(
    *,
    model_key: str,
    run_id: str,
    weights_root: str | Path,
    eval_dir: str | Path | None = None,
    submissions_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    recipe_path: str | Path | None = None,
    tier: str = "results",
) -> list[tuple[Path, str]]:
    """Return (local path, path within the repo) for everything in scope."""
    if tier not in ARTIFACT_TIERS:
        raise ValueError(f"tier must be one of {ARTIFACT_TIERS}, got {tier!r}")
    weights_root = Path(weights_root)
    prefix = f"{model_key}/{run_id}"
    items: list[tuple[Path, str]] = []

    def add(path: Path, repo_path: str) -> None:
        if path.is_file():
            items.append((path, f"{prefix}/{repo_path}"))

    state = weights_root / ".state" / model_key
    for name in STATE_FILES:
        add(state / name, f"state/{name}")
    if (state / "logs").is_dir():
        for log in sorted((state / "logs").glob("*.log")):
            add(log, f"state/logs/{log.name}")

    if data_dir:
        add(Path(data_dir) / "versions.json", "provenance/versions.json")
    if recipe_path:
        add(Path(recipe_path), f"provenance/{Path(recipe_path).name}")

    runs = weights_root / ".runs" / model_key
    if runs.is_dir():
        for dataset_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            for variant in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
                base = f"runs/{dataset_dir.name}/{variant.name}"
                for filename in RUN_FILES:
                    add(variant / filename, f"{base}/{filename}")
                if tier in ("checkpoints", "all"):
                    add(variant / "weights" / "best.pt", f"{base}/weights/best.pt")
                if tier == "all":
                    add(variant / "weights" / "last.pt", f"{base}/weights/last.pt")

    if weights_root.is_dir():
        for dataset_dir in sorted(p for p in weights_root.iterdir()
                                  if p.is_dir() and not p.name.startswith(".")):
            add(dataset_dir / "stats.json", f"stats/{dataset_dir.name}.json")

    if eval_dir:
        eval_root = Path(eval_dir)
        if eval_root.is_dir():
            for path in sorted(eval_root.rglob("*")):
                if path.is_file() and not path.name.startswith("."):
                    add(path, f"eval/{path.relative_to(eval_root).as_posix()}")

    if submissions_dir:
        for path in sorted(Path(submissions_dir).glob("*.json")):
            add(path, f"submissions/{path.name}")

    return items


def select_prediction_dumps(
    eval_root: str | Path,
    fingerprint_prefix: str = "",
) -> list[Path]:
    """Find prediction dumps, refusing to double-count a dataset.

    A dataset evaluated more than once leaves one dump per fingerprint side by
    side. Averaging them would silently count that dataset twice and corrupt the
    headline mean, so this raises unless the caller disambiguates.
    """
    eval_root = Path(eval_root)
    dumps = sorted(eval_root.rglob("*.predictions.json.gz"))
    if fingerprint_prefix:
        dumps = [d for d in dumps if d.name.startswith(fingerprint_prefix)]
        if not dumps:
            raise ValueError(f"no prediction dumps match fingerprint {fingerprint_prefix!r}")
    by_dataset: dict[str, list[Path]] = {}
    for dump in dumps:
        by_dataset.setdefault(dump.parent.name, []).append(dump)
    ambiguous = sorted(name for name, paths in by_dataset.items() if len(paths) > 1)
    if ambiguous:
        raise ValueError(
            "several prediction dumps exist for the same dataset "
            f"({', '.join(ambiguous)}); pass a fingerprint prefix to choose the run"
        )
    return dumps


def load_predictions(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("fingerprint", ""), payload["detections"]


def rescore_from_predictions(
    *,
    eval_root: str | Path,
    data_dir: str | Path,
    split: str = "test",
    max_det: int = 500,
    fingerprint_prefix: str = "",
    verify: bool = True,
    evaluator: Callable[..., dict[str, float]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Recompute metrics from saved detections. No GPU, no model, no box.

    When ``verify`` is set, each recomputed metric is compared against what the
    evaluating machine recorded beside the dump. A mismatch means one of the two
    evaluations is wrong and is surfaced rather than averaged away.
    """
    from pycocotools.coco import COCO

    if evaluator is None:
        from .coco_eval import evaluate_coco as evaluator  # noqa: PLC0415

    dumps = select_prediction_dumps(eval_root, fingerprint_prefix)
    if not dumps:
        raise ValueError(f"no prediction dumps under {eval_root}")

    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for dump in dumps:
        dataset = dump.parent.name
        gt_path = Path(data_dir) / dataset / split / "_annotations.coco.json"
        if not gt_path.exists():
            if progress:
                progress(f"{dataset}: SKIP, no ground truth at {gt_path}")
            continue
        fingerprint, detections = load_predictions(dump)
        coco_gt = COCO(str(gt_path))
        metrics = evaluator(
            coco_gt, detections, image_ids=sorted(coco_gt.getImgIds()), max_det=max_det
        )
        row = {
            "dataset": dataset,
            "fingerprint": fingerprint,
            "num_detections": len(detections),
            "mAP_50_95": metrics["mAP"],
            "mAP_50": metrics["mAP50"],
        }
        recorded_path = dump.with_name(dump.name.replace(".predictions.json.gz", ".json"))
        if verify and recorded_path.exists():
            recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
            was = recorded.get("result", {}).get("metrics", {})
            if "mAP" in was:
                delta = abs(float(was["mAP"]) - metrics["mAP"])
                row["recorded_mAP_50_95"] = float(was["mAP"])
                row["delta"] = delta
                if delta > RESCORE_TOLERANCE:
                    mismatches.append(
                        {"dataset": dataset, "recorded": float(was["mAP"]),
                         "rescored": metrics["mAP"], "delta": delta}
                    )
        rows.append(row)
        if progress:
            progress(f"{dataset}: AP50:95={metrics['mAP']:.4f} AP50={metrics['mAP50']:.4f}")

    if not rows:
        raise ValueError("nothing scored: no dataset had matching ground truth")

    return {
        "schema_version": RESCORE_SCHEMA,
        "split": split,
        "max_det": max_det,
        "num_datasets": len(rows),
        "is_full_benchmark": len(rows) == 100,
        "mean_mAP_50_95": statistics.fmean(r["mAP_50_95"] for r in rows),
        "mean_mAP_50": statistics.fmean(r["mAP_50"] for r in rows),
        "mismatches": mismatches,
        "datasets": rows,
    }


def upload_artifacts(
    items: Iterable[tuple[Path, str]],
    *,
    repo: str,
    token: str,
    private: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Upload, skipping anything already present so a partial sync resumes."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    existing = set(api.list_repo_files(repo, repo_type="dataset"))

    uploaded = skipped = 0
    for path, repo_path in items:
        if repo_path in existing:
            skipped += 1
            continue
        api.upload_file(path_or_fileobj=str(path), path_in_repo=repo_path,
                        repo_id=repo, repo_type="dataset")
        uploaded += 1
        if progress:
            progress(f"+ {repo_path}")
    return {"uploaded": uploaded, "skipped": skipped}
