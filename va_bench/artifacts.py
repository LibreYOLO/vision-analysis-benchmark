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
    # GPU telemetry. Measured at 29 bytes per second per dataset gzipped, so a
    # 100 dataset campaign adds about 4.4 MB, less than one checkpoint.
    "gpu_trace.jsonl.gz",
    "gpu_summary.json",
)
STATE_FILES = ("summary.json", "rerun.json", "failures.json", "manifest.json")
MANIFEST_SCHEMA = "rf100vl.manifest.v1"
MANIFEST_FILENAME = "manifest.json"


def package_provenance(name: str) -> dict[str, Any]:
    """Version and exact commit of an installed package.

    A campaign box pip-installs from git and has no .git directory, so asking
    git is useless after the fact. pip records the resolved commit in
    ``direct_url.json`` (PEP 610) for VCS installs, which is the only reliable
    way to answer "which code produced these numbers" once the box is gone.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    # Import name and distribution name differ for this harness.
    aliases = {
        "va-bench": ("vision-analysis-benchmark", "va-bench", "va_bench"),
    }.get(name, (name,))
    info: dict[str, Any] = {"package": name}
    dist = None
    for candidate in aliases:
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
        if vcs.get("requested_revision"):
            info["requested_revision"] = vcs["requested_revision"]
        if direct.get("url"):
            info["url"] = direct["url"]
    return info


def build_manifest(
    *,
    model_key: str,
    run_id: str,
    state_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    recipe_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Everything needed to say what produced this upload, and reproduce it.

    Results that cannot be traced to an exact commit are not evidence, they are
    anecdotes. This travels beside the numbers so a reader a year from now can
    identify the code, the recipe, the dataset lock and the hardware without
    access to the box, which will be long destroyed.
    """
    from datetime import datetime, timezone

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "model_key": model_key,
        "run_id": run_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "packages": [package_provenance("libreyolo"), package_provenance("va-bench")],
    }

    try:
        import platform

        manifest["host"] = {
            "node": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
    except Exception:
        pass

    try:
        import torch

        devices = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        manifest["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": devices,
            "gpu_count": len(devices),
        }
    except Exception:
        pass

    if recipe_path and Path(recipe_path).is_file():
        recipe_file = Path(recipe_path)
        manifest["recipe"] = {
            "filename": recipe_file.name,
            "sha256": _sha256(recipe_file),
        }
        try:
            manifest["recipe"]["protocol"] = json.loads(
                recipe_file.read_text(encoding="utf-8")
            ).get("protocol")
        except Exception:
            pass

    if data_dir:
        versions = Path(data_dir) / "versions.json"
        if versions.is_file():
            # Use the harness's own canonical hash, not a hash of the raw bytes.
            # They differ (one hashes parsed content, the other the file), and
            # publishing both invites a reader to conclude the lock was tampered
            # with when only the hash DEFINITION differs.
            try:
                from .rf100vl_data import version_lock_sha256

                digest = version_lock_sha256(
                    json.loads(versions.read_text(encoding="utf-8"))
                )
            except Exception:
                digest = _sha256(versions)
            manifest["dataset_versions"] = {
                "filename": "versions.json",
                "sha256": digest,
                "hash_of": "canonical version-lock content",
            }

    if state_dir:
        state = Path(state_dir)
        summary = state / "summary.json"
        if summary.is_file():
            try:
                data = json.loads(summary.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            manifest["campaign"] = {
                key: data.get(key)
                for key in (
                    "completed",
                    "failed",
                    "interrupted",
                    "skipped_done",
                    "protocol_conformant",
                    "libreyolo_capabilities",
                )
                if data.get(key) is not None
            }
        # Status files carry the hashes the workers actually used, which beats
        # re-deriving them here: a mismatch is exactly what we want visible.
        # They are also the truth about how much of the campaign is finished:
        # summary.json's "completed" counts only THIS invocation, so a resumed
        # campaign reported 0 completed while seven datasets were done.
        states: dict[str, int] = {}
        for status_file in sorted(state.glob("*.json")):
            if status_file.stem in {"summary", "rerun", "failures", "manifest"}:
                continue
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if status.get("schema_version", "").startswith("rf100vl.train-status"):
                key = str(status.get("state", "unknown"))
                states[key] = states.get(key, 0) + 1
            observed = {
                key: status[key]
                for key in ("recipe_sha256", "versions_sha256")
                if status.get(key)
            }
            if observed:
                manifest.setdefault("observed_hashes", observed)
        if states:
            manifest["dataset_states"] = dict(sorted(states.items()))
            manifest["datasets_total"] = sum(states.values())
    return manifest


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(state_dir: str | Path, manifest: dict[str, Any]) -> Path:
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


class BackgroundSyncer:
    """Upload artifacts as the campaign produces them, never blocking it.

    An interruptible box can vanish between one dataset and the next, and the
    old advice was "run sync-artifacts after each dataset", which is toil a
    human at 3am will skip. This does it automatically.

    Three properties matter more than throughput:

    * **It cannot fail the campaign.** Every exception is swallowed and
      recorded. Losing an upload costs one re-sync; losing a campaign costs
      GPU-hours.
    * **It coalesces.** Several datasets finishing together produce one sync,
      not one per dataset, because each sync is a full collect and the uploader
      already skips unchanged files by size.
    * **It never runs two uploads at once.** One worker thread, so concurrent
      commits cannot race each other into a 412 on the hub.
    """

    def __init__(
        self,
        *,
        collect: Callable[[], list[tuple[Path, str]]],
        repo: str,
        token: str | None = None,
        private: bool = False,
        min_interval_seconds: float = 60.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        import queue as _queue
        import threading as _threading

        self._collect = collect
        self._repo = repo
        self._token = token
        self._private = private
        self._min_interval = min_interval_seconds
        self._log = log or (lambda message: None)
        self._queue: _queue.Queue[str | None] = _queue.Queue()
        self._thread: _threading.Thread | None = None
        self._stopping = _threading.Event()
        self.syncs = 0
        self.uploaded = 0
        self.failures = 0
        self.last_error: str | None = None

    def start(self) -> None:
        import threading as _threading

        self._thread = _threading.Thread(
            target=self._run, name="artifact-syncer", daemon=True
        )
        self._thread.start()

    def notify(self, dataset: str) -> None:
        self._queue.put(dataset)

    def stop(self, timeout: float = 600.0) -> None:
        """Drain, then run one final sync so the last dataset is never lost."""
        self._stopping.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        import queue as _queue
        import time as _time

        last_sync = 0.0
        pending: list[str] = []
        while True:
            try:
                item = self._queue.get(timeout=5.0)
                if item is None:
                    self._sync(pending, final=True)
                    return
                pending.append(item)
            except _queue.Empty:
                if self._stopping.is_set():
                    self._sync(pending, final=True)
                    return
                if not pending:
                    continue
            # Drain anything that arrived while we were waiting.
            while True:
                try:
                    extra = self._queue.get_nowait()
                except _queue.Empty:
                    break
                if extra is None:
                    self._sync(pending, final=True)
                    return
                pending.append(extra)
            if pending and _time.time() - last_sync >= self._min_interval:
                self._sync(pending)
                pending = []
                last_sync = _time.time()

    def _sync(self, datasets: list[str], final: bool = False) -> None:
        label = "final" if final else f"after {', '.join(datasets[:3])}"
        try:
            items = self._collect()
            result = upload_artifacts(
                items, repo=self._repo, token=self._token, private=self._private
            )
            self.syncs += 1
            self.uploaded += result["uploaded"]
            if result["uploaded"]:
                self._log(
                    f"auto-sync ({label}): uploaded {result['uploaded']}, "
                    f"skipped {result['skipped']}"
                )
        except Exception as exc:  # never fail a campaign over an upload
            self.failures += 1
            self.last_error = str(exc)
            self._log(f"auto-sync ({label}) FAILED, will retry later: {exc}")


def default_run_id(manifest: dict[str, Any]) -> str:
    """A run id nobody has to invent, and nobody can accidentally reuse.

    ``--run-id`` was a required free-text string, and reusing one is silently
    destructive: the second campaign's files land on the first's paths, and with
    same-size files they are skipped rather than overwritten, so you read run A
    under run B's name. Deriving it from the date plus the code identity makes
    a genuine repeat collide only when the code, recipe and day are identical,
    which is the one case where sharing a folder is correct.
    """
    import hashlib

    parts = [
        entry.get("commit", entry.get("version", "?"))
        for entry in manifest.get("packages", [])
    ]
    parts.append((manifest.get("recipe") or {}).get("sha256", ""))
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8]
    day = str(manifest.get("created_at", ""))[:10].replace("-", "")
    return f"{day}-{manifest.get('model_key', 'model')}-{digest}"


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
    token: str | None = None,
    private: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """Upload, skipping anything already present so a partial sync resumes.

    ``token=None`` lets huggingface_hub resolve credentials the standard way:
    an explicit token, then ``HF_TOKEN``, then the file written by
    ``hf auth login``. Do not re-implement that lookup.

    Everything lands in ONE commit. A results-tier sync is several hundred
    files, and one commit per file makes the repo history unusable and invites
    rate limiting.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    # A campaign token should be fine-grained and scoped to this one repo, which
    # means it may legitimately lack permission to CREATE repos. Creation is
    # therefore best-effort: if the repo is already there and writable, a
    # refused create is not an error.
    try:
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    except Exception:
        try:
            api.repo_info(repo, repo_type="dataset")
        except Exception as error:
            raise RuntimeError(
                f"cannot create or reach {repo!r}: {error}. Create the dataset "
                "repo once by hand, then scope the token to it with write access."
            ) from error
    # Compare against the remote tree by SIZE, not mere presence. Skipping any
    # path that already exists is right for resuming an interrupted upload and
    # wrong for everything a live campaign rewrites: summary.json, the manifest,
    # and a dataset's metrics/results/log/status once it resumes and finishes.
    # Presence-only skipping silently froze a dataset's status at "interrupted"
    # forever and made a corrected manifest un-uploadable.
    remote_size: dict[str, int] = {}
    try:
        for entry in api.list_repo_tree(
            repo, repo_type="dataset", recursive=True, expand=True
        ):
            size = getattr(entry, "size", None)
            if size is not None:
                remote_size[entry.path] = int(size)
    except Exception:
        remote_size = {}
    existing = set(remote_size) or set(api.list_repo_files(repo, repo_type="dataset"))

    operations = []
    skipped = 0
    for path, repo_path in items:
        if repo_path in existing:
            known = remote_size.get(repo_path)
            try:
                unchanged = known is not None and known == path.stat().st_size
            except OSError:
                unchanged = True
            if unchanged:
                skipped += 1
                continue
        operations.append(
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(path))
        )
        if progress:
            progress(f"+ {repo_path}")
    if operations:
        api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Add {len(operations)} RF100-VL campaign artifacts",
        )
    return {"uploaded": len(operations), "skipped": skipped}
