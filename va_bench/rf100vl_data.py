"""RF100-VL dataset identity, version locking, and domain-manifest helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Iterable

ANNOTATION_FILENAME = "_annotations.coco.json"
VERSION_LOCK_FILENAME = "versions.json"
VERSION_LOCK_SCHEMA = "rf100vl.versions.v1"
RF100VL_REPOSITORY = "https://github.com/roboflow/rf100-vl"
RF100VL_SOURCE_COMMIT = "1c1ecd84cf2865270702121311e1d9e5f2a861aa"

_FETCHERS = {
    "rf100vl": "get_rf100vl_projects",
    "rf20vl": "get_rf20vl_full_projects",
    "rf100vl-fsod": "get_rf100vl_fsod_projects",
    "rf20vl-fsod": "get_rf20vl_fsod_projects",
}


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable canonical encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def long_path(path: Path) -> str:
    """Windows-safe path string.

    The per-dataset result cache nests a 64-hex-char filename six levels under
    the data dir, and the atomic temp name adds ``.<pid>.tmp`` on top, which
    clears the 260-char MAX_PATH on ordinary layouts (a scratch dir was enough).
    mkdir succeeds and only the write fails, so it surfaces as a confusing
    "No such file or directory" for a directory that plainly exists. The
    ``\\\\?\\`` prefix opts the call out of MAX_PATH; it needs an absolute path
    with native separators and no relative components.
    """
    if os.name != "nt":
        return str(path)
    resolved = os.path.abspath(str(path))
    if resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):  # UNC share
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Atomically replace ``path`` with pretty, deterministic JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(long_path(temporary), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(long_path(temporary), long_path(path))
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    path = Path(path)
    with open(long_path(path), encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("rf100vl")
    except importlib.metadata.PackageNotFoundError:
        return None


def _named_datasets(subset: str, api_key: str | None = None) -> dict[str, Any]:
    """Fetch RF100-VL wrappers and key them by name, never package index."""
    if subset not in _FETCHERS:
        raise ValueError(f"Unknown subset {subset!r}. Options: {sorted(_FETCHERS)}")
    try:
        from rf100vl import roboflow100vl
    except ImportError as exc:
        raise RuntimeError(
            "The 'rf100vl' package is required to download RF100-VL. "
            "Install with: pip install 'vision-analysis-benchmark[rf100vl]'"
        ) from exc

    fetcher = getattr(roboflow100vl, _FETCHERS[subset], None)
    if fetcher is None:
        raise RuntimeError(
            f"Installed rf100vl package has no {_FETCHERS[subset]}(); "
            "upgrade with: pip install -U rf100vl"
        )

    wrappers = list(fetcher(api_key))
    by_name: dict[str, Any] = {}
    for wrapper in wrappers:
        name = str(wrapper.name)
        if name in by_name:
            raise RuntimeError(f"RF100-VL workspace returned duplicate dataset name {name!r}")
        by_name[name] = wrapper
    return dict(sorted(by_name.items()))


def _lock_identity(lock: dict[str, Any]) -> dict[str, Any]:
    """Return immutable version-selection fields, excluding download progress."""
    return {
        "schema_version": lock.get("schema_version"),
        "subset": lock.get("subset"),
        "source": lock.get("source"),
        "datasets": lock.get("datasets"),
    }


def version_lock_sha256(lock: dict[str, Any]) -> str:
    """Hash only the immutable dataset/version selection."""
    return canonical_json_sha256(_lock_identity(lock))


def validate_version_lock(
    lock: dict[str, Any],
    *,
    subset: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize a replayable RF100-VL version lock."""
    if lock.get("schema_version") != VERSION_LOCK_SCHEMA:
        raise ValueError(
            f"Unsupported RF100-VL version-lock schema: {lock.get('schema_version')!r}"
        )
    if subset is not None and lock.get("subset") != subset:
        raise ValueError(f"Version lock is for {lock.get('subset')!r}, requested {subset!r}")
    records = lock.get("datasets")
    if not isinstance(records, dict) or not records:
        raise ValueError("RF100-VL version lock has no dataset records")
    for name, record in records.items():
        if not isinstance(name, str) or not name:
            raise ValueError("RF100-VL version lock contains an invalid dataset name")
        if not isinstance(record, dict):
            raise ValueError(f"Version-lock record for {name!r} is not an object")
        version_id = record.get("version_id")
        if isinstance(version_id, bool) or not isinstance(version_id, int) or version_id < 1:
            raise ValueError(
                f"Version-lock record for {name!r} has invalid version_id {version_id!r}"
            )
    downloaded = lock.get("downloaded", [])
    if not isinstance(downloaded, list) or not all(isinstance(name, str) for name in downloaded):
        raise ValueError("RF100-VL version lock 'downloaded' must be a list of names")
    unknown = set(downloaded) - set(records)
    if unknown:
        raise ValueError(
            "RF100-VL version lock marks unknown datasets downloaded: " + ", ".join(sorted(unknown))
        )
    return lock


def load_version_lock(
    data_dir_or_path: str | Path,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    """Load ``versions.json`` from a data directory or an explicit path."""
    path = Path(data_dir_or_path)
    if path.is_dir() or path.suffix.lower() != ".json":
        path = path / VERSION_LOCK_FILENAME
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"RF100-VL version lock not found: {path}. "
                "Download with the harness before a protocol run."
            )
        return None
    return validate_version_lock(load_json(path))


def roboflow_version_number(version: Any) -> int:
    """Extract the numeric version from the SDK's number or project slug."""
    raw = getattr(version, "version", None)
    if raw is None:
        raw = getattr(version, "id", None)
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"Roboflow version has no numeric identity: {raw!r}")
    if isinstance(raw, int):
        number = raw
    else:
        tail = str(raw).rstrip("/").rsplit("/", 1)[-1]
        if not tail.isdigit():
            raise ValueError(f"Roboflow version has invalid identity: {raw!r}")
        number = int(tail)
    if number < 1:
        raise ValueError(f"Roboflow version must be positive, got {number}")
    return number


def _new_version_lock(subset: str, wrappers: dict[str, Any]) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for name, wrapper in wrappers.items():
        versions = list(wrapper.rf_project.versions())
        if not versions:
            raise RuntimeError(f"RF100-VL project {name!r} has no downloadable versions")
        selected = max(versions, key=roboflow_version_number)
        records[name] = {
            "project_name": str(getattr(wrapper.rf_project, "name", name)),
            "version_id": roboflow_version_number(selected),
        }
    return {
        "schema_version": VERSION_LOCK_SCHEMA,
        "subset": subset,
        "source": {
            "repository": RF100VL_REPOSITORY,
            "commit": RF100VL_SOURCE_COMMIT,
            "package_version": _package_version(),
        },
        "datasets": records,
        "downloaded": [],
        "selection_complete": True,
    }


def _dataset_materialized(dataset_dir: Path) -> bool:
    return all(
        (dataset_dir / split / ANNOTATION_FILENAME).exists() for split in ("train", "valid", "test")
    )


def download_version_locked_datasets(
    data_dir: str | Path,
    *,
    subset: str = "rf100vl",
    api_key: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Download exact RF100-VL versions and atomically persist/replay their IDs.

    A new run resolves every project's latest version first and writes the
    complete lock before downloading any dataset. Once the lock exists, later
    runs replay those IDs instead of resolving latest again.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / VERSION_LOCK_FILENAME
    wrappers = _named_datasets(subset, api_key=api_key)

    if lock_path.exists():
        lock = validate_version_lock(load_json(lock_path), subset=subset)
        locked_names = set(lock["datasets"])
        workspace_names = set(wrappers)
        if locked_names != workspace_names:
            missing = sorted(locked_names - workspace_names)
            extra = sorted(workspace_names - locked_names)
            raise RuntimeError(
                "RF100-VL workspace no longer matches the version lock "
                f"(missing={missing}, extra={extra})"
            )
    else:
        lock = _new_version_lock(subset, wrappers)
        atomic_write_json(lock_path, lock)

    downloaded = set(lock.get("downloaded", []))
    for position, name in enumerate(sorted(lock["datasets"]), start=1):
        target = data_dir / name
        materialized = _dataset_materialized(target)
        if not overwrite and materialized:
            if name not in downloaded:
                # The previous process completed the SDK download but died
                # before advancing progress. Reconcile without downloading.
                downloaded.add(name)
                lock["downloaded"] = sorted(downloaded)
                atomic_write_json(lock_path, lock)
            if verbose:
                print(f"[{position}/{len(wrappers)}] {name}: locked dataset already present")
            continue

        wrapper = wrappers[name]
        version_id = int(lock["datasets"][name]["version_id"])
        versions = {
            roboflow_version_number(version): version
            for version in wrapper.rf_project.versions()
        }
        version = versions.get(version_id)
        if version is None:
            # Public Roboflow API supports exact project.version(id). Keeping
            # this fallback explicit avoids any use of the package's latest-
            # version helper.
            version = wrapper.rf_project.version(version_id)

        if verbose:
            print(f"[{position}/{len(wrappers)}] {name}: downloading locked version {version_id}")
        version.download(
            location=str(target),
            model_format="coco",
            overwrite=overwrite or target.exists(),
        )
        wrapper.clean_coco_dataset(str(target))
        if not _dataset_materialized(target):
            raise RuntimeError(
                f"RF100-VL dataset {name!r} version {version_id} is missing a "
                "train, valid, or test COCO annotation file"
            )
        downloaded.add(name)
        lock["downloaded"] = sorted(downloaded)
        atomic_write_json(lock_path, lock)

    return data_dir


def dataset_version(
    lock: dict[str, Any] | None,
    dataset_name: str,
) -> int | None:
    """Return one dataset's locked version id, if available."""
    if lock is None:
        return None
    record = lock.get("datasets", {}).get(dataset_name)
    return int(record["version_id"]) if isinstance(record, dict) else None


def domain_manifest_path() -> Path:
    return Path(__file__).with_name("data") / "rf100vl_domains.json"


def load_domain_manifest() -> dict[str, Any]:
    """Load and validate the vendored canonical RF100-VL domain manifest."""
    manifest = load_json(domain_manifest_path())
    datasets = manifest.get("datasets")
    if manifest.get("schema_version") != "rf100vl.domains.v1":
        raise ValueError("Unsupported RF100-VL domain-manifest schema")
    if not isinstance(datasets, dict) or len(datasets) != 100:
        raise ValueError("RF100-VL domain manifest must contain exactly 100 datasets")
    allowed = {
        "Aerial",
        "Document",
        "Flora and Fauna",
        "Industrial",
        "Medical",
        "Other",
        "Sports",
    }
    unknown_domains = set(datasets.values()) - allowed
    if unknown_domains:
        raise ValueError(
            "RF100-VL domain manifest has non-canonical domains: "
            + ", ".join(sorted(unknown_domains))
        )
    return manifest


def validate_dataset_names(names: Iterable[str]) -> list[str]:
    """Return dataset names absent from the vendored 100-dataset manifest."""
    known = set(load_domain_manifest()["datasets"])
    return sorted(set(names) - known)
