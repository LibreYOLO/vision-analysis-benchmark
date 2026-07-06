"""
Reproducibility provenance for Vision Analysis benchmarks.

Collects the facts a third party needs to reproduce a published number:
the harness commit, the exact command, a verifiable fingerprint of the
evaluated image set, the weights hash, and (for ONNX/TensorRT) the export
manifest. These are emitted under the ``repro`` block of every submission.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

from . import __version__

# Default label for the canonical accuracy set. This is an operator-asserted
# label; the verifiable identity is ``dataset.image_id_sha256`` below.
DEFAULT_DATASET_ID = "LibreYOLO/coco-val2017-mini500"
DEFAULT_DATASET_REVISION = "main"


def _repo_root() -> Path | None:
    """Return the harness git repo root, or None if not a checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def harness_git_info() -> dict[str, Any]:
    """Return the harness commit and whether its working tree is dirty.

    ``dirty`` is True when the checkout has uncommitted changes, so a number
    produced from an unclean tree cannot be silently passed off as pinned.
    """
    root = _repo_root()
    if root is None:
        return {"commit": "unknown", "dirty": None}

    commit = _git(root, "rev-parse", "HEAD") or "unknown"
    status = _git(root, "status", "--porcelain")
    dirty = None if status is None else bool(status.strip())
    return {"commit": commit, "dirty": dirty}


def file_sha256(path: str | Path | None) -> str | None:
    """Return the SHA-256 of a file, or None if it is missing/unreadable."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def image_id_sha256(image_ids: Iterable[int]) -> str:
    """Return a stable fingerprint of the evaluated image-id set.

    Two runs over the same images produce the same hash regardless of order,
    so this verifiably identifies the eval subset (mini500 vs full val2017 vs
    any other slice) without trusting an operator-supplied label.
    """
    canonical = "\n".join(str(i) for i in sorted(image_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reconstruct_command(argv: list[str]) -> str:
    """Return a copy-pasteable command line from the parsed argv tail."""
    return "va-bench " + shlex.join(argv)


def read_export_manifest(weights_path: str | Path | None) -> dict[str, Any] | None:
    """Read an export manifest sidecar for an ONNX/TensorRT artifact.

    Looks for ``<weights>.json`` next to the artifact (e.g. ``model.onnx.json``
    or ``model.engine.json``). The sidecar is operator-authored and records how
    the artifact was produced (export command, opset, TensorRT version, builder
    flags). Returned verbatim so it travels with the submission.
    """
    if weights_path is None:
        return None
    sidecar = Path(str(weights_path) + ".json")
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def onnx_opset(onnx_path: str | Path | None) -> int | None:
    """Return the default-domain opset of an ONNX model, or None."""
    if onnx_path is None:
        return None
    try:
        import onnx
    except ImportError:
        return None
    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception:
        return None
    for opset in model.opset_import:
        # The empty domain is the default (ai.onnx) operator set.
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return None


def build_dataset_repro(
    image_ids: Iterable[int],
    dataset_id: str | None,
    dataset_revision: str | None,
) -> dict[str, Any]:
    """Assemble the dataset provenance sub-block."""
    return {
        "image_id_sha256": image_id_sha256(image_ids),
        "hf_dataset": dataset_id or DEFAULT_DATASET_ID,
        "hf_revision": dataset_revision or DEFAULT_DATASET_REVISION,
    }


def build_weights_repro(
    weight_file: str,
    resolved_path: str | Path | None,
    source: str,
    export_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the weights provenance sub-block.

    ``resolved_path`` is the concrete file that was executed. For ONNX/TensorRT
    that is the export artifact; for PyTorch it is the ``.pt`` if the location
    could be resolved, else None (the run stays pinned by the model name plus
    ``benchmark.libreyolo_commit``).
    """
    block: dict[str, Any] = {
        "file": weight_file,
        "sha256": file_sha256(resolved_path),
        "source": source,
    }
    manifest = read_export_manifest(export_artifact_path)
    if manifest is not None:
        block["export_manifest"] = manifest
    opset = onnx_opset(export_artifact_path) if export_artifact_path else None
    if opset is not None:
        block["onnx_opset"] = opset
    return block


def run_repro(dataset: dict[str, Any], weights: dict[str, Any]) -> dict[str, Any]:
    """Assemble the per-run (verifiable) part of the repro block.

    Invocation-level fields (harness commit, argv, command) are merged in by
    the CLI, which is where the real command line is known.
    """
    return {
        "harness_version": __version__,
        "dataset": dataset,
        "weights": weights,
    }
