"""Preflight for RF100-VL campaigns: is this box ready to burn money?

One command that validates everything a campaign depends on BEFORE the first
GPU-hour is spent: the LibreYOLO build can enforce the protocol, the dataset
snapshot matches its version lock, every split file exists, the recipe
resolves, torch actually has kernels for the installed GPUs, and there is
disk to write to. Modeled on the compliance-checker habit of hardware
benchmarks: renting a box and discovering a broken stack an hour in is the
most expensive way to run a check that takes seconds.

Every check is independent and reports PASS/FAIL with a one-line detail;
the process exits non-zero if any check fails.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .models import get_spec
from .provenance import file_sha256
from .rf100vl_data import ANNOTATION_FILENAME, load_version_lock

_SPLITS = ("train", "valid", "test")
_MIN_FREE_GB = 20.0
# Where the dataset and the published artifacts live. A rented host that
# cannot reach this is useless for a campaign even when its GPUs are perfect,
# and the failure looks like a hung download rather than an error.
_HUB_URL = "https://huggingface.co/api/datasets/LibreYOLO/rf100-vl"
_HUB_TIMEOUT_SECONDS = 20.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _check_libreyolo() -> Check:
    from .rf100vl_train import require_libreyolo_protocol_capabilities

    try:
        capabilities = require_libreyolo_protocol_capabilities()
    except Exception as exc:
        return Check("libreyolo", False, str(exc))
    extras = []
    if capabilities.get("cuda_graph"):
        extras.append("cuda_graph")
    if capabilities.get("cache"):
        extras.append("cache")
    extra = (", " + "+".join(extras)) if extras else ", (no cuda_graph/cache)"
    return Check(
        "libreyolo",
        True,
        f"version {capabilities.get('version')}, eval_max_det "
        f"{capabilities.get('eval_max_det')}, amp_dtype "
        f"{capabilities.get('amp_dtype')}{extra}",
    )


def _check_data(data_dir: Path) -> tuple[Check, list[str]]:
    try:
        version_lock = load_version_lock(data_dir, required=True)
    except Exception as exc:
        return Check("data", False, str(exc)), []
    locked = sorted(version_lock["datasets"])
    on_disk = {
        child.name
        for child in data_dir.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    }
    missing = sorted(set(locked) - on_disk)
    extra = sorted(on_disk - set(locked))
    if missing:
        preview = ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else "")
        return (
            Check("data", False, f"{len(missing)} locked datasets absent: {preview}"),
            locked,
        )
    detail = f"{len(locked)} locked datasets present ({version_lock.get('subset')})"
    if extra:
        detail += f"; {len(extra)} unlocked extra dirs ignored"
    return Check("data", True, detail), locked


def _check_splits(data_dir: Path, names: list[str]) -> Check:
    if not names:
        return Check("splits", False, "no datasets to check")
    missing: list[str] = []
    for name in names:
        for split in _SPLITS:
            if not (data_dir / name / split / ANNOTATION_FILENAME).is_file():
                missing.append(f"{name}/{split}")
    if missing:
        preview = ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else "")
        return Check("splits", False, f"{len(missing)} split files absent: {preview}")
    return Check("splits", True, f"train/valid/test annotations present x{len(names)}")


def _check_recipe(model_key: str, recipe: str | None) -> Check:
    from .rf100vl_train import load_recipe, recipe_path_for_family

    try:
        spec = get_spec(model_key)
        recipe_path = Path(recipe or recipe_path_for_family(spec.family)).resolve()
        load_recipe(recipe_path, family=spec.family)
    except Exception as exc:
        return Check("recipe", False, str(exc))
    return Check(
        "recipe", True, f"{recipe_path.name} sha {file_sha256(recipe_path)[:12]}"
    )


def _parse_arch(arch: str) -> tuple[int, int] | None:
    """'sm_86' -> (8, 6). Ignores PTX ('compute_86') and suffixed ('sm_90a')."""
    if not arch.startswith("sm_"):
        return None
    digits = arch[3:]
    if not digits.isdigit():  # sm_90a and friends are arch-conditional, not portable
        return None
    return int(digits[:-1]), int(digits[-1])


def _runs_on(device: tuple[int, int], arch_list: Iterable[str]) -> bool:
    """Does any shipped cubin actually run on this device?

    CUDA guarantees binary compatibility FORWARD across minor versions only:
    a cubin built for compute capability X.y runs on X.z when z >= y, and never
    across a major version. So an sm_86 build runs a 4090 (sm_89) fine, and
    requiring exact membership rejects perfectly good hardware. Verified on a
    rented 8x4090 on 2026-07-31: a torch build shipping sm_86 but not sm_89
    trained and evaluated a full dataset without a single launch failure.
    """
    major, minor = device
    for arch in arch_list:
        parsed = _parse_arch(arch)
        if parsed and parsed[0] == major and parsed[1] <= minor:
            return True
    return False


def _check_gpu() -> Check:
    try:
        import torch
    except Exception as exc:
        return Check("gpu", False, f"torch import failed: {exc}")
    if not torch.cuda.is_available():
        return Check("gpu", False, "torch.cuda.is_available() is False")
    arch_list = set(torch.cuda.get_arch_list())
    names = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        sm = f"sm_{properties.major}{properties.minor}"
        if not _runs_on((properties.major, properties.minor), arch_list):
            return Check(
                "gpu",
                False,
                f"device {index} ({properties.name}) is {sm} but this torch build "
                f"only ships {sorted(arch_list)}; kernels will not launch",
            )
        names.append(
            f"{properties.name} ({properties.total_memory / 2**30:.0f}GB)"
        )
    return Check("gpu", True, f"{len(names)}x " + "; ".join(sorted(set(names))))


def _check_hub() -> Check:
    """Can this box reach the artifact hub at all?

    Rented hosts have been observed resolving huggingface.co to IPv6 only with
    no IPv6 egress, and others block it outright while PyPI and GitHub work
    fine. Either way ``snapshot_download`` simply hangs with no output, which
    is expensive to diagnose after a box is already provisioned.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(_HUB_URL, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_HUB_TIMEOUT_SECONDS) as response:
            code = response.status
    except urllib.error.HTTPError as error:
        # The host answered; auth or rate limits are not a reachability problem.
        return Check("hub", True, f"huggingface.co reachable (HTTP {error.code})")
    except Exception as error:
        return Check(
            "hub",
            False,
            f"cannot reach huggingface.co ({type(error).__name__}: {error}); "
            "staging and artifact upload will hang on this host",
        )
    return Check("hub", True, f"huggingface.co reachable (HTTP {code})")


def _check_disk(weights_root: Path) -> Check:
    probe = weights_root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / 2**30
    except OSError as exc:
        return Check("disk", False, str(exc))
    if free_gb < _MIN_FREE_GB:
        return Check(
            "disk", False, f"{free_gb:.0f}GB free at {probe}; need >= {_MIN_FREE_GB:.0f}GB"
        )
    return Check("disk", True, f"{free_gb:.0f}GB free at {probe}")


def _check_writable(weights_root: Path) -> Check:
    try:
        weights_root.mkdir(parents=True, exist_ok=True)
        probe = weights_root / f".preflight-{uuid.uuid4().hex[:8]}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("write", False, f"weights root not writable: {exc}")
    return Check("write", True, str(weights_root))


def run_preflight(
    model_key: str,
    data_dir: Path,
    weights_root: Path,
    recipe: str | None = None,
    skip: tuple[str, ...] = (),
) -> list[Check]:
    """Run every campaign precondition check; ``skip`` names checks to omit."""
    data_dir = Path(data_dir)
    weights_root = Path(weights_root)
    checks: list[Check] = []
    if "libreyolo" not in skip:
        checks.append(_check_libreyolo())
    locked_names: list[str] = []
    if "data" not in skip:
        data_check, locked_names = _check_data(data_dir)
        checks.append(data_check)
        if "splits" not in skip:
            if data_check.ok:
                checks.append(_check_splits(data_dir, locked_names))
            else:
                checks.append(Check("splits", False, "skipped: data check failed"))
    if "recipe" not in skip:
        checks.append(_check_recipe(model_key, recipe))
    if "gpu" not in skip:
        checks.append(_check_gpu())
    if "hub" not in skip:
        checks.append(_check_hub())
    if "disk" not in skip:
        checks.append(_check_disk(weights_root))
    if "write" not in skip:
        checks.append(_check_writable(weights_root))
    return checks


def has_failure(checks: list[Check]) -> bool:
    return any(not check.ok for check in checks)


def render(checks: list[Check]) -> str:
    width = max(len(check.name) for check in checks) if checks else 0
    lines = [
        f"{'PASS' if check.ok else 'FAIL'}  {check.name.ljust(width)}  {check.detail}"
        for check in checks
    ]
    verdict = (
        "preflight FAILED: fix the above before spending GPU-hours"
        if has_failure(checks)
        else "preflight passed: the box is ready"
    )
    return "\n".join(lines + [verdict])
