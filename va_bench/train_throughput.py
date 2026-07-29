"""Training-throughput benchmark path for Vision Analysis.

Sibling to the inference path (benchmark.py). Measures how fast a given
(model, GPU + rig, software stack) trains: it drives LibreYOLO's *real*
``model.train()`` over a fixed COCO subset (coco1000) for a few epochs,
captures per-epoch wall-time via the supported ``callbacks=`` hook, discards
the first epoch as warmup, and reports steady-state training throughput
(img/s). From img/s it projects time- and dollars-per-epoch for full COCO.

Why epoch-granularity and the real train loop: the data-dependent cost
(label assignment / matcher / loss) is only authentic with the real augmented
dataloader and the family's real loss — so we measure at the granularity the
trainer already exposes (epoch_seconds) rather than reimplementing a step.

The unit of measurement is a *configuration*, not a bare GPU:
(model, GPU + rig label, provider, precision, effective batch). GPU
utilisation is recorded so a reader can tell whether a number is
compute-bound (clean silicon fact) or host/dataloader-bound (a property of
that rented offering).
"""

from __future__ import annotations

import hashlib
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import __version__
from .hardware import collect_all
from .models import get_spec, load_model
from .output import detect_hardware_id
from .provenance import build_weights_repro, run_repro

# COCO train2017 size — the projection target for "$/epoch of full COCO".
COCO_FULL_TRAIN_IMAGES = 118_287


class _EpochTimeCollector:
    """Object-style TrainCallback that records per-epoch wall-time."""

    def __init__(self) -> None:
        self.epoch_seconds: list[float] = []

    def on_train_start(self, event: Any) -> None:  # noqa: D401
        return None

    def on_train_epoch_end(self, event: Any) -> None:
        self.epoch_seconds.append(float(event.epoch_seconds))

    def on_train_end(self, event: Any) -> None:
        return None

    def on_train_exception(self, event: Any) -> None:
        return None


def _gpu_util_sampler(stop: threading.Event, out: list[int], interval: float = 0.25) -> None:
    """Poll nvidia-smi for GPU utilisation until stopped."""
    while not stop.is_set():
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            for tok in res.stdout.split():
                if tok.strip().isdigit():
                    out.append(int(tok.strip()))
                    break
        except Exception:
            pass
        stop.wait(interval)


def _train_image_files(data: str) -> list[str]:
    """Resolve the dataset and return the exact training image list."""
    from libreyolo.data import load_data_config

    cfg = load_data_config(data, autodownload=True, allow_scripts=False)
    return [str(path) for path in (cfg.get("train_img_files") or [])]


def _count_train_images(data: str) -> int:
    """Backward-compatible count helper used by callers and tests."""
    return len(_train_image_files(data))


def _dataset_id(data: str) -> str:
    """Derive a truthful local dataset label from a name or data YAML path."""
    path = Path(data)
    if path.exists() and path.name.lower() in {"data.yaml", "data.yml"}:
        return path.parent.name
    return path.stem or str(data)


def _image_name_sha256(files: list[str]) -> str:
    """Fingerprint the ordered-independent set of training image names."""
    canonical = "\n".join(sorted(Path(path).name for path in files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def benchmark_train_throughput(
    model_key: str,
    *,
    data: str = "coco1000",
    device: str = "auto",
    batch: int = 16,
    imgsz: int | None = None,
    warmup_epochs: int = 1,
    measure_epochs: int = 3,
    workers: int = 8,
    amp: bool = False,
    amp_dtype: str = "float16",
    nbs: int | None = None,
    dollars_per_hour: float | None = None,
    rig_label: str | None = None,
    provider: str = "local",
    project_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Measure steady-state training throughput for one model on this machine.

    Args:
        model_key: Registry key (e.g. "yolov9t").
        data: Dataset yaml/name (default coco1000 — the portable fixture).
        device: "auto"/"0"/"cpu".
        batch: Per-step micro-batch.
        imgsz: Input size; defaults to the model's native size.
        warmup_epochs: Leading epochs discarded (kernel autotune, loader warmup,
            thermal ramp).
        measure_epochs: Steady-state epochs averaged for the reported img/s.
        workers: Dataloader workers (part of the measured configuration).
        amp: Use CUDA AMP instead of fp32.
        amp_dtype: CUDA AMP dtype, ``float16`` or ``bfloat16``.
        nbs: Nominal/effective batch for gradient accumulation. None = no accum.
        dollars_per_hour: Rental price of this configuration; enables $/epoch.
        rig_label: Human label for the (GPU + host) box, e.g. "home-5070ti".
        provider: Where it ran ("local", "modal", "runpod", ...).
        project_dir: Where the trainer writes its (discarded) run artifacts.
    """
    spec = get_spec(model_key)
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError(f"amp_dtype must be 'float16' or 'bfloat16', got {amp_dtype!r}")
    imgsz = imgsz or spec.input_size
    effective_batch = nbs if nbs else batch
    accum = max(1, round(effective_batch / batch)) if nbs else 1

    train_image_files = _train_image_files(data)
    n_images = len(train_image_files)
    dataset_id = _dataset_id(data)
    if n_images < 1:
        raise ValueError("Training dataset contains no images.")
    # LibreYOLO drops the final partial batch only when at least one full batch
    # exists. Tiny smoke datasets therefore still produce one partial batch.
    if n_images < batch:
        steps_per_epoch = 1
        images_per_epoch = n_images
    else:
        steps_per_epoch = n_images // batch
        images_per_epoch = steps_per_epoch * batch

    model, _ = load_model(model_key, device=device)

    use_cuda = torch.cuda.is_available() and str(device).lower() != "cpu"
    dev_type = "cuda" if use_cuda else "cpu"
    train_device = "0" if use_cuda else "cpu"
    if amp and not use_cuda:
        raise ValueError("amp=True requires a CUDA device for a truthful benchmark")
    if amp and amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("amp_dtype='bfloat16' is not supported by this CUDA device")
    precision = amp_dtype if amp else "fp32"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    if project_dir is None:
        project_dir = Path.home() / ".cache" / "va_train_bench"
    Path(project_dir).mkdir(parents=True, exist_ok=True)

    collector = _EpochTimeCollector()
    stop = threading.Event()
    util_samples: list[int] = []
    sampler = threading.Thread(target=_gpu_util_sampler, args=(stop, util_samples), daemon=True)

    total_epochs = warmup_epochs + measure_epochs
    if verbose:
        print(
            f"[train-bench] {model_key}: {total_epochs} epochs "
            f"({warmup_epochs} warmup + {measure_epochs} measured), "
            f"batch={batch} imgsz={imgsz} precision={precision} "
            f"steps/epoch={steps_per_epoch} on {n_images} images"
        )

    train_kwargs: dict[str, Any] = dict(
        data=data,
        epochs=total_epochs,
        batch=batch,
        imgsz=imgsz,
        device=train_device,
        workers=workers,
        amp=amp,
        amp_dtype=amp_dtype,
        seed=0,
        no_aug_epochs=0,  # keep aug regime constant across measured epochs
        eval_interval=10**9,  # never validate during the benchmark
        save_period=10**9,  # no periodic checkpoints
        patience=0,  # no early-stop validation pass
        save_plots=False,
        project=str(project_dir),
        name=f"trainbench_{model_key}",
        exist_ok=True,
        callbacks=collector,
    )
    if nbs:
        train_kwargs["nbs"] = nbs

    sampler.start()
    t0 = time.time()
    try:
        model.train(**train_kwargs)
    finally:
        stop.set()
        sampler.join(timeout=2.0)
        wall = time.time() - t0

    epoch_seconds = collector.epoch_seconds
    if len(epoch_seconds) < total_epochs:
        # Trainer may have stopped early; measure whatever steady epochs we got.
        warmup_epochs = min(warmup_epochs, max(0, len(epoch_seconds) - 1))
    measured = epoch_seconds[warmup_epochs:]
    if not measured:
        raise RuntimeError(f"No measured epochs captured (got {len(epoch_seconds)} epoch times).")

    per_epoch_img_s = [images_per_epoch / s for s in measured]
    img_s_median = statistics.median(per_epoch_img_s)
    img_s_mean = statistics.fmean(per_epoch_img_s)

    sec_per_full_epoch = COCO_FULL_TRAIN_IMAGES / img_s_median
    dollars_per_epoch = sec_per_full_epoch / 3600.0 * dollars_per_hour if dollars_per_hour else None

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else None
    gpu_util_mean = round(statistics.fmean(util_samples), 1) if util_samples else None

    meta = collect_all()
    hardware = meta["hardware"]
    software = meta["software"]
    hardware_id = detect_hardware_id(hardware)
    rig = rig_label or f"{hardware_id}@{provider}"

    now = datetime.now(timezone.utc)
    created_at = now.isoformat().replace("+00:00", "Z")
    projection_valid = dataset_id.lower() == "coco1000"
    repro = run_repro(
        dataset={
            "id": dataset_id,
            "source": str(data),
            "num_images": n_images,
            "image_id_sha256": _image_name_sha256(train_image_files),
            "identity_scheme": "sha256(sorted training image filenames)",
        },
        weights=build_weights_repro(
            weight_file=spec.weight_file,
            resolved_path=getattr(model, "model_path", None),
            source="libreyolo-managed",
        ),
    )

    result = {
        "schema_version": "va.train.v1",
        "submission_id": (
            f"{model_key}-train-{provider}-{hardware_id}-{now.strftime('%Y%m%dT%H%M%SZ')}"
        ),
        "created_at": created_at,
        "benchmark": {
            "harness": "vision-analysis-benchmark",
            "harness_version": __version__,
            "mode": "train-throughput",
            "libreyolo_version": software.get("libreyolo", "unknown"),
            "libreyolo_commit": software.get("libreyolo_commit", "unknown"),
            "libreyolo_dirty": software.get("libreyolo_dirty"),
        },
        "model": {
            "id": spec.key,
            "family": spec.family,
            "variant": spec.variant,
            "input_size": imgsz,
        },
        "dataset": {
            "id": dataset_id,
            "source": str(data),
            "benchmark_images": n_images,
            "projection_target": "coco2017 train",
            "projection_images": COCO_FULL_TRAIN_IMAGES,
            "projection_valid": projection_valid,
        },
        "config": {
            "micro_batch": batch,
            "effective_batch": effective_batch,
            "accum_steps": accum,
            "input_size": imgsz,
            "amp": amp,
            "amp_dtype": amp_dtype if amp else None,
            "precision": precision,
            "seed": 0,
            "workers": workers,
            "warmup_epochs": warmup_epochs,
            "measure_epochs": len(measured),
        },
        "hardware": {**hardware, "id": hardware_id, "rig_label": rig},
        "software": software,
        "measurement": {
            "steps_per_epoch": steps_per_epoch,
            "images_per_epoch": images_per_epoch,
            "epoch_seconds_all": [round(s, 3) for s in epoch_seconds],
            "epoch_seconds_measured": [round(s, 3) for s in measured],
            "img_per_s_median": round(img_s_median, 2),
            "img_per_s_mean": round(img_s_mean, 2),
            "img_per_s_min": round(min(per_epoch_img_s), 2),
            "img_per_s_max": round(max(per_epoch_img_s), 2),
            "gpu_util_mean_pct": gpu_util_mean,
            "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else None,
            "wall_seconds": round(wall, 1),
        },
        "derived": {
            "coco_full_train_images": COCO_FULL_TRAIN_IMAGES,
            "sec_per_full_epoch": round(sec_per_full_epoch, 1),
            "hours_per_full_epoch": round(sec_per_full_epoch / 3600.0, 4),
            "dollars_per_hour": dollars_per_hour,
            "dollars_per_epoch": (round(dollars_per_epoch, 4) if dollars_per_epoch else None),
            "projection_valid": projection_valid,
            "projection_note": (
                None
                if projection_valid
                else "Smoke dataset is not coco1000; do not publish the full-COCO projection."
            ),
        },
        "runtime": {
            "format": "pytorch-train",
            "precision": precision,
            "provider": provider,
            "device": dev_type,
        },
        "repro": repro,
    }

    if verbose:
        m = result["measurement"]
        d = result["derived"]
        print(
            f"[train-bench] {model_key}: {m['img_per_s_median']} img/s "
            f"(min {m['img_per_s_min']} / max {m['img_per_s_max']}), "
            f"util {m['gpu_util_mean_pct']}%, peak VRAM {m['peak_vram_mb']} MB"
        )
        print(
            f"[train-bench] projected full-COCO epoch: "
            f"{d['hours_per_full_epoch']} h"
            + (
                f" = ${d['dollars_per_epoch']}/epoch @ ${d['dollars_per_hour']}/h"
                if d["dollars_per_epoch"]
                else ""
            )
        )

    return result


def save_train_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Persist a training-throughput result JSON."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = result.get("created_at", "").replace("-", "").replace(":", "")
    fname = (
        f"{result['model']['id']}__train__{result['runtime']['provider']}"
        f"__{result['hardware']['id']}__{ts}.json"
    )
    path = output_dir / fname
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path
