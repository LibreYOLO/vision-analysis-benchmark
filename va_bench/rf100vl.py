"""
RF100-VL benchmark support for Vision Analysis.

Roboflow100-VL (https://rf100-vl.org, NeurIPS 2025 D&B) is 100 object
detection datasets across 7 domains, distributed in COCO JSON format under
Apache 2.0. Unlike COCO val2017, the headline regime is *fine-tuned*: a model
is trained per dataset, evaluated on that dataset's test split, and the
published score is the mean AP50 / AP50:95 across all datasets.

This module implements the evaluation half of that protocol:

* ``download_datasets``: fetch RF100-VL (or the RF20-VL subset) via the
  ``rf100vl`` pip package into a local directory, one sub-folder per dataset.
* ``benchmark_model_rf100vl``: run one registered model over every dataset's
  chosen split, score each dataset independently with pycocotools, and emit a
  single submission JSON whose accuracy block is the across-dataset mean plus
  a per-dataset breakdown under ``rf100vl``.

Fine-tuned checkpoints are supplied per dataset via ``--weights-root``:

    <weights_root>/<dataset_name>/<spec.weight_file>          (pytorch)
    <weights_root>/<dataset_name>/<spec weight stem>.onnx     (onnx)

Running COCO-pretrained weights across RF100-VL is meaningless for closed-
vocabulary detectors (the class spaces differ), so it is refused unless
``allow_pretrained=True``; that escape hatch exists for smoke tests and for
future open-vocabulary models whose class space is prompt-defined.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .coco_eval import evaluate_coco
from .hardware import collect_all as collect_hw
from .hardware import get_runtime_device_name
from .models import ModelSpec, get_spec
from .output import assemble_result
from .provenance import file_sha256, harness_git_info, image_id_sha256, run_repro
from .rf100vl_data import (
    ANNOTATION_FILENAME,
    atomic_write_json,
    canonical_json_sha256,
    dataset_version,
    domain_manifest_path,
    download_version_locked_datasets,
    load_domain_manifest,
    load_json,
    load_version_lock,
    long_path,
    validate_dataset_names,
    version_lock_sha256,
)
from .timing import compute_stats

DATASET_ID = "rf100_vl"
EXPECTED_DATASETS = 100
PROTOCOL_IOU = 0.65
PROTOCOL_MAX_DET = 500
PROTOCOL_VERSION = "rf100vl.libreyolo.v1"
PER_DATASET_RESULT_SCHEMA = "rf100vl.dataset-result.v1"
TRAIN_STATS_SCHEMA = "rf100vl.train-stats.v1"
PROTOCOL_EPOCHS = 100
PROTOCOL_SEED = 0

# Metrics averaged across datasets (keys of coco_eval.evaluate_coco output).
_METRIC_KEYS = (
    "mAP",
    "mAP50",
    "mAP75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_max_det",
    "AR_small",
    "AR_medium",
    "AR_large",
)


# =============================================================================
# Dataset acquisition / discovery
# =============================================================================


def download_datasets(
    data_dir: str | Path,
    subset: str = "rf100vl",
    verbose: bool = True,
) -> Path:
    """Download exact RF100-VL versions and persist/replay ``versions.json``."""
    return download_version_locked_datasets(
        data_dir,
        subset=subset,
        verbose=verbose,
    )


def discover_datasets(data_dir: str | Path, split: str = "test") -> list[Path]:
    """Return sorted dataset directories containing ``<split>/_annotations.coco.json``."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"RF100-VL data dir not found: {data_dir}. "
            "Download first with: va-bench rf100vl --download"
        )
    found = sorted(
        d for d in data_dir.iterdir() if d.is_dir() and (d / split / ANNOTATION_FILENAME).exists()
    )
    if not found:
        raise FileNotFoundError(f"No datasets with {split}/{ANNOTATION_FILENAME} under {data_dir}.")
    return found


def load_dataset_split(dataset_dir: Path, split: str, verbose: bool = False):
    """Load one RF100-VL dataset split as (coco_gt, img_ids, img_dir).

    Roboflow COCO exports keep images and ``_annotations.coco.json`` in the
    same split directory.
    """
    import contextlib
    import io

    from pycocotools.coco import COCO

    ann_file = dataset_dir / split / ANNOTATION_FILENAME
    img_dir = dataset_dir / split

    if verbose:
        coco_gt = COCO(str(ann_file))
    else:
        # pycocotools prints unconditionally; keep per-dataset output quiet.
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(str(ann_file))
    img_ids = sorted(coco_gt.getImgIds())
    return coco_gt, img_ids, img_dir


def category_mapping(coco_gt: Any) -> list[int]:
    """Model class index -> ground-truth category id, by ascending category id.

    LibreYOLO training on a Roboflow export enumerates classes in ascending
    category-id order, so index i corresponds to the i-th smallest category id.
    (Roboflow COCO exports usually reserve id 0 for a placeholder
    supercategory; when present it has no annotations and simply occupies
    index 0 on both sides, keeping the mapping aligned.)
    """
    return sorted(coco_gt.getCatIds())


# =============================================================================
# Per-dataset weight resolution
# =============================================================================


def resolve_finetuned_weights(
    spec: ModelSpec,
    weights_root: str | Path,
    dataset_name: str,
    fmt: str,
) -> Path | None:
    """Path to this model's fine-tuned checkpoint for one dataset, or None."""
    stem = Path(spec.weight_file).stem
    dataset_dir = Path(weights_root) / dataset_name
    if fmt == "pytorch":
        candidates = [dataset_dir / spec.weight_file, dataset_dir / f"{stem}.pt"]
    elif fmt == "onnx":
        candidates = [dataset_dir / f"{stem}.onnx"]
    else:
        raise ValueError(f"Unsupported RF100-VL format: {fmt!r} (use pytorch or onnx)")
    for c in candidates:
        if c.exists():
            return c
    return None


# =============================================================================
# Core loop
# =============================================================================


def _predictions_for_split(
    predict: Callable[[Image.Image], tuple[Any, Any, Any]],
    coco_gt: Any,
    img_ids: list[int],
    img_dir: Path,
    cat_ids: list[int],
    desc: str,
    verbose: bool,
) -> tuple[list[dict], list[float]]:
    """Run ``predict`` over a split; return COCO predictions + per-image ms."""
    predictions: list[dict] = []
    total_times: list[float] = []

    pbar = tqdm(img_ids, desc=desc, disable=not verbose, leave=False)
    for img_id in pbar:
        img_info = coco_gt.loadImgs(img_id)[0]
        with Image.open(img_dir / img_info["file_name"]) as source_image:
            pil_img = source_image.convert("RGB")

        t0 = time.perf_counter()
        boxes, scores, classes = predict(pil_img)
        total_times.append((time.perf_counter() - t0) * 1000.0)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = (float(v) for v in box)
            cls_int = int(cls)
            cat_id = cat_ids[cls_int] if 0 <= cls_int < len(cat_ids) else cls_int
            predictions.append(
                {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return predictions, total_times


def _make_pytorch_predict(model: Any, conf: float, iou: float, max_det: int):
    import torch

    imgsz = model._get_input_size()

    def predict(pil_img: Image.Image):
        input_tensor, _orig, original_size, ratio = model._preprocess(
            pil_img,
            "rgb",
            input_size=imgsz,
        )
        input_tensor = input_tensor.to(model.device)
        with torch.no_grad():
            output = model._forward(input_tensor)
        det = model._postprocess(
            output,
            conf,
            iou,
            original_size,
            max_det=max_det,
            ratio=ratio,
        )
        if det["num_detections"] == 0:
            return [], [], []

        def _np(v):
            return v.cpu().numpy() if isinstance(v, torch.Tensor) else v

        return _np(det["boxes"]), _np(det["scores"]), _np(det["classes"])

    return predict, imgsz


def _make_onnx_predict(backend: Any, conf: float, iou: float, max_det: int):
    def predict(pil_img: Image.Image):
        result = backend.predict(
            pil_img,
            conf=conf,
            iou=iou,
            max_det=max_det,
            color_format="rgb",
        )
        b = result.boxes
        if len(b.xyxy) == 0:
            return [], [], []
        return b.xyxy, b.conf, b.cls

    return predict, backend.imgsz


def _load_for_dataset(
    spec: ModelSpec,
    fmt: str,
    weights_path: Path | None,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
):
    """Load the model/backend for one dataset; return (predict_fn, imgsz, params_m, device_str)."""
    from libreyolo import LibreYOLO

    if fmt == "pytorch":
        model = LibreYOLO(
            model_path=str(weights_path) if weights_path else spec.weight_file,
            size=spec.constructor_size,
            device=device,
        )
        predict, imgsz = _make_pytorch_predict(model, conf, iou, max_det)
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
        return predict, imgsz, params_m, str(model.device)

    if fmt == "onnx":
        if weights_path is None:
            raise ValueError("RF100-VL onnx runs need per-dataset .onnx weights")
        backend = LibreYOLO(model_path=str(weights_path), device=device)
        predict, imgsz = _make_onnx_predict(backend, conf, iou, max_det)
        return predict, imgsz, 0.0, str(backend.device)

    raise ValueError(f"Unsupported RF100-VL format: {fmt!r} (use pytorch or onnx)")


def _aggregate_metrics(per_dataset: list[dict[str, float]]) -> dict[str, float]:
    """Unweighted mean of each COCO metric across datasets (RF100-VL protocol)."""
    if not per_dataset:
        raise ValueError("No datasets were evaluated; cannot aggregate.")
    return {key: float(np.mean([m[key] for m in per_dataset])) for key in _METRIC_KEYS}


def _manifest_sha256(records: list[dict[str, Any]]) -> str:
    """Hash a canonical JSON manifest for dataset/checkpoint provenance."""
    return canonical_json_sha256(records)


def _recipe_repro(
    recipe_path: str | Path | None,
    weights_root: str | Path | None,
    dataset_names: list[str],
    *,
    version_lock: dict[str, Any] | None = None,
    model_key: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve one campaign recipe hash from an explicit file or train stats."""
    reasons: list[str] = []
    explicit: dict[str, Any] | None = None
    if recipe_path is not None:
        path = Path(recipe_path)
        if not path.is_file():
            raise FileNotFoundError(f"RF100-VL recipe not found: {path}")
        explicit = {
            "file": str(path),
            "sha256": file_sha256(path),
            "source": "explicit",
        }

    if weights_root is None:
        if explicit is not None:
            return explicit, reasons
        return (
            {
                "file": None,
                "sha256": None,
                "source": "unavailable",
            },
            ["no training recipe hash was provided"],
        )

    hashes: dict[str, list[str]] = {}
    files: set[str] = set()
    missing: list[str] = []
    nonconformant: list[str] = []
    wrong_protocol: list[str] = []
    # dataset name -> the metadata fields that disagree with this campaign, so
    # the reason can name them instead of printing one generic sentence for
    # every possible cause.
    metadata_mismatch: dict[str, set[str]] = {}
    expected_versions_sha256 = (
        version_lock_sha256(version_lock) if version_lock is not None else None
    )
    for name in dataset_names:
        stats_path = Path(weights_root) / name / "stats.json"
        if not stats_path.exists():
            missing.append(name)
            continue
        try:
            stats = load_json(stats_path)
        except (OSError, ValueError, json.JSONDecodeError):
            missing.append(name)
            continue
        recipe = stats.get("recipe", {})
        recipe_sha = recipe.get("sha256") if isinstance(recipe, dict) else None
        if not isinstance(recipe_sha, str) or len(recipe_sha) != 64:
            missing.append(name)
            continue
        hashes.setdefault(recipe_sha, []).append(name)
        if stats.get("protocol_conformant") is not True:
            nonconformant.append(name)
        if stats.get("protocol_version") != PROTOCOL_VERSION:
            wrong_protocol.append(name)
        expected_version = dataset_version(version_lock, name)
        expected_metadata = {
            "schema_version": TRAIN_STATS_SCHEMA,
            "dataset": name,
            "dataset_version": expected_version,
            "model_key": model_key,
            "seed": PROTOCOL_SEED,
            "epochs_requested": PROTOCOL_EPOCHS,
            "versions_sha256": expected_versions_sha256,
        }
        mismatched_fields = {
            key
            for key, expected in expected_metadata.items()
            if expected is not None and stats.get(key) != expected
        }
        if stats.get("precision") not in {"fp32", "bfloat16"}:
            mismatched_fields.add("precision")
        capabilities = stats.get("libreyolo_capabilities")
        if (
            not isinstance(capabilities, dict)
            or capabilities.get("validated") is not True
            or capabilities.get("eval_max_det") != PROTOCOL_MAX_DET
            or capabilities.get("default_eval_max_det") != 100
        ):
            mismatched_fields.add("libreyolo_capabilities")
        if mismatched_fields:
            metadata_mismatch[name] = mismatched_fields
        recipe_file = recipe.get("file")
        if isinstance(recipe_file, str):
            files.add(recipe_file)

    if missing:
        reasons.append(f"{len(missing)} evaluated datasets lack a training recipe hash")
    if len(hashes) > 1:
        reasons.append("evaluated checkpoints were trained with multiple recipe hashes")
    if nonconformant:
        reasons.append(
            f"{len(nonconformant)} evaluated checkpoints are marked non-protocol training runs"
        )
    if wrong_protocol:
        reasons.append(
            f"{len(wrong_protocol)} evaluated checkpoints have the wrong training protocol version"
        )
    if metadata_mismatch:
        offending = sorted({field for fields in metadata_mismatch.values() for field in fields})
        reasons.append(
            f"{len(metadata_mismatch)} evaluated checkpoints have training metadata "
            f"that does not match this campaign: {', '.join(offending)}"
        )
    recipe_sha = next(iter(hashes)) if len(hashes) == 1 else None
    if explicit is not None:
        if hashes and set(hashes) != {explicit["sha256"]}:
            reasons.append("explicit recipe hash does not match per-dataset training stats")
        explicit["dataset_count"] = sum(len(names) for names in hashes.values())
        return explicit, reasons
    return {
        "file": next(iter(files)) if len(files) == 1 else None,
        "sha256": recipe_sha,
        "source": "per-dataset stats.json",
        "dataset_count": sum(len(names) for names in hashes.values()),
    }, reasons


def _dataset_cache_path(
    root: Path,
    dataset_name: str,
    fingerprint: str,
) -> Path:
    return root / dataset_name / f"{fingerprint}.json"


PREDICTIONS_SCHEMA = "rf100vl.predictions.v1"


def _write_predictions(
    cache_path: Path,
    fingerprint: str,
    predictions: list[dict[str, Any]],
) -> Path:
    """Persist the raw COCO detections beside the per-dataset result.

    Without these, the published reproducibility story ("rescore from the JSONs
    with pycocotools, no GPU needed") does not hold: the detections were built
    in memory and discarded, so nobody could recheck a score without repeating
    the whole campaign. Gzipped because at conf 0.001 and max_det 500 these are
    the largest artifact the run produces.
    """
    # Store only the four fields COCO detection scoring reads. pycocotools'
    # loadRes MUTATES the prediction dicts in place, stamping on id, iscrowd,
    # area and a segmentation polygon that merely re-encodes the box corners.
    # Since these are written after evaluation, that derived padding would
    # otherwise be persisted: measured at 81% of the file, or 431 MB of waste
    # across a 100-dataset run. Box coordinates are rounded to 1/100 px, far
    # below any IoU threshold's sensitivity; scores keep FULL precision because
    # AP depends on their ranking and rounding could introduce ties.
    slim = [
        {
            "image_id": d["image_id"],
            "category_id": d["category_id"],
            "bbox": [round(float(v), 2) for v in d["bbox"]],
            "score": float(d["score"]),
        }
        for d in predictions
    ]

    path = cache_path.with_name(f"{cache_path.stem}.predictions.json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(long_path(temporary), "wt", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "schema_version": PREDICTIONS_SCHEMA,
                "fingerprint": fingerprint,
                "detections": slim,
            },
            handle,
        )
    os.replace(long_path(temporary), long_path(path))
    return path


def _load_cached_dataset_result(
    path: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    # Path.exists() answers False (rather than raising) for a path over the
    # Windows MAX_PATH limit, which would silently defeat resume rather than
    # failing loudly, so ask through the same long-path shim the writer uses.
    if not os.path.exists(long_path(path)):
        return None
    try:
        cached = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.warn(f"Ignoring unreadable RF100-VL result cache {path}: {exc}")
        return None
    if (
        cached.get("schema_version") != PER_DATASET_RESULT_SCHEMA
        or cached.get("fingerprint") != fingerprint
    ):
        return None
    result = cached.get("result")
    return result if isinstance(result, dict) else None


def benchmark_model_rf100vl(
    model_key: str,
    data_dir: str | Path,
    fmt: str = "pytorch",
    weights_root: str | Path | None = None,
    device: str = "auto",
    conf: float = 0.001,
    iou: float = PROTOCOL_IOU,
    max_det: int = PROTOCOL_MAX_DET,
    split: str = "test",
    limit: int | None = None,
    limit_datasets: int | None = None,
    datasets: list[str] | None = None,
    allow_pretrained: bool = False,
    versions_path: str | Path | None = None,
    recipe_path: str | Path | None = None,
    per_dataset_dir: str | Path | None = None,
    save_predictions: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate one model across all RF100-VL datasets and aggregate.

    Args:
        model_key: Registry key (e.g. "yolov9t").
        data_dir: Directory of RF100-VL datasets (one sub-folder each).
        fmt: "pytorch" or "onnx".
        weights_root: Root of per-dataset fine-tuned checkpoints
            (``<weights_root>/<dataset>/<weight file>``). Datasets without a
            checkpoint are skipped and listed in the result.
        split: Which split to score ("test" per the RF100-VL protocol).
        limit: Max images per dataset (smoke runs only; not submittable).
        limit_datasets: Evaluate only the first N datasets (smoke runs only).
        datasets: Evaluate only these datasets by name (partial run; not
            submittable). Names must exist under ``data_dir``.
        allow_pretrained: Permit registry COCO weights when weights_root is
            not given. Scores are meaningless for closed-vocab models; exists
            for smoke tests and open-vocabulary models.
        versions_path: Optional explicit ``versions.json``. By default the
            evaluator reads ``<data_dir>/versions.json``.
        recipe_path: Optional explicit training recipe. Otherwise recipe
            hashes are read from each checkpoint's sibling ``stats.json``.
        per_dataset_dir: Directory for atomic per-dataset result files. The
            default is ``<weights_root>/.eval/<model>/<format>/<split>``, so the
            dataset directory can be a read-only shared mount. Falls back to
            ``<data_dir>/.va-bench/eval/...`` only when no weights_root is given
            (allow_pretrained smoke runs, which are never submittable). Caches
            written by earlier versions live at the old path; pass it here to
            reuse them, otherwise those datasets simply re-evaluate.

    Returns:
        Submission dict: schema va.submission.v1 with dataset id "rf100_vl",
        accuracy = across-dataset means, and an ``rf100vl`` breakdown block.
    """
    spec = get_spec(model_key)
    if max_det < 1:
        raise ValueError(f"max_det must be >= 1, got {max_det}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if limit_datasets is not None and limit_datasets < 1:
        raise ValueError(f"limit_datasets must be >= 1, got {limit_datasets}")
    if weights_root is None and not allow_pretrained:
        raise ValueError(
            "RF100-VL is a fine-tuned benchmark: pass weights_root with one "
            "checkpoint per dataset, or set allow_pretrained=True to force "
            "COCO-pretrained weights (smoke tests / open-vocab only)."
        )

    data_dir = Path(data_dir)
    dataset_dirs = discover_datasets(data_dir, split=split)
    num_datasets_discovered = len(dataset_dirs)
    if datasets is not None:
        requested = set(datasets)
        if not requested:
            raise ValueError("datasets filter must name at least one dataset")
        by_name = {path.name: path for path in dataset_dirs}
        unknown = sorted(requested - set(by_name))
        if unknown:
            raise ValueError(
                "Requested datasets not found under data_dir: " + ", ".join(unknown)
            )
        dataset_dirs = [by_name[name] for name in sorted(requested)]
    if limit_datasets is not None:
        dataset_dirs = dataset_dirs[:limit_datasets]
    selected_names = [path.name for path in dataset_dirs]

    version_lock = load_version_lock(
        versions_path if versions_path is not None else data_dir,
        required=False,
    )
    versions_sha256 = version_lock_sha256(version_lock) if version_lock is not None else None
    recipe_repro, recipe_invalid_reasons = _recipe_repro(
        recipe_path,
        weights_root,
        selected_names,
        version_lock=version_lock,
        model_key=model_key,
    )
    domain_manifest = load_domain_manifest()
    unknown_manifest_names = validate_dataset_names(selected_names)
    hw_sw = collect_hw()
    harness_identity = harness_git_info()
    # The dataset directory is a READ-ONLY MOUNT as far as this harness is
    # concerned: it is shared between boxes, and on a campaign it is a snapshot
    # pulled from HuggingFace. Harness state therefore lives under the output
    # root, not inside the data.
    #
    # MIGRATION: caches written by earlier versions live at
    # <data_dir>/.va-bench/eval/<model>/<fmt>/<split>. They are not read from
    # the new location, so an existing campaign either re-evaluates (cheap, the
    # checkpoints are untouched) or passes the old path via per_dataset_dir.
    if per_dataset_dir is not None:
        cache_root = Path(per_dataset_dir)
    elif weights_root is not None:
        cache_root = Path(weights_root) / ".eval" / model_key / fmt / split
    else:
        # Only reachable with allow_pretrained, which is never submittable.
        cache_root = data_dir / ".va-bench" / "eval" / model_key / fmt / split
    cache_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'=' * 70}")
        print(
            f"RF100-VL: {spec.display_name} ({spec.key}) [{fmt}] "
            f"- {len(dataset_dirs)} datasets, split={split}"
        )
        print(f"{'=' * 70}")

    per_dataset_metrics: list[dict[str, float]] = []
    per_dataset_records: list[dict[str, Any]] = []
    dataset_repro_records: list[dict[str, Any]] = []
    weights_repro_records: list[dict[str, Any]] = []
    skipped: list[str] = []
    all_times: list[float] = []
    total_images = 0
    imgsz = spec.input_size
    params_m = 0.0
    provider = "cpu"

    for i, dataset_dir in enumerate(dataset_dirs):
        name = dataset_dir.name
        dataset_started = time.perf_counter()

        weights_path = None
        if weights_root is not None:
            weights_path = resolve_finetuned_weights(spec, weights_root, name, fmt)
            if weights_path is None:
                skipped.append(name)
                if verbose:
                    print(f"  [{i + 1}/{len(dataset_dirs)}] {name}: no checkpoint, skipped")
                continue

        coco_gt, img_ids, img_dir = load_dataset_split(dataset_dir, split)
        if limit is not None:
            img_ids = img_ids[:limit]
        cat_ids = category_mapping(coco_gt)
        weights_sha256 = file_sha256(weights_path)
        image_ids_sha256 = image_id_sha256(img_ids)
        version_id = dataset_version(version_lock, name)
        dataset_repro_records.append(
            {
                "dataset": name,
                "num_images": len(img_ids),
                "image_id_sha256": image_ids_sha256,
                "version_id": version_id,
            }
        )
        weights_repro_records.append(
            {
                "dataset": name,
                "file": (str(weights_path) if weights_path is not None else spec.weight_file),
                "sha256": weights_sha256,
                "source": (
                    "per-dataset-finetuned"
                    if weights_path is not None
                    else "libreyolo-managed-pretrained"
                ),
            }
        )

        cache_inputs = {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": model_key,
            "format": fmt,
            "dataset": name,
            "split": split,
            "image_id_sha256": image_ids_sha256,
            "dataset_version": version_id,
            "weights_sha256": weights_sha256,
            "weights_file": spec.weight_file if weights_path is None else None,
            "recipe_sha256": recipe_repro.get("sha256"),
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "limit": limit,
            "model_spec": {
                "constructor_size": spec.constructor_size,
                "input_size": spec.input_size,
                "weight_file": spec.weight_file,
            },
            "implementation_sha256": {
                "rf100vl": file_sha256(__file__),
                "coco_eval": file_sha256(Path(__file__).with_name("coco_eval.py")),
            },
            "runtime": {
                "requested_device": device,
                "hardware": hw_sw["hardware"],
                "software": hw_sw["software"],
                "harness": harness_identity,
            },
        }
        cache_fingerprint = canonical_json_sha256(cache_inputs)
        cache_path = _dataset_cache_path(cache_root, name, cache_fingerprint)
        cached = _load_cached_dataset_result(cache_path, cache_fingerprint)

        if cached is None:
            predict, imgsz, params_m, provider = _load_for_dataset(
                spec,
                fmt,
                weights_path,
                device,
                conf,
                iou,
                max_det,
            )

            predictions, times = _predictions_for_split(
                predict,
                coco_gt,
                img_ids,
                img_dir,
                cat_ids,
                desc=f"[{i + 1}/{len(dataset_dirs)}] {name}",
                verbose=verbose,
            )
            metrics = evaluate_coco(
                coco_gt,
                predictions,
                image_ids=img_ids,
                max_det=max_det,
            )
            dataset_wall_seconds = time.perf_counter() - dataset_started
            cached = {
                "metrics": metrics,
                "timing_ms": [float(value) for value in times],
                "imgsz": imgsz,
                "params_m": float(params_m),
                "provider": provider,
                "num_images": len(img_ids),
                "num_classes": len(cat_ids),
                "wall_seconds": dataset_wall_seconds,
            }
            if save_predictions:
                predictions_path = _write_predictions(
                    cache_path, cache_fingerprint, predictions
                )
                cached["predictions_file"] = str(predictions_path)
                cached["num_detections"] = len(predictions)
            atomic_write_json(
                cache_path,
                {
                    "schema_version": PER_DATASET_RESULT_SCHEMA,
                    "fingerprint": cache_fingerprint,
                    "inputs": cache_inputs,
                    "result": cached,
                },
            )
            resumed_from_cache = False
        else:
            metrics = {key: float(cached["metrics"][key]) for key in _METRIC_KEYS}
            times = [float(value) for value in cached["timing_ms"]]
            imgsz = cached["imgsz"]
            params_m = float(cached["params_m"])
            provider = str(cached["provider"])
            dataset_wall_seconds = float(cached["wall_seconds"])
            resumed_from_cache = True

        per_dataset_metrics.append(metrics)
        per_dataset_records.append(
            {
                "dataset": name,
                "num_images": len(img_ids),
                "num_classes": len(cat_ids),
                "dataset_version": version_id,
                "weights": str(weights_path) if weights_path else spec.weight_file,
                "weights_sha256": weights_sha256,
                "result_file": str(cache_path),
                "predictions_file": cached.get("predictions_file"),
                "resumed_from_cache": resumed_from_cache,
                "wall_seconds": round(dataset_wall_seconds, 3),
                "mAP_50": metrics["mAP50"],
                "mAP_50_95": metrics["mAP"],
                "mAP_75": metrics["mAP75"],
                "AR100": metrics["AR100"],
                "AR_max_det": metrics["AR_max_det"],
            }
        )
        all_times.extend(times)
        total_images += len(img_ids)

        if verbose:
            resume_note = " [resumed]" if resumed_from_cache else ""
            print(
                f"  [{i + 1}/{len(dataset_dirs)}] {name}{resume_note}: "
                f"AP50={metrics['mAP50']:.4f} AP50:95={metrics['mAP']:.4f} "
                f"({len(img_ids)} images)"
            )

    mean_metrics = _aggregate_metrics(per_dataset_metrics)
    total_stats = compute_stats(np.array(all_times))
    fps_mean = 1000.0 / total_stats["mean"] if total_stats["mean"] > 0 else 0.0
    fps_p50 = 1000.0 / total_stats["p50"] if total_stats["p50"] > 0 else 0.0

    if verbose:
        print(f"\nRF100-VL mean over {len(per_dataset_metrics)} datasets:")
        print(f"  AP50:    {mean_metrics['mAP50']:.4f}")
        print(f"  AP50:95: {mean_metrics['mAP']:.4f}")
        if skipped:
            print(f"  Skipped (no checkpoint): {len(skipped)}")

    regime = "fine-tuned" if weights_root is not None else "pretrained-forced"
    if regime == "pretrained-forced":
        warnings.warn(
            "RF100-VL scored with COCO-pretrained weights: numbers are NOT "
            "comparable to published fine-tuned results and must not be "
            "submitted."
        )

    device_str = provider if provider in ("cuda", "cpu", "mps") else provider.split(":")[0]
    repro = run_repro(
        dataset={
            "id": DATASET_ID,
            "split": split,
            "image_id_sha256": _manifest_sha256(dataset_repro_records),
            "identity_scheme": (
                "sha256(canonical JSON of dataset name, image count, "
                "per-dataset image-id SHA-256, and locked version id)"
            ),
            "datasets": dataset_repro_records,
            "versions": {
                "file": (
                    str(
                        Path(versions_path)
                        if versions_path is not None
                        else data_dir / "versions.json"
                    )
                    if version_lock is not None
                    else None
                ),
                "sha256": versions_sha256,
                "schema_version": (
                    version_lock.get("schema_version") if version_lock is not None else None
                ),
                "subset": (version_lock.get("subset") if version_lock is not None else None),
                "datasets": (version_lock.get("datasets") if version_lock is not None else None),
            },
            "domain_manifest": {
                "file": str(domain_manifest_path()),
                "sha256": file_sha256(domain_manifest_path()),
                "source": domain_manifest["source"],
            },
        },
        weights={
            "file": "per-dataset checkpoints",
            "sha256": _manifest_sha256(
                [
                    {
                        "dataset": record["dataset"],
                        "sha256": record["sha256"],
                    }
                    for record in weights_repro_records
                ]
            ),
            "source": regime,
            "identity_scheme": (
                "sha256(canonical JSON of dataset name and checkpoint file SHA-256)"
            ),
            "datasets": weights_repro_records,
        },
    )
    repro["protocol_version"] = PROTOCOL_VERSION
    repro["recipe"] = recipe_repro
    result = assemble_result(
        spec=spec,
        coco_metrics=mean_metrics,
        total_stats=total_stats,
        preprocess_ms=None,
        inference_ms=None,
        postprocess_ms=None,
        fps_mean=round(fps_mean, 2),
        fps_p50=round(fps_p50, 2),
        num_images=total_images,
        measured_params_m=round(params_m, 2),
        peak_vram_mb=None,
        peak_ram_mb=0.0,
        device_type=get_runtime_device_name(device_str),
        provider=device_str,
        hardware=hw_sw["hardware"],
        software=hw_sw["software"],
        actual_input_size=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        fmt=fmt,
        repro=repro,
    )

    # Re-stamp the COCO defaults from assemble_result with RF100-VL identity.
    result["submission_id"] = (
        f"{spec.key}-{DATASET_ID}-" + result["submission_id"].split(f"{spec.key}-", 1)[1]
    )
    result["dataset"] = {
        "id": DATASET_ID,
        "split": split,
        "num_images": total_images,
        "num_datasets": len(per_dataset_metrics),
        "num_datasets_discovered": num_datasets_discovered,
        "num_datasets_selected": len(dataset_dirs),
    }
    result["eval"] = {
        "dataset": DATASET_ID,
        "split": split,
        "numImages": total_images,
        "maxDets": sorted({1, 10, 100, max_det}),
    }
    result["accuracy"]["AR_max_det"] = mean_metrics["AR_max_det"]

    invalid_reasons = list(recipe_invalid_reasons)
    if regime != "fine-tuned":
        invalid_reasons.append("registry pretrained weights were forced")
    if split != "test":
        invalid_reasons.append(f"split is {split!r}, not 'test'")
    if num_datasets_discovered != EXPECTED_DATASETS:
        invalid_reasons.append(
            f"discovered {num_datasets_discovered} datasets, expected {EXPECTED_DATASETS}"
        )
    if skipped:
        invalid_reasons.append(f"{len(skipped)} datasets had no checkpoint")
    if limit is not None or limit_datasets is not None:
        invalid_reasons.append("an image or dataset smoke-test limit was applied")
    if datasets is not None:
        invalid_reasons.append("an explicit dataset name filter was applied")
    if not np.isclose(iou, PROTOCOL_IOU, rtol=0.0, atol=1e-12):
        invalid_reasons.append(f"NMS IoU is {iou}, protocol requires {PROTOCOL_IOU}")
    if max_det != PROTOCOL_MAX_DET:
        invalid_reasons.append(f"max_det is {max_det}, protocol requires {PROTOCOL_MAX_DET}")
    if version_lock is None:
        invalid_reasons.append("dataset versions.json lock is missing")
    else:
        if version_lock.get("subset") != "rf100vl":
            invalid_reasons.append("dataset version lock is not for the full rf100vl subset")
        locked_names = set(version_lock["datasets"])
        discovered_names = {path.name for path in discover_datasets(data_dir, split=split)}
        if locked_names != discovered_names:
            invalid_reasons.append(
                "dataset version lock does not exactly match discovered datasets"
            )
    if unknown_manifest_names:
        invalid_reasons.append(
            f"{len(unknown_manifest_names)} datasets are absent from the vendored domain manifest"
        )
    if harness_identity.get("dirty") is True:
        invalid_reasons.append("benchmark harness working tree is dirty")
    if hw_sw["software"].get("libreyolo_dirty") is True:
        invalid_reasons.append("LibreYOLO working tree is dirty")

    result["rf100vl"] = {
        "regime": regime,
        "aggregation": "unweighted mean across datasets",
        "protocol": {
            "version": PROTOCOL_VERSION,
            "nms_iou": iou,
            "max_det": max_det,
            "expected_datasets": EXPECTED_DATASETS,
        },
        "dataset_versions_sha256": versions_sha256,
        "recipe_sha256": recipe_repro.get("sha256"),
        "per_dataset_results_dir": str(cache_root),
        "valid_submission": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "datasets": per_dataset_records,
        "skipped_datasets": skipped,
    }
    if limit is not None or limit_datasets is not None or datasets is not None:
        result["rf100vl"]["subset_run"] = {
            "limit_images": limit,
            "limit_datasets": limit_datasets,
            "datasets": sorted(datasets) if datasets is not None else None,
            "note": "SUBSET - not a valid RF100-VL submission",
        }
    return result
