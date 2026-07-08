"""Unit tests for the instance segmentation path (registry, RLE, output)."""

from __future__ import annotations

import numpy as np
import pytest

from va_bench.benchmark import (
    _append_mask_predictions,
    _extract_masks_or_fail,
    COCO_80_TO_91,
)
from va_bench.coco_eval import evaluate_coco
from va_bench.models import MODEL_REGISTRY, get_spec
from va_bench.output import assemble_result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_all_detect_models_default_to_detect_task():
    for key, spec in MODEL_REGISTRY.items():
        if "-seg-" not in key:
            assert spec.task == "detect", key


def test_seg_models_registered_with_segment_task():
    seg_keys = [k for k, s in MODEL_REGISTRY.items() if s.task == "segment"]
    assert sorted(seg_keys) == [
        "dfine-seg-l", "dfine-seg-m", "dfine-seg-n", "dfine-seg-s", "dfine-seg-x",
        "ec-seg-l", "ec-seg-m", "ec-seg-s", "ec-seg-x",
        "rfdetr-seg-l", "rfdetr-seg-m", "rfdetr-seg-n", "rfdetr-seg-s",
    ]


def test_seg_weight_files_carry_seg_suffix():
    for spec in MODEL_REGISTRY.values():
        if spec.task == "segment":
            assert spec.weight_file.endswith("-seg.pt"), spec.key


def test_seg_family_stays_base_family():
    assert get_spec("rfdetr-seg-s").family == "rfdetr"
    assert get_spec("dfine-seg-n").family == "dfine"
    assert get_spec("ec-seg-s").family == "ec"


# ---------------------------------------------------------------------------
# Mask prediction encoding
# ---------------------------------------------------------------------------

def test_append_mask_predictions_encodes_rle():
    masks = np.zeros((2, 32, 48), dtype=bool)
    masks[0, 4:10, 6:20] = True   # 6 x 14 = 84 px
    masks[1, 20:30, 0:8] = True   # 10 x 8 = 80 px
    preds: list[dict] = []
    _append_mask_predictions(preds, masks, [0.9, 0.5], [0, 17], img_id=42)

    assert len(preds) == 2
    for p in preds:
        assert p["image_id"] == 42
        assert "bbox" not in p  # segm-only list, see docstring
        assert isinstance(p["segmentation"]["counts"], str)
        assert p["segmentation"]["size"] == [32, 48]
    assert preds[0]["area"] == 84.0
    assert preds[1]["area"] == 80.0
    assert preds[0]["category_id"] == COCO_80_TO_91[0]
    assert preds[1]["category_id"] == COCO_80_TO_91[17]
    assert preds[0]["score"] == pytest.approx(0.9)


def test_append_mask_predictions_rejects_bad_shape():
    with pytest.raises(ValueError, match="N, H, W"):
        _append_mask_predictions([], np.zeros((32, 48)), [0.9], [0], img_id=1)


def test_extract_masks_or_fail_raises_on_none():
    with pytest.raises(RuntimeError, match="rfdetr-seg-s.*no masks"):
        _extract_masks_or_fail(None, "rfdetr-seg-s")


def test_extract_masks_or_fail_passes_through():
    masks = np.ones((1, 4, 4), dtype=bool)
    assert _extract_masks_or_fail(masks, "x") is masks


# ---------------------------------------------------------------------------
# COCO eval plumbing
# ---------------------------------------------------------------------------

def test_evaluate_coco_rejects_unknown_iou_type():
    with pytest.raises(ValueError, match="iou_type"):
        evaluate_coco(None, [{"image_id": 1}], iou_type="keypoints")


def test_evaluate_coco_empty_predictions_returns_zero_metrics():
    metrics = evaluate_coco(None, [], iou_type="segm")
    assert metrics["mAP"] == 0.0
    assert set(metrics) == {
        "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
    }


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _metrics(v: float) -> dict[str, float]:
    return {
        "mAP": v, "mAP50": v, "mAP75": v,
        "mAP_small": v, "mAP_medium": v, "mAP_large": v,
        "AR1": v, "AR10": v, "AR100": v,
        "AR_small": v, "AR_medium": v, "AR_large": v,
    }


def _assemble(spec_key: str, mask_metrics: dict | None):
    return assemble_result(
        spec=get_spec(spec_key),
        coco_metrics=_metrics(0.40),
        mask_metrics=mask_metrics,
        total_stats={"mean": 10.0, "std": 1.0, "p50": 10.0, "p95": 11.0, "p99": 12.0},
        preprocess_ms=1.0,
        inference_ms=8.0,
        postprocess_ms=1.0,
        fps_mean=100.0,
        fps_p50=100.0,
        num_images=5000,
        measured_params_m=10.0,
        peak_vram_mb=100.0,
        peak_ram_mb=500.0,
        device_type="gpu",
        provider="cuda",
        hardware={"gpu": "NVIDIA GeForce RTX 5080", "gpu_memory_gb": 16.0},
        software={"python": "3.12", "torch": "2.11"},
        actual_input_size=384,
        conf=0.001,
        iou=0.6,
        max_det=300,
        fmt="pytorch",
    )


def test_seg_result_headline_is_mask_map_with_bbox_secondary():
    result = _assemble("rfdetr-seg-s", mask_metrics=_metrics(0.35))
    acc = result["accuracy"]
    assert result["model"]["task"] == "segment"
    assert acc["mAP_50_95"] == 0.35       # headline = mask mAP
    assert acc["bbox_mAP_50_95"] == 0.40  # box mAP kept, prefixed
    assert acc["bbox_mAP_small"] == 0.40
    assert result["model"]["id"] == "rfdetr-seg-s"
    assert result["model"]["weights"] == "LibreRFDETRs-seg.pt"


def test_detect_result_has_no_bbox_prefixed_keys():
    result = _assemble("yolox-s", mask_metrics=None)
    acc = result["accuracy"]
    assert result["model"]["task"] == "detect"
    assert acc["mAP_50_95"] == 0.40
    assert not any(k.startswith("bbox_") for k in acc)


def test_assemble_rejects_task_metric_mismatch():
    with pytest.raises(ValueError, match="mask_metrics"):
        _assemble("rfdetr-seg-s", mask_metrics=None)
    with pytest.raises(ValueError, match="mask_metrics"):
        _assemble("yolox-s", mask_metrics=_metrics(0.35))


def test_seg_result_roundtrips_mask_eval():
    """End-to-end micro check: RLE predictions -> segm COCOeval -> perfect mAP.

    Builds a one-image, one-instance COCO gt in memory and feeds the exact
    prediction dicts produced by _append_mask_predictions.
    """
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO

    masks = np.zeros((1, 64, 64), dtype=bool)
    masks[0, 10:30, 10:30] = True
    gt_rle = mask_utils.encode(np.asfortranarray(masks[0].astype(np.uint8)))
    gt_rle["counts"] = gt_rle["counts"].decode("utf-8")

    gt = COCO()
    gt.dataset = {
        "images": [{"id": 1, "width": 64, "height": 64}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "segmentation": gt_rle,
            "area": 400.0, "bbox": [10, 10, 20, 20], "iscrowd": 0,
        }],
        "categories": [{"id": 1, "name": "person"}],
    }
    gt.createIndex()

    preds: list[dict] = []
    _append_mask_predictions(preds, masks, [0.99], [0], img_id=1)

    metrics = evaluate_coco(gt, preds, image_ids=[1], iou_type="segm")
    assert metrics["mAP"] == pytest.approx(1.0)
