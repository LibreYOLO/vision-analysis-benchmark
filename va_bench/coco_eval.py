"""
COCO evaluation wrapper for Vision Analysis benchmarks.

Runs pycocotools COCOeval and extracts all 12 standard metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_coco(
    coco_gt: Any,
    predictions: list[dict],
    image_ids: list[int] | None = None,
    max_det: int = 100,
) -> dict[str, float]:
    """Run COCO evaluation at an explicit detection cap.

    Args:
        coco_gt: A pycocotools.coco.COCO ground truth object.
        predictions: List of dicts with keys:
            image_id (int), category_id (int), bbox [x,y,w,h], score (float).
        image_ids: Optional subset of image IDs to evaluate on.
        max_det: Detection cap used for headline AP and size-stratified AP/AR.
            A genuine AR@100 compatibility metric is retained separately.

    Returns:
        Dict with mAP, mAP50, mAP75, mAP_small, mAP_medium, mAP_large,
        AR1, AR10, AR100, AR_max_det, AR_small, AR_medium, AR_large.

    ``pycocotools.COCOeval.summarize()`` hard-codes the headline AP lookup to
    maxDets=100. Calling it after replacing ``params.maxDets`` with
    ``[1, 10, 500]`` therefore returns -1 for overall AP. Read the accumulated
    arrays directly so the requested cap is used consistently.
    """
    if max_det < 1:
        raise ValueError(f"max_det must be >= 1, got {max_det}")
    if not predictions:
        return _empty_metrics(max_det)

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    if image_ids is not None:
        coco_eval.params.imgIds = image_ids

    # Keep 100 in the axis so AR100 remains a real metric even when the
    # benchmark protocol requests a larger cap such as RF100-VL's 500.
    coco_eval.params.maxDets = sorted({1, 10, 100, int(max_det)})
    coco_eval.evaluate()
    coco_eval.accumulate()

    m_ap = _summarize_metric(coco_eval, ap=True, max_det=max_det)
    m_ap50 = _summarize_metric(coco_eval, ap=True, max_det=max_det, iou_thr=0.5)
    m_ap75 = _summarize_metric(coco_eval, ap=True, max_det=max_det, iou_thr=0.75)
    ar_100 = _summarize_metric(coco_eval, ap=False, max_det=100)
    ar_max_det = _summarize_metric(coco_eval, ap=False, max_det=max_det)
    return {
        "max_det": int(max_det),
        "mAP": m_ap,
        "mAP50": m_ap50,
        "mAP75": m_ap75,
        "mAP_small": _summarize_metric(coco_eval, ap=True, max_det=max_det, area="small"),
        "mAP_medium": _summarize_metric(coco_eval, ap=True, max_det=max_det, area="medium"),
        "mAP_large": _summarize_metric(coco_eval, ap=True, max_det=max_det, area="large"),
        "AR1": _summarize_metric(coco_eval, ap=False, max_det=1),
        "AR10": _summarize_metric(coco_eval, ap=False, max_det=10),
        "AR100": ar_100,
        "AR_max_det": ar_max_det,
        "AR_small": _summarize_metric(coco_eval, ap=False, max_det=max_det, area="small"),
        "AR_medium": _summarize_metric(coco_eval, ap=False, max_det=max_det, area="medium"),
        "AR_large": _summarize_metric(coco_eval, ap=False, max_det=max_det, area="large"),
    }


def _summarize_metric(
    coco_eval: Any,
    *,
    ap: bool,
    max_det: int,
    iou_thr: float | None = None,
    area: str = "all",
) -> float:
    """Read one scalar from COCOeval's accumulated precision/recall arrays."""
    params = coco_eval.params
    try:
        area_index = list(params.areaRngLbl).index(area)
        max_det_index = list(params.maxDets).index(max_det)
    except ValueError:
        return -1.0

    key = "precision" if ap else "recall"
    values = coco_eval.eval[key][..., area_index, max_det_index]
    if iou_thr is not None:
        iou_indices = np.flatnonzero(np.isclose(params.iouThrs, iou_thr))
        values = values[iou_indices]
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else -1.0


def _empty_metrics(max_det: int = 100) -> dict[str, float]:
    return {
        "max_det": int(max_det),
        "mAP": 0.0,
        "mAP50": 0.0,
        "mAP75": 0.0,
        "mAP_small": 0.0,
        "mAP_medium": 0.0,
        "mAP_large": 0.0,
        "AR1": 0.0,
        "AR10": 0.0,
        "AR100": 0.0,
        "AR_max_det": 0.0,
        "AR_small": 0.0,
        "AR_medium": 0.0,
        "AR_large": 0.0,
    }
