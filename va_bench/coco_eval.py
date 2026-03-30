"""
COCO evaluation wrapper for Vision Analysis benchmarks.

Runs pycocotools COCOeval and extracts all 12 standard metrics.
"""

from __future__ import annotations

from typing import Any


def evaluate_coco(
    coco_gt: Any,
    predictions: list[dict],
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    """Run COCO evaluation and return all 12 metrics.

    Args:
        coco_gt: A pycocotools.coco.COCO ground truth object.
        predictions: List of dicts with keys:
            image_id (int), category_id (int), bbox [x,y,w,h], score (float).
        image_ids: Optional subset of image IDs to evaluate on.

    Returns:
        Dict with mAP, mAP50, mAP75, mAP_small, mAP_medium, mAP_large,
        AR1, AR10, AR100, AR_small, AR_medium, AR_large.
    """
    if not predictions:
        return _empty_metrics()

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    if image_ids is not None:
        coco_eval.params.imgIds = image_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats layout: [mAP, mAP50, mAP75, AP_s, AP_m, AP_l,
    #                AR1, AR10, AR100, AR_s, AR_m, AR_l]
    s = coco_eval.stats
    return {
        "mAP": float(s[0]),
        "mAP50": float(s[1]),
        "mAP75": float(s[2]),
        "mAP_small": float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large": float(s[5]),
        "AR1": float(s[6]),
        "AR10": float(s[7]),
        "AR100": float(s[8]),
        "AR_small": float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large": float(s[11]),
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0,
        "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
        "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
        "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0,
    }
