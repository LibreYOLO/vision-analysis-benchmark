"""Contract tests for configurable COCO maxDets evaluation."""

from __future__ import annotations

import pytest
from pycocotools.coco import COCO

from va_bench.coco_eval import evaluate_coco


def _dense_fixture(count: int) -> tuple[COCO, list[dict]]:
    annotations = []
    predictions = []
    for index in range(count):
        x = float((index % 25) * 20)
        y = float((index // 25) * 20)
        bbox = [x, y, 10.0, 10.0]
        annotations.append(
            {
                "id": index + 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "area": 100.0,
                "iscrowd": 0,
            }
        )
        predictions.append(
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": bbox,
                "score": 1.0 - index / max(1, count * 2),
            }
        )

    coco = COCO()
    coco.dataset = {
        "images": [{"id": 1, "file_name": "dense.jpg", "width": 500, "height": 400}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }
    coco.createIndex()
    return coco, predictions


def test_ground_truth_self_eval_is_one_at_500():
    coco, predictions = _dense_fixture(1)

    metrics = evaluate_coco(coco, predictions, image_ids=[1], max_det=500)

    assert metrics["max_det"] == 500
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["AR_max_det"] == pytest.approx(1.0)


def test_dense_image_proves_ap500_exceeds_ap100():
    coco, predictions = _dense_fixture(500)

    metrics_100 = evaluate_coco(coco, predictions, image_ids=[1], max_det=100)
    metrics_500 = evaluate_coco(coco, predictions, image_ids=[1], max_det=500)

    assert metrics_500["mAP"] == pytest.approx(1.0)
    assert metrics_500["mAP"] > metrics_100["mAP"]
    assert metrics_500["AR_max_det"] > metrics_500["AR100"]


def test_empty_predictions_preserve_requested_cap():
    coco, _ = _dense_fixture(1)

    metrics = evaluate_coco(coco, [], image_ids=[1], max_det=500)

    assert metrics["max_det"] == 500
    assert metrics["mAP"] == 0.0
    assert metrics["AR_max_det"] == 0.0


def test_invalid_max_det_is_rejected():
    coco, predictions = _dense_fixture(1)

    with pytest.raises(ValueError, match="max_det must be >= 1"):
        evaluate_coco(coco, predictions, max_det=0)
