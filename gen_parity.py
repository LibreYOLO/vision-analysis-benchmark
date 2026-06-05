"""Generate the website parity dataset for visionanalysis.org.

Measured values prefer the fresh parity re-eval in results_parity/ and fall
back to the main sweep in results_full_gpu/. Reference values and sources come
from reference_map.json. Cite papers, official repos, model zoos, or HF cards
only.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "reference_map.json"
PARITY_RESULTS = HERE / "results_parity"
SWEEP_RESULTS = HERE / "results_full_gpu"
DEFAULT_VISION_ANALYSIS_REPO = HERE.parent / "vision-analysis"
OUT = Path(
    os.environ.get(
        "VISION_ANALYSIS_PARITY_OUT",
        DEFAULT_VISION_ANALYSIS_REPO / "website" / "src" / "data" / "metadata" / "parity.json",
    )
)

CHECKED_DATE = "2026-06-05"
HARDWARE = "NVIDIA RTX 5070 Ti"
DATASET = "COCO val2017 (5000 images)"
METHODOLOGY = (
    "LibreYOLO PyTorch, fp32, per-image inference at each model's native input "
    "size. NMS IoU per family (YOLOX 0.65, YOLOv9 0.7, others 0.6; DETR families "
    "are NMS-free). Reference = each model's original paper / repo README / model "
    "zoo / HF card claimed COCO box mAP@50-95 (source linked per value)."
)

# Families being fixed on another branch. Excluded until their fixes land;
# then re-run this generator to add them.
IN_PROGRESS = {"rtmdet"}

FAMILY_LABELS = {
    "yolox": "YOLOX",
    "yolov9": "YOLOv9",
    "yolov9-e2e": "YOLOv9-E2E",
    "rfdetr": "RF-DETR",
    "rtdetr": "RT-DETR",
    "rtdetrv2": "RT-DETRv2",
    "rtdetrv4": "RT-DETRv4",
    "deim": "DEIM",
    "deimv2": "DEIMv2",
    "dfine": "D-FINE",
    "picodet": "PicoDet",
    "ec": "EC (EdgeCrafter)",
    "damoyolo": "DAMO-YOLO",
    "rtmdet": "RTMDet",
}


def source_label(url: str) -> str:
    if not url:
        return "-"
    if "arxiv.org" in url:
        m = url.rstrip("/").split("/")[-1]
        return f"arXiv:{m}"
    if "huggingface.co" in url:
        return "HF model card"
    if "github.com" in url:
        parts = url.split("github.com/")[1].split("/")
        return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else "GitHub"
    return url.replace("https://", "").split("/")[0]


def load_measured() -> dict[str, float]:
    """Latest measured mAP per model id; parity re-eval overrides the sweep."""
    measured = {}
    for directory in (SWEEP_RESULTS, PARITY_RESULTS):
        for filename in sorted(glob.glob(str(directory / "*.json"))):
            try:
                with open(filename, encoding="utf-8") as handle:
                    result = json.load(handle)
            except Exception:
                continue
            model_id = result.get("model", {}).get("id")
            if model_id:
                measured[model_id] = result["accuracy"]["mAP_50_95"] * 100.0
    return measured


def family_for_model(model_id: str) -> str:
    if model_id.startswith("yolov9") and not model_id.startswith("yolov9e2e"):
        return "yolov9"
    for family in FAMILY_LABELS:
        if model_id.startswith(family + "-") or model_id == family:
            return family
    stem = model_id.rsplit("-", 1)[0]
    return stem if stem in FAMILY_LABELS else model_id.split("-")[0]


def main() -> None:
    with open(REF, encoding="utf-8") as handle:
        ref = json.load(handle)
    measured = load_measured()

    families_by_key = defaultdict(list)
    for model_id, reference in ref.items():
        if model_id.startswith("_") or not isinstance(reference, dict):
            continue
        if model_id not in measured:
            continue

        family = family_for_model(model_id)
        claimed = (
            round(reference["paper_map"] * 100, 2)
            if isinstance(reference.get("paper_map"), (int, float))
            else None
        )
        measured_map = round(measured[model_id], 2)
        delta = round(measured_map - claimed, 2) if claimed is not None else None
        families_by_key[family].append(
            {
                "id": model_id,
                "measured": measured_map,
                "claimed": claimed,
                "delta": delta,
                "dataset": reference.get("dataset", "val2017"),
                "source": reference.get("source", ""),
                "sourceLabel": source_label(reference.get("source", "")),
                "confidence": reference.get("confidence", "high"),
                "note": reference.get("notes", ""),
            }
        )

    families = []
    for family, label in FAMILY_LABELS.items():
        if family in IN_PROGRESS:
            continue
        if family not in families_by_key:
            continue
        variants = sorted(
            families_by_key[family],
            key=lambda value: value["claimed"] if value["claimed"] is not None else -1,
        )
        families.append({"family": family, "displayName": label, "variants": variants})

    variant_count = sum(len(family["variants"]) for family in families)
    payload = {
        "schema": "va.parity.v1",
        "checkedDate": CHECKED_DATE,
        "hardware": HARDWARE,
        "dataset": DATASET,
        "harness": "vision-analysis-benchmark",
        "methodology": METHODOLOGY,
        "variantCount": variant_count,
        "families": families,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {OUT}: {len(families)} families, {variant_count} variants")


if __name__ == "__main__":
    main()
