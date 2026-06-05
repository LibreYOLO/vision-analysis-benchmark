"""
Model registry for Vision Analysis benchmarks.

Each entry contains paper-reported specs and LibreYOLO loading parameters.
GFLOPs are hardcoded from papers/official repos — thop is unreliable for
transformer-based models (RF-DETR, RT-DETR).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SUPPORTED_FORMATS = ("pytorch", "onnx")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    variant: str
    weight_file: str
    constructor_size: str
    input_size: int
    paper_params_m: float
    paper_flops_g: float


# Specs sourced from vision-analysis website models.json + LibreYOLO model classes.
# YOLOX nano/tiny use 416; s/m/l/x use 640 (from YOLOXModel.INPUT_SIZES).
# RF-DETR uses per-variant sizes (from RFDETRModel.INPUT_SIZES).
# YOLOv9 uses 640 for all variants.

MODEL_REGISTRY: dict[str, ModelSpec] = {}

def _register(*specs: ModelSpec) -> None:
    for s in specs:
        MODEL_REGISTRY[s.key] = s


_register(
    # --- YOLOX (6 variants) ---
    ModelSpec("yolox-nano", "YOLOX-Nano", "yolox", "nano", "LibreYOLOXn.pt", "n", 416, 0.91, 1.32),
    ModelSpec("yolox-tiny", "YOLOX-Tiny", "yolox", "tiny", "LibreYOLOXt.pt", "t", 416, 5.06, 7.68),
    ModelSpec("yolox-s", "YOLOX-S", "yolox", "s", "LibreYOLOXs.pt", "s", 640, 8.97, 13.46),
    ModelSpec("yolox-m", "YOLOX-M", "yolox", "m", "LibreYOLOXm.pt", "m", 640, 25.33, 36.99),
    ModelSpec("yolox-l", "YOLOX-L", "yolox", "l", "LibreYOLOXl.pt", "l", 640, 54.21, 78.01),
    ModelSpec("yolox-x", "YOLOX-X", "yolox", "x", "LibreYOLOXx.pt", "x", 640, 99.07, 141.23),
    # --- YOLOv9 (4 variants) ---
    ModelSpec("yolov9t", "YOLOv9-T", "yolov9", "t", "LibreYOLO9t.pt", "t", 640, 2.04, 3.98),
    ModelSpec("yolov9s", "YOLOv9-S", "yolov9", "s", "LibreYOLO9s.pt", "s", 640, 7.23, 13.52),
    ModelSpec("yolov9m", "YOLOv9-M", "yolov9", "m", "LibreYOLO9m.pt", "m", 640, 20.12, 38.68),
    ModelSpec("yolov9c", "YOLOv9-C", "yolov9", "c", "LibreYOLO9c.pt", "c", 640, 25.50, 51.75),
    # --- RF-DETR (4 variants, per-variant input sizes) ---
    ModelSpec("rfdetr-n", "RF-DETR-N", "rfdetr", "n", "LibreRFDETRn.pt", "n", 384, 0.0, 0.0),
    ModelSpec("rfdetr-s", "RF-DETR-S", "rfdetr", "s", "LibreRFDETRs.pt", "s", 512, 0.0, 0.0),
    ModelSpec("rfdetr-m", "RF-DETR-M", "rfdetr", "m", "LibreRFDETRm.pt", "m", 576, 0.0, 0.0),
    ModelSpec("rfdetr-l", "RF-DETR-L", "rfdetr", "l", "LibreRFDETRl.pt", "l", 704, 128.0, 340.0),
)

# ---------------------------------------------------------------------------
# Families added for the full LibreYOLO COCO sweep.
#
# input_size / constructor_size mirror each family's LibreYOLO model class
# (INPUT_SIZES). paper_params_m / paper_flops_g are left 0.0 (= "unknown",
# disables check_params) until reconciled with the vision-analysis website
# models.json - we do not fabricate paper specs here. Measured params are
# still recorded at run time from the loaded model.
# ---------------------------------------------------------------------------
_register(
    # --- YOLOv9 end-to-end (4 variants, NMS-free) ---
    ModelSpec("yolov9e2e-t", "YOLOv9-E2E-T", "yolov9-e2e", "t", "LibreYOLO9E2Et.pt", "t", 640, 0.0, 0.0),
    ModelSpec("yolov9e2e-s", "YOLOv9-E2E-S", "yolov9-e2e", "s", "LibreYOLO9E2Es.pt", "s", 640, 0.0, 0.0),
    ModelSpec("yolov9e2e-m", "YOLOv9-E2E-M", "yolov9-e2e", "m", "LibreYOLO9E2Em.pt", "m", 640, 0.0, 0.0),
    ModelSpec("yolov9e2e-c", "YOLOv9-E2E-C", "yolov9-e2e", "c", "LibreYOLO9E2Ec.pt", "c", 640, 0.0, 0.0),
    # --- RT-DETR (7 variants) ---
    ModelSpec("rtdetr-r18", "RT-DETR-R18", "rtdetr", "r18", "LibreRTDETRr18.pt", "r18", 640, 0.0, 0.0),
    ModelSpec("rtdetr-r34", "RT-DETR-R34", "rtdetr", "r34", "LibreRTDETRr34.pt", "r34", 640, 0.0, 0.0),
    ModelSpec("rtdetr-r50", "RT-DETR-R50", "rtdetr", "r50", "LibreRTDETRr50.pt", "r50", 640, 0.0, 0.0),
    ModelSpec("rtdetr-r50m", "RT-DETR-R50m", "rtdetr", "r50m", "LibreRTDETRr50m.pt", "r50m", 640, 0.0, 0.0),
    ModelSpec("rtdetr-r101", "RT-DETR-R101", "rtdetr", "r101", "LibreRTDETRr101.pt", "r101", 640, 0.0, 0.0),
    ModelSpec("rtdetr-l", "RT-DETR-L", "rtdetr", "l", "LibreRTDETRl.pt", "l", 640, 0.0, 0.0),
    ModelSpec("rtdetr-x", "RT-DETR-X", "rtdetr", "x", "LibreRTDETRx.pt", "x", 640, 0.0, 0.0),
    # --- DEIM (5 variants) ---
    ModelSpec("deim-n", "DEIM-N", "deim", "n", "LibreDEIMn.pt", "n", 640, 0.0, 0.0),
    ModelSpec("deim-s", "DEIM-S", "deim", "s", "LibreDEIMs.pt", "s", 640, 0.0, 0.0),
    ModelSpec("deim-m", "DEIM-M", "deim", "m", "LibreDEIMm.pt", "m", 640, 0.0, 0.0),
    ModelSpec("deim-l", "DEIM-L", "deim", "l", "LibreDEIMl.pt", "l", 640, 0.0, 0.0),
    ModelSpec("deim-x", "DEIM-X", "deim", "x", "LibreDEIMx.pt", "x", 640, 0.0, 0.0),
    # --- DEIMv2 (8 variants, per-variant input sizes) ---
    ModelSpec("deimv2-atto", "DEIMv2-Atto", "deimv2", "atto", "LibreDEIMv2atto.pt", "atto", 320, 0.0, 0.0),
    ModelSpec("deimv2-femto", "DEIMv2-Femto", "deimv2", "femto", "LibreDEIMv2femto.pt", "femto", 416, 0.0, 0.0),
    ModelSpec("deimv2-pico", "DEIMv2-Pico", "deimv2", "pico", "LibreDEIMv2pico.pt", "pico", 640, 0.0, 0.0),
    ModelSpec("deimv2-n", "DEIMv2-N", "deimv2", "n", "LibreDEIMv2n.pt", "n", 640, 0.0, 0.0),
    ModelSpec("deimv2-s", "DEIMv2-S", "deimv2", "s", "LibreDEIMv2s.pt", "s", 640, 0.0, 0.0),
    ModelSpec("deimv2-m", "DEIMv2-M", "deimv2", "m", "LibreDEIMv2m.pt", "m", 640, 0.0, 0.0),
    ModelSpec("deimv2-l", "DEIMv2-L", "deimv2", "l", "LibreDEIMv2l.pt", "l", 640, 0.0, 0.0),
    ModelSpec("deimv2-x", "DEIMv2-X", "deimv2", "x", "LibreDEIMv2x.pt", "x", 640, 0.0, 0.0),
    # --- D-FINE (5 variants) ---
    ModelSpec("dfine-n", "D-FINE-N", "dfine", "n", "LibreDFINEn.pt", "n", 640, 0.0, 0.0),
    ModelSpec("dfine-s", "D-FINE-S", "dfine", "s", "LibreDFINEs.pt", "s", 640, 0.0, 0.0),
    ModelSpec("dfine-m", "D-FINE-M", "dfine", "m", "LibreDFINEm.pt", "m", 640, 0.0, 0.0),
    ModelSpec("dfine-l", "D-FINE-L", "dfine", "l", "LibreDFINEl.pt", "l", 640, 0.0, 0.0),
    ModelSpec("dfine-x", "D-FINE-X", "dfine", "x", "LibreDFINEx.pt", "x", 640, 0.0, 0.0),
    # --- PicoDet (3 variants, per-variant input sizes) ---
    ModelSpec("picodet-s", "PicoDet-S", "picodet", "s", "LibrePICODETs.pt", "s", 320, 0.0, 0.0),
    ModelSpec("picodet-m", "PicoDet-M", "picodet", "m", "LibrePICODETm.pt", "m", 416, 0.0, 0.0),
    ModelSpec("picodet-l", "PicoDet-L", "picodet", "l", "LibrePICODETl.pt", "l", 640, 0.0, 0.0),
    # --- EC / EdgeCrafter (4 variants) ---
    ModelSpec("ec-s", "EC-S", "ec", "s", "LibreECs.pt", "s", 640, 0.0, 0.0),
    ModelSpec("ec-m", "EC-M", "ec", "m", "LibreECm.pt", "m", 640, 0.0, 0.0),
    ModelSpec("ec-l", "EC-L", "ec", "l", "LibreECl.pt", "l", 640, 0.0, 0.0),
    ModelSpec("ec-x", "EC-X", "ec", "x", "LibreECx.pt", "x", 640, 0.0, 0.0),
)

# ---------------------------------------------------------------------------
# Families present on libreyolo dev but absent from the earlier branch the
# registry was first built against. Added for the extended sweep. paper
# params/flops left 0.0 (unknown) until sourced. Only open, downloadable
# variants are registered.
# ---------------------------------------------------------------------------
_register(
    # DAMO-YOLO (6 open variants; -l is gated 401)
    ModelSpec("damoyolo-ns", "DAMO-YOLO-Ns", "damoyolo", "ns", "LibreDAMOYOLOns.pt", "ns", 416, 0.0, 0.0),
    ModelSpec("damoyolo-nm", "DAMO-YOLO-Nm", "damoyolo", "nm", "LibreDAMOYOLOnm.pt", "nm", 416, 0.0, 0.0),
    ModelSpec("damoyolo-nl", "DAMO-YOLO-Nl", "damoyolo", "nl", "LibreDAMOYOLOnl.pt", "nl", 416, 0.0, 0.0),
    ModelSpec("damoyolo-t", "DAMO-YOLO-T", "damoyolo", "t", "LibreDAMOYOLOt.pt", "t", 640, 0.0, 0.0),
    ModelSpec("damoyolo-s", "DAMO-YOLO-S", "damoyolo", "s", "LibreDAMOYOLOs.pt", "s", 640, 0.0, 0.0),
    ModelSpec("damoyolo-m", "DAMO-YOLO-M", "damoyolo", "m", "LibreDAMOYOLOm.pt", "m", 640, 0.0, 0.0),
    # RT-DETRv2 (5)
    ModelSpec("rtdetrv2-r18", "RT-DETRv2-R18", "rtdetrv2", "r18", "LibreRTDETRv2r18.pt", "r18", 640, 0.0, 0.0),
    ModelSpec("rtdetrv2-r34", "RT-DETRv2-R34", "rtdetrv2", "r34", "LibreRTDETRv2r34.pt", "r34", 640, 0.0, 0.0),
    ModelSpec("rtdetrv2-r50", "RT-DETRv2-R50", "rtdetrv2", "r50", "LibreRTDETRv2r50.pt", "r50", 640, 0.0, 0.0),
    ModelSpec("rtdetrv2-r50m", "RT-DETRv2-R50m", "rtdetrv2", "r50m", "LibreRTDETRv2r50m.pt", "r50m", 640, 0.0, 0.0),
    ModelSpec("rtdetrv2-r101", "RT-DETRv2-R101", "rtdetrv2", "r101", "LibreRTDETRv2r101.pt", "r101", 640, 0.0, 0.0),
    # RT-DETRv4 (4)
    ModelSpec("rtdetrv4-s", "RT-DETRv4-S", "rtdetrv4", "s", "LibreRTDETRv4s.pt", "s", 640, 0.0, 0.0),
    ModelSpec("rtdetrv4-m", "RT-DETRv4-M", "rtdetrv4", "m", "LibreRTDETRv4m.pt", "m", 640, 0.0, 0.0),
    ModelSpec("rtdetrv4-l", "RT-DETRv4-L", "rtdetrv4", "l", "LibreRTDETRv4l.pt", "l", 640, 0.0, 0.0),
    ModelSpec("rtdetrv4-x", "RT-DETRv4-X", "rtdetrv4", "x", "LibreRTDETRv4x.pt", "x", 640, 0.0, 0.0),
    # RTMDet (5)
    ModelSpec("rtmdet-t", "RTMDet-T", "rtmdet", "t", "LibreRTMDett.pt", "t", 640, 0.0, 0.0),
    ModelSpec("rtmdet-s", "RTMDet-S", "rtmdet", "s", "LibreRTMDets.pt", "s", 640, 0.0, 0.0),
    ModelSpec("rtmdet-m", "RTMDet-M", "rtmdet", "m", "LibreRTMDetm.pt", "m", 640, 0.0, 0.0),
    ModelSpec("rtmdet-l", "RTMDet-L", "rtmdet", "l", "LibreRTMDetl.pt", "l", 640, 0.0, 0.0),
    ModelSpec("rtmdet-x", "RTMDet-X", "rtmdet", "x", "LibreRTMDetx.pt", "x", 640, 0.0, 0.0),
)


def list_models() -> list[str]:
    """Return sorted list of all model keys."""
    return sorted(MODEL_REGISTRY.keys())


def list_families() -> list[str]:
    """Return sorted unique family names."""
    return sorted({s.family for s in MODEL_REGISTRY.values()})


def get_spec(key: str) -> ModelSpec:
    """Look up a model spec by key. Raises KeyError if not found."""
    if key not in MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{key}'. Available: {available}")
    return MODEL_REGISTRY[key]


def load_model(key: str, device: str = "auto"):
    """Load a LibreYOLO PyTorch model by registry key.

    Returns the loaded model instance and its spec.
    """
    from libreyolo import LibreYOLO

    spec = get_spec(key)
    model = LibreYOLO(model_path=spec.weight_file, size=spec.constructor_size, device=device)
    return model, spec


def resolve_onnx_weights(spec: ModelSpec, weights_dir: str | Path) -> Path:
    """Return the path to a user-supplied ONNX file for this spec.

    Looks for ``{weight_file_stem}.onnx`` in ``weights_dir``. Raises
    ``FileNotFoundError`` if missing — we never auto-export from ``.pt``.
    """
    stem = Path(spec.weight_file).stem
    onnx_path = Path(weights_dir) / f"{stem}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX weights not found for {spec.key}: expected {onnx_path}. "
            f"Export with LibreYOLO's .export() and place it in the weights dir."
        )
    return onnx_path


def load_onnx(key: str, weights_dir: str | Path, device: str = "auto"):
    """Load a LibreYOLO ONNX backend by registry key.

    Returns the backend instance and its spec.
    """
    from libreyolo import LibreYOLO

    spec = get_spec(key)
    onnx_path = resolve_onnx_weights(spec, weights_dir)
    backend = LibreYOLO(model_path=str(onnx_path), device=device)
    return backend, spec


def count_onnx_params(onnx_path: str | Path) -> float:
    """Count parameters in an ONNX model by summing initializer sizes.

    Returns params in millions. Requires the ``onnx`` package.
    """
    import numpy as np
    import onnx

    model_proto = onnx.load(str(onnx_path))
    total = 0
    for init in model_proto.graph.initializer:
        dims = list(init.dims)
        if dims:
            total += int(np.prod(dims))
    return total / 1e6


def check_params(model, spec: ModelSpec, tolerance: float = 0.05) -> Optional[str]:
    """Compare measured param count to paper-reported value.

    Returns a warning string if deviation exceeds tolerance, else None.
    """
    if spec.paper_params_m == 0.0:
        return None
    measured = sum(p.numel() for p in model.model.parameters()) / 1e6
    deviation = abs(measured - spec.paper_params_m) / spec.paper_params_m
    if deviation > tolerance:
        return (
            f"Parameter count mismatch for {spec.key}: "
            f"measured {measured:.2f}M vs paper {spec.paper_params_m:.2f}M "
            f"({deviation:.1%} deviation)"
        )
    return None
