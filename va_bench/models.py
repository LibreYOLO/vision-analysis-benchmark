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
