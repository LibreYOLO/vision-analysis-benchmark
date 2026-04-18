# Vision Analysis Benchmark

Benchmarking suite that powers [visionanalysis.org](https://visionanalysis.org). Runs object detection models on COCO val2017, measures timing/accuracy/memory, and outputs JSON that the website consumes directly.

Models are loaded through [LibreYOLO](https://github.com/LibreYOLO/libreyolo). Currently supports YOLOX, YOLOv9, and RF-DETR in both PyTorch and ONNX.

## Two modes

| Mode | Command | Status | What it does |
|---|---|---|---|
| **Simple** | `va-bench run` | Active | Benchmark one model on one machine, emit a result JSON. |
| **VA** | `va-bench score` | Dormant | Combine result JSONs from multiple machines into a composite VA v1 Score. |

The simple mode has no hardware requirements — any GPU, any CPU, PyTorch or ONNX. The VA mode expects results from both an RTX 5080 and a Raspberry Pi 5 and is parked for now; the code still works if you want to use it.

## Setup

Base install (PyTorch only):
```
pip install -e .
```

With ONNX support — pick **one** based on the hardware you're benchmarking:
```
pip install -e ".[onnx]"        # CPU ONNX Runtime — use on Raspberry Pi 5
pip install -e ".[onnx-gpu]"    # CUDA ONNX Runtime — use on RTX 5080 / other NVIDIA GPUs
```

You need COCO val2017 somewhere on disk:
```
coco/
├── annotations/
│   └── instances_val2017.json
└── images/
    └── val2017/
        └── *.jpg
```

## Usage

### Simple mode (`va-bench run`)

List available models:
```
va-bench list
```

Benchmark with PyTorch (weights auto-download from HuggingFace):
```
va-bench run --models yolov9t yolox-s --coco-dir /path/to/coco
```

Benchmark with ONNX (user-supplied weights — we do not auto-export):
```
va-bench run --models yolov9t yolox-s --format onnx \
  --weights-dir /path/to/onnx-weights --coco-dir /path/to/coco
```

The `--weights-dir` is expected to contain files named `LibreYOLO9t.onnx`, `LibreYOLOXs.onnx`, etc. — same stems as the PyTorch weights.

Run all 14 models:
```
va-bench run --all --coco-dir /path/to/coco --output-dir ./results
```

Each run produces a JSON named `{model}_{hardware}.json` for PyTorch or `{model}_{hardware}_onnx.json` for ONNX. The JSON matches the `RawBenchmark` schema from the website and includes accuracy (12 COCO metrics), timing, FPS, memory usage, and hardware/software info.

### VA mode (`va-bench score`, dormant)

Once you have result JSONs from both an RTX 5080 and a Raspberry Pi 5 for the same set of models, compute the composite:
```
va-bench score --results-dir ./results
```

The VA v1 Score (0–100) ranks models across 6 metrics: mAP@50, mAP@50-95, mAP_small, FPS on RTX 5080, FPS on RPi5, and mAP/GFLOP. Each metric is min-max normalized across qualifying models, then averaged. A model needs benchmarks on **both** hardware platforms to get a score.

## Notes on the ONNX path

* **User-supplied weights only.** Export your `.onnx` from a `.pt` via LibreYOLO's `BaseModel.export()`. We never auto-export.
* **No phase-split timing.** PyTorch reports preprocess/inference/postprocess separately; ONNX reports a single total per image (the `preprocess_ms`/`inference_ms`/`postprocess_ms` fields are `null`).
* **No VRAM stats.** `torch.cuda.max_memory_allocated()` can't see the ONNX runtime allocator; `peak_vram_mb` is `null` for ONNX. RAM delta still works.
* **Execution provider** is chosen by `--device`: `auto`/`cuda` uses `CUDAExecutionProvider` when available, otherwise falls back to `CPUExecutionProvider`.

## Tests

```
pip install -e ".[dev]"
pytest
```
