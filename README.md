# Vision Analysis Benchmark

Produces benchmark JSONs for `visionanalysis.org`.

The harness records the exact LibreYOLO version and commit in each emitted
result JSON. For public submissions, validate the result JSON rather than
assuming the local editable install points at the intended branch.

## Model / Backend Support

The registry covers 77 model variants from two sources: 67 open LibreYOLO
detection variants plus 10 Ultralytics-source models (see
[Ultralytics Source](#ultralytics-source-subprocess-driver) below).

| Family | Variants | PyTorch | ONNX | Notes |
|---|---:|---:|---:|---|
| YOLOX | 6 | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| YOLOv9 | 4 | Yes | Yes | Standard NMS variants. |
| YOLOv9-E2E | 4 | Yes | Yes | End-to-end variants. |
| RF-DETR | 4 | Yes* | Yes | `PyTorch` requires optional RF-DETR dependencies. |
| RT-DETR | 7 | Yes | Yes | Includes ResNet and HGNetv2 variants. |
| RT-DETRv2 | 5 | Yes | Yes |  |
| RT-DETRv4 | 4 | Yes | Yes |  |
| DEIM | 5 | Yes | Yes |  |
| DEIMv2 | 8 | Yes | Yes |  |
| D-FINE | 5 | Yes | Yes |  |
| PicoDet | 3 | Yes | Yes |  |
| EC / EdgeCrafter | 4 | Yes | Yes |  |
| RTMDet | 5 | Yes | Yes |  |
| YOLO-NAS | 3 | Registered | Registered | Replaced DAMO-YOLO in the registry (c2d7c3d); no benchmark results produced through this harness yet. |

`TensorRT` (FP16) is supported for any registered variant for which you supply a
LibreYOLO-built `.engine` (plus its `.engine.json` sidecar) in `--weights-dir`,
on the same backend path as `ONNX`.

## Ultralytics Source (subprocess driver)

The registry also lists 10 Ultralytics-source models — YOLO11 n/s/m/l/x and
YOLOv8 n/s/m/l/x (keys `uly-yolo11n` … `uly-yolov8x`). They are benchmarked
through a standalone subprocess driver; the harness process never imports the
`ultralytics` package (AGPL isolation boundary).

| Family | Variants | PyTorch | ONNX | TensorRT | Notes |
|---|---:|---:|---:|---:|---|
| YOLO11 (`uly-yolo11*`) | 5 | Yes (driver) | No | No | `uly-yolo11n` validated end-to-end (see below). |
| YOLOv8 (`uly-yolov8*`) | 5 | Yes (driver) | No | No | Same driver code path; not individually validated yet. |

- **PyTorch only**, batch=1. Official `.pt` weights are pre-downloaded from
  GitHub release assets by `drivers/ultralytics/fetch_weights.py`; the driver
  itself never downloads anything.
- Runs in a separate pinned venv (`.venv-ultralytics/`,
  `ultralytics==8.4.60`) via `drivers/ultralytics/uly_driver.py`. The harness
  exchanges JSON files with it over a subprocess boundary; point
  `--uly-python` (or the `VA_ULY_PYTHON` env var) at the driver venv's
  python.
- **Telemetry is disabled and proven off at runtime**: the driver pre-writes
  a complete `settings.json` with `sync: false`, runs with
  `YOLO_OFFLINE=true`, and blocks all sockets for the entire run — any
  network attempt fails the run loudly. Audit:
  [docs/ULTRALYTICS_TELEMETRY.md](docs/ULTRALYTICS_TELEMETRY.md); driver
  details: [drivers/ultralytics/README.md](drivers/ultralytics/README.md).
- Default NMS IoU is per source: 0.6 for LibreYOLO models, 0.7 for
  Ultralytics models (their shipped predict default); override with `--iou`.
  Phase timings come from ultralytics' own per-result `speed` instrumentation;
  total/FPS come from the driver's device-synced wall clock.
- **What has been exercised**: `uly-yolo11n` end-to-end on full COCO val2017
  (RTX 5070 Ti, fp32, conf=0.001 / iou=0.7 / max_det=300) — measured
  mAP50-95 38.7 vs the published 39.5, inside the ±1.0 acceptance gate
  (`predict`-per-image protocol, not their `val` path). The other nine
  registry rows run the identical driver code path but have not been
  individually validated.

## Runtime / Hardware Support

| Runtime | Hardware | Status | Notes |
|---|---|---:|---|
| `PyTorch` | CPU | Yes | Implemented in the harness. |
| `PyTorch` | NVIDIA CUDA | Yes | Full timing path; CUDA VRAM stats are recorded. |
| `PyTorch` | Apple MPS | Partial | Runs through the MPS path, but memory reporting is incomplete. |
| `PyTorch` | AMD / ROCm | No | Not a declared support target for this harness. |
| `ONNX Runtime` | CPU | Yes | Uses `CPUExecutionProvider`. |
| `ONNX Runtime` | NVIDIA CUDA | Yes | Uses `CUDAExecutionProvider` when available. |
| `ONNX Runtime` | Apple GPU / MPS | No | No MPS / CoreML / Metal path in this harness. |
| `ONNX Runtime` | AMD / DirectML / ROCm | No | No provider support in this harness. |
| `TensorRT` | NVIDIA CUDA | Yes | FP16 engines via LibreYOLO's native TensorRT backend. Expects a `.engine` plus its `.engine.json` sidecar in `--weights-dir`. Requires the `tensorrt` package + CUDA. |

## Out Of Scope Today

| Item | Status |
|---|---:|
| OpenVINO benchmarking in this harness | No |
| ncnn benchmarking in this harness | No |

Notes:
- This harness supports fewer things than LibreYOLO itself.
- `va-bench run` is the active path that generates website data.
- `va-bench score` is dormant and currently assumes paired `RTX 5080` and `Raspberry Pi 5` results.

## NVIDIA Note

For community CUDA runs, use a clean virtualenv and avoid user-site contamination:

```bash
python3 -m venv .venv
source .venv/bin/activate
export PIP_USER=0
export PYTHONNOUSERSITE=1
```

- Install the pinned LibreYOLO build shown above, not an arbitrary local branch.
- Match PyTorch CUDA wheels to the host driver/runtime. On CUDA 12.4 hosts, use the `cu124` wheel set if the default install pulls a newer incompatible runtime.
- For `ONNX Runtime + CUDA`, the harness now expects `CUDAExecutionProvider` to be available and fails fast if the runtime only exposes CPU or non-CUDA providers.
