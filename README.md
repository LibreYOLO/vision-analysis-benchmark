# Vision Analysis Benchmark

Produces benchmark JSONs for `visionanalysis.org`.

Pinned upstream:
`libreyolo @ 1c70efb05a78d1a6e82f29478283883fc9bf38f9`

## Model / Backend Support

| Model Key | Family | PyTorch | ONNX | Notes |
|---|---|---:|---:|---|
| `yolox-nano` | YOLOX | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolox-tiny` | YOLOX | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolox-s` | YOLOX | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolox-m` | YOLOX | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolox-l` | YOLOX | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolox-x` | YOLOX | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolov9t` | YOLOv9 | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolov9s` | YOLOv9 | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolov9m` | YOLOv9 | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `yolov9c` | YOLOv9 | Yes | Yes | ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `rfdetr-n` | RF-DETR | Yes* | Yes | `PyTorch` requires `libreyolo[rfdetr]`. ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `rfdetr-s` | RF-DETR | Yes* | Yes | `PyTorch` requires `libreyolo[rfdetr]`. ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `rfdetr-m` | RF-DETR | Yes* | Yes | `PyTorch` requires `libreyolo[rfdetr]`. ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |
| `rfdetr-l` | RF-DETR | Yes* | Yes | `PyTorch` requires `libreyolo[rfdetr]`. ONNX expects a LibreYOLO-exported `.onnx` with embedded metadata. |

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

## Out Of Scope Today

| Item | Status |
|---|---:|
| RT-DETR in this harness | No |
| TensorRT benchmarking in this harness | No |
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
