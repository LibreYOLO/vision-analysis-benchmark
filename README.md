# Vision Analysis Benchmark

Produces benchmark JSONs for `visionanalysis.org`.

The harness records the exact LibreYOLO version and commit in each emitted
result JSON. For public submissions, validate the result JSON rather than
assuming the local editable install points at the intended branch.

## Model / Backend Support

The registry covers 70 open LibreYOLO detection variants plus 13 instance
segmentation variants:

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
| DAMO-YOLO | 6 | Yes | Yes | Open variants only. |
| RTMDet | 5 | Yes | Yes |  |

YOLO-NAS is intentionally excluded because the weights are gated.

### Instance Segmentation

Segmentation models are separate registry keys (different checkpoints,
`-seg` weight suffix, `task="segment"` in the spec and submission):

| Family | Keys | PyTorch | ONNX | Notes |
|---|---|---:|---:|---|
| RF-DETR-Seg | `rfdetr-seg-{n,s,m,l}` | Yes | Untested | Per-variant native sizes (312/384/432/504). |
| D-FINE-Seg | `dfine-seg-{n,s,m,l,x}` | Yes | Untested | ArgoHA D-FINE-seg heads (Apache-2.0). |
| EC-Seg | `ec-seg-{s,m,l,x}` | Yes | Untested | |

Scoring: the run collects boxes and per-instance masks, then evaluates
COCO twice (bbox + segm). The submission's unprefixed `accuracy.mAP_*`
keys carry **mask mAP** (the task's headline metric); box mAP is kept
under `accuracy.bbox_mAP_*`. Mask upsampling to source resolution runs
inside the model's postprocess and is included in postprocess timing;
RLE encoding for evaluation is outside the timed region.

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
