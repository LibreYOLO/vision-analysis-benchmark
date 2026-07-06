# Vision Analysis Benchmark

Produces benchmark JSONs for `visionanalysis.org`.

The harness records the exact LibreYOLO version and commit in each emitted
result JSON. For public submissions, validate the result JSON rather than
assuming the local editable install points at the intended branch.

## Reproducibility

Every emitted JSON carries a `repro` block so a third party can reproduce the
number without guessing how it was run:

- `harness_commit` / `harness_dirty`: the `vision-analysis-benchmark` commit
  the run used, and whether its working tree had uncommitted changes. A run
  from a dirty tree is flagged, not silently passed off as pinned.
- `command` / `argv`: the exact `va-bench` command line, ready to copy and paste.
- `dataset.image_id_sha256`: a verifiable fingerprint of the evaluated image-id
  set. This is the ground truth for which images were scored (mini500 vs full
  val2017 vs any subset), independent of the human-readable `hf_dataset` label.
- `weights`: the weights filename, its SHA-256 (for user-supplied ONNX/TensorRT
  artifacts; `null` for LibreYOLO-managed `.pt` files, which stay pinned by the
  model name plus `benchmark.libreyolo_commit`), and, for ONNX/TensorRT, the
  export manifest read from a `<weights>.json` sidecar plus the detected opset.

The `libreyolo` commit is still the pin for model behavior; the `repro` block
adds the harness side of the story.

### Export manifests (ONNX / TensorRT)

ONNX and TensorRT runs benchmark a user-supplied artifact. To make those
reproducible too, place a `<weights>.json` sidecar next to the artifact
(e.g. `LibreYOLO9t.engine.json`) recording how it was built: the export or
`trtexec` command, source opset, TensorRT version, and builder flags. The
harness embeds it verbatim under `repro.weights.export_manifest`.

### Releases

Tag harness releases so `harness_version` maps to an immutable point in
history: `git tag v2.1.0 && git push --tags`. `harness_commit` remains the
exact pin regardless of tagging.

## Model / Backend Support

The registry covers 70 open LibreYOLO detection variants:

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

`TensorRT` (FP16) is supported for any registered variant for which you supply a
LibreYOLO-built `.engine` (plus its `.engine.json` sidecar) in `--weights-dir`,
on the same backend path as `ONNX`.

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
