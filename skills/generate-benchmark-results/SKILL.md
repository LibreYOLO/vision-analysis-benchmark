---
name: generate-benchmark-results
description: >-
  Run supported Vision Analysis benchmarks with va-bench, generate benchmark
  result JSONs, and sanity-check the emitted output. Use when asked to
  benchmark one or more models, smoke-test the harness, or produce a result
  file that can later be submitted through the vision-analysis repo.
---

# Generate Benchmark Results

Use this skill only in `vision-analysis-benchmark`.

## Scope

- Active backends: `pytorch`, `onnx`
- Output contract: `va.submission.v1`
- This skill is for generating result JSONs, not for opening website PRs

## Dataset & protocol (canonical)

Accuracy and latency are measured on **different splits, on purpose**:

- **Accuracy (mAP) — full val2017 (5000 images), measured once per model + precision.**
  mAP is hardware-independent: the same weights + precision + images produce the same
  detections on any device. So establish a model's accuracy with **one** run over the full
  5000 COCO val2017 images, in the format/precision that preserves its best performance
  (typically PyTorch fp32). Reuse that mAP for every row of the model at that precision, on
  any hardware. A lower-precision path (e.g. INT8/Hailo) gets its **own** full-val reference
  run, because quantization changes accuracy.
- **Latency — mini500, measured on every hardware × format × precision.** Latency is
  hardware-dependent, so measure it per device; 500 images is enough for stable latency
  percentiles and keeps runs fast on slow hardware (Pi/CPU).
- A published row therefore **combines accuracy from the full-val reference run with timing
  from the mini500 run on that hardware** (the harness records both in one JSON; when they
  come from different runs, take `accuracy` from the full-val run and `timing`/`throughput`
  from the mini500 run).

The latency (mini500) runs use the **same 500-image COCO subset**, not full val2017:

- **Dataset: `LibreYOLO/coco-val2017-mini500`** on the HF LibreYOLO org (deterministic
  first-500 of COCO val2017). Materialize it with `huggingface_hub.snapshot_download`
  (repo_type=`dataset`). Point `--coco-dir` at the result.
- **Layout the harness expects:** `<coco-dir>/annotations/instances_val2017.json` and
  `<coco-dir>/images/val2017/`. The HF annotation file is named
  `instances_val2017_mini500.json`, so symlink it:
  `ln -sf instances_val2017_mini500.json annotations/instances_val2017.json`.
- **Do NOT use `--limit`** to fake a subset: the harness defaults to full 5000 and
  brands `--limit` runs "not a valid submission". The canonical protocol is the
  mini500 dataset over all its images.
- **Protocol config** (the harness defaults match): input 640, batch 1, `conf=0.001`,
  `iou=0.6`, `max_det=300`.

The website schema (`vision-analysis/support-matrix.json` + `runtimes.json`) accepts
runtimes `pytorch` (cpu/cuda/mps), `onnx` (cpu/cuda), `tensorrt` (cuda, fp16/fp32).
ncnn and hailo are not in the schema yet. `benchmark.libreyolo_commit` must be in
`supported_libreyolo_identifier` OR be `"unknown"` (a plain PyPI install resolves to
`"unknown"`, which passes validation).

## Device pre-flight (before any run)

A benchmark's latency is only valid on an **uncontended** device, and the GPU box may
be **shared** — assume someone else could be training on it. Checking occupancy first is
not just courtesy; it is **correctness**.

- **CUDA runs — check GPU occupancy first.** Run
  `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`
  and `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader`.
  If a compute process that is **not yours** is on the GPU (a real training/inference job,
  or a large `used_gpu_memory` / non-idle utilization), **do not launch** — surface it to
  the user and wait. Launching anyway does two kinds of harm: it competes for VRAM and can
  **OOM-kill the other job** (and yours), and contention **silently corrupts the latency
  numbers** the benchmark exists to measure. Proceed only on a genuinely free GPU (idle
  util; only OS/desktop graphics apps present, which show `[N/A]` memory).
- **VRAM headroom.** Confirm free VRAM comfortably exceeds the largest variant's footprint
  before a multi-model campaign, so a big model doesn't OOM mid-run.
- **Keep the machine quiet during the run.** Don't run other heavy work — including on the
  same CPU/edge device — while a latency benchmark is measuring; it skews the timings.
- On Windows, `nvidia-smi` is often not on the shell PATH; use the full path
  (`/c/Windows/System32/nvidia-smi.exe`).

## Workflow

1. Read the current support surface before running anything:
   - `README.md`
   - `va_bench/cli.py`
   - `va_bench/output.py`

2. Confirm the requested model/backend pair is supported.
   - For ONNX, require `--weights-dir`.
   - Do not improvise TensorRT, OpenVINO, or other backends from this repo.
   - On NVIDIA machines, prefer a clean venv with `PIP_USER=0` and `PYTHONNOUSERSITE=1`.
   - Make sure the installed `libreyolo` matches the pinned support commit in `README.md`.
   - Make sure PyTorch CUDA wheels match the host driver/runtime before starting a long run.

3. Device pre-flight (see "Device pre-flight" above).
   - CUDA: confirm the GPU is free — no other user's compute process — before launching.
   - Confirm VRAM headroom for the largest variant in the run.

4. Run the harness.
   - Example:
     `python -m va_bench.cli run --models yolov9t --coco-dir /path/to/coco --output-dir ./results --format pytorch`
   - Add `--device`, `--weights-dir`, `--conf`, `--iou`, and `--max-det` only when needed.

5. Inspect the emitted JSON manually.
   - Use `references/output-checklist.md`.
   - Make sure the file contains the real runtime/provider/device and the actual input size used by the run.
   - **Latency sanity:** within a model family, a larger variant should not measure
     *faster* than a smaller one. If it does — or a run OOM'd — suspect device contention:
     re-check GPU occupancy and re-run on a free device. Do not publish contaminated latencies.

6. If the user is changing harness code, run:
   - `pytest -q`

7. If the user wants to publish the result, hand the emitted JSON off to the `vision-analysis` repo submission flow rather than inventing an upload path here.

## Hard rules

- Do not call a benchmark complete if the CLI prints an exception for the model.
- Do not accept ONNX runs without a real exported `.onnx` file in `--weights-dir`.
- Do not attempt `onnx + cuda` if ONNX Runtime does not expose `CUDAExecutionProvider`.
- Do not hand-edit result JSONs unless the user explicitly asks; regenerate them from the harness instead.
- Do not start a CUDA benchmark while another user's compute process is on the GPU — wait for a free device.
- Do not publish latencies from a run that shared the GPU with other compute; non-monotonic latency-vs-size within a family is the tell.

## Reference files

- Output review checklist: `references/output-checklist.md`
