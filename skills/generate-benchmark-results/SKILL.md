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

Every published row uses the **same 500-image COCO subset**, not full val2017:

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

3. Run the harness.
   - Example:
     `python -m va_bench.cli run --models yolov9t --coco-dir /path/to/coco --output-dir ./results --format pytorch`
   - Add `--device`, `--weights-dir`, `--conf`, `--iou`, and `--max-det` only when needed.

4. Inspect the emitted JSON manually.
   - Use `references/output-checklist.md`.
   - Make sure the file contains the real runtime/provider/device and the actual input size used by the run.

5. If the user is changing harness code, run:
   - `pytest -q`

6. If the user wants to publish the result, hand the emitted JSON off to the `vision-analysis` repo submission flow rather than inventing an upload path here.

## Hard rules

- Do not call a benchmark complete if the CLI prints an exception for the model.
- Do not accept ONNX runs without a real exported `.onnx` file in `--weights-dir`.
- Do not attempt `onnx + cuda` if ONNX Runtime does not expose `CUDAExecutionProvider`.
- Do not hand-edit result JSONs unless the user explicitly asks; regenerate them from the harness instead.

## Reference files

- Output review checklist: `references/output-checklist.md`
