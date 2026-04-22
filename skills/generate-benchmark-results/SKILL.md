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

## Workflow

1. Read the current support surface before running anything:
   - `README.md`
   - `va_bench/cli.py`
   - `va_bench/output.py`

2. Confirm the requested model/backend pair is supported.
   - For ONNX, require `--weights-dir`.
   - Do not improvise TensorRT, OpenVINO, or other backends from this repo.

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
- Do not hand-edit result JSONs unless the user explicitly asks; regenerate them from the harness instead.

## Reference files

- Output review checklist: `references/output-checklist.md`
