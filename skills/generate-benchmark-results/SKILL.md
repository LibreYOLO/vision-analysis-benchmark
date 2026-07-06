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
   - On NVIDIA machines, prefer a clean venv with `PIP_USER=0` and `PYTHONNOUSERSITE=1`.
   - Make sure the installed `libreyolo` matches the pinned support commit in `README.md`.
   - Make sure PyTorch CUDA wheels match the host driver/runtime before starting a long run.
   - For ONNX / TensorRT, drop a `<weights>.json` export-manifest sidecar next to the
     artifact (export/`trtexec` command, source opset, TensorRT version, builder flags).
     The harness embeds it in `repro.weights.export_manifest` so the artifact is reproducible.

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

## Reproducing a published result

Every emitted JSON (harness >= 2.1.0) self-describes how it was produced via the
`repro` block, so reproducing a past number does not require guessing:

1. Open the target submission JSON (in this repo's output, or under
   `submissions/` in the `vision-analysis` repo).
2. Install LibreYOLO at `benchmark.libreyolo_commit` and this harness at
   `repro.harness_commit`.
3. Fetch the eval set: the canonical `LibreYOLO/coco-val2017-mini500` for 500-image
   runs, or full COCO val2017 for 5000-image runs.
4. Run `repro.command` verbatim (adjust only `--coco-dir` to your local path).
5. Confirm `repro.dataset.image_id_sha256` in the new run matches the original.
   This proves you scored the identical images. ONNX / TensorRT reruns also need the
   exported artifact described by `repro.weights` (hash + export manifest).

If `repro.harness_dirty` was `true` on the original, the run came from an unclean
harness tree and is not a clean pin. Treat it as unreproducible and regenerate.

## Hard rules

- Do not call a benchmark complete if the CLI prints an exception for the model.
- Do not accept ONNX runs without a real exported `.onnx` file in `--weights-dir`.
- Do not attempt `onnx + cuda` if ONNX Runtime does not expose `CUDAExecutionProvider`.
- Do not hand-edit result JSONs unless the user explicitly asks; regenerate them from the harness instead.

## Reference files

- Output review checklist: `references/output-checklist.md`
