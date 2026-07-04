# Plan: Ultralytics as a second model source in va-bench

Goal: benchmark Ultralytics models (YOLO11, YOLOv8 to start) on visionanalysis.org
alongside LibreYOLO models, with defensible numbers, zero Ultralytics telemetry,
and no AGPL linkage into this MIT harness.

## Scope

**In:** PyTorch format only, batch=1, COCO val2017, GPU + CPU. Models:
YOLO11 n/s/m/l/x and YOLOv8 n/s/m/l/x (10 registry rows).

**Out (for now):** their ONNX/TensorRT export paths, their RT-DETR/v10/v12,
training benchmarks, website ingestion changes (the emitted JSON stays
`va.submission.v1`-compatible; the site decides presentation later).

## Architecture: subprocess driver, separate venv

The harness never imports `ultralytics`. A standalone driver script lives in
`drivers/ultralytics/` and runs in its own venv (`.venv-ultralytics/`,
gitignored). The harness invokes it via subprocess, passing a JSON config and
receiving a JSON result. This one decision buys three things:

1. **License isolation** — no AGPL code linked into the MIT harness process.
   The driver script itself imports ultralytics, so it carries an
   `SPDX-License-Identifier: AGPL-3.0-or-later` header and a README note in
   `drivers/ultralytics/README.md` stating why.
2. **Dependency isolation** — ultralytics' torch/opencv stack never conflicts
   with the harness venv.
3. **Telemetry containment** — settings are killed inside one controlled venv
   (see below), verifiable in one place.

**Version pin:** `ultralytics==8.4.60` — the exact version of the local source
checkout at `C:/Users/Usuario/Documents/github/ultralytics` (audit target ==
runtime). Torch in the driver venv: CUDA build matching the harness venv's
major torch version (RTX 5070 Ti needs cu128 wheels, torch >= 2.7).

## Telemetry policy

- Before the FIRST ever import of ultralytics in the driver venv, the driver
  pre-creates their settings file with `sync: false` (exact file location and
  format per the telemetry audit doc, `docs/ULTRALYTICS_TELEMETRY.md`).
- After import, the driver asserts the effective setting is off and hard-fails
  the run otherwise. The emitted JSON records `telemetry.settings_sync`.
- Weights are pre-downloaded (GitHub releases — GitHub infra, not their
  endpoints). Benchmark runs then need no network; runbook says to run
  campaigns offline.

## Driver contract

Invocation:
```
<venv-ultralytics>/Scripts/python.exe drivers/ultralytics/uly_driver.py \
    --config <run_config.json> --output <driver_result.json>
```

`run_config.json`:
```json
{
  "weights": "<abs path to .pt>",
  "images": ["<abs image paths, ordered>"],
  "imgsz": 640,
  "conf": 0.001,
  "iou": 0.7,
  "max_det": 300,
  "device": "cuda:0",          // or "cpu"
  "warmup_iters": 10,           // 3 on cpu — harness decides, driver obeys
  "half": false
}
```

`driver_result.json`:
```json
{
  "driver_version": "1",
  "ultralytics_version": "8.4.60",
  "torch_version": "...",
  "device_name": "...",
  "telemetry": {"settings_sync": false, "settings_file": "<abs path>"},
  "model": {"params_millions": 2.62, "num_classes": 80},
  "per_image": [
    {
      "image": "000000000139.jpg",       // basename, same order as config
      "detections": [
        {"bbox_xywh": [x, y, w, h], "score": 0.91, "class_index": 0}
      ],
      "speed_ms": {"preprocess": 1.2, "inference": 5.4, "postprocess": 0.9},
      "wall_ms": 8.1
    }
  ],
  "peak_vram_mb": 812.0,
  "peak_ram_mb": 1234.0
}
```

Contract rules:
- `bbox_xywh` is COCO-style absolute pixels in ORIGINAL image coordinates
  (ultralytics `boxes.xyxy` is already rescaled to original; driver converts
  xyxy -> xywh).
- `class_index` is the contiguous 0-79 COCO-80 index. The harness maps to
  91-style category ids with its existing `COCO_80_TO_91` table
  (`benchmark.py::_append_predictions`).
- `speed_ms` phases come from ultralytics' own per-result `speed` dict (their
  instrumentation — nobody can claim we mistimed them); `wall_ms` is the
  driver's device-synced wall clock around the predict call, used for the
  harness `total_ms` stats and FPS. Both are recorded.
- Driver prints progress to stdout; machine-readable data goes ONLY to the
  output JSON file.

## Harness changes (file by file)

- `va_bench/models.py` — `ModelSpec` gains `source: str = "libreyolo"`.
  10 new rows, keys `uly-yolo11n..x`, `uly-yolov8n..x`, weight files
  `yolo11n.pt` etc., input_size 640, paper params/GFLOPs from the Ultralytics
  model docs (verify against the local checkout's docs, do not invent).
- `va_bench/benchmark.py` — new `run_ultralytics_benchmark(...)`: builds the
  image list from `_load_coco`, writes the config JSON to a temp dir, invokes
  the driver, parses the result, feeds `_append_predictions` + `evaluate_coco`
  + `compute_stats` (over `wall_ms`), then `assemble_result`.
- `va_bench/output.py` — `assemble_result` gains optional `source` /
  `impl_provider` / `impl_version` / extra-software args, defaulting to
  today's behavior. For ultralytics rows: `model.source = "ultralytics"`,
  `implementation = {"provider": "ultralytics", "version": "8.4.60"}`,
  `software["ultralytics"] = "8.4.60"`. Everything else keeps its shape.
- `va_bench/cli.py` — routing by `spec.source`; new flag
  `--uly-python <path>` (driver venv python; also honors env
  `VA_ULY_PYTHON`). `va-bench list` shows the source column. Note: cli.py has
  UNCOMMITTED train-throughput edits in the MAIN checkout — this worktree is
  clean; do not copy those edits here.
- `tests/test_ultralytics_backend.py` — unit tests using a FAKE driver (tiny
  python script emitting a canned `driver_result.json`; no ultralytics
  needed), covering: config written correctly, detections mapped to 91-ids,
  timing stats from wall_ms, result JSON has source/provider/version set,
  schema fields intact, driver failure surfaces as a clean error.
- `README.md` — support-matrix row, added ONLY after the live smoke passes
  (repo rule: no support claims that aren't exercised honestly).
- `.gitignore` — `.venv-ultralytics/`, driver weight downloads.

## Eval protocol & acceptance

- Protocol: conf=0.001, iou=0.7, max_det=300, imgsz=640, batch=1, their own
  preprocessing/NMS as shipped, via `model.predict` per image.
- Data: `C:/Users/Usuario/datasets/coco` (full val2017),
  `C:/Users/Usuario/datasets/coco-val2017-mini500` (500-image subset tier).
- Acceptance gate: full-val2017 mAP50-95 for `uly-yolo11n` within ±1.0 of the
  published 39.5. If outside, investigate protocol drift (rect padding, NMS
  settings, class mapping) before publishing anything.
- Published reference mAPs (verify from local checkout docs):
  YOLO11 n 39.5 / s 47.0 / m 51.5 / l 53.4 / x 54.7;
  YOLOv8 n 37.3 / s 44.9 / m 50.2 / l 52.9 / x 53.9.

## Work breakdown

| Phase | Agent | Owns (files) |
|---|---|---|
| A. Telemetry audit | Explore (read-only on local ultralytics checkout) | report only |
| B. Driver + venv + weights + single-image smoke | general-purpose | `drivers/ultralytics/**`, `docs/ULTRALYTICS_TELEMETRY.md`, `.gitignore` |
| C. Harness integration + unit tests (fake driver) | general-purpose | `va_bench/*.py`, `tests/test_ultralytics_backend.py` |
| D. End-to-end: real driver through CLI, mini500 then full val, README row | general-purpose | fixes anywhere + `README.md` |

B and C run in parallel against this contract; D integrates.

## Git policy

- All work in THIS worktree
  (`C:/Users/Usuario/Documents/github/vision-analysis-benchmark-ultralytics`),
  branch `add-ultralytics-source`. Agents do NOT commit or push; the
  orchestrator commits once at the end after review. Never `git add -A`.
