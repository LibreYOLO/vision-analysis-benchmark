# Ultralytics driver (subprocess source)

This directory holds the standalone subprocess driver that lets va-bench
benchmark Ultralytics models (YOLO11, YOLOv8). The harness never imports
`ultralytics`; it invokes `uly_driver.py` in a separate, pinned venv and
exchanges JSON files with it.

## Licensing boundary

`uly_driver.py` imports the `ultralytics` package, which is licensed
AGPL-3.0-or-later — so that one file honestly carries an
`SPDX-License-Identifier: AGPL-3.0-or-later` header. The rest of the
repository does not import it: the harness communicates with the driver
exclusively over a subprocess boundary with a JSON-file interface
(`--config` in, `--output` out), which keeps AGPL code out of the MIT
harness process. `fetch_weights.py` and `setup_venv.py` are stdlib/pip-only
and never import ultralytics.

## Telemetry containment

Before importing ultralytics the driver sets `YOLO_OFFLINE=true`,
`YOLO_AUTOINSTALL=false`, unsets `ULTRALYTICS_API_KEY`, points
`YOLO_CONFIG_DIR` at `drivers/ultralytics/config/` with a pre-written
complete 19-key `settings.json` (`sync: false`, fixed non-MAC uuid), and
replaces `socket.socket` so any network attempt raises
`RuntimeError("network blocked by va-bench driver")`. After import it
asserts telemetry is off and refuses to run otherwise. Full audit:
`docs/ULTRALYTICS_TELEMETRY.md`.

Because the driver is fully offline, weights must be fetched beforehand by
`fetch_weights.py` (plain urllib against GitHub release assets).

## Setup

```bash
# 1. create .venv-ultralytics at the repo root (cu128 torch + ultralytics==8.4.60)
python drivers/ultralytics/setup_venv.py

# 2. pre-fetch weights (stdlib-only, no ultralytics import)
python drivers/ultralytics/fetch_weights.py yolo11n.pt
```

## Smoke test

Write a config JSON:

```json
{
  "weights": "<abs path>/drivers/ultralytics/weights/yolo11n.pt",
  "images": ["<abs path to a .jpg>"],
  "imgsz": 640, "conf": 0.001, "iou": 0.7, "max_det": 300,
  "device": "cuda:0", "warmup_iters": 3, "half": false
}
```

then:

```bash
.venv-ultralytics/Scripts/python.exe drivers/ultralytics/uly_driver.py \
    --config cfg.json --output out.json
```

`out.json` must show `telemetry.settings_sync: false` and a
`telemetry.settings_file` under `drivers/ultralytics/config/`. Progress goes
to stdout; results only to the output file; failures exit nonzero.

Gitignored (local-only): `../../.venv-ultralytics/`, `weights/`, `config/`,
`scratch/`.
