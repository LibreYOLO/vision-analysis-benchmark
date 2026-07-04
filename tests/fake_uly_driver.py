"""Fake ultralytics driver for harness unit tests. Stdlib only.

Mimics the CLI contract of drivers/ultralytics/uly_driver.py
(see PLAN_ultralytics_source.md): reads --config/--output and writes a
canned, contract-valid driver_result.json. Never imports ultralytics or
torch, so it runs under the harness venv python.

Environment knobs (all optional):
    FAKE_ULY_CONFIG_DUMP  Copy the received run config JSON to this path
                          (lets tests assert what the harness wrote).
    FAKE_ULY_FAIL         Print to stderr and exit(3) without output.
    FAKE_ULY_BAD_JSON     Write invalid JSON to the output path.
    FAKE_ULY_TELEMETRY_ON Report telemetry.settings_sync = true.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Deterministic per-image wall clock values, cycled over the image list.
WALL_MS = [10.0, 20.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    dump_path = os.environ.get("FAKE_ULY_CONFIG_DUMP")
    if dump_path:
        Path(dump_path).write_text(json.dumps(config), encoding="utf-8")

    if os.environ.get("FAKE_ULY_FAIL"):
        print("fake driver exploding: CUDA out of imagination", file=sys.stderr)
        sys.exit(3)

    if os.environ.get("FAKE_ULY_BAD_JSON"):
        Path(args.output).write_text("{this is not json", encoding="utf-8")
        return

    per_image = []
    for i, image_path in enumerate(config["images"]):
        per_image.append({
            "image": Path(image_path).name,
            "detections": [
                {"bbox_xywh": [10.0, 20.0, 30.0, 40.0], "score": 0.9, "class_index": 0},
                {"bbox_xywh": [5.0, 5.0, 10.0, 10.0], "score": 0.5, "class_index": 16},
            ],
            "speed_ms": {
                "preprocess": 1.0 + i,
                "inference": 5.0 + i,
                "postprocess": 0.5,
            },
            "wall_ms": WALL_MS[i % len(WALL_MS)],
        })

    result = {
        "driver_version": "1",
        "ultralytics_version": "8.4.60",
        "torch_version": "2.7.0-fake",
        "device_name": "Fake GPU 9000",
        "telemetry": {
            "settings_sync": bool(os.environ.get("FAKE_ULY_TELEMETRY_ON")),
            "settings_file": "C:/fake/ultralytics/settings.json",
        },
        "model": {"params_millions": 2.62, "num_classes": 80},
        "per_image": per_image,
        "peak_vram_mb": 812.0,
        "peak_ram_mb": 1234.0,
    }

    print(f"fake driver: processed {len(per_image)} images")
    Path(args.output).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
