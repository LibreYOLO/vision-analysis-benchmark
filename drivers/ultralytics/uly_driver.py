"""va-bench subprocess driver for Ultralytics models.

Runs inference with the `ultralytics` package (AGPL-3.0) in an isolated
subprocess + venv so the MIT harness never links against it. Communicates
with the harness ONLY via JSON files (--config in, --output out).

Telemetry containment (see docs/ULTRALYTICS_TELEMETRY.md for the audit):
before `ultralytics` is ever imported this process
  1. sets YOLO_OFFLINE=true / YOLO_AUTOINSTALL=false and drops
     ULTRALYTICS_API_KEY from the environment,
  2. points YOLO_CONFIG_DIR at drivers/ultralytics/config and pre-writes a
     COMPLETE 19-key settings.json (sync=false, fixed non-MAC uuid) at
     config/Ultralytics/settings.json — a partial file would be silently
     reset to defaults with sync=true,
  3. replaces socket.socket / socket.create_connection with stubs that raise,
     so ANY network attempt (including the GA4 beacon's raw urllib POST)
     fails loudly for the entire run.
After import it asserts SETTINGS["sync"] is False and that the active
settings file lives under our config dir, refusing to run otherwise.

Invocation:
    python uly_driver.py --config run_config.json --output driver_result.json

Progress goes to stdout; machine-readable data goes only to --output;
failures exit nonzero with a message on stderr.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file imports `ultralytics` (AGPL-3.0-or-later) and is therefore
# licensed AGPL-3.0-or-later. It is an isolation boundary: the rest of the
# repository does not import this file or ultralytics; the harness talks to
# this script over a subprocess + JSON-file interface only.

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

DRIVER_VERSION = "1"
DRIVER_DIR = Path(__file__).resolve().parent
CONFIG_DIR = DRIVER_DIR / "config"           # YOLO_CONFIG_DIR
SCRATCH_DIR = DRIVER_DIR / "scratch"         # datasets/weights/runs sandboxes
SETTINGS_VERSION = "0.0.6"
# Fixed placeholder identity: sha256("va-bench-driver"). Pre-writing it means
# ultralytics never generates its MAC-derived machine id.
FIXED_UUID = hashlib.sha256(b"va-bench-driver").hexdigest()

NETWORK_BLOCKED_MSG = "network blocked by va-bench driver"


def _fail(msg: str, code: int = 1) -> None:
    print(f"uly_driver: ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Hardening — MUST run before the first `import ultralytics`.
# ---------------------------------------------------------------------------

def write_settings_file() -> Path:
    """Pre-write the complete 19-key Ultralytics settings.json (sync off).

    Ultralytics appends an "Ultralytics" subdir to YOLO_CONFIG_DIR, and its
    _validate_settings resets ANY partial/atypical file back to defaults
    (sync=true) — so this file must carry the exact full key set with correct
    types and settings_version "0.0.6". Completeness is load-bearing.
    """
    datasets_dir = SCRATCH_DIR / "datasets"
    weights_dir = SCRATCH_DIR / "weights"
    runs_dir = SCRATCH_DIR / "runs"
    for d in (datasets_dir, weights_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    settings = {  # all 19 keys — do not trim
        "settings_version": SETTINGS_VERSION,
        "datasets_dir": str(datasets_dir),
        "weights_dir": str(weights_dir),
        "runs_dir": str(runs_dir),
        "uuid": FIXED_UUID,
        "sync": False,
        "api_key": "",
        "openai_api_key": "",
        "clearml": False,
        "comet": False,
        "dvc": False,
        "hub": False,
        "mlflow": False,
        "neptune": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": False,
        "vscode_msg": False,
        "openvino_msg": False,
    }
    settings_path = CONFIG_DIR / "Ultralytics" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings_path


def install_socket_guard() -> None:
    """Block ALL network at the socket level for the rest of the process.

    The GA4 beacon uses raw urllib (not requests), so patching must happen at
    the socket layer. The guard is permanent for the run — no escape hatch.
    """

    def _blocked(*_args, **_kwargs):
        raise RuntimeError(NETWORK_BLOCKED_MSG)

    class _BlockedSocket(socket.socket):
        def __init__(self, *_args, **_kwargs):  # noqa: D401
            raise RuntimeError(NETWORK_BLOCKED_MSG)

    socket.socket = _BlockedSocket
    socket.create_connection = _blocked


def install_hardening() -> Path:
    """Full pre-import hardening, in the audited order. Returns settings path."""
    # (a) environment switches — kills import-time DNS probe, autoinstall,
    #     and the API-key-armed platform channel.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    # (b) private config dir + complete settings.json with sync=false.
    os.environ["YOLO_CONFIG_DIR"] = str(CONFIG_DIR)
    settings_path = write_settings_file()
    # (c) socket-level network block, on for the whole run.
    install_socket_guard()
    return settings_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run(config: dict, expected_settings_path: Path) -> dict:
    # (d) only now is ultralytics imported, into a hardened process.
    import torch  # noqa: E402
    import ultralytics  # noqa: E402
    import ultralytics.utils  # noqa: E402

    settings = ultralytics.utils.SETTINGS
    settings_file = getattr(ultralytics.utils, "SETTINGS_FILE", None)
    if settings_file is None:
        _fail("ultralytics.utils.SETTINGS_FILE not found - version drift, "
              "cannot verify telemetry isolation")
    settings_file = Path(str(settings_file)).resolve()

    if settings["sync"] is not False:
        _fail(f"SETTINGS['sync'] is {settings['sync']!r}, expected False - "
              "telemetry isolation failed, refusing to run")
    try:
        settings_file.relative_to(CONFIG_DIR.resolve())
    except ValueError:
        _fail(f"active settings file {settings_file} is not under "
              f"{CONFIG_DIR} - YOLO_CONFIG_DIR was not honored, refusing to run")
    if settings_file != expected_settings_path.resolve():
        _fail(f"active settings file {settings_file} != pre-written "
              f"{expected_settings_path} - refusing to run")
    print(f"telemetry: sync={settings['sync']} settings_file={settings_file}",
          flush=True)

    from ultralytics import YOLO  # noqa: E402

    weights = str(config["weights"])
    images = [str(p) for p in config["images"]]
    imgsz = int(config["imgsz"])
    conf = float(config["conf"])
    iou = float(config["iou"])
    max_det = int(config["max_det"])
    device = str(config["device"])
    warmup_iters = int(config["warmup_iters"])
    half = bool(config.get("half", False))

    if not images:
        _fail("config['images'] is empty")

    use_cuda = device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        _fail(f"config requests device {device!r} but torch.cuda.is_available() "
              "is False")
    torch_device = torch.device(device)

    torch.set_grad_enabled(False)

    print(f"loading model: {weights}", flush=True)
    model = YOLO(weights)

    predict_kwargs = dict(
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        half=half,
        verbose=False,
        save=False,
    )

    def _sync():
        if use_cuda:
            torch.cuda.synchronize(torch_device)

    # Warmup on the first image, then reset the VRAM peak counter so the
    # reported peak reflects steady-state inference.
    print(f"warmup: {warmup_iters} iters on {Path(images[0]).name}", flush=True)
    for _ in range(warmup_iters):
        model.predict(images[0], **predict_kwargs)
    _sync()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(torch_device)

    per_image = []
    n = len(images)
    for i, img_path in enumerate(images):  # given order, no reordering
        _sync()
        t0 = time.perf_counter()
        results = model.predict(img_path, **predict_kwargs)
        _sync()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        res = results[0]
        boxes = res.boxes
        detections = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()   # original-image coords
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes):
                detections.append({
                    "bbox_xywh": [float(x1), float(y1),
                                  float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                    "class_index": int(c),
                })
        speed = res.speed or {}
        entry = {
            "image": Path(img_path).name,
            "detections": detections,
            "speed_ms": {
                "preprocess": float(speed.get("preprocess") or 0.0),
                "inference": float(speed.get("inference") or 0.0),
                "postprocess": float(speed.get("postprocess") or 0.0),
            },
            "wall_ms": wall_ms,
        }
        per_image.append(entry)
        print(f"[{i + 1}/{n}] {entry['image']}: {len(detections)} dets, "
              f"{wall_ms:.1f} ms", flush=True)

    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated(torch_device) / (1024 ** 2)
        device_name = torch.cuda.get_device_name(torch_device)
    else:
        peak_vram_mb = None
        device_name = platform.processor() or "cpu"

    import psutil  # ultralytics dependency, present in the driver venv
    mem = psutil.Process().memory_info()
    # Windows exposes the true high-water mark as peak_wset; fall back to rss.
    peak_ram_mb = float(getattr(mem, "peak_wset", 0) or mem.rss) / (1024 ** 2)

    params = sum(p.numel() for p in model.model.parameters())

    return {
        "driver_version": DRIVER_VERSION,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device_name": device_name,
        "telemetry": {
            "settings_sync": bool(settings["sync"]),
            "settings_file": str(settings_file),
        },
        "model": {
            "params_millions": round(params / 1e6, 3),
            "num_classes": len(model.names),
        },
        "per_image": per_image,
        "peak_vram_mb": peak_vram_mb,
        "peak_ram_mb": peak_ram_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="va-bench Ultralytics subprocess driver")
    parser.add_argument("--config", required=True,
                        help="path to run_config.json")
    parser.add_argument("--output", required=True,
                        help="path to write driver_result.json")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError) as e:
        _fail(f"cannot read config {args.config}: {e}")

    settings_path = install_hardening()

    try:
        result = run(config, settings_path)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - contract: nonzero exit + stderr
        _fail(f"{type(e).__name__}: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
