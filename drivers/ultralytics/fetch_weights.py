"""Pre-fetch Ultralytics weight files for the va-bench driver.

Stdlib-only (urllib) and NEVER imports ultralytics — weight downloads happen
in this separate process so the hardened driver never needs network access.
Files come from GitHub release assets (GitHub infrastructure, not Ultralytics
endpoints):

    https://github.com/ultralytics/assets/releases/download/v8.4.0/<name>

Usage:
    python fetch_weights.py yolo11n.pt [yolov8s.pt ...]

Idempotent: existing non-empty files are not re-downloaded. Prints the
sha256 and size of every requested file either way.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> Path:
    dest = WEIGHTS_DIR / name
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"{name}: already present, skipping download")
        return dest
    url = BASE_URL + name
    print(f"{name}: downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "va-bench-fetch"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)  # atomic-ish: no partial file ever sits at dest
    return dest


def main() -> None:
    names = sys.argv[1:]
    if not names:
        print("usage: python fetch_weights.py <weight.pt> [...]",
              file=sys.stderr)
        sys.exit(2)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in names:
        try:
            dest = fetch(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed = True
            continue
        size = dest.stat().st_size
        print(f"{name}: sha256={sha256_of(dest)} size={size} bytes "
              f"({size / (1024 ** 2):.2f} MiB) -> {dest}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
