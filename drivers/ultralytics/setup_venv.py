"""Create the isolated driver venv (.venv-ultralytics) at the repo root.

Installs pinned dependencies in the order that keeps the CUDA torch build
intact on Blackwell (RTX 5070 Ti needs cu128 wheels, torch >= 2.7):

    1. torch + torchvision from the cu128 index
    2. ultralytics==8.4.60
    3. if step 2's resolver replaced torch with a non-cu128 build, redo step 1
    4. verify torch reports +cu128 and CUDA availability

This script NEVER imports ultralytics (pip install does not import the
package; the only place ultralytics is ever imported is the hardened
uly_driver.py). The final version report uses `pip show`, not an import.

Usage:
    python drivers/ultralytics/setup_venv.py [--base-python <python.exe>]
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_DIR = REPO_ROOT / ".venv-ultralytics"
DEFAULT_BASE_PYTHON = (
    "C:/Users/Usuario/Documents/github/libreyolo/.venv/Scripts/python.exe"
)
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ULTRALYTICS_PIN = "ultralytics==8.4.60"


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def create_venv(base_python: str) -> None:
    if venv_python().is_file():
        print(f"venv already exists at {VENV_DIR}")
        return
    try:
        run([base_python, "-m", "venv", str(VENV_DIR)])
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"base python {base_python} failed ({e}); trying py -3.12",
              flush=True)
        run(["py", "-3.12", "-m", "venv", str(VENV_DIR)])


def install_cu128_torch(py: Path) -> None:
    run([py, "-m", "pip", "install", "torch", "torchvision",
         "--index-url", TORCH_INDEX])


def torch_is_cu128(py: Path) -> bool:
    out = subprocess.run(
        [str(py), "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True)
    version = out.stdout.strip()
    print(f"torch version check: {version or out.stderr.strip()}")
    return out.returncode == 0 and "+cu128" in version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-python", default=DEFAULT_BASE_PYTHON)
    args = parser.parse_args()

    create_venv(args.base_python)
    py = venv_python()

    # Order matters: cu128 torch first, then ultralytics; re-pin torch if the
    # resolver swapped it for a CPU/default-index build.
    install_cu128_torch(py)
    run([py, "-m", "pip", "install", ULTRALYTICS_PIN])
    if not torch_is_cu128(py):
        print("ultralytics install replaced torch; re-installing cu128 build",
              flush=True)
        install_cu128_torch(py)
        if not torch_is_cu128(py):
            sys.exit("ERROR: could not keep a +cu128 torch installed")

    run([py, "-c",
         "import torch; print('torch', torch.__version__, "
         "'cuda_available', torch.cuda.is_available())"])
    # Report ultralytics version WITHOUT importing it (telemetry discipline).
    run([py, "-m", "pip", "show", "ultralytics"])
    print(f"done: {VENV_DIR}")


if __name__ == "__main__":
    main()
