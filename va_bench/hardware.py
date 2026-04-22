"""
Hardware and software detection for Vision Analysis benchmarks.

Detects GPU (NVIDIA, Apple Silicon, RPi5), CPU info, RAM, and package versions.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def get_gpu_info() -> dict[str, Any]:
    """Detect GPU/accelerator and return metadata."""
    gpu_name = "CPU"
    memory_gb = 0.0
    driver = "N/A"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        parts = result.stdout.strip().split(", ")
        if len(parts) >= 3:
            gpu_name = parts[0]
            mem_str = parts[1]
            if "MiB" in mem_str or "MB" in mem_str:
                memory_gb = float(mem_str.split()[0]) / 1024
            driver = parts[2]
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Check for Raspberry Pi
        try:
            with open("/proc/device-tree/model") as f:
                gpu_name = f.read().strip().replace("\x00", "")
        except (FileNotFoundError, OSError):
            if platform.system() == "Darwin":
                gpu_name = _get_mac_chip()
            else:
                gpu_name = "CPU"

    import torch
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"

    return {
        "gpu": gpu_name,
        "gpu_memory_gb": round(memory_gb, 1),
        "driver_version": driver,
        "cuda_version": cuda_version,
    }


def _get_mac_chip() -> str:
    """Get Apple Silicon chip name on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return platform.processor() or "Apple Silicon"


def get_cpu_info() -> tuple[str, int]:
    """Return (cpu_model, core_count)."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                lines = f.readlines()
            model_lines = [l for l in lines if "model name" in l]
            cpu_model = model_lines[0].split(":")[1].strip() if model_lines else "Unknown"
            cpu_cores = len([l for l in lines if "processor" in l])
        elif platform.system() == "Darwin":
            cpu_model = _get_mac_chip()
            cpu_cores = os.cpu_count() or 0
        else:
            cpu_model = platform.processor() or "Unknown"
            cpu_cores = os.cpu_count() or 0
    except Exception:
        cpu_model = "Unknown"
        cpu_cores = os.cpu_count() or 0

    return cpu_model, cpu_cores


def get_system_memory_gb() -> int:
    """Return total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 ** 3)
    except ImportError:
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            return int(line.split()[1]) // (1024 ** 2)
            except Exception:
                pass
    return 0


def get_software_info() -> dict[str, str]:
    """Return Python, PyTorch, LibreYOLO, and (if present) ONNX Runtime versions."""
    import torch

    libreyolo_version = "unknown"
    libreyolo_commit = "unknown"
    try:
        import libreyolo
        libreyolo_version = getattr(libreyolo, "__version__", "dev")
        libreyolo_commit = _resolve_git_commit(getattr(libreyolo, "__file__", None)) or "unknown"
    except ImportError:
        pass

    onnxruntime_version = "not-installed"
    try:
        import onnxruntime
        onnxruntime_version = onnxruntime.__version__
    except ImportError:
        pass

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "libreyolo": libreyolo_version,
        "libreyolo_commit": libreyolo_commit,
        "onnxruntime": onnxruntime_version,
    }


def _resolve_git_commit(module_file: str | None) -> str | None:
    """Best-effort git commit lookup for editable/local installs."""
    if not module_file:
        return None

    path = Path(module_file).resolve()
    for parent in path.parents:
        if not (parent / ".git").exists():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            return commit or None
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    return None


def get_runtime_device_name(device_type: str) -> str:
    """Map torch device type to a human-readable runtime device string."""
    return {"cuda": "gpu", "mps": "gpu", "cpu": "cpu"}.get(device_type, device_type)


def collect_all() -> dict[str, Any]:
    """Collect all hardware and software metadata."""
    gpu_info = get_gpu_info()
    cpu_model, cpu_cores = get_cpu_info()
    ram_gb = get_system_memory_gb()
    software = get_software_info()

    return {
        "hardware": {
            "gpu": gpu_info["gpu"],
            "gpu_memory_gb": gpu_info["gpu_memory_gb"],
            "driver_version": gpu_info["driver_version"],
            "cuda_version": gpu_info["cuda_version"],
            "cpu": cpu_model,
            "cpu_cores": cpu_cores,
            "ram_gb": ram_gb,
        },
        "software": software,
    }
