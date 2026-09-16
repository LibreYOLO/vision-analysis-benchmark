"""Hardware probes must distinguish unavailable VRAM from unified memory."""

from types import SimpleNamespace
from unittest.mock import mock_open

import pytest

from va_bench import hardware
from va_bench.output import detect_hardware_id


@pytest.mark.parametrize("memory,expected", [("[N/A]", None), ("16384 MiB", 16.0)])
def test_nvidia_memory_probe(monkeypatch, memory, expected):
    monkeypatch.setattr(
        hardware.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=f"NVIDIA GB10, {memory}, 580.173.02\n"),
    )
    assert hardware.get_gpu_info()["gpu_memory_gb"] == expected


def test_spark_uses_os_visible_shared_memory(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "get_gpu_info",
        lambda: dict(
            gpu="NVIDIA GB10", gpu_memory_gb=None, driver_version="580", cuda_version="13.0"
        ),
    )
    monkeypatch.setattr(hardware, "get_cpu_info", lambda: ("aarch64", 20))
    monkeypatch.setattr(hardware, "get_system_memory_gb", lambda: 121)
    monkeypatch.setattr(hardware, "get_software_info", lambda: {})
    result = hardware.collect_all()["hardware"]
    assert result["gpu_memory_gb"] is None
    assert result["unified_memory_gb"] == 121
    assert result["memory_type"] == "unified"
    assert result["cpu"] == "NVIDIA GB10 Arm CPU"
    assert detect_hardware_id(result) == "dgx_spark"


def test_discrete_gpu_retains_dedicated_memory(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "get_gpu_info",
        lambda: dict(
            gpu="NVIDIA RTX 5080", gpu_memory_gb=16.0, driver_version="580", cuda_version="13.0"
        ),
    )
    monkeypatch.setattr(hardware, "get_cpu_info", lambda: ("CPU", 16))
    monkeypatch.setattr(hardware, "get_system_memory_gb", lambda: 64)
    monkeypatch.setattr(hardware, "get_software_info", lambda: {})
    result = hardware.collect_all()["hardware"]
    assert result["gpu_memory_gb"] == 16.0
    assert "unified_memory_gb" not in result


def test_arm_cpu_without_x86_model_name(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "builtins.open",
        mock_open(read_data="processor : 0\nCPU implementer : 0x41\nprocessor : 1\n"),
    )
    monkeypatch.setattr(hardware.Path, "read_text", lambda self: "NVIDIA DGX Spark\x00")
    assert hardware.get_cpu_info() == ("NVIDIA DGX Spark", 2)
