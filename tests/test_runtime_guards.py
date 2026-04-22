"""Regression tests for environment guards added to the benchmark harness."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from va_bench import benchmark, hardware


def test_resolve_direct_url_commit_reads_pep610_metadata(monkeypatch):
    class FakeDist:
        def read_text(self, name: str) -> str:
            assert name == "direct_url.json"
            return '{"vcs_info": {"commit_id": "abc123"}}'

    monkeypatch.setattr(hardware.importlib_metadata, "distribution", lambda _: FakeDist())

    assert hardware._resolve_direct_url_commit("libreyolo") == "abc123"


def test_assert_supported_pytorch_model_api_raises_clear_error():
    incompatible = SimpleNamespace(_forward=lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError) as exc:
        benchmark._assert_supported_pytorch_model_api(incompatible)

    message = str(exc.value)
    assert "_preprocess" in message
    assert "_postprocess" in message
    assert benchmark.SUPPORTED_LIBREYOLO_COMMIT in message


def test_assert_onnx_cuda_provider_available_rejects_cpu_only_runtime(monkeypatch):
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["AzureExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(RuntimeError) as exc:
        benchmark._assert_onnx_cuda_provider_available()

    assert "CUDAExecutionProvider" in str(exc.value)


def test_assert_onnx_cuda_provider_available_accepts_cuda_runtime(monkeypatch):
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    benchmark._assert_onnx_cuda_provider_available()
