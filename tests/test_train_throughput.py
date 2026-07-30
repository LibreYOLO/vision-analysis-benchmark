"""Training-throughput protocol tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from va_bench import train_throughput


def test_bfloat16_is_forwarded_and_recorded(monkeypatch, tmp_path):
    captured = {}
    spec = SimpleNamespace(
        key="fake-n",
        family="fake",
        variant="n",
        input_size=64,
        weight_file="fake.pt",
    )

    class _Model:
        def train(self, **kwargs):
            captured.update(kwargs)
            callback = kwargs["callbacks"]
            for _ in range(kwargs["epochs"]):
                callback.on_train_epoch_end(SimpleNamespace(epoch_seconds=1.0))

    monkeypatch.setattr(train_throughput, "get_spec", lambda key: spec)
    monkeypatch.setattr(
        train_throughput,
        "_train_image_files",
        lambda data: [f"image-{index}.jpg" for index in range(8)],
    )
    monkeypatch.setattr(train_throughput, "load_model", lambda key, device: (_Model(), spec))
    monkeypatch.setattr(train_throughput, "_gpu_util_sampler", lambda stop, out: None)
    monkeypatch.setattr(
        train_throughput,
        "collect_all",
        lambda: {
            "hardware": {"gpu": "NVIDIA GeForce RTX 5070 Ti", "cpu": "test"},
            "software": {"libreyolo": "test", "libreyolo_commit": "abc"},
        },
    )
    monkeypatch.setattr(train_throughput.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_throughput.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(train_throughput.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(train_throughput.torch.cuda, "max_memory_allocated", lambda: 1_000_000)

    result = train_throughput.benchmark_train_throughput(
        "fake-n",
        data="synthetic",
        device="0",
        batch=2,
        warmup_epochs=0,
        measure_epochs=1,
        workers=0,
        amp=True,
        amp_dtype="bfloat16",
        project_dir=tmp_path,
        verbose=False,
    )

    assert captured["amp"] is True
    assert captured["amp_dtype"] == "bfloat16"
    assert result["config"]["precision"] == "bfloat16"
    assert result["config"]["amp_dtype"] == "bfloat16"
    assert result["runtime"]["precision"] == "bfloat16"
    assert result["dataset"]["id"] == "synthetic"
    assert result["dataset"]["projection_valid"] is False
    assert result["repro"]["dataset"]["num_images"] == 8


def test_invalid_amp_dtype_is_rejected(monkeypatch):
    monkeypatch.setattr(
        train_throughput,
        "get_spec",
        lambda key: SimpleNamespace(input_size=64),
    )

    with pytest.raises(ValueError, match="amp_dtype"):
        train_throughput.benchmark_train_throughput(
            "fake-n",
            amp=True,
            amp_dtype="tf32",
            verbose=False,
        )


def test_tiny_dataset_uses_one_partial_batch(monkeypatch, tmp_path):
    captured = {}
    spec = SimpleNamespace(
        key="fake-n",
        family="fake",
        variant="n",
        input_size=64,
        weight_file="fake.pt",
    )

    class _Model:
        def train(self, **kwargs):
            captured.update(kwargs)
            kwargs["callbacks"].on_train_epoch_end(SimpleNamespace(epoch_seconds=1.0))

    monkeypatch.setattr(train_throughput, "get_spec", lambda key: spec)
    monkeypatch.setattr(
        train_throughput,
        "_train_image_files",
        lambda data: ["only-image.jpg"],
    )
    monkeypatch.setattr(
        train_throughput,
        "load_model",
        lambda key, device: (_Model(), spec),
    )
    monkeypatch.setattr(train_throughput, "_gpu_util_sampler", lambda stop, out: None)
    monkeypatch.setattr(
        train_throughput,
        "collect_all",
        lambda: {
            "hardware": {"gpu": "none", "cpu": "test"},
            "software": {"libreyolo": "test", "libreyolo_commit": "abc"},
        },
    )
    monkeypatch.setattr(train_throughput.torch.cuda, "is_available", lambda: False)

    result = train_throughput.benchmark_train_throughput(
        "fake-n",
        data="tiny",
        device="cpu",
        batch=8,
        warmup_epochs=0,
        measure_epochs=1,
        workers=0,
        project_dir=tmp_path,
        verbose=False,
    )

    assert captured["batch"] == 8
    assert result["measurement"]["steps_per_epoch"] == 1
    assert result["measurement"]["images_per_epoch"] == 1
    assert result["measurement"]["img_per_s_median"] == pytest.approx(1.0)
