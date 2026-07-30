"""CLI failure and RF100-VL dataset-name contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from va_bench import cli, rf100vl


def test_rf100vl_cli_exits_nonzero_when_every_model_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        rf100vl,
        "benchmark_model_rf100vl",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("dead campaign")),
    )
    args = SimpleNamespace(
        download=False,
        all=False,
        models=["yolov9t"],
        format="pytorch",
        data_dir=str(tmp_path),
        weights_root=None,
        output_dir=str(tmp_path / "results"),
        device="cpu",
        conf=0.001,
        iou=0.65,
        max_det=500,
        split="test",
        limit=None,
        limit_datasets=None,
        allow_pretrained=True,
        versions=None,
        recipe=None,
        per_dataset_dir=None,
        quiet=True,
        debug=False,
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_rf100vl(args)

    assert exc.value.code == 1


def test_main_accepts_leading_dash_dataset_name(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "cmd_rf100vl_train",
        lambda args: captured.update({"datasets": args.datasets}),
    )

    cli.main(
        [
            "rf100vl-train",
            "--model",
            "yolov9s",
            "--data-dir",
            "data",
            "--weights-root",
            "weights",
            "--datasets",
            "-grccs",
            "aerial-cows",
        ]
    )

    assert captured["datasets"] == ["-grccs", "aerial-cows"]
