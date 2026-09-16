"""Group selection, local weight routing, and incomplete sweep failures."""

from types import SimpleNamespace

import pytest

from va_bench import benchmark, cli, models


def test_group_selection_uses_canonical_family_and_every_size(monkeypatch):
    required = {("yolo9", "t"), ("yolo9_p2", "s"), ("tinyformer", "xl")}
    monkeypatch.setattr(models, "_required_group_pairs", lambda groups: required)
    assert models.list_models(["g0", "g1"]) == ["tinyformer-xl", "yolov9p2-s", "yolov9t"]


def test_group_selection_rejects_silent_coverage_gaps(monkeypatch):
    monkeypatch.setattr(models, "_required_group_pairs", lambda groups: {("tinyformer", "future")})
    with pytest.raises(ValueError, match="future"):
        models.list_models(["g1"])


def test_run_selection_is_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--groups", "g0", "--all", "--coco-dir", "unused"])
    assert exc.value.code == 2


def test_partial_sweep_exits_nonzero_but_finishes_other_models(monkeypatch, tmp_path):
    calls = []
    saved = []

    def run(**kwargs):
        calls.append(kwargs["model_key"])
        if kwargs["model_key"] == "yolov9t":
            raise RuntimeError("bad checkpoint")
        return {}

    monkeypatch.setattr(benchmark, "benchmark_model", run)
    monkeypatch.setattr("va_bench.output.save_result", lambda result, output: saved.append(result))
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--models", "yolov9t", "rfdetr-s", "--coco-dir", str(tmp_path)])
    assert exc.value.code == 1
    assert calls == ["yolov9t", "rfdetr-s"]
    assert len(saved) == 1


def test_local_checkpoint_is_used_and_coco_identity_checked(monkeypatch, tmp_path):
    from libreyolo.utils.general import COCO_CLASSES

    names = dict(enumerate(COCO_CLASSES))
    path = tmp_path / "LibreYOLO9P2t.pt"
    path.write_bytes(b"fixture")
    calls = []
    fake = SimpleNamespace(FAMILY="yolo9_p2", task="detect", size="t", nb_classes=80, names=names)

    def load(**kwargs):
        calls.append(kwargs)
        return fake

    monkeypatch.setattr("libreyolo.LibreYOLO", load)
    assert models.load_model("yolov9p2-t", "cpu", tmp_path)[0] is fake
    assert calls[0]["model_path"] == str(path)
    fake.nb_classes = 10
    with pytest.raises(ValueError, match="80 classes"):
        models.load_model("yolov9p2-t", "cpu", tmp_path)
    fake.nb_classes = 80
    fake.names = dict(reversed(list(names.items())))
    fake.names[0] = "not-person"
    with pytest.raises(ValueError, match="class names/order"):
        models.load_model("yolov9p2-t", "cpu", tmp_path)


def test_missing_p2_does_not_load_random_or_visdrone_weights(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no published COCO checkpoint"):
        models.load_model("yolov9p2-t")


def test_pytorch_dispatch_forwards_local_weight_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "_benchmark_pytorch", lambda *args, **kwargs: kwargs)
    result = benchmark.benchmark_model("yolov9p2-t", tmp_path, weights_dir=tmp_path)
    assert result["weights_dir"] == tmp_path
