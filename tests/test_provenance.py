"""Unit tests for reproducibility provenance collection."""

from __future__ import annotations

import json

from va_bench import provenance


def test_image_id_sha256_is_order_independent():
    a = provenance.image_id_sha256([3, 1, 2])
    b = provenance.image_id_sha256([1, 2, 3])
    assert a == b


def test_image_id_sha256_distinguishes_subsets():
    mini = provenance.image_id_sha256(range(500))
    full = provenance.image_id_sha256(range(5000))
    assert mini != full


def test_file_sha256_roundtrip(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"libreyolo weights")
    digest = provenance.file_sha256(f)
    assert digest is not None and len(digest) == 64


def test_file_sha256_missing_returns_none(tmp_path):
    assert provenance.file_sha256(tmp_path / "nope.bin") is None
    assert provenance.file_sha256(None) is None


def test_reconstruct_command_is_copy_pasteable():
    cmd = provenance.reconstruct_command(
        ["run", "--models", "yolov9t", "--coco-dir", "/data/coco", "--format", "pytorch"]
    )
    assert cmd.startswith("va-bench run ")
    assert "--models yolov9t" in cmd


def test_reconstruct_command_quotes_spaces():
    cmd = provenance.reconstruct_command(["run", "--coco-dir", "/my data/coco"])
    assert "'/my data/coco'" in cmd


def test_read_export_manifest_sidecar(tmp_path):
    engine = tmp_path / "LibreYOLO9t.engine"
    engine.write_bytes(b"engine")
    (tmp_path / "LibreYOLO9t.engine.json").write_text(
        json.dumps({"trtexec": "trtexec --fp16", "tensorrt": "10.7"})
    )
    manifest = provenance.read_export_manifest(engine)
    assert manifest is not None
    assert manifest["tensorrt"] == "10.7"


def test_read_export_manifest_absent_returns_none(tmp_path):
    engine = tmp_path / "x.engine"
    engine.write_bytes(b"engine")
    assert provenance.read_export_manifest(engine) is None
    assert provenance.read_export_manifest(None) is None


def test_build_dataset_repro_defaults():
    block = provenance.build_dataset_repro([1, 2, 3], None, None)
    assert block["hf_dataset"] == provenance.DEFAULT_DATASET_ID
    assert block["hf_revision"] == provenance.DEFAULT_DATASET_REVISION
    assert len(block["image_id_sha256"]) == 64


def test_build_dataset_repro_operator_label():
    block = provenance.build_dataset_repro([1], "LibreYOLO/custom", "abc123")
    assert block["hf_dataset"] == "LibreYOLO/custom"
    assert block["hf_revision"] == "abc123"


def test_build_weights_repro_hashes_artifact(tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"not really onnx")
    block = provenance.build_weights_repro(
        weight_file="m.onnx",
        resolved_path=onnx,
        source="user-supplied",
        export_artifact_path=onnx,
    )
    assert block["file"] == "m.onnx"
    assert block["sha256"] is not None
    assert block["source"] == "user-supplied"
    # No sidecar and unreadable onnx -> no manifest/opset keys.
    assert "export_manifest" not in block


def test_build_weights_repro_null_hash_when_unmanaged():
    block = provenance.build_weights_repro(
        weight_file="LibreYOLO9t.pt",
        resolved_path=None,
        source="libreyolo-managed",
    )
    assert block["sha256"] is None
    assert block["source"] == "libreyolo-managed"


def test_run_repro_shape():
    block = provenance.run_repro(
        dataset=provenance.build_dataset_repro([1, 2], None, None),
        weights=provenance.build_weights_repro("w.pt", None, "libreyolo-managed"),
    )
    assert set(block) == {"harness_version", "dataset", "weights"}


def test_harness_git_info_shape():
    info = provenance.harness_git_info()
    assert set(info) == {"commit", "dirty"}
    assert isinstance(info["commit"], str)
