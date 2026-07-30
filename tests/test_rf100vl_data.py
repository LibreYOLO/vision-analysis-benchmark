"""Contracts for RF100-VL version locking and vendored domain identity."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from va_bench import rf100vl_data


class _Version:
    def __init__(self, version_id, calls):
        self.id = f"workspace/project/{version_id}"
        self.version = version_id
        self._calls = calls

    def download(self, *, location, model_format, overwrite):
        self._calls.append((self.version, location, model_format, overwrite))
        for split in ("train", "valid", "test"):
            split_dir = rf100vl_data.Path(location) / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / rf100vl_data.ANNOTATION_FILENAME).write_text(
                "{}",
                encoding="utf-8",
            )
        return SimpleNamespace(location=location)


class _Project:
    def __init__(self, name, versions):
        self.name = name
        self._versions = versions

    def versions(self):
        return list(self._versions)

    def version(self, version_id):
        return next(
            item
            for item in self._versions
            if rf100vl_data.roboflow_version_number(item) == version_id
        )


class _Wrapper:
    def __init__(self, name, versions):
        self.name = name
        self.rf_project = _Project(name, versions)

    def clean_coco_dataset(self, path):
        return None


def test_download_writes_complete_lock_then_replays_exact_version(tmp_path, monkeypatch):
    calls = []
    old = _Version(3, calls)
    wrapper = _Wrapper("z-dataset", [_Version(1, calls), old])
    monkeypatch.setattr(
        rf100vl_data,
        "_named_datasets",
        lambda subset, api_key=None: {"z-dataset": wrapper},
    )

    rf100vl_data.download_version_locked_datasets(tmp_path, verbose=False)
    lock = rf100vl_data.load_version_lock(tmp_path)
    assert lock["selection_complete"] is True
    assert lock["datasets"]["z-dataset"]["version_id"] == 3
    assert lock["downloaded"] == ["z-dataset"]
    assert calls[0][0] == 3

    # A newer workspace version appears. Replay must still choose id 3.
    calls.clear()
    wrapper.rf_project._versions.append(_Version(9, calls))
    rf100vl_data.download_version_locked_datasets(
        tmp_path,
        overwrite=True,
        verbose=False,
    )
    assert calls == [(3, str(tmp_path / "z-dataset"), "coco", True)]


def test_version_number_parses_real_sdk_slug_without_numeric_version_attribute():
    version = SimpleNamespace(id="workspace/project/10")
    assert rf100vl_data.roboflow_version_number(version) == 10


def test_version_number_rejects_non_numeric_slug():
    with pytest.raises(ValueError, match="invalid identity"):
        rf100vl_data.roboflow_version_number(
            SimpleNamespace(id="workspace/project/latest")
        )


def test_version_lock_hash_ignores_mutable_download_progress():
    base = {
        "schema_version": rf100vl_data.VERSION_LOCK_SCHEMA,
        "subset": "rf100vl",
        "source": {"commit": "abc"},
        "datasets": {"a": {"version_id": 1}},
        "downloaded": [],
    }
    finished = json.loads(json.dumps(base))
    finished["downloaded"] = ["a"]
    assert rf100vl_data.version_lock_sha256(base) == rf100vl_data.version_lock_sha256(finished)


def test_domain_manifest_is_complete_and_uses_canonical_names():
    manifest = rf100vl_data.load_domain_manifest()
    domains = list(manifest["datasets"].values())
    assert len(domains) == 100
    assert set(domains) == {
        "Aerial",
        "Document",
        "Flora and Fauna",
        "Industrial",
        "Medical",
        "Other",
        "Sports",
    }
    assert domains.count("Medical") == 13
    assert domains.count("Other") == 15
    assert domains.count("Sports") == 6
