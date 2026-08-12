from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tdwm.training.experiment import (
    canonical_hash,
    dataset_signature,
    prepare_run_directory,
)


def _prepare(tmp_path: Path, identity: dict, run_id: str = "fixed"):
    return prepare_run_directory(
        run_root=tmp_path,
        requested_run_id=run_id,
        method="lewm",
        environment="pusht",
        seed=42,
        identity=identity,
    )


def test_canonical_hash_does_not_depend_on_mapping_order() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_dataset_signature_changes_when_file_changes(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.h5"
    dataset.write_bytes(b"first")
    before = dataset_signature(dataset)
    dataset.write_bytes(b"second-version")
    after = dataset_signature(dataset)

    assert before["size_bytes"] != after["size_bytes"]


def test_declared_dataset_hash_is_verified(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.h5"
    dataset.write_bytes(b"dataset")
    digest = hashlib.sha256(b"dataset").hexdigest()

    assert dataset_signature(dataset, digest)["sha256"] == digest
    with pytest.raises(ValueError, match="mismatch"):
        dataset_signature(dataset, "0" * 64)


def test_run_directory_rejects_a_different_experiment(tmp_path: Path) -> None:
    _, run_dir, fingerprint = _prepare(tmp_path, {"seed": 1})
    (run_dir / "metadata.json").write_text(
        json.dumps({"experiment_fingerprint": fingerprint}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="different experiment"):
        _prepare(tmp_path, {"seed": 2})


def test_nonempty_run_without_metadata_is_not_reused(tmp_path: Path) -> None:
    _, run_dir, _ = _prepare(tmp_path, {"seed": 1})
    (run_dir / "orphan.ckpt").write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="no metadata"):
        _prepare(tmp_path, {"seed": 1})


def test_run_id_cannot_escape_the_run_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path component"):
        _prepare(tmp_path, {"seed": 1}, run_id="../outside")
