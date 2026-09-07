"""Pruebas de la promoción registrada del checkpoint servible."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.training.checkpoint_promotion import (
    FROZEN_BEST_CHECKPOINT_SHA256,
    CheckpointPromotionError,
    expected_best_checkpoint_sha256,
    promote_checkpoint,
    promoted_checkpoint_path,
    read_registry,
    registry_path,
)


def _trained_checkpoint(output_root: Path, payload: bytes) -> Path:
    checkpoint = (
        output_root
        / "leaf_detection"
        / "segmenter"
        / "yolo26n_seg_baseline"
        / "weights"
        / "best.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(payload)
    return checkpoint


def test_frozen_identity_is_used_without_promotion(tmp_path: Path) -> None:
    assert expected_best_checkpoint_sha256(tmp_path) == FROZEN_BEST_CHECKPOINT_SHA256


def test_promotion_copies_and_registers_the_identity(tmp_path: Path) -> None:
    source = _trained_checkpoint(tmp_path, b"pesos-entrenados")
    registry = promote_checkpoint(tmp_path)

    destination = promoted_checkpoint_path(tmp_path)
    assert destination.is_file()
    assert destination.read_bytes() == source.read_bytes()
    assert registry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert registry["status"] == "promoted"
    assert registry["replaces"] is None
    assert read_registry(tmp_path) == registry
    assert registry_path(tmp_path).is_file()


def test_registered_identity_replaces_the_frozen_default(tmp_path: Path) -> None:
    _trained_checkpoint(tmp_path, b"pesos-entrenados")
    registry = promote_checkpoint(tmp_path)
    assert expected_best_checkpoint_sha256(tmp_path) == registry["sha256"]
    assert expected_best_checkpoint_sha256(tmp_path) != FROZEN_BEST_CHECKPOINT_SHA256


def test_environment_override_wins_over_the_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trained_checkpoint(tmp_path, b"pesos-entrenados")
    promote_checkpoint(tmp_path)
    monkeypatch.setenv("SEGMENTATION_EXPECTED_BEST_SHA256", "b" * 64)
    assert expected_best_checkpoint_sha256(tmp_path) == "b" * 64


def test_malformed_override_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEGMENTATION_EXPECTED_BEST_SHA256", "no-es-un-sha")
    with pytest.raises(CheckpointPromotionError):
        expected_best_checkpoint_sha256(tmp_path)


def test_missing_trained_checkpoint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(CheckpointPromotionError):
        promote_checkpoint(tmp_path)


def test_replacing_another_identity_requires_force(tmp_path: Path) -> None:
    _trained_checkpoint(tmp_path, b"primer-entrenamiento")
    first = promote_checkpoint(tmp_path)

    _trained_checkpoint(tmp_path, b"segundo-entrenamiento")
    with pytest.raises(CheckpointPromotionError):
        promote_checkpoint(tmp_path)

    replaced = promote_checkpoint(tmp_path, force=True)
    assert replaced["replaces"] == first["sha256"]
    assert replaced["sha256"] != first["sha256"]
    assert expected_best_checkpoint_sha256(tmp_path) == replaced["sha256"]
