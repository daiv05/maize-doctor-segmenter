"""Selection-threshold calibration helpers for validation-only evaluation."""

from __future__ import annotations

from typing import Mapping, Sequence


class ThresholdCalibrationError(RuntimeError):
    """Raised when threshold candidates do not contain usable validation metrics."""


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ThresholdCalibrationError(f"{name} no es numérico")
    try:
        return float(value)
    except ValueError as exc:
        raise ThresholdCalibrationError(f"{name} no es numérico") from exc


def choose_selection_threshold(
    candidates: Sequence[Mapping[str, object]],
    *,
    minimum_single_leaf_recall: float = 0.97,
) -> Mapping[str, object]:
    """Choose maximum Dice after enforcing the leaf-recall requirement.

    Selection is based only on the single-instance validation subset because a
    selected target mask is not directly comparable with a multi-leaf union.
    """
    if not candidates:
        raise ThresholdCalibrationError("No hay candidatos de umbral")
    if not 0.0 <= minimum_single_leaf_recall <= 1.0:
        raise ValueError("minimum_single_leaf_recall debe estar entre 0 y 1")

    def metrics(candidate: Mapping[str, object]) -> Mapping[str, object]:
        value = candidate.get("single_leaf")
        if not isinstance(value, Mapping):
            raise ThresholdCalibrationError("Faltan métricas single_leaf")
        return value

    viable = [
        candidate
        for candidate in candidates
        if _as_float(metrics(candidate)["mean_leaf_pixel_recall"], "mean_leaf_pixel_recall")
        >= minimum_single_leaf_recall
    ]

    def constrained_rank(candidate: Mapping[str, object]) -> tuple[float, ...]:
        row = metrics(candidate)
        return (
            _as_float(row["mean_dice"], "mean_dice"),
            _as_float(row["mean_leaf_pixel_recall"], "mean_leaf_pixel_recall"),
            _as_float(row["mean_leaf_pixel_precision"], "mean_leaf_pixel_precision"),
            -_as_float(
                row["images_without_detection_rate"], "images_without_detection_rate"
            ),
            _as_float(candidate["selection_threshold"], "selection_threshold"),
        )

    def fallback_rank(candidate: Mapping[str, object]) -> tuple[float, ...]:
        row = metrics(candidate)
        return (
            _as_float(row["mean_leaf_pixel_recall"], "mean_leaf_pixel_recall"),
            _as_float(row["mean_dice"], "mean_dice"),
            -_as_float(
                row["images_without_detection_rate"], "images_without_detection_rate"
            ),
            _as_float(candidate["selection_threshold"], "selection_threshold"),
        )

    return max(viable, key=constrained_rank) if viable else max(candidates, key=fallback_rank)
