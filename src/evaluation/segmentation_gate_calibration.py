"""Reproducible calibration of the segmentation quality gate.

The calibration consumes only the frozen, human-reviewed reliability audit.
It never evaluates the internal test split and never changes annotations.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.segmentation.quality import SegmentationQualityGateConfig

GOOD_LABEL = "GOOD"
NON_GOOD_LABELS = {"AMBIGUOUS", "BAD"}


class GateCalibrationError(RuntimeError):
    """Raised when human-reviewed audit evidence is incomplete or invalid."""


def _result_float(result: Mapping[str, object], key: str) -> float:
    value = result[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise GateCalibrationError(f"Resultado no numérico: {key}")
    try:
        return float(value)
    except ValueError as exc:
        raise GateCalibrationError(f"Resultado no numérico: {key}") from exc


def load_reviewed_audit(path: Path) -> list[dict[str, str]]:
    """Load the frozen reliability audit and validate its human labels."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise GateCalibrationError(f"La auditoría está vacía: {path}")
    labels = {row.get("quality_label", "") for row in rows}
    unknown = labels - {GOOD_LABEL, *NON_GOOD_LABELS}
    if unknown:
        raise GateCalibrationError(f"quality_label inválido: {sorted(unknown)}")
    return rows


def _bool(row: Mapping[str, str], key: str) -> bool:
    value = row.get(key, "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise GateCalibrationError(f"{key} no es booleano en {row.get('image_id', '?')}")


def _float(row: Mapping[str, str], key: str, *, optional: bool = False) -> float | None:
    raw = row.get(key, "").strip()
    if optional and not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise GateCalibrationError(f"{key} no es numérico en {row.get('image_id', '?')}") from exc
    if not math.isfinite(value):
        raise GateCalibrationError(f"{key} no es finito en {row.get('image_id', '?')}")
    return value


def _integer(row: Mapping[str, str], key: str) -> int:
    try:
        value = int(row.get(key, ""))
    except ValueError as exc:
        raise GateCalibrationError(f"{key} no es entero en {row.get('image_id', '?')}") from exc
    if value < 0:
        raise GateCalibrationError(f"{key} es negativo en {row.get('image_id', '?')}")
    return value


def calibrated_status(
    row: Mapping[str, str],
    gate: SegmentationQualityGateConfig,
    *,
    reject_multiple_eligible: bool = False,
) -> str:
    """Replay the production quality-gate rules from persisted audit metrics."""
    if not _bool(row, "segmentation_available") or _bool(row, "fallback_used"):
        return "failed"
    eligible = _integer(row, "eligible_instances")
    if eligible < 1:
        return "failed"
    if eligible > 1:
        if reject_multiple_eligible:
            return "uncertain"
        margin = _float(row, "instance_score_margin", optional=True)
        if margin is None or margin < gate.min_multi_instance_score_margin:
            return "uncertain"
    area = _float(row, "mask_area_ratio")
    bbox_ratio = _float(row, "mask_bbox_ratio")
    normalized_perimeter = _float(row, "normalized_perimeter")
    if area is None or bbox_ratio is None or normalized_perimeter is None:
        raise ValueError(
            "Fila de calibración incompleta: se requieren mask_area_ratio, "
            "mask_bbox_ratio y normalized_perimeter"
        )
    if area >= gate.max_mask_area_ratio:
        return "uncertain"
    if (
        area >= gate.large_mask_area_ratio
        and bbox_ratio < gate.min_large_mask_bbox_ratio
        and normalized_perimeter > gate.max_large_mask_normalized_perimeter
    ):
        return "uncertain"
    return "reliable"


def evaluate_gate(
    rows: Sequence[Mapping[str, str]],
    gate: SegmentationQualityGateConfig,
    *,
    reject_multiple_eligible: bool = False,
) -> dict[str, object]:
    """Score one gate against human labels without changing the reviewed sample."""
    statuses = [
        calibrated_status(row, gate, reject_multiple_eligible=reject_multiple_eligible)
        for row in rows
    ]
    reliable = [row for row, status in zip(rows, statuses, strict=True) if status == "reliable"]
    reliable_good = [row for row in reliable if row["quality_label"] == GOOD_LABEL]
    false_reliable = [row for row in reliable if row["quality_label"] in NON_GOOD_LABELS]
    good_total = sum(row["quality_label"] == GOOD_LABEL for row in rows)
    good_rejected = [
        row
        for row, status in zip(rows, statuses, strict=True)
        if row["quality_label"] == GOOD_LABEL and status != "reliable"
    ]
    return {
        **gate.to_metadata(),
        "total_audited": len(rows),
        "status_counts": dict(sorted(Counter(statuses).items())),
        "reliable_good": len(reliable_good),
        "false_reliable": len(false_reliable),
        "false_reliable_ids": [row["image_id"] for row in false_reliable],
        "good_masks_rejected": len(good_rejected),
        "good_masks_rejected_ids": [row["image_id"] for row in good_rejected],
        "reliability_precision": (len(reliable_good) / len(reliable) if reliable else 0.0),
        "segmentation_coverage": len(reliable) / len(rows),
        "good_mask_coverage": len(reliable_good) / good_total if good_total else 0.0,
    }


CANDIDATE_GRID: dict[str, tuple[float, ...]] = {
    "max_mask_area_ratio": (0.995, 0.999),
    "large_mask_area_ratio": (0.25, 0.35, 0.50, 0.60),
    "min_large_mask_bbox_ratio": (0.60, 0.70, 0.80, 0.90),
    "max_large_mask_normalized_perimeter": (6.0, 7.0, 8.0, 9.0),
    "min_multi_instance_score_margin": (0.20, 0.30, 0.33, 0.40, 0.50),
}


def candidate_gates() -> Iterable[SegmentationQualityGateConfig]:
    """Yield the documented, bounded grid used for audit calibration."""
    for values in product(*CANDIDATE_GRID.values()):
        yield SegmentationQualityGateConfig(*values)


def _distance(
    candidate: SegmentationQualityGateConfig,
    baseline: SegmentationQualityGateConfig,
) -> float:
    """Distancia entre gates con cada eje normalizado por el rango de su rejilla.

    Sin normalizar, el perímetro —que vive en ``(6.0 … 9.0)``— domina el desempate
    frente a los cuatro umbrales que viven en ``[0, 1]``.

    @param {SegmentationQualityGateConfig} candidate Gate candidato.
    @param {SegmentationQualityGateConfig} baseline Gate de referencia.
    @returns {float} Distancia adimensional en ``[0, número de ejes]``.
    """
    left = candidate.to_metadata()
    right = baseline.to_metadata()
    total = 0.0
    for key in left:
        span = CANDIDATE_GRID.get(key)
        width = (max(span) - min(span)) if span else 0.0
        difference = abs(float(left[key]) - float(right[key]))
        total += difference / width if width else difference
    return total


def calibrate_gate(
    rows: Sequence[Mapping[str, str]],
    *,
    baseline: SegmentationQualityGateConfig,
    minimum_precision: float = 0.95,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Choose the least disruptive gate maximizing reviewed GOOD coverage."""
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision debe estar entre 0 y 1")
    results = [evaluate_gate(rows, gate) for gate in candidate_gates()]
    viable = [
        result
        for result in results
        if _result_float(result, "reliability_precision") >= minimum_precision
    ]
    if not viable:
        raise GateCalibrationError(
            f"Ningún candidato alcanza precision mínima {minimum_precision:.3f}"
        )

    def rank(result: Mapping[str, object]) -> tuple[float, ...]:
        gate = SegmentationQualityGateConfig.from_mapping(result)
        return (
            _result_float(result, "reliable_good"),
            _result_float(result, "good_mask_coverage"),
            _result_float(result, "reliability_precision"),
            _result_float(result, "segmentation_coverage"),
            -_distance(gate, baseline),
        )

    ordered = sorted(results, key=rank, reverse=True)
    recommended = max(viable, key=rank)
    return recommended, ordered
