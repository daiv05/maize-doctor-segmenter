"""Auditable quality gates for leaf-segmentation results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypedDict

from src.preprocessing.leaf_mask import MaskGeometry, mask_geometry

if TYPE_CHECKING:
    from src.preprocessing.segmented_leaf_processor import SegmentedLeafProcessingResult

QUALITY_GATE_VERSION = "1.0.0"


def _mapping_float(raw: Mapping[str, object], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{key} debe ser numérico")
    try:
        converted = float(value)
    except ValueError as exc:
        raise ValueError(f"{key} debe ser numérico") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{key} debe ser finito")
    return converted


class SegmentationStatus(str, Enum):
    RELIABLE = "reliable"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class _AssessmentCommon(TypedDict):
    number_of_instances: int
    eligible_instances: int
    selected_instance: int | None
    confidence: float | None
    mask_area_ratio: float | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class SegmentationQualityGateConfig:
    """Umbrales calibrados sobre la revisión humana congelada de 42 imágenes.

    Los valores por defecto son exactamente los de ``config/segmentation.yaml``: un
    consumidor que instancie esta clase sin pasar por :func:`from_mapping` obtiene el
    mismo gate que produce la calibración, nunca uno más permisivo.
    """

    max_mask_area_ratio: float = 0.999
    large_mask_area_ratio: float = 0.25
    min_large_mask_bbox_ratio: float = 0.80
    max_large_mask_normalized_perimeter: float = 8.0
    min_multi_instance_score_margin: float = 0.33

    def __post_init__(self) -> None:
        ratios = {
            "max_mask_area_ratio": self.max_mask_area_ratio,
            "large_mask_area_ratio": self.large_mask_area_ratio,
            "min_large_mask_bbox_ratio": self.min_large_mask_bbox_ratio,
            "min_multi_instance_score_margin": self.min_multi_instance_score_margin,
        }
        for name, value in ratios.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1")
        if self.max_mask_area_ratio <= self.large_mask_area_ratio:
            raise ValueError("max_mask_area_ratio debe ser mayor que large_mask_area_ratio")
        if (
            not math.isfinite(self.max_large_mask_normalized_perimeter)
            or self.max_large_mask_normalized_perimeter <= 0.0
        ):
            raise ValueError("max_large_mask_normalized_perimeter debe ser positivo y finito")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> SegmentationQualityGateConfig:
        return cls(
            max_mask_area_ratio=_mapping_float(raw, "max_mask_area_ratio"),
            large_mask_area_ratio=_mapping_float(raw, "large_mask_area_ratio"),
            min_large_mask_bbox_ratio=_mapping_float(raw, "min_large_mask_bbox_ratio"),
            max_large_mask_normalized_perimeter=_mapping_float(
                raw, "max_large_mask_normalized_perimeter"
            ),
            min_multi_instance_score_margin=_mapping_float(
                raw, "min_multi_instance_score_margin"
            ),
        )

    def to_metadata(self) -> dict[str, float]:
        return {
            "max_mask_area_ratio": self.max_mask_area_ratio,
            "large_mask_area_ratio": self.large_mask_area_ratio,
            "min_large_mask_bbox_ratio": self.min_large_mask_bbox_ratio,
            "max_large_mask_normalized_perimeter": self.max_large_mask_normalized_perimeter,
            "min_multi_instance_score_margin": self.min_multi_instance_score_margin,
        }


@dataclass(frozen=True)
class SegmentationAssessment:
    status: SegmentationStatus
    reason: str | None
    number_of_instances: int
    eligible_instances: int
    selected_instance: int | None
    confidence: float | None
    mask_area_ratio: float | None
    metadata: dict[str, object]
    quality_gate_reasons: tuple[str, ...] = ()
    quality_gate_metrics: dict[str, object] = field(default_factory=dict)
    quality_gate_thresholds: dict[str, float] = field(default_factory=dict)
    quality_gate_version: str = QUALITY_GATE_VERSION

    def to_metadata(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "number_of_instances": self.number_of_instances,
            "eligible_instances": self.eligible_instances,
            "selected_instance": self.selected_instance,
            "confidence": self.confidence,
            "mask_area_ratio": self.mask_area_ratio,
            "metadata": self.metadata,
            "quality_gate": {
                "version": self.quality_gate_version,
                "reasons": list(self.quality_gate_reasons),
                "metrics": self.quality_gate_metrics,
                "thresholds": self.quality_gate_thresholds,
            },
        }


def _failure_reason(result: SegmentedLeafProcessingResult) -> str:
    if result.number_of_instances == 0:
        return "no_detection"
    reasons = tuple(trace.reason or "" for trace in result.selection_traces if not trace.eligible)
    if reasons and all("confidence por debajo" in reason for reason in reasons):
        return "low_segmentation_confidence"
    if any(
        marker in reason
        for reason in reasons
        for marker in ("máscara corrupta", "máscara degenerada", "demasiado pequeña")
    ):
        return "invalid_mask"
    return "no_reliable_instance"


def _instance_score_margin(result: SegmentedLeafProcessingResult) -> float | None:
    scores = sorted(
        (
            float(trace.score)
            for trace in result.selection_traces
            if trace.eligible and trace.score is not None
        ),
        reverse=True,
    )
    if len(scores) < 2:
        return None
    return scores[0] - scores[1]


def _quality_gate_metrics(
    result: SegmentedLeafProcessingResult,
) -> tuple[MaskGeometry | None, dict[str, object]]:
    geometry = mask_geometry(result.mask) if result.mask is not None else None
    metrics = geometry.to_metadata() if geometry is not None else {}
    metrics["instance_score_margin"] = _instance_score_margin(result)
    return geometry, metrics


def _assessment(
    *,
    status: SegmentationStatus,
    reason: str | None,
    common: _AssessmentCommon,
    metrics: dict[str, object],
    thresholds: dict[str, float],
    reasons: tuple[str, ...] | None = None,
) -> SegmentationAssessment:
    return SegmentationAssessment(
        status=status,
        reason=reason,
        quality_gate_reasons=reasons if reasons is not None else ((reason,) if reason else ()),
        quality_gate_metrics=metrics,
        quality_gate_thresholds=thresholds,
        **common,
    )


def assess_segmentation_legacy(
    result: SegmentedLeafProcessingResult,
    *,
    reject_multiple_eligible: bool = False,
) -> SegmentationAssessment:
    """Reproduce the pre-quality-gate policy for before/after audits."""
    eligible = sum(trace.eligible for trace in result.selection_traces)
    _, metrics = _quality_gate_metrics(result)
    common: _AssessmentCommon = {
        "number_of_instances": result.number_of_instances,
        "eligible_instances": eligible,
        "selected_instance": result.selected_instance,
        "confidence": result.confidence,
        "mask_area_ratio": result.mask_area_ratio,
        "metadata": result.to_metadata(),
    }
    if result.fallback_used or result.processed_image is None:
        status = SegmentationStatus.FAILED
        reason = _failure_reason(result)
    elif reject_multiple_eligible and eligible > 1:
        status = SegmentationStatus.UNCERTAIN
        reason = "ambiguous_multiple_eligible_leaves"
    elif (
        result.mask is None
        or result.selected_instance is None
        or result.confidence is None
        or eligible != 1
    ):
        status = SegmentationStatus.FAILED
        reason = "incomplete_segmentation_evidence"
    else:
        status = SegmentationStatus.RELIABLE
        reason = None
    assessment = _assessment(
        status=status,
        reason=reason,
        common=common,
        metrics=metrics,
        thresholds={},
    )
    return SegmentationAssessment(
        status=assessment.status,
        reason=assessment.reason,
        number_of_instances=assessment.number_of_instances,
        eligible_instances=assessment.eligible_instances,
        selected_instance=assessment.selected_instance,
        confidence=assessment.confidence,
        mask_area_ratio=assessment.mask_area_ratio,
        metadata=assessment.metadata,
        quality_gate_reasons=assessment.quality_gate_reasons,
        quality_gate_metrics=assessment.quality_gate_metrics,
        quality_gate_thresholds={},
        quality_gate_version="legacy_pre_v1",
    )


def assess_segmentation(
    result: SegmentedLeafProcessingResult,
    *,
    reject_multiple_eligible: bool = False,
    quality_gate: SegmentationQualityGateConfig | None = None,
) -> SegmentationAssessment:
    """Map selector and transparent mask-quality evidence to a status."""
    gate = quality_gate or SegmentationQualityGateConfig()
    eligible = sum(trace.eligible for trace in result.selection_traces)
    geometry, metrics = _quality_gate_metrics(result)
    thresholds = gate.to_metadata()
    common: _AssessmentCommon = {
        "number_of_instances": result.number_of_instances,
        "eligible_instances": eligible,
        "selected_instance": result.selected_instance,
        "confidence": result.confidence,
        "mask_area_ratio": result.mask_area_ratio,
        "metadata": result.to_metadata(),
    }
    if result.fallback_used or result.processed_image is None:
        return _assessment(
            status=SegmentationStatus.FAILED,
            reason=_failure_reason(result),
            common=common,
            metrics=metrics,
            thresholds=thresholds,
        )
    if (
        result.mask is None
        or result.selected_instance is None
        or result.confidence is None
        or eligible < 1
        or geometry is None
    ):
        return _assessment(
            status=SegmentationStatus.FAILED,
            reason="incomplete_segmentation_evidence",
            common=common,
            metrics=metrics,
            thresholds=thresholds,
        )
    if eligible > 1:
        margin = metrics["instance_score_margin"]
        if reject_multiple_eligible:
            return _assessment(
                status=SegmentationStatus.UNCERTAIN,
                reason="ambiguous_multiple_eligible_leaves",
                common=common,
                metrics=metrics,
                thresholds=thresholds,
            )
        if not isinstance(margin, float) or margin < gate.min_multi_instance_score_margin:
            return _assessment(
                status=SegmentationStatus.UNCERTAIN,
                reason="ambiguous_instance_score_margin",
                common=common,
                metrics=metrics,
                thresholds=thresholds,
            )
    geometry_reasons: list[str] = []
    if geometry.mask_area_ratio >= gate.max_mask_area_ratio:
        geometry_reasons.append("excessive_mask_area_ratio")
    if (
        geometry.mask_area_ratio >= gate.large_mask_area_ratio
        and geometry.mask_bbox_ratio < gate.min_large_mask_bbox_ratio
        and geometry.normalized_perimeter > gate.max_large_mask_normalized_perimeter
    ):
        geometry_reasons.append("suspicious_large_mask_geometry")
    if geometry_reasons:
        return _assessment(
            status=SegmentationStatus.UNCERTAIN,
            reason=geometry_reasons[0],
            reasons=tuple(geometry_reasons),
            common=common,
            metrics=metrics,
            thresholds=thresholds,
        )
    return _assessment(
        status=SegmentationStatus.RELIABLE,
        reason=None,
        common=common,
        metrics=metrics,
        thresholds=thresholds,
    )
