"""High-level segmentation, target selection, masking, fallback, and debug flow."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageOps

from src.preprocessing.leaf_mask import (
    InvalidLeafMaskError,
    apply_leaf_mask,
    binary_mask_image,
    mask_area_ratio,
    mask_bbox,
    mask_centroid,
)
from src.preprocessing.leaf_roi import (
    BoundingBox,
    crop_leaf_region,
    crop_square_centered,
    image_to_rgb,
    normalize_rgb_color,
)
from src.preprocessing.letterbox import letterbox_image, validate_target_size
from src.segmentation.leaf_segmenter import LeafInstance, LeafSegmenter

MASK_BLACK = "mask_black"
BBOX_CROP = "bbox_crop"
CROP_MASK_BLACK = "crop_mask_black"
CROP_MASK_LETTERBOX = "crop_mask_letterbox"
SQUARE_CROP = "square_crop"
SUPPORTED_MASK_PROFILES = frozenset(
    {
        MASK_BLACK,
        BBOX_CROP,
        CROP_MASK_BLACK,
        CROP_MASK_LETTERBOX,
        SQUARE_CROP,
    }
)
FALLBACK_ORIGINAL = "original"
FALLBACK_REJECT = "reject"
SUPPORTED_MASK_FALLBACKS = frozenset({FALLBACK_ORIGINAL, FALLBACK_REJECT})
MASK_PROCESSOR_VERSION = "1.0.0"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} debe ser numérico")
    try:
        converted = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} debe ser numérico") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} debe ser finito")
    return converted


def _ratio(value: object, name: str, *, allow_zero: bool = True) -> float:
    converted = _finite_number(value, name)
    valid_lower = converted >= 0.0 if allow_zero else converted > 0.0
    if not valid_lower or converted > 1.0:
        raise ValueError(f"{name} debe estar entre 0 y 1")
    return converted


def _background_channel(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"canal de background_value inválido: {value!r}")
    try:
        channel = float(value)
    except ValueError as exc:
        raise ValueError(f"canal de background_value inválido: {value!r}") from exc
    if not math.isfinite(channel) or not channel.is_integer():
        raise ValueError(f"canal de background_value inválido: {value!r}")
    return int(channel)


@dataclass(frozen=True)
class LeafMaskProcessorConfig:
    """Configuration for auditable, opt-in leaf-mask preprocessing."""

    processing_profile: str = MASK_BLACK
    confidence_threshold: float = 0.50
    min_mask_area_ratio: float = 0.01
    near_full_warning_ratio: float = 0.98
    area_weight: float = 0.45
    center_weight: float = 0.35
    confidence_weight: float = 0.20
    background_value: tuple[int, int, int] = (0, 0, 0)
    target_size: tuple[int, int] = (224, 224)
    fallback: str = FALLBACK_ORIGINAL

    def __post_init__(self) -> None:
        if self.processing_profile not in SUPPORTED_MASK_PROFILES:
            raise ValueError(
                f"processing_profile desconocido: {self.processing_profile!r}"
            )
        _ratio(self.confidence_threshold, "confidence_threshold")
        _ratio(self.min_mask_area_ratio, "min_mask_area_ratio")
        near_full = _ratio(
            self.near_full_warning_ratio,
            "near_full_warning_ratio",
            allow_zero=False,
        )
        if near_full <= self.min_mask_area_ratio:
            raise ValueError(
                "near_full_warning_ratio debe ser mayor que min_mask_area_ratio"
            )
        weights = (
            _ratio(self.area_weight, "area_weight"),
            _ratio(self.center_weight, "center_weight"),
            _ratio(self.confidence_weight, "confidence_weight"),
        )
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("los pesos de selección deben sumar exactamente 1.0")
        normalize_rgb_color(self.background_value)
        validate_target_size(self.target_size)
        if self.fallback not in SUPPORTED_MASK_FALLBACKS:
            raise ValueError(f"fallback desconocido: {self.fallback!r}")


def _configured_target_size(segmentation: Mapping[str, object]) -> tuple[int, int]:
    """Resuelve el tamaño de entrega al clasificador declarado en la configuración.

    El contrato de tamaño entre segmentador y clasificador debe estar declarado, no
    hardcodeado: ``config/segmentation.yaml`` fija ``target_size`` igual que el
    ``config/dataset.yaml`` del clasificador.

    @param {Mapping[str, object]} segmentation Sección ``segmentation`` del YAML.
    @returns {tuple[int, int]} Tamaño ``(alto, ancho)`` de entrega.
    """
    raw = segmentation.get("target_size")
    if raw is None:
        raise ValueError(
            "config/segmentation.yaml debe declarar segmentation.target_size; "
            "es el contrato de tamaño con el clasificador"
        )
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("target_size debe ser una pareja [alto, ancho]")
    height, width = raw
    if not isinstance(height, int) or not isinstance(width, int):
        raise ValueError("target_size debe contener enteros")
    return (height, width)


def mask_processor_config_from_mapping(
    segmentation: Mapping[str, object],
    *,
    processing_profile: str | None = None,
    confidence_threshold: float | None = None,
    selection_confidence_threshold: float | None = None,
    target_size: tuple[int, int] | None = None,
) -> LeafMaskProcessorConfig:
    """Build mask-output settings from ``config/segmentation.yaml``."""
    raw_weights = segmentation.get("selection_weights")
    if not isinstance(raw_weights, Mapping):
        raise ValueError("selection_weights debe ser un mapping")
    raw_background = segmentation.get("background_value")
    if not isinstance(raw_background, (list, tuple)) or len(raw_background) != 3:
        raise ValueError("background_value debe contener tres canales")
    background = tuple(_background_channel(channel) for channel in raw_background)
    configured_target = target_size or _configured_target_size(segmentation)
    if confidence_threshold is not None and selection_confidence_threshold is not None:
        raise ValueError(
            "use confidence_threshold o selection_confidence_threshold, no ambos"
        )
    configured_selection_confidence = (
        selection_confidence_threshold
        if selection_confidence_threshold is not None
        else confidence_threshold
    )
    if configured_selection_confidence is None:
        configured_selection_confidence = segmentation.get(
            "selection_confidence_threshold",
            segmentation.get("confidence_threshold"),
        )
    if configured_selection_confidence is None:
        raise ValueError("falta selection_confidence_threshold")
    return LeafMaskProcessorConfig(
        processing_profile=(
            processing_profile
            if processing_profile is not None
            else str(segmentation["output_profile"])
        ),
        confidence_threshold=_finite_number(
            configured_selection_confidence,
            "selection_confidence_threshold",
        ),
        min_mask_area_ratio=_finite_number(
            segmentation["min_mask_area_ratio"], "min_mask_area_ratio"
        ),
        near_full_warning_ratio=_finite_number(
            segmentation["near_full_warning_ratio"], "near_full_warning_ratio"
        ),
        area_weight=_finite_number(raw_weights["area"], "selection_weights.area"),
        center_weight=_finite_number(raw_weights["center"], "selection_weights.center"),
        confidence_weight=_finite_number(
            raw_weights["confidence"], "selection_weights.confidence"
        ),
        background_value=(background[0], background[1], background[2]),
        target_size=configured_target,
        fallback=str(segmentation["fallback"]),
    )


@dataclass(frozen=True)
class InstanceSelectionTrace:
    """Per-instance evidence used by the deterministic target selector."""

    source_index: int
    confidence: float | None
    area_ratio: float | None
    relative_area: float | None
    center_proximity: float | None
    score: float | None
    eligible: bool
    reason: str | None

    def to_metadata(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "confidence": self.confidence,
            "area_ratio": self.area_ratio,
            "relative_area": self.relative_area,
            "center_proximity": self.center_proximity,
            "score": self.score,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TargetSelectionResult:
    selected: LeafInstance | None
    selected_bbox: BoundingBox | None
    traces: tuple[InstanceSelectionTrace, ...]
    reason: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SegmentedLeafProcessingResult:
    """Pixels and trace returned by :class:`SegmentedLeafProcessor`."""

    original_image: Image.Image
    mask: Image.Image | None
    masked_image: Image.Image | None
    processed_image: Image.Image | None
    crop_image: Image.Image | None
    bbox: BoundingBox | None
    confidence: float | None
    number_of_instances: int
    selected_instance: int | None
    selection_traces: tuple[InstanceSelectionTrace, ...]
    preprocessing_strategy: str
    fallback_used: bool
    fallback_reason: str | None
    warnings: tuple[str, ...]
    source_image: str | None
    segmenter_metadata: dict[str, object]

    @property
    def mask_area_ratio(self) -> float | None:
        return mask_area_ratio(self.mask) if self.mask is not None else None

    def to_metadata(self) -> dict[str, object]:
        return {
            "source_image": self.source_image,
            **self.segmenter_metadata,
            "confidence": self.confidence,
            "mask_area_ratio": self.mask_area_ratio,
            "number_of_instances": self.number_of_instances,
            "selected_instance": self.selected_instance,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "preprocessing_strategy": self.preprocessing_strategy,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "original_size": list(self.original_image.size),
            "processed_size": (
                list(self.processed_image.size)
                if self.processed_image is not None
                else None
            ),
            "selection": [trace.to_metadata() for trace in self.selection_traces],
            "warnings": list(self.warnings),
            "processor_version": MASK_PROCESSOR_VERSION,
        }


@dataclass(frozen=True)
class _EligibleInstance:
    instance: LeafInstance
    bbox: BoundingBox
    area_ratio: float
    center_proximity: float


def _center_proximity(mask: Image.Image, image_size: tuple[int, int]) -> float:
    centroid = mask_centroid(mask)
    if centroid is None:
        return 0.0
    width, height = image_size
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    distance = math.hypot(centroid[0] - center_x, centroid[1] - center_y)
    maximum = math.hypot(max(center_x, 0.5), max(center_y, 0.5))
    return max(0.0, min(1.0, 1.0 - distance / maximum))


def select_target_leaf(
    instances: Sequence[LeafInstance],
    image_size: tuple[int, int],
    config: LeafMaskProcessorConfig,
) -> TargetSelectionResult:
    """Select the largest centered credible instance with a fully traced score."""
    eligible: list[_EligibleInstance] = []
    rejected: dict[int, InstanceSelectionTrace] = {}
    warnings: list[str] = []
    for position, instance in enumerate(instances):
        index = int(getattr(instance, "source_index", position))
        confidence: float | None = None
        area: float | None = None
        try:
            confidence = float(instance.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence inválida")
            normalized_mask = binary_mask_image(
                instance.mask,
                expected_size=image_size,
            )
            area = mask_area_ratio(normalized_mask)
            bbox = mask_bbox(normalized_mask)
            if bbox is None:
                raise InvalidLeafMaskError("máscara vacía")
        except (InvalidLeafMaskError, TypeError, ValueError) as exc:
            rejected[index] = InstanceSelectionTrace(
                index,
                confidence,
                area,
                None,
                None,
                None,
                False,
                f"máscara corrupta: {exc}",
            )
            continue
        if instance.class_id != 0:
            rejected[index] = InstanceSelectionTrace(
                index,
                confidence,
                area,
                None,
                None,
                None,
                False,
                f"class_id inesperado: {instance.class_id}",
            )
        elif confidence < config.confidence_threshold:
            rejected[index] = InstanceSelectionTrace(
                index,
                confidence,
                area,
                None,
                None,
                None,
                False,
                "confidence por debajo del umbral",
            )
        elif area < config.min_mask_area_ratio:
            rejected[index] = InstanceSelectionTrace(
                index,
                confidence,
                area,
                None,
                None,
                None,
                False,
                "máscara demasiado pequeña",
            )
        elif math.isclose(area, 1.0, abs_tol=1e-12):
            rejected[index] = InstanceSelectionTrace(
                index,
                confidence,
                area,
                None,
                None,
                None,
                False,
                "máscara degenerada: cubre exactamente toda la imagen",
            )
        else:
            if area >= config.near_full_warning_ratio:
                warnings.append(
                    f"instancia {index} cubre {area:.2%}; se acepta porque el dataset "
                    "contiene hojas válidas casi a cuadro completo"
                )
            eligible.append(
                _EligibleInstance(
                    instance=LeafInstance(
                        mask=normalized_mask,
                        confidence=confidence,
                        bbox=bbox,
                        source_index=index,
                        class_id=instance.class_id,
                    ),
                    bbox=bbox,
                    area_ratio=area,
                    center_proximity=_center_proximity(normalized_mask, image_size),
                )
            )

    if not eligible:
        traces = tuple(rejected[key] for key in sorted(rejected))
        reason = "sin instancias" if not instances else "sin instancias elegibles"
        return TargetSelectionResult(None, None, traces, reason, tuple(warnings))

    largest = max(item.area_ratio for item in eligible)
    scored: list[tuple[float, _EligibleInstance, InstanceSelectionTrace]] = []
    for item in eligible:
        relative_area = item.area_ratio / largest
        score = (
            config.area_weight * relative_area
            + config.center_weight * item.center_proximity
            + config.confidence_weight * item.instance.confidence
        )
        trace = InstanceSelectionTrace(
            source_index=item.instance.source_index,
            confidence=item.instance.confidence,
            area_ratio=item.area_ratio,
            relative_area=relative_area,
            center_proximity=item.center_proximity,
            score=score,
            eligible=True,
            reason=None,
        )
        scored.append((score, item, trace))
    scored.sort(
        key=lambda row: (
            row[0],
            row[1].area_ratio,
            row[1].instance.confidence,
            -row[1].instance.source_index,
        ),
        reverse=True,
    )
    selected = scored[0][1]
    if len(eligible) > 1:
        warnings.append(
            f"{len(eligible)} hojas elegibles; se seleccionó la instancia "
            f"{selected.instance.source_index} con la estrategia largest_centered_confident"
        )
    traces_by_index = {trace.source_index: trace for _, _, trace in scored}
    traces_by_index.update(rejected)
    traces = tuple(traces_by_index[key] for key in sorted(traces_by_index))
    return TargetSelectionResult(
        selected.instance,
        selected.bbox,
        traces,
        None,
        tuple(warnings),
    )


def _segmenter_metadata(segmenter: LeafSegmenter) -> dict[str, object]:
    method = getattr(segmenter, "to_metadata", None)
    if callable(method):
        metadata = method()
        if not isinstance(metadata, dict):
            raise TypeError("segmenter.to_metadata() debe devolver dict")
        return dict(metadata)
    return {"segmenter_model": type(segmenter).__name__}


def _overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image_to_rgb(image)
    tint = Image.new("RGB", base.size, (0, 180, 80))
    blended = Image.blend(base, tint, 0.42)
    return Image.composite(blended, base, mask)


def _comparison_panel(result: SegmentedLeafProcessingResult) -> Image.Image:
    mask_panel = (
        result.mask.convert("RGB")
        if result.mask is not None
        else Image.new("RGB", result.original_image.size, 0)
    )
    overlay_panel = (
        _overlay(result.original_image, result.mask)
        if result.mask is not None
        else result.original_image.copy()
    )
    final_panel = result.processed_image or result.original_image
    sources = (result.original_image, overlay_panel, mask_panel, final_panel)
    panel_size = (320, 240)
    canvas = Image.new("RGB", (panel_size[0] * len(sources), panel_size[1]), "white")
    for index, source in enumerate(sources):
        fitted = ImageOps.contain(image_to_rgb(source), panel_size)
        offset = (
            index * panel_size[0] + (panel_size[0] - fitted.width) // 2,
            (panel_size[1] - fitted.height) // 2,
        )
        canvas.paste(fitted, offset)
    return canvas


def save_debug_artifacts(
    result: SegmentedLeafProcessingResult,
    output_dir: Path,
) -> None:
    """Create one immutable debug bundle; never overwrite an earlier bundle."""
    output_dir.mkdir(parents=True, exist_ok=False)
    result.original_image.save(output_dir / "original.jpg", quality=95)
    if result.mask is not None:
        result.mask.save(output_dir / "mask.png")
        _overlay(result.original_image, result.mask).save(
            output_dir / "overlay.jpg", quality=95
        )
    if result.masked_image is not None:
        # PNG preserves exact RGB(0, 0, 0); JPEG would introduce non-zero ringing.
        result.masked_image.save(output_dir / "masked_black.png")
    if result.crop_image is not None:
        result.crop_image.save(output_dir / "crop.png")
    _comparison_panel(result).save(output_dir / "comparison.jpg", quality=95)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            result.to_metadata(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


class SegmentedLeafProcessor:
    """Opt-in entry point: ``processor.process(image)``."""

    def __init__(
        self,
        segmenter: LeafSegmenter,
        config: LeafMaskProcessorConfig | None = None,
    ) -> None:
        self.segmenter = segmenter
        self.config = config or LeafMaskProcessorConfig()

    def _metadata(self) -> dict[str, object]:
        return {
            **_segmenter_metadata(self.segmenter),
            "selection_confidence_threshold": self.config.confidence_threshold,
        }

    def _fallback(
        self,
        original: Image.Image,
        *,
        instances: Sequence[LeafInstance],
        selection: TargetSelectionResult,
        source_image: str | None,
    ) -> SegmentedLeafProcessingResult:
        reason = selection.reason or "segmentación no confiable"
        processed = original.copy() if self.config.fallback == FALLBACK_ORIGINAL else None
        return SegmentedLeafProcessingResult(
            original_image=original,
            mask=None,
            masked_image=None,
            processed_image=processed,
            crop_image=None,
            bbox=None,
            confidence=None,
            number_of_instances=len(instances),
            selected_instance=None,
            selection_traces=selection.traces,
            preprocessing_strategy=self.config.processing_profile,
            fallback_used=True,
            fallback_reason=reason,
            warnings=selection.warnings
            + (f"fallback {self.config.fallback}: {reason}",),
            source_image=source_image,
            segmenter_metadata=self._metadata(),
        )

    def process(
        self,
        image: Image.Image,
        *,
        source_image: str | Path | None = None,
        debug_dir: Path | None = None,
    ) -> SegmentedLeafProcessingResult:
        if not isinstance(image, Image.Image):
            raise TypeError("image debe ser una instancia de PIL.Image.Image")
        original = image_to_rgb(image, self.config.background_value)
        source = str(Path(source_image).resolve()) if source_image is not None else None

        instances = tuple(self.segmenter.segment(original))
        selection = select_target_leaf(instances, original.size, self.config)
        if selection.selected is None or selection.selected_bbox is None:
            result = self._fallback(
                original,
                instances=instances,
                selection=selection,
                source_image=source,
            )
            if debug_dir is not None:
                save_debug_artifacts(result, debug_dir)
            return result

        selected = selection.selected
        mask = binary_mask_image(selected.mask, expected_size=original.size)
        masked = apply_leaf_mask(original, mask, self.config.background_value)
        bbox = selection.selected_bbox
        masked_crop = crop_leaf_region(masked, bbox)
        if self.config.processing_profile == MASK_BLACK:
            processed = masked.copy()
        elif self.config.processing_profile == BBOX_CROP:
            processed = crop_leaf_region(original, bbox)
        elif self.config.processing_profile == CROP_MASK_BLACK:
            processed = masked_crop.copy()
        elif self.config.processing_profile == CROP_MASK_LETTERBOX:
            processed = letterbox_image(
                masked_crop,
                self.config.target_size,
                padding_value=self.config.background_value,
            ).image
        elif self.config.processing_profile == SQUARE_CROP:
            processed = crop_square_centered(
                original,
                bbox,
                margin_ratio=0.15,
                target_size=self.config.target_size,
            )
        else:  # guarded by config validation
            raise AssertionError(self.config.processing_profile)

        result = SegmentedLeafProcessingResult(
            original_image=original,
            mask=mask,
            masked_image=masked,
            processed_image=processed,
            crop_image=masked_crop,
            bbox=bbox,
            confidence=selected.confidence,
            number_of_instances=len(instances),
            selected_instance=selected.source_index,
            selection_traces=selection.traces,
            preprocessing_strategy=self.config.processing_profile,
            fallback_used=False,
            fallback_reason=None,
            warnings=selection.warnings,
            source_image=source,
            segmenter_metadata=self._metadata(),
        )
        if debug_dir is not None:
            save_debug_artifacts(result, debug_dir)
        return result
