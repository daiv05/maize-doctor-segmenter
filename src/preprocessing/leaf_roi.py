"""Bounding-box validation, margin expansion, and in-memory ROI cropping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias

from PIL import Image

BoundingBox: TypeAlias = tuple[int, int, int, int]
BoundingBoxInput: TypeAlias = Sequence[int | float | str]
RGBColor: TypeAlias = tuple[int, int, int]


class InvalidBoundingBoxError(ValueError):
    """Raised when an operation requires a geometrically valid bounding box."""


@dataclass(frozen=True)
class LeafDetectionResult:
    """Validated location and provenance of one candidate leaf region."""

    detected: bool
    bbox: BoundingBox | None
    confidence: float
    area_ratio: float
    source: str
    fallback_used: bool = False
    reason: str | None = None


def _invalid_detection(
    reason: str,
    *,
    confidence: float,
    source: str,
) -> LeafDetectionResult:
    return LeafDetectionResult(
        detected=False,
        bbox=None,
        confidence=confidence,
        area_ratio=0.0,
        source=source,
        reason=reason,
    )


def _coerce_finite_number(value: int | float | str, name: str) -> float:
    if isinstance(value, bool):
        raise InvalidBoundingBoxError(f"{name} no puede ser booleano")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidBoundingBoxError(f"{name} debe ser numérico") from exc
    if not math.isfinite(converted):
        raise InvalidBoundingBoxError(f"{name} debe ser finito")
    return converted


def _coerce_bbox_outward(bbox: BoundingBoxInput) -> BoundingBox:
    if isinstance(bbox, (str, bytes, Path)):
        raise InvalidBoundingBoxError("bbox debe contener cuatro coordenadas")
    try:
        values = tuple(bbox)
    except TypeError as exc:
        raise InvalidBoundingBoxError("bbox debe ser una secuencia de cuatro valores") from exc
    if len(values) != 4:
        raise InvalidBoundingBoxError(
            f"bbox debe contener cuatro coordenadas; recibió {len(values)}"
        )
    x1, y1, x2, y2 = (
        _coerce_finite_number(value, name)
        for value, name in zip(values, ("x1", "y1", "x2", "y2"), strict=True)
    )
    if x2 <= x1:
        raise InvalidBoundingBoxError("x2 debe ser mayor que x1")
    if y2 <= y1:
        raise InvalidBoundingBoxError("y2 debe ser mayor que y1")
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def _validate_image_size(image_width: int, image_height: int) -> str | None:
    if isinstance(image_width, bool) or isinstance(image_height, bool):
        return "las dimensiones de imagen no pueden ser booleanas"
    if not isinstance(image_width, int) or not isinstance(image_height, int):
        return "image_width e image_height deben ser enteros"
    if image_width <= 0 or image_height <= 0:
        return "image_width e image_height deben ser mayores que cero"
    return None


def _validate_ratio(value: float, name: str, *, maximum: float = 1.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} no puede ser booleano")
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} debe ser numérico") from exc
    if not math.isfinite(ratio) or not 0.0 <= ratio <= maximum:
        raise ValueError(f"{name} debe estar entre 0 y {maximum:g}")
    return ratio


def clip_bbox(
    bbox: BoundingBoxInput,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Convert coordinates outward and clip them to Pillow crop boundaries.

    Coordinates follow Pillow's half-open convention: ``x2`` may equal the image
    width and ``y2`` may equal its height.
    """
    image_error = _validate_image_size(image_width, image_height)
    if image_error:
        raise InvalidBoundingBoxError(image_error)
    x1, y1, x2, y2 = _coerce_bbox_outward(bbox)
    clipped = (
        min(max(x1, 0), image_width),
        min(max(y1, 0), image_height),
        min(max(x2, 0), image_width),
        min(max(y2, 0), image_height),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise InvalidBoundingBoxError("bbox queda vacío después de limitarlo a la imagen")
    return clipped


def bbox_requires_clipping(
    bbox: BoundingBoxInput,
    image_width: int,
    image_height: int,
) -> bool:
    """Return whether a valid bbox extends beyond the image boundaries."""
    raw_bbox = _coerce_bbox_outward(bbox)
    return raw_bbox != clip_bbox(raw_bbox, image_width, image_height)


def bbox_width(bbox: BoundingBox) -> int:
    """Return bbox width, rejecting empty or inverted coordinates."""
    width = bbox[2] - bbox[0]
    if width <= 0:
        raise InvalidBoundingBoxError("el ancho del bbox debe ser mayor que cero")
    return width


def bbox_height(bbox: BoundingBox) -> int:
    """Return bbox height, rejecting empty or inverted coordinates."""
    height = bbox[3] - bbox[1]
    if height <= 0:
        raise InvalidBoundingBoxError("el alto del bbox debe ser mayor que cero")
    return height


def bbox_area(bbox: BoundingBox) -> int:
    """Return bbox area in pixels."""
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_area_ratio(bbox: BoundingBox, image_width: int, image_height: int) -> float:
    """Return the fraction of image area covered by a validated bbox."""
    image_error = _validate_image_size(image_width, image_height)
    if image_error:
        raise ValueError(image_error)
    return bbox_area(bbox) / float(image_width * image_height)


def validate_bbox(
    image_width: int,
    image_height: int,
    bbox: BoundingBoxInput,
    min_area_ratio: float,
    *,
    confidence: float = 1.0,
    source: str = "manual",
) -> LeafDetectionResult:
    """Validate, clip, and measure a bbox without opening an image.

    Invalid candidate regions produce ``detected=False`` with a descriptive
    reason. Invalid configuration values such as ``min_area_ratio > 1`` raise
    ``ValueError`` because they are programmer or configuration errors.
    """
    minimum = _validate_ratio(min_area_ratio, "min_area_ratio")
    try:
        safe_confidence = _coerce_finite_number(confidence, "confidence")
    except InvalidBoundingBoxError as exc:
        return _invalid_detection(str(exc), confidence=0.0, source=source)
    if not 0.0 <= safe_confidence <= 1.0:
        return _invalid_detection(
            "confidence debe estar entre 0 y 1",
            confidence=safe_confidence,
            source=source,
        )

    image_error = _validate_image_size(image_width, image_height)
    if image_error:
        return _invalid_detection(image_error, confidence=safe_confidence, source=source)
    try:
        clipped = clip_bbox(bbox, image_width, image_height)
    except InvalidBoundingBoxError as exc:
        return _invalid_detection(str(exc), confidence=safe_confidence, source=source)

    area_ratio = bbox_area_ratio(clipped, image_width, image_height)
    if area_ratio < minimum:
        return LeafDetectionResult(
            detected=False,
            bbox=clipped,
            confidence=safe_confidence,
            area_ratio=area_ratio,
            source=source,
            reason=(
                f"area_ratio {area_ratio:.6f} es menor que min_area_ratio {minimum:.6f}"
            ),
        )
    return LeafDetectionResult(
        detected=True,
        bbox=clipped,
        confidence=safe_confidence,
        area_ratio=area_ratio,
        source=source,
    )


def expand_bbox(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> BoundingBox:
    """Expand a bbox by a fraction of its own width and height on every side.

    ``margin_ratio`` is accepted from 0 through 1. Values greater than 1 are
    rejected as likely configuration errors instead of silently swallowing most
    of the original photograph.
    """
    margin = _validate_ratio(margin_ratio, "margin_ratio")
    clipped = clip_bbox(bbox, image_width, image_height)
    if margin == 0.0:
        return clipped
    horizontal = bbox_width(clipped) * margin
    vertical = bbox_height(clipped) * margin
    return (
        max(0, math.floor(clipped[0] - horizontal)),
        max(0, math.floor(clipped[1] - vertical)),
        min(image_width, math.ceil(clipped[2] + horizontal)),
        min(image_height, math.ceil(clipped[3] + vertical)),
    )


def normalize_rgb_color(value: int | Sequence[int]) -> RGBColor:
    """Normalize a scalar or three-channel padding value to an RGB tuple."""
    if isinstance(value, bool):
        raise ValueError("el color RGB no puede ser booleano")
    if isinstance(value, int):
        channels = (value, value, value)
    else:
        try:
            channels = tuple(value)
        except TypeError as exc:
            raise ValueError("el color debe ser un entero o tres canales RGB") from exc
        if len(channels) != 3:
            raise ValueError("el color RGB debe contener exactamente tres canales")
    if any(isinstance(channel, bool) or not isinstance(channel, int) for channel in channels):
        raise ValueError("los canales RGB deben ser enteros")
    if any(not 0 <= channel <= 255 for channel in channels):
        raise ValueError("los canales RGB deben estar entre 0 y 255")
    return channels[0], channels[1], channels[2]


def image_to_rgb(image: Image.Image, background: int | Sequence[int] = 0) -> Image.Image:
    """Return an independent RGB image, compositing transparency over a color."""
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("la imagen debe tener dimensiones mayores que cero")
    background_rgb = normalize_rgb_color(background)
    has_transparency = "A" in image.getbands() or "transparency" in image.info
    if has_transparency:
        foreground = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (*background_rgb, 255))
        return Image.alpha_composite(canvas, foreground).convert("RGB")
    return image.convert("RGB") if image.mode != "RGB" else image.copy()


def crop_leaf_region(
    image: Image.Image,
    bbox: BoundingBox,
    *,
    transparency_background: int | Sequence[int] = 0,
) -> Image.Image:
    """Crop a validated bbox from an in-memory image and return a new RGB image."""
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    clipped = clip_bbox(bbox, image.width, image.height)
    rgb_image = image_to_rgb(image, transparency_background)
    return rgb_image.crop(clipped)


def crop_square_centered(
    image: Image.Image,
    bbox: BoundingBoxInput,
    *,
    margin_ratio: float = 0.15,
    target_size: Sequence[int] | None = None,
    transparency_background: int | Sequence[int] = 0,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Crop a 1:1 square centered around the leaf while preserving natural background.

    Calculates a square window covering the largest dimension of the leaf plus an
    expandable margin, shifting the window to remain within the original image boundaries
    (without adding artificial black masks or pixels). If target_size is given, resizes
    the square crop to the target dimensions preserving 1:1 aspect ratio.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    clipped = clip_bbox(bbox, image.width, image.height)
    x1, y1, x2, y2 = clipped
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    w_img, h_img = image.size
    max_dim = max(bw, bh)
    margin = _validate_ratio(margin_ratio, "margin_ratio")
    side = max_dim * (1.0 + margin)
    side = min(side, min(w_img, h_img))
    half = side / 2.0

    sq_x1 = cx - half
    sq_y1 = cy - half
    sq_x2 = cx + half
    sq_y2 = cy + half

    if sq_x1 < 0:
        shift = -sq_x1
        sq_x1 = 0.0
        sq_x2 = min(float(w_img), sq_x2 + shift)
    if sq_y1 < 0:
        shift = -sq_y1
        sq_y1 = 0.0
        sq_y2 = min(float(h_img), sq_y2 + shift)
    if sq_x2 > w_img:
        shift = sq_x2 - float(w_img)
        sq_x2 = float(w_img)
        sq_x1 = max(0.0, sq_x1 - shift)
    if sq_y2 > h_img:
        shift = sq_y2 - float(h_img)
        sq_y2 = float(h_img)
        sq_y1 = max(0.0, sq_y1 - shift)

    crop_coords = (
        int(math.floor(sq_x1)),
        int(math.floor(sq_y1)),
        int(math.ceil(sq_x2)),
        int(math.ceil(sq_y2)),
    )
    crop_coords = clip_bbox(crop_coords, w_img, h_img)
    rgb = image_to_rgb(image, transparency_background)
    cropped = rgb.crop(crop_coords)

    if target_size is not None:
        values = tuple(target_size)
        if len(values) != 2:
            raise ValueError("target_size debe contener exactamente (alto, ancho)")
        th, tw = int(values[0]), int(values[1])
        if th <= 0 or tw <= 0:
            raise ValueError("alto y ancho objetivo deben ser mayores que cero")
        cropped = cropped.resize((tw, th), resample=resample)
    return cropped

