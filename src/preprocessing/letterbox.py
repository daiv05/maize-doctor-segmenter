"""Aspect-ratio-preserving resize with centered RGB padding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from src.preprocessing.leaf_roi import RGBColor, image_to_rgb, normalize_rgb_color


@dataclass(frozen=True)
class LetterboxResult:
    """Imagen con letterbox y geometría para trazar coordenadas después.

    Las dos convenciones conviven a propósito y por eso están marcadas en el nombre:
    ``*_width_height`` sigue el orden de Pillow y ``target_size`` el ``(alto, ancho)``
    del proyecto. Con tamaños no cuadrados confundirlas deja de ser inocuo.
    """

    image: Image.Image
    original_size_width_height: tuple[int, int]
    resized_size_width_height: tuple[int, int]
    target_size: tuple[int, int]
    padding: tuple[int, int, int, int]
    scale: float


def validate_target_size(target_size: Sequence[int]) -> tuple[int, int]:
    """Validate and return target size using project convention ``(height, width)``."""
    try:
        values = tuple(target_size)
    except TypeError as exc:
        raise ValueError("target_size debe contener alto y ancho") from exc
    if len(values) != 2:
        raise ValueError("target_size debe contener exactamente (alto, ancho)")
    height, width = values
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
    ):
        raise ValueError("alto y ancho objetivo deben ser enteros")
    if height <= 0 or width <= 0:
        raise ValueError("alto y ancho objetivo deben ser mayores que cero")
    return height, width


def validate_resample(resample: int | Image.Resampling) -> Image.Resampling:
    """Return a Pillow interpolation method or raise a clear configuration error."""
    try:
        return Image.Resampling(resample)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"método de interpolación Pillow inválido: {resample!r}") from exc


def _resized_dimensions(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int, float]:
    scale = min(target_width / source_width, target_height / source_height)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("la escala de letterbox debe ser finita y mayor que cero")
    resized_width = min(target_width, max(1, round(source_width * scale)))
    resized_height = min(target_height, max(1, round(source_height * scale)))
    return resized_width, resized_height, scale


def letterbox_image(
    image: Image.Image,
    target_size: Sequence[int],
    *,
    padding_value: int | Sequence[int] = 0,
    resample: int | Image.Resampling = Image.Resampling.BILINEAR,
) -> LetterboxResult:
    """Resize an image proportionally and center it on an exact-size RGB canvas.

    ``target_size`` follows the repository's ``(height, width)`` convention.
    Upscaling is allowed when the requested canvas is larger; calculated resize
    dimensions are always clamped to at least one pixel and at most the target.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("la imagen debe tener dimensiones mayores que cero")
    target_height, target_width = validate_target_size(target_size)
    color: RGBColor = normalize_rgb_color(padding_value)
    interpolation = validate_resample(resample)
    source = image_to_rgb(image, color)
    resized_width, resized_height, scale = _resized_dimensions(
        source.width,
        source.height,
        target_width,
        target_height,
    )
    resized = source.resize((resized_width, resized_height), resample=interpolation)
    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    left = horizontal_padding // 2
    right = horizontal_padding - left
    top = vertical_padding // 2
    bottom = vertical_padding - top
    canvas = Image.new("RGB", (target_width, target_height), color)
    canvas.paste(resized, (left, top))
    return LetterboxResult(
        image=canvas,
        original_size_width_height=source.size,
        resized_size_width_height=resized.size,
        target_size=(target_height, target_width),
        padding=(left, top, right, bottom),
        scale=scale,
    )
