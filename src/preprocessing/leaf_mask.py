"""Binary leaf-mask validation and exact black-background application."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from PIL import Image

from src.preprocessing.leaf_roi import (
    BoundingBox,
    image_to_rgb,
    normalize_rgb_color,
)

MaskInput: TypeAlias = Image.Image | np.ndarray


class InvalidLeafMaskError(ValueError):
    """Raised when a mask is empty, non-binary, or spatially misaligned."""


@dataclass(frozen=True)
class MaskGeometry:
    """Transparent geometry used to audit a selected segmentation mask."""

    area_pixels: int
    image_area_pixels: int
    mask_area_ratio: float
    bbox: BoundingBox
    bbox_area_pixels: int
    bbox_area_ratio: float
    mask_bbox_ratio: float
    bbox_width_ratio: float
    bbox_height_ratio: float
    bbox_aspect_ratio: float
    bbox_center_x_ratio: float
    bbox_center_y_ratio: float
    border_contact_count: int
    connected_components: int
    largest_component_ratio: float
    perimeter_edges: int
    perimeter_area_ratio: float
    normalized_perimeter: float

    def to_metadata(self) -> dict[str, object]:
        return {
            "area_pixels": self.area_pixels,
            "image_area_pixels": self.image_area_pixels,
            "mask_area_ratio": self.mask_area_ratio,
            "bbox": list(self.bbox),
            "bbox_area_pixels": self.bbox_area_pixels,
            "bbox_area_ratio": self.bbox_area_ratio,
            "mask_bbox_ratio": self.mask_bbox_ratio,
            "bbox_width_ratio": self.bbox_width_ratio,
            "bbox_height_ratio": self.bbox_height_ratio,
            "bbox_aspect_ratio": self.bbox_aspect_ratio,
            "bbox_center_x_ratio": self.bbox_center_x_ratio,
            "bbox_center_y_ratio": self.bbox_center_y_ratio,
            "border_contact_count": self.border_contact_count,
            "connected_components": self.connected_components,
            "largest_component_ratio": self.largest_component_ratio,
            "perimeter_edges": self.perimeter_edges,
            "perimeter_area_ratio": self.perimeter_area_ratio,
            "normalized_perimeter": self.normalized_perimeter,
        }


def binary_mask_array(
    mask: MaskInput,
    *,
    expected_size: tuple[int, int] | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    """Return an independent ``bool[height, width]`` mask without resizing.

    Accepted values are boolean, ``0/1`` or ``0/255``. Rejecting intermediate
    values prevents an accidentally bilinear-resized mask from being treated as
    valid geometry.
    """
    if isinstance(mask, Image.Image):
        if mask.mode not in {"1", "L"}:
            raise InvalidLeafMaskError("la máscara PIL debe usar modo 1 o L")
        array = np.asarray(mask.convert("L"))
    elif isinstance(mask, np.ndarray):
        array = np.asarray(mask)
    else:
        raise TypeError("mask debe ser PIL.Image.Image o numpy.ndarray")
    if array.ndim != 2:
        raise InvalidLeafMaskError(
            f"la máscara debe tener shape (alto, ancho); recibió {array.shape!r}"
        )
    height, width = array.shape
    if height <= 0 or width <= 0:
        raise InvalidLeafMaskError("la máscara debe tener dimensiones positivas")
    if expected_size is not None and (width, height) != expected_size:
        raise InvalidLeafMaskError(
            "la máscara no coincide espacialmente con la imagen: "
            f"{(width, height)} != {expected_size}"
        )
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise InvalidLeafMaskError("la máscara contiene NaN o infinito")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({False, True, 0, 1, 255}):
        raise InvalidLeafMaskError(
            f"la máscara debe ser binaria (0/1 o 0/255); valores encontrados={sorted(unique)[:10]}"
        )
    binary = np.array(array != 0, dtype=np.bool_, copy=True)
    if not allow_empty and not binary.any():
        raise InvalidLeafMaskError("la máscara está vacía")
    return binary


def binary_mask_image(
    mask: MaskInput,
    *,
    expected_size: tuple[int, int] | None = None,
    allow_empty: bool = False,
) -> Image.Image:
    array = binary_mask_array(
        mask,
        expected_size=expected_size,
        allow_empty=allow_empty,
    )
    return Image.fromarray(array.astype(np.uint8) * 255, mode="L")


def mask_area_ratio(mask: MaskInput) -> float:
    array = binary_mask_array(mask, allow_empty=True)
    return float(array.mean())


def mask_bbox(mask: MaskInput) -> BoundingBox | None:
    array = binary_mask_array(mask, allow_empty=True)
    ys, xs = np.nonzero(array)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_centroid(mask: MaskInput) -> tuple[float, float] | None:
    array = binary_mask_array(mask, allow_empty=True)
    ys, xs = np.nonzero(array)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def _component_areas(binary: np.ndarray) -> tuple[int, ...]:
    """Return exact 4-connected areas using scanline runs, without SciPy/OpenCV."""
    parents: list[int] = []
    run_areas: list[int] = []
    previous: list[tuple[int, int, int]] = []

    def find(node: int) -> int:
        root = node
        while parents[root] != root:
            root = parents[root]
        while parents[node] != node:
            parent = parents[node]
            parents[node] = root
            node = parent
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for row in binary:
        padded = np.pad(row.astype(np.int8, copy=False), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start_value, end_value in zip(starts.tolist(), ends.tolist(), strict=True):
            node = len(parents)
            parents.append(node)
            run_areas.append(end_value - start_value)
            while previous_index < len(previous) and previous[previous_index][1] <= start_value:
                previous_index += 1
            overlap_index = previous_index
            while overlap_index < len(previous) and previous[overlap_index][0] < end_value:
                union(node, previous[overlap_index][2])
                overlap_index += 1
            current.append((start_value, end_value, node))
        previous = current

    totals: dict[int, int] = {}
    for node, area in enumerate(run_areas):
        root = find(node)
        totals[root] = totals.get(root, 0) + area
    return tuple(sorted(totals.values(), reverse=True))


def mask_geometry(mask: MaskInput) -> MaskGeometry:
    """Measure mask geometry exactly in source-image coordinates.

    The perimeter is the number of foreground/background grid edges. Components
    use 4-connectivity. Both definitions are dependency-free and deterministic.

    ``connected_components`` y ``largest_component_ratio`` valen 1 y 1.0 para toda
    máscara procedente de ``rasterize_instance_polygon``, que rellena un único
    polígono simple; sólo se separan con polígonos auto-intersectantes. No sirven
    como evidencia de que una hoja esté fragmentada: Ultralytics entrega un contorno
    por instancia, así que un fragmento perdido por oclusión no llega hasta aquí.
    """
    binary = binary_mask_array(mask)
    height, width = binary.shape
    bbox = mask_bbox(binary)
    if bbox is None:  # guarded by binary_mask_array(allow_empty=False)
        raise InvalidLeafMaskError("la máscara está vacía")
    left, top, right, bottom = bbox
    area = int(binary.sum())
    image_area = width * height
    bbox_width = right - left
    bbox_height = bottom - top
    bbox_area = bbox_width * bbox_height
    perimeter = int(binary[0, :].sum() + binary[-1, :].sum())
    perimeter += int(binary[:, 0].sum() + binary[:, -1].sum())
    perimeter += int(np.count_nonzero(binary[1:, :] != binary[:-1, :]))
    perimeter += int(np.count_nonzero(binary[:, 1:] != binary[:, :-1]))
    components = _component_areas(binary)
    return MaskGeometry(
        area_pixels=area,
        image_area_pixels=image_area,
        mask_area_ratio=area / image_area,
        bbox=bbox,
        bbox_area_pixels=bbox_area,
        bbox_area_ratio=bbox_area / image_area,
        mask_bbox_ratio=area / bbox_area,
        bbox_width_ratio=bbox_width / width,
        bbox_height_ratio=bbox_height / height,
        bbox_aspect_ratio=bbox_width / bbox_height,
        bbox_center_x_ratio=((left + right) / 2.0) / width,
        bbox_center_y_ratio=((top + bottom) / 2.0) / height,
        border_contact_count=sum(
            (
                bool(binary[0, :].any()),
                bool(binary[-1, :].any()),
                bool(binary[:, 0].any()),
                bool(binary[:, -1].any()),
            )
        ),
        connected_components=len(components),
        largest_component_ratio=components[0] / area,
        perimeter_edges=perimeter,
        perimeter_area_ratio=perimeter / area,
        normalized_perimeter=perimeter / math.sqrt(area),
    )


def apply_leaf_mask(
    image: Image.Image,
    mask: MaskInput,
    background_value: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Preserve leaf pixels and set every background pixel to an exact RGB value."""
    if not isinstance(image, Image.Image):
        raise TypeError("image debe ser una instancia de PIL.Image.Image")
    rgb = image_to_rgb(image, background_value)
    binary = binary_mask_array(mask, expected_size=rgb.size)
    color = np.asarray(normalize_rgb_color(background_value), dtype=np.uint8)
    source = np.asarray(rgb, dtype=np.uint8)
    output = np.empty_like(source)
    output[...] = color
    output[binary] = source[binary]
    return Image.fromarray(output, mode="RGB")
