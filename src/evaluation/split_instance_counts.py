"""Conteos de instancias de un split YOLO-seg, tal como los ve el dataset y el cargador.

El dataset declara una anotación por polígono. Ultralytics, en cambio, deduplica las
filas ``[clase, cx, cy, w, h]`` idénticas al construir su cargador, de modo que dos
polígonos distintos con el mismo bounding box cuentan como una sola instancia. Ambos
números son correctos y describen cosas distintas; este módulo los deriva del propio
dataset para que ninguno tenga que congelarse como constante.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LABEL_SUFFIX = ".txt"
BBOX_DECIMALS = 6


class InstanceCountError(RuntimeError):
    """Se lanza cuando una etiqueta YOLO-seg no puede interpretarse."""


@dataclass(frozen=True)
class SplitInstanceCounts:
    """Conteos de un split y el detalle de la deduplicación del cargador."""

    image_count: int
    annotation_count: int
    loader_instance_count: int
    deduplicated_images: tuple[str, ...]

    @property
    def deduplicated_annotations(self) -> int:
        """Anotaciones que el cargador colapsa por compartir clase y bounding box."""
        return self.annotation_count - self.loader_instance_count


def _polygon_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """Devuelve el bounding box normalizado ``(cx, cy, w, h)`` de un polígono.

    @param {list[float]} values Coordenadas normalizadas intercaladas ``x, y``.
    @returns {tuple[float, float, float, float]} Centro y tamaño del bounding box.
    """
    if len(values) < 6 or len(values) % 2 != 0:
        raise InstanceCountError(
            f"Polígono con {len(values)} coordenadas; se requieren pares y al menos tres puntos"
        )
    xs = values[0::2]
    ys = values[1::2]
    minimum_x, maximum_x = min(xs), max(xs)
    minimum_y, maximum_y = min(ys), max(ys)
    return (
        (minimum_x + maximum_x) / 2,
        (minimum_y + maximum_y) / 2,
        maximum_x - minimum_x,
        maximum_y - minimum_y,
    )


def _label_rows(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Convierte un archivo de etiquetas en filas ``(clase, bounding box)``."""
    rows: list[tuple[int, tuple[float, float, float, float]]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        try:
            class_id = int(tokens[0])
            coordinates = [float(token) for token in tokens[1:]]
        except ValueError as exc:
            raise InstanceCountError(f"{path.name}:{number} no es una fila YOLO-seg") from exc
        bbox = _polygon_bbox(coordinates)
        rows.append((class_id, tuple(round(value, BBOX_DECIMALS) for value in bbox)))
    return rows


def count_split_instances(dataset_root: Path, split: str) -> SplitInstanceCounts:
    """Deriva los conteos de anotaciones y de instancias efectivas de un split.

    @param {Path} dataset_root Raíz del dataset YOLO-seg.
    @param {str} split Nombre del split (``train``, ``val`` o ``test``).
    @returns {SplitInstanceCounts} Conteos derivados y detalle de la deduplicación.
    """
    label_dir = dataset_root / "labels" / split
    if not label_dir.is_dir():
        raise InstanceCountError(f"Falta el directorio de etiquetas: {label_dir}")

    image_count = 0
    annotation_count = 0
    loader_instance_count = 0
    deduplicated: list[str] = []
    for path in sorted(label_dir.glob(f"*{LABEL_SUFFIX}")):
        image_count += 1
        rows = _label_rows(path)
        unique = len(set(rows))
        annotation_count += len(rows)
        loader_instance_count += unique
        if unique < len(rows):
            deduplicated.append(path.stem)

    return SplitInstanceCounts(
        image_count=image_count,
        annotation_count=annotation_count,
        loader_instance_count=loader_instance_count,
        deduplicated_images=tuple(deduplicated),
    )
