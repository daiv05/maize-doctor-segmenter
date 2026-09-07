"""Métricas de conservación de hoja para el segmentador.

mAP penaliza por igual el exceso y el defecto de máscara. Para DoctorMaiz no son
equivalentes: recortar tejido elimina señal visual que no puede recuperarse, mientras
que una máscara algo amplia deja pasar fondo; estas métricas hacen visible ese efecto.

Por eso este módulo prioriza el recall de píxel de hoja y la sub-segmentación
sobre la precisión, y desagrega por fuente para vigilar la brecha de dominio
entre ``corn`` (155 imágenes a 224x224) y ``corn_leaf_diseases_classification``
(1000 imágenes con 35 resoluciones distintas).

El cálculo parte de etiquetas en formato YOLO-seg, no de un modelo cargado, para
que sea reproducible y auditable sin GPU ni Ultralytics.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from src.data.segmentation_audit import parse_yolo_segmentation_line

SPLITS = ("train", "val", "test")
DEFAULT_RASTER_SIZE = 640
DEFAULT_MIN_AREA = 0.01
ROW_COLUMNS = (
    "filename",
    "source_dataset",
    "orientation",
    "image_width",
    "image_height",
    "native_megapixels",
    "resolution_bin",
    "truth_instances",
    "predicted_instances",
    "multi_leaf",
    "leaf_count_group",
    "truth_touches_border",
    "border_contact",
    "detected",
    "leaf_pixel_recall",
    "leaf_pixel_precision",
    "iou",
    "dice",
    "under_segmentation_ratio",
    "over_segmentation_ratio",
    "cropped_leaf_percent",
    "truth_area_fraction",
    "predicted_area_fraction",
)


class DownstreamMetricsError(RuntimeError):
    """Raised when the evidence needed for a downstream metric is missing."""


@dataclass(frozen=True)
class MaskPair:
    """Ground truth and prediction for one image, already rasterized."""

    filename: str
    source_dataset: str
    orientation: str
    truth: np.ndarray
    prediction: np.ndarray
    truth_instances: int
    predicted_instances: int
    image_width: int = 0
    image_height: int = 0

    @property
    def truth_area(self) -> float:
        return float(self.truth.sum())

    @property
    def predicted_area(self) -> float:
        return float(self.prediction.sum())

    @property
    def intersection(self) -> float:
        return float(np.logical_and(self.truth, self.prediction).sum())

    @property
    def union(self) -> float:
        return float(np.logical_or(self.truth, self.prediction).sum())


def rasterize_polygons(
    polygons: Sequence[Sequence[tuple[float, float]]],
    size: int = DEFAULT_RASTER_SIZE,
) -> np.ndarray:
    """Rasterize normalized polygons into one boolean mask of ``size`` squared."""
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        draw.polygon(
            [(x * (size - 1), y * (size - 1)) for x, y in polygon],
            fill=255,
        )
    return np.asarray(canvas, dtype=np.uint8) > 0


def read_yolo_polygons(path: Path) -> list[list[tuple[float, float]]]:
    """Read every valid segmentation polygon from a YOLO-seg label file."""
    if not path.is_file():
        return []
    polygons: list[list[tuple[float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = parse_yolo_segmentation_line(line)
        if parsed.valid and parsed.points:
            polygons.append([(float(x), float(y)) for x, y in parsed.points])
    return polygons


def image_metrics(pair: MaskPair, *, minimum_area: float = DEFAULT_MIN_AREA) -> dict[str, object]:
    """Compute per-image downstream metrics for one ground-truth/prediction pair."""
    truth_area = pair.truth_area
    if truth_area <= 0:
        raise DownstreamMetricsError(
            f"La verdad de referencia no tiene área rasterizada: {pair.filename}"
        )
    predicted_area = pair.predicted_area
    intersection = pair.intersection
    union = pair.union
    total_pixels = float(pair.truth.size)
    detected = predicted_area / total_pixels >= minimum_area
    leaf_pixel_recall = intersection / truth_area
    image_height, image_width = pair.truth.shape
    native_width = pair.image_width or image_width
    native_height = pair.image_height or image_height
    touches_border = bool(
        pair.truth[0, :].any()
        or pair.truth[-1, :].any()
        or pair.truth[:, 0].any()
        or pair.truth[:, -1].any()
    )
    return {
        "filename": pair.filename,
        "source_dataset": pair.source_dataset,
        "orientation": pair.orientation,
        "image_width": native_width,
        "image_height": native_height,
        "native_megapixels": native_width * native_height / 1_000_000,
        "resolution_bin": _resolution_bin(native_width, native_height),
        "truth_instances": pair.truth_instances,
        "predicted_instances": pair.predicted_instances,
        "multi_leaf": pair.truth_instances > 1,
        "leaf_count_group": "multiple" if pair.truth_instances > 1 else "single",
        "truth_touches_border": touches_border,
        "border_contact": "touching" if touches_border else "interior",
        "detected": detected,
        # Prioritaria: fracción del tejido foliar real que sobrevive al recorte.
        "leaf_pixel_recall": leaf_pixel_recall,
        "leaf_pixel_precision": (intersection / predicted_area) if predicted_area else 0.0,
        "iou": (intersection / union) if union else 0.0,
        "dice": (
            (2 * intersection / (truth_area + predicted_area))
            if (truth_area + predicted_area)
            else 0.0
        ),
        # Tejido perdido: el error que no puede recuperarse después.
        "under_segmentation_ratio": 1.0 - leaf_pixel_recall,
        # Fondo añadido, relativo al tamaño de la hoja: el error tolerable.
        "over_segmentation_ratio": (predicted_area - intersection) / truth_area,
        "cropped_leaf_percent": 100.0 * (1.0 - leaf_pixel_recall),
        "truth_area_fraction": truth_area / total_pixels,
        "predicted_area_fraction": predicted_area / total_pixels,
    }


def _area_bin(fraction: float) -> str:
    return "small" if fraction < 0.05 else "large" if fraction > 0.50 else "medium"


def _resolution_bin(width: int, height: int) -> str:
    megapixels = width * height / 1_000_000
    if megapixels <= 0.10:
        return "low_<=0.10mp"
    if megapixels <= 1.0:
        return "medium_<=1.00mp"
    return "high_>1.00mp"


def _as_float(value: object) -> float:
    """Convert serialized metric values without weakening the row type to Any."""
    if isinstance(value, (int, float, str)):
        return float(value)
    raise DownstreamMetricsError(f"Valor numérico inválido: {value!r}")


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate per-image metrics into the summary the project reports."""
    if not rows:
        raise DownstreamMetricsError("No hay filas de métricas para agregar")
    detected = [row for row in rows if row["detected"]]
    return {
        "images": len(rows),
        "images_without_detection": sum(1 for row in rows if not row["detected"]),
        "images_without_detection_rate": (
            sum(1 for row in rows if not row["detected"]) / len(rows)
        ),
        "mean_leaf_pixel_recall": mean(_as_float(row["leaf_pixel_recall"]) for row in rows),
        "mean_leaf_pixel_precision": mean(
            _as_float(row["leaf_pixel_precision"]) for row in rows
        ),
        "mean_iou": mean(_as_float(row["iou"]) for row in rows),
        "mean_dice": mean(_as_float(row["dice"]) for row in rows),
        "mean_under_segmentation_ratio": mean(
            _as_float(row["under_segmentation_ratio"]) for row in rows
        ),
        "mean_over_segmentation_ratio": mean(
            _as_float(row["over_segmentation_ratio"]) for row in rows
        ),
        "mean_cropped_leaf_percent": mean(
            _as_float(row["cropped_leaf_percent"]) for row in rows
        ),
        "worst_cropped_leaf_percent": max(
            _as_float(row["cropped_leaf_percent"]) for row in rows
        ),
        "mean_iou_detected_only": (
            mean(_as_float(row["iou"]) for row in detected) if detected else 0.0
        ),
        "multi_leaf_images": sum(1 for row in rows if row["multi_leaf"]),
        "mean_leaf_pixel_recall_multi_leaf": (
            mean(
                _as_float(row["leaf_pixel_recall"])
                for row in rows
                if row["multi_leaf"]
            )
            if any(row["multi_leaf"] for row in rows)
            else None
        ),
    }


def group_by(
    rows: Sequence[Mapping[str, object]],
    key: str,
) -> dict[str, dict[str, object]]:
    """Aggregate the same metrics inside each value of one grouping column."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {value: aggregate(items) for value, items in sorted(grouped.items())}


def group_by_area_bin(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Aggregate by ground-truth mask size, which drives small-object behaviour."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_area_bin(_as_float(row["truth_area_fraction"]))].append(row)
    return {value: aggregate(items) for value, items in sorted(grouped.items())}


def load_manifest_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    """Load the frozen split manifest rows for one split."""
    if split not in SPLITS:
        raise DownstreamMetricsError(f"Split inválido: {split}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise DownstreamMetricsError(f"El manifiesto no tiene filas para {split}")
    return rows


def build_pairs(
    manifest_rows: Iterable[Mapping[str, str]],
    dataset_root: Path,
    prediction_root: Path,
    *,
    raster_size: int = DEFAULT_RASTER_SIZE,
    minimum_area: float = DEFAULT_MIN_AREA,
) -> list[MaskPair]:
    """Pair every frozen ground-truth label with its predicted label file."""
    pairs: list[MaskPair] = []
    for row in manifest_rows:
        truth_path = dataset_root / row["materialized_label_path"]
        prediction_path = prediction_root / f"{Path(row['filename']).stem}.txt"
        truth_polygons = read_yolo_polygons(truth_path)
        if not truth_polygons:
            raise DownstreamMetricsError(f"Sin geometría de referencia: {truth_path}")
        predicted_polygons = read_yolo_polygons(prediction_path)
        truth = rasterize_polygons(truth_polygons, raster_size)
        prediction = rasterize_polygons(predicted_polygons, raster_size)
        try:
            image_width = int(row["width"])
            image_height = int(row["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DownstreamMetricsError(
                f"Dimensiones nativas inválidas para {row['filename']}"
            ) from exc
        if image_width <= 0 or image_height <= 0:
            raise DownstreamMetricsError(
                f"Dimensiones nativas inválidas para {row['filename']}: "
                f"{image_width}x{image_height}"
            )
        pairs.append(
            MaskPair(
                filename=row["filename"],
                source_dataset=row["source_dataset"],
                orientation=row["orientation"],
                truth=truth,
                prediction=prediction,
                truth_instances=int(row["instance_count"]),
                predicted_instances=len(predicted_polygons),
                image_width=image_width,
                image_height=image_height,
            )
        )
    return pairs


def evaluate_downstream(
    *,
    dataset_root: Path,
    prediction_root: Path,
    split: str,
    raster_size: int = DEFAULT_RASTER_SIZE,
    minimum_area: float = DEFAULT_MIN_AREA,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return per-image rows and the grouped summary for one split."""
    manifest_rows = load_manifest_rows(
        dataset_root / "manifests" / "split_manifest.csv", split
    )
    pairs = build_pairs(
        manifest_rows,
        dataset_root,
        prediction_root,
        raster_size=raster_size,
        minimum_area=minimum_area,
    )
    rows = [image_metrics(pair, minimum_area=minimum_area) for pair in pairs]
    summary = {
        "schema_version": 2,
        "split": split,
        "raster_size": raster_size,
        "minimum_area_fraction": minimum_area,
        "prediction_root": str(prediction_root),
        "overall": aggregate(rows),
        "by_source": group_by(rows, "source_dataset"),
        "by_orientation": group_by(rows, "orientation"),
        "by_truth_area_bin": group_by_area_bin(rows),
        "by_border_contact": group_by(rows, "border_contact"),
        "by_resolution_bin": group_by(rows, "resolution_bin"),
        "by_leaf_count": group_by(rows, "leaf_count_group"),
        "priority_note": (
            "leaf_pixel_recall y under_segmentation_ratio tienen prioridad sobre "
            "precision: recortar tejido enfermo destruye la señal diagnóstica, "
            "mientras que una máscara amplia sólo añade fondo."
        ),
    }
    return rows, summary
