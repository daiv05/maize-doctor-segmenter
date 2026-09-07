"""Tests for the downstream segmentation metrics that guide DoctorMaiz."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.segmentation_downstream import (
    DownstreamMetricsError,
    MaskPair,
    aggregate,
    build_pairs,
    evaluate_downstream,
    group_by,
    group_by_area_bin,
    image_metrics,
    rasterize_polygons,
    read_yolo_polygons,
)

SQUARE = [(0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6)]
SHIFTED = [(0.35, 0.1), (0.85, 0.1), (0.85, 0.6), (0.35, 0.6)]
WIDER = [(0.05, 0.05), (0.65, 0.05), (0.65, 0.65), (0.05, 0.65)]
HALF = [(0.1, 0.1), (0.35, 0.1), (0.35, 0.6), (0.1, 0.6)]


def make_pair(truth, prediction, *, size=64, instances=1, source="corn"):
    truth_mask = rasterize_polygons(truth, size)
    predicted_mask = rasterize_polygons(prediction, size)
    return MaskPair(
        filename="image.jpg",
        source_dataset=source,
        orientation="square",
        truth=truth_mask,
        prediction=predicted_mask,
        truth_instances=instances,
        predicted_instances=len(prediction),
        image_width=224,
        image_height=224,
    )


def test_rasterize_produces_expected_area_fraction() -> None:
    mask = rasterize_polygons([SQUARE], 200)
    fraction = mask.sum() / mask.size
    assert fraction == pytest.approx(0.25, abs=0.01)


def test_rasterize_ignores_degenerate_polygons() -> None:
    mask = rasterize_polygons([[(0.1, 0.1), (0.2, 0.2)]], 64)
    assert mask.sum() == 0


def test_perfect_prediction_scores_one() -> None:
    metrics = image_metrics(make_pair([SQUARE], [SQUARE]))
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["leaf_pixel_recall"] == pytest.approx(1.0)
    assert metrics["under_segmentation_ratio"] == pytest.approx(0.0)
    assert metrics["cropped_leaf_percent"] == pytest.approx(0.0)
    assert metrics["detected"] is True


def test_cropping_leaf_is_penalised_by_recall_not_by_precision() -> None:
    """Una máscara que corta la hoja debe verse en recall y sub-segmentación."""
    metrics = image_metrics(make_pair([SQUARE], [HALF]))
    assert metrics["leaf_pixel_recall"] < 0.6
    assert metrics["under_segmentation_ratio"] > 0.4
    assert metrics["cropped_leaf_percent"] > 40.0
    # La predicción cae dentro de la hoja: la precisión no delata el problema.
    assert metrics["leaf_pixel_precision"] == pytest.approx(1.0, abs=0.02)
    assert metrics["over_segmentation_ratio"] == pytest.approx(0.0, abs=0.02)


def test_wider_mask_keeps_all_tissue_and_only_adds_background() -> None:
    """Exceder es el error tolerable: conserva el tejido y sólo añade fondo."""
    metrics = image_metrics(make_pair([SQUARE], [WIDER]))
    assert metrics["leaf_pixel_recall"] == pytest.approx(1.0, abs=0.02)
    assert metrics["under_segmentation_ratio"] == pytest.approx(0.0, abs=0.02)
    assert metrics["over_segmentation_ratio"] > 0.3
    assert metrics["leaf_pixel_precision"] < 1.0


def test_cropping_and_widening_are_not_symmetric_for_doctormaiz() -> None:
    """El recorte pierde tejido; el exceso no. Las métricas deben distinguirlo."""
    cropped = image_metrics(make_pair([SQUARE], [HALF]))
    widened = image_metrics(make_pair([SQUARE], [WIDER]))
    assert cropped["leaf_pixel_recall"] < widened["leaf_pixel_recall"]
    assert cropped["under_segmentation_ratio"] > widened["under_segmentation_ratio"]


def test_partial_overlap_is_between_zero_and_one() -> None:
    metrics = image_metrics(make_pair([SQUARE], [SHIFTED]))
    assert 0.0 < metrics["iou"] < 1.0
    assert metrics["dice"] > metrics["iou"]


def test_empty_prediction_counts_as_missing_detection() -> None:
    pair = make_pair([SQUARE], [])
    metrics = image_metrics(pair)
    assert metrics["detected"] is False
    assert metrics["leaf_pixel_recall"] == pytest.approx(0.0)
    assert metrics["cropped_leaf_percent"] == pytest.approx(100.0)


def test_empty_ground_truth_is_rejected() -> None:
    pair = MaskPair(
        filename="x.jpg",
        source_dataset="corn",
        orientation="square",
        truth=np.zeros((8, 8), dtype=bool),
        prediction=np.ones((8, 8), dtype=bool),
        truth_instances=0,
        predicted_instances=1,
    )
    with pytest.raises(DownstreamMetricsError):
        image_metrics(pair)


def test_aggregate_reports_a_single_missing_detection_rate() -> None:
    rows = [
        image_metrics(make_pair([SQUARE], [SQUARE])),
        image_metrics(make_pair([SQUARE], [])),
    ]
    summary = aggregate(rows)
    assert summary["images"] == 2
    assert summary["images_without_detection"] == 1
    assert summary["images_without_detection_rate"] == pytest.approx(0.5)
    assert "fallback_rate" not in summary
    assert summary["worst_cropped_leaf_percent"] == pytest.approx(100.0)


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(DownstreamMetricsError):
        aggregate([])


def test_group_by_source_separates_the_domain_gap() -> None:
    rows = [
        image_metrics(make_pair([SQUARE], [SQUARE], source="corn")),
        image_metrics(make_pair([SQUARE], [HALF], source="corn_leaf_diseases")),
    ]
    grouped = group_by(rows, "source_dataset")
    assert set(grouped) == {"corn", "corn_leaf_diseases"}
    assert grouped["corn"]["mean_leaf_pixel_recall"] > (
        grouped["corn_leaf_diseases"]["mean_leaf_pixel_recall"]
    )


def test_group_by_area_bin_uses_ground_truth_size() -> None:
    tiny = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]
    huge = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    rows = [
        image_metrics(make_pair([tiny], [tiny])),
        image_metrics(make_pair([huge], [huge])),
    ]
    grouped = group_by_area_bin(rows)
    assert "small" in grouped
    assert "large" in grouped


def test_per_image_metrics_expose_border_resolution_and_leaf_count_groups() -> None:
    border_polygon = [(0.0, 0.1), (0.5, 0.1), (0.5, 0.9), (0.0, 0.9)]
    pair = make_pair([border_polygon], [border_polygon], instances=2)
    metrics = image_metrics(pair)

    assert metrics["truth_touches_border"] is True
    assert metrics["border_contact"] == "touching"
    assert metrics["resolution_bin"] == "low_<=0.10mp"
    assert metrics["leaf_count_group"] == "multiple"


def test_multi_leaf_subset_is_reported_separately() -> None:
    rows = [
        image_metrics(make_pair([SQUARE], [SQUARE], instances=1)),
        image_metrics(make_pair([SQUARE], [HALF], instances=3)),
    ]
    summary = aggregate(rows)
    assert summary["multi_leaf_images"] == 1
    assert summary["mean_leaf_pixel_recall_multi_leaf"] < 0.6


def test_read_yolo_polygons_skips_missing_file_and_invalid_lines(tmp_path: Path) -> None:
    assert read_yolo_polygons(tmp_path / "missing.txt") == []
    label = tmp_path / "label.txt"
    label.write_text(
        "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6\n"
        "\n"
        "0 0.1 0.1\n",
        encoding="utf-8",
    )
    polygons = read_yolo_polygons(label)
    assert len(polygons) == 1
    assert len(polygons[0]) == 4


def _write_fixture(root: Path) -> Path:
    dataset = root / "detector_dataset"
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "manifests").mkdir(parents=True)
    polygon = "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6\n"
    for name in ("a", "b"):
        (dataset / "labels" / "val" / f"{name}.txt").write_text(polygon, encoding="utf-8")
    with (dataset / "manifests" / "split_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split", "filename", "source_dataset", "orientation",
                "instance_count", "materialized_label_path", "width", "height",
            ),
        )
        writer.writeheader()
        for name, source in (("a.jpg", "corn"), ("b.jpg", "corn_leaf_diseases")):
            writer.writerow({
                "split": "val",
                "filename": name,
                "source_dataset": source,
                "orientation": "square",
                "instance_count": "1",
                "materialized_label_path": f"labels/val/{Path(name).stem}.txt",
                "width": "224",
                "height": "224",
            })
        writer.writerow({
            "split": "train",
            "filename": "ignored.jpg",
            "source_dataset": "corn",
            "orientation": "square",
            "instance_count": "1",
            "materialized_label_path": "labels/val/a.txt",
            "width": "224",
            "height": "224",
        })
    return dataset


def test_evaluate_downstream_reads_only_the_requested_split(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "a.txt").write_text(
        "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6\n", encoding="utf-8"
    )
    rows, summary = evaluate_downstream(
        dataset_root=dataset,
        prediction_root=predictions,
        split="val",
        raster_size=64,
    )
    assert len(rows) == 2
    assert summary["overall"]["images"] == 2
    # "b" no tiene predicción: cuenta como fallback y sin detección.
    assert summary["overall"]["images_without_detection_rate"] == pytest.approx(0.5)
    assert set(summary["by_source"]) == {"corn", "corn_leaf_diseases"}
    assert summary["by_source"]["corn"]["mean_iou"] == pytest.approx(1.0, abs=0.02)
    assert set(summary["by_border_contact"]) == {"interior"}
    assert set(summary["by_resolution_bin"]) == {"low_<=0.10mp"}
    assert set(summary["by_leaf_count"]) == {"single"}


def test_build_pairs_requires_ground_truth_geometry(tmp_path: Path) -> None:
    dataset = _write_fixture(tmp_path)
    (dataset / "labels" / "val" / "a.txt").write_text("", encoding="utf-8")
    with pytest.raises(DownstreamMetricsError):
        build_pairs(
            [{
                "filename": "a.jpg",
                "source_dataset": "corn",
                "orientation": "square",
                "instance_count": "1",
                "materialized_label_path": "labels/val/a.txt",
                "width": "224",
                "height": "224",
            }],
            dataset,
            tmp_path / "predictions",
            raster_size=64,
        )
