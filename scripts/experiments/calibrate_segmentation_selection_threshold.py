#!/usr/bin/env python3
"""Calibrate target-selection confidence on the frozen validation split.

Inference runs once per validation image at the proposal floor. Candidate
selection thresholds are replayed over the same predictions. The internal test
split and the external retained pilot are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import yaml
from PIL import Image

from src.config import get_output_root, get_project_data_root
from src.data.loader import load_and_normalize_image
from src.evaluation.segmentation_downstream import (
    ROW_COLUMNS,
    MaskPair,
    aggregate,
    group_by,
    image_metrics,
    load_manifest_rows,
    rasterize_polygons,
    read_yolo_polygons,
)
from src.evaluation.segmentation_threshold_calibration import (
    choose_selection_threshold,
)
from src.preprocessing.segmented_leaf_processor import (
    LeafMaskProcessorConfig,
    mask_processor_config_from_mapping,
    select_target_leaf,
)
from src.segmentation.leaf_segmenter import LeafInstance, UltralyticsLeafSegmenter

DEFAULT_DATASET = get_project_data_root() / "leaf_detection" / "detector_dataset"
DEFAULT_CHECKPOINT = (
    get_output_root() / "leaf_detection" / "models" / "doctor_maiz_leaf_segmenter_best.pt"
)
DEFAULT_OUTPUT = (
    get_output_root()
    / "leaf_detection"
    / "validation_real_pipeline"
    / "selection_threshold_calibration_v1"
)


def _metric_row(
    manifest: Mapping[str, str],
    raw_instances: tuple[LeafInstance, ...],
    *,
    truth: np.ndarray,
    processor_config: LeafMaskProcessorConfig,
    threshold: float,
    raster_size: int,
    rasterized_predictions: dict[int, np.ndarray],
) -> dict[str, object]:
    image_size = (int(manifest["width"]), int(manifest["height"]))
    selection = select_target_leaf(raw_instances, image_size, processor_config)
    if selection.selected is None:
        prediction = np.zeros_like(truth)
        selected_confidence = None
    else:
        source_index = selection.selected.source_index
        if source_index not in rasterized_predictions:
            rasterized_predictions[source_index] = (
                np.asarray(
                    selection.selected.mask.resize(
                        (raster_size, raster_size),
                        resample=Image.Resampling.NEAREST,
                    ),
                    dtype=np.uint8,
                )
                > 0
            )
        prediction = rasterized_predictions[source_index]
        selected_confidence = selection.selected.confidence
    metrics = image_metrics(
        MaskPair(
            filename=manifest["filename"],
            source_dataset=manifest["source_dataset"],
            orientation=manifest["orientation"],
            truth=truth,
            prediction=prediction,
            truth_instances=int(manifest["instance_count"]),
            predicted_instances=1 if selection.selected is not None else 0,
            fallback=selection.selected is None,
            image_width=image_size[0],
            image_height=image_size[1],
        )
    )
    metrics["selection_threshold"] = threshold
    metrics["selected_confidence"] = selected_confidence
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=Path("config/segmentation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--proposal-threshold", type=float, default=0.10)
    parser.add_argument(
        "--selection-thresholds",
        type=float,
        nargs="+",
        default=(0.20, 0.30, 0.40, 0.50, 0.60),
    )
    parser.add_argument("--raster-size", type=int, default=640)
    parser.add_argument("--minimum-single-leaf-recall", type=float, default=0.97)
    args = parser.parse_args()
    thresholds = tuple(sorted(set(args.selection_thresholds)))
    if not thresholds or args.proposal_threshold > min(thresholds):
        raise SystemExit("proposal-threshold debe ser <= todos los selection-thresholds")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    segmentation = config["segmentation"]
    runtime_cache = args.output_dir / "runtime_cache"
    runtime_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_cache / "matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_cache / "ultralytics"))
    manifest_rows = load_manifest_rows(
        args.dataset_root / "manifests" / "split_manifest.csv", "val"
    )
    segmenter = UltralyticsLeafSegmenter(
        args.checkpoint,
        image_size=int(segmentation["image_size"]),
        proposal_confidence_threshold=args.proposal_threshold,
        iou_threshold=float(segmentation["iou_threshold"]),
        max_detections=int(segmentation["max_detections"]),
        device=args.device,
        expected_version=str(segmentation["ultralytics_version"]),
    )
    base_processor_config = mask_processor_config_from_mapping(segmentation)
    processor_configs = {
        threshold: replace(base_processor_config, confidence_threshold=threshold)
        for threshold in thresholds
    }
    rows_by_threshold: dict[float, list[dict[str, object]]] = {
        threshold: [] for threshold in thresholds
    }
    for index, manifest in enumerate(manifest_rows, start=1):
        image_path = args.dataset_root / manifest["materialized_image_path"]
        image = load_and_normalize_image(str(image_path))
        raw_instances = tuple(segmenter.segment(image))
        truth = rasterize_polygons(
            read_yolo_polygons(args.dataset_root / manifest["materialized_label_path"]),
            args.raster_size,
        )
        rasterized_predictions: dict[int, np.ndarray] = {}
        for threshold in thresholds:
            rows_by_threshold[threshold].append(
                _metric_row(
                    manifest,
                    raw_instances,
                    truth=truth,
                    processor_config=processor_configs[threshold],
                    threshold=threshold,
                    raster_size=args.raster_size,
                    rasterized_predictions=rasterized_predictions,
                )
            )
        if index % 20 == 0 or index == len(manifest_rows):
            print(f"inferencia val: {index}/{len(manifest_rows)}", flush=True)

    candidates: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        rows = rows_by_threshold[threshold]
        single = [row for row in rows if not row["multi_leaf"]]
        multiple = [row for row in rows if row["multi_leaf"]]
        candidates.append(
            {
                "selection_threshold": threshold,
                "overall": aggregate(rows),
                "single_leaf": aggregate(single),
                "multi_leaf_union_diagnostic": aggregate(multiple),
                "by_source": group_by(rows, "source_dataset"),
            }
        )
        all_rows.extend(rows)
    recommended = choose_selection_threshold(
        candidates,
        minimum_single_leaf_recall=args.minimum_single_leaf_recall,
    )
    summary = {
        "schema_version": 1,
        "calibration_split": "val",
        "test_read": False,
        "pilot_read": False,
        "checkpoint": segmenter.to_metadata(),
        "proposal_threshold": args.proposal_threshold,
        "selection_thresholds": list(thresholds),
        "minimum_single_leaf_recall": args.minimum_single_leaf_recall,
        "selection_rule": (
            "on single-leaf val images, meet minimum pixel recall then maximize Dice"
        ),
        "recommended": recommended,
        "candidates": candidates,
        "multi_leaf_note": (
            "Multi-leaf rows compare one selected target against the union truth mask "
            "and are diagnostic only; they do not select the threshold."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ("selection_threshold", "selected_confidence", *ROW_COLUMNS)
    with (args.output_dir / "per_image_threshold_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cast(Any, all_rows))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recommended, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
