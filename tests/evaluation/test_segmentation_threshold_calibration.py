"""Tests for validation-only selection-threshold choice."""

from __future__ import annotations

import pytest

from src.evaluation.segmentation_threshold_calibration import (
    ThresholdCalibrationError,
    choose_selection_threshold,
)


def _candidate(threshold: float, recall: float, dice: float, precision: float):
    return {
        "selection_threshold": threshold,
        "single_leaf": {
            "mean_leaf_pixel_recall": recall,
            "mean_dice": dice,
            "mean_leaf_pixel_precision": precision,
            "images_without_detection_rate": 0.0,
        },
    }


def test_choice_enforces_recall_then_maximizes_dice() -> None:
    candidates = [
        _candidate(0.2, 0.98, 0.91, 0.88),
        _candidate(0.4, 0.97, 0.94, 0.93),
        _candidate(0.6, 0.95, 0.96, 0.97),
    ]
    selected = choose_selection_threshold(candidates, minimum_single_leaf_recall=0.97)
    assert selected["selection_threshold"] == 0.4


def test_choice_falls_back_to_maximum_recall_when_constraint_is_unmet() -> None:
    candidates = [_candidate(0.2, 0.95, 0.90, 0.88), _candidate(0.5, 0.90, 0.94, 0.96)]
    selected = choose_selection_threshold(candidates, minimum_single_leaf_recall=0.97)
    assert selected["selection_threshold"] == 0.2


def test_choice_rejects_empty_candidates_and_invalid_constraint() -> None:
    with pytest.raises(ThresholdCalibrationError):
        choose_selection_threshold([])
    with pytest.raises(ValueError):
        choose_selection_threshold([_candidate(0.5, 1.0, 1.0, 1.0)], minimum_single_leaf_recall=1.1)
