"""El gate por defecto debe ser exactamente el calibrado que declara el YAML."""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from src.evaluation import segmentation_gate_calibration
from src.segmentation import quality
from src.segmentation.quality import SegmentationQualityGateConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "segmentation.yaml"


def _configured_gate() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return payload["segmentation"]["quality_gate"]


def test_dataclass_defaults_match_the_calibrated_yaml() -> None:
    configured = _configured_gate()
    defaults = SegmentationQualityGateConfig()
    for name, expected in configured.items():
        if name == "reject_multiple_eligible":
            continue
        assert getattr(defaults, name) == expected, name


def test_reject_multiple_eligible_has_a_single_default() -> None:
    configured = _configured_gate()["reject_multiple_eligible"]
    assert configured is False

    functions = (
        quality.assess_segmentation,
        quality.assess_segmentation_legacy,
        segmentation_gate_calibration.calibrated_status,
        segmentation_gate_calibration.evaluate_gate,
    )
    for function in functions:
        parameter = inspect.signature(function).parameters["reject_multiple_eligible"]
        assert parameter.default is configured, function.__qualname__
