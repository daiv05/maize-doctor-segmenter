from __future__ import annotations

import base64
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from unittest import TestCase

import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing.segmented_leaf_processor import (
    MASK_BLACK,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessor,
)
from src.segmentation.leaf_segmenter import LeafInstance
from src.segmentation.quality import (
    SegmentationQualityGateConfig,
    SegmentationStatus,
    assess_segmentation,
)


def _mask(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(
        (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=255
    )
    return mask


def _instance(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    confidence: float,
    source_index: int,
) -> LeafInstance:
    return LeafInstance(_mask(size, bbox), confidence, bbox, source_index)


def _instance_from_mask(
    mask: Image.Image,
    confidence: float = 0.95,
    source_index: int = 0,
) -> LeafInstance:
    return LeafInstance(
        mask=mask,
        confidence=confidence,
        bbox=mask.getbbox() or (0, 0, 1, 1),
        source_index=source_index,
    )


class _StaticSegmenter:
    def __init__(self, instances: Sequence[LeafInstance]) -> None:
        self.instances = tuple(instances)

    def segment(self, image: Image.Image) -> tuple[LeafInstance, ...]:
        return self.instances

    def to_metadata(self) -> dict[str, object]:
        return {"segmenter_model": "static", "proposal_confidence_threshold": 0.5}


class SegmentationQualityTests(TestCase):
    def setUp(self) -> None:
        self.size = (100, 80)
        self.image = Image.new("RGB", self.size, (20, 120, 40))

    def _process(self, instances: Sequence[LeafInstance]):
        processor = SegmentedLeafProcessor(
            _StaticSegmenter(instances),
            LeafMaskProcessorConfig(processing_profile=MASK_BLACK),
        )
        return processor.process(self.image)

    def test_clear_leaf_is_reliable(self) -> None:
        result = assess_segmentation(
            self._process((_instance(self.size, (20, 10, 80, 70), 0.95, 0),))
        )

        self.assertEqual(result.status, SegmentationStatus.RELIABLE)
        self.assertIsNone(result.reason)

    def test_no_detection_and_low_confidence_fail(self) -> None:
        no_detection = assess_segmentation(self._process(()))
        low_confidence = assess_segmentation(
            self._process((_instance(self.size, (20, 10, 80, 70), 0.49, 0),))
        )

        self.assertEqual(no_detection.reason, "no_detection")
        self.assertEqual(low_confidence.reason, "low_segmentation_confidence")

    def test_multiple_leaves_use_the_configured_policy(self) -> None:
        instances = (
            _instance(self.size, (5, 5, 45, 60), 0.90, 0),
            _instance(self.size, (50, 10, 95, 70), 0.88, 1),
        )
        processing = self._process(instances)

        margin_checked = assess_segmentation(processing)
        rejected = assess_segmentation(processing, reject_multiple_eligible=True)

        self.assertEqual(margin_checked.reason, "ambiguous_instance_score_margin")
        self.assertEqual(rejected.reason, "ambiguous_multiple_eligible_leaves")

    def test_clearly_selected_multiple_leaf_result_can_be_reliable(self) -> None:
        instances = (
            _instance(self.size, (10, 5, 80, 75), 0.98, 0),
            _instance(self.size, (85, 2, 98, 15), 0.70, 1),
        )

        result = assess_segmentation(
            self._process(instances), reject_multiple_eligible=False
        )

        self.assertEqual(result.status, SegmentationStatus.RELIABLE)
        self.assertGreaterEqual(
            result.quality_gate_metrics["instance_score_margin"], 0.33
        )

    def test_large_or_irregular_masks_are_uncertain(self) -> None:
        near_full_array = np.ones((self.size[1], self.size[0]), dtype=np.uint8) * 255
        near_full_array[0, 0] = 0
        near_full = assess_segmentation(
            self._process((_instance_from_mask(Image.fromarray(near_full_array)),))
        )

        irregular_array = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
        for start in range(0, self.size[0], 5):
            irregular_array[:, start : start + 3] = 255
        irregular = assess_segmentation(
            self._process((_instance_from_mask(Image.fromarray(irregular_array)),))
        )

        self.assertEqual(near_full.reason, "excessive_mask_area_ratio")
        self.assertEqual(irregular.reason, "suspicious_large_mask_geometry")

    def test_real_problematic_mask_is_not_reliable(self) -> None:
        fixture = Path(__file__).parents[1] / "fixtures" / "segmentation" / "67220087_mask.png.b64"
        mask = Image.open(BytesIO(base64.b64decode(fixture.read_text().strip()))).convert("L")
        self.image = Image.new("RGB", mask.size, (20, 120, 40))

        result = assess_segmentation(
            self._process((_instance_from_mask(mask, confidence=0.6492254137992859),))
        )

        self.assertEqual(result.status, SegmentationStatus.UNCERTAIN)
        self.assertEqual(result.reason, "suspicious_large_mask_geometry")

    def test_config_mapping_and_metadata_are_auditable(self) -> None:
        config = SegmentationQualityGateConfig.from_mapping(
            {
                "max_mask_area_ratio": 0.999,
                "large_mask_area_ratio": 0.50,
                "min_large_mask_bbox_ratio": 0.70,
                "max_large_mask_normalized_perimeter": 8.0,
                "min_multi_instance_score_margin": 0.33,
            }
        )
        result = assess_segmentation(
            self._process(
                (
                    _instance(self.size, (5, 5, 45, 60), 0.90, 0),
                    _instance(self.size, (50, 10, 95, 70), 0.88, 1),
                )
            ),
            reject_multiple_eligible=False,
            quality_gate=config,
        )
        gate = result.to_metadata()["quality_gate"]

        self.assertEqual(gate["reasons"], ["ambiguous_instance_score_margin"])
        self.assertEqual(gate["thresholds"]["min_multi_instance_score_margin"], 0.33)
