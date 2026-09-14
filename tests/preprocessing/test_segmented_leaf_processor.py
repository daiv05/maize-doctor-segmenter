"""Representative normal and edge cases for segmentation-first preprocessing."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

from PIL import Image, ImageDraw

from src.preprocessing.segmented_leaf_processor import (
    BBOX_CROP,
    CROP_MASK_LETTERBOX,
    FALLBACK_REJECT,
    MASK_BLACK,
    SQUARE_CROP,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessor,
    mask_processor_config_from_mapping,
)
from src.segmentation.leaf_segmenter import LeafInstance


def _mask(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    result = Image.new("L", size, 0)
    ImageDraw.Draw(result).rectangle(
        (bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1),
        fill=255,
    )
    return result


def _instance(
    size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    confidence: float,
    index: int,
) -> LeafInstance:
    return LeafInstance(_mask(size, bbox), confidence, bbox, index)


class _StaticSegmenter:
    def __init__(self, instances: tuple[LeafInstance, ...]) -> None:
        self.instances = instances
        self.calls = 0

    def segment(self, image: Image.Image) -> tuple[LeafInstance, ...]:
        self.calls += 1
        return self.instances

    def to_metadata(self) -> dict[str, object]:
        return {
            "segmenter_model": "static-test-segmenter",
            "segmenter_checkpoint": "fixture.pt",
            "segmenter_checkpoint_sha256": "fixture-sha256",
        }


class SegmentedLeafProcessorTests(TestCase):
    def setUp(self) -> None:
        self.size = (100, 80)
        self.image = Image.new("RGB", self.size, (25, 120, 45))

    def _processor(
        self,
        instances: tuple[LeafInstance, ...],
        **config: Any,
    ) -> SegmentedLeafProcessor:
        return SegmentedLeafProcessor(
            _StaticSegmenter(instances),
            LeafMaskProcessorConfig(**config),
        )

    def test_config_mapping_can_override_selection_confidence(self) -> None:
        segmentation = {
            "proposal_confidence_threshold": 0.2,
            "selection_confidence_threshold": 0.5,
            "output_profile": "mask_black",
            "min_mask_area_ratio": 0.01,
            "near_full_warning_ratio": 0.98,
            "background_value": [0, 0, 0],
            "target_size": [224, 224],
            "selection_weights": {
                "area": 0.45,
                "center": 0.35,
                "confidence": 0.20,
            },
            "fallback": "original",
        }

        config = mask_processor_config_from_mapping(
            segmentation,
            selection_confidence_threshold=0.125,
        )

        self.assertEqual(config.confidence_threshold, 0.125)

    def test_config_mapping_supports_legacy_confidence_key(self) -> None:
        segmentation = {
            "confidence_threshold": 0.5,
            "output_profile": "mask_black",
            "min_mask_area_ratio": 0.01,
            "near_full_warning_ratio": 0.98,
            "background_value": [0, 0, 0],
            "target_size": [224, 224],
            "selection_weights": {
                "area": 0.45,
                "center": 0.35,
                "confidence": 0.20,
            },
            "fallback": "original",
        }

        config = mask_processor_config_from_mapping(segmentation)

        self.assertEqual(config.confidence_threshold, 0.5)

    def test_one_clear_leaf_produces_black_background(self) -> None:
        processor = self._processor((_instance(self.size, (20, 10, 80, 70), 0.95, 0),))

        result = processor.process(self.image)

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.selected_instance, 0)
        self.assertEqual(result.bbox, (20, 10, 80, 70))
        self.assertEqual(result.processed_image.size, self.size)  # type: ignore[union-attr]
        self.assertEqual(result.processed_image.getpixel((0, 0)), (0, 0, 0))  # type: ignore[union-attr]
        self.assertEqual(result.processed_image.getpixel((50, 40)), (25, 120, 45))  # type: ignore[union-attr]
        self.assertEqual(result.to_metadata()["selection_confidence_threshold"], 0.5)

    def test_multiple_leaves_selects_large_centered_target(self) -> None:
        instances = (
            _instance(self.size, (0, 0, 20, 20), 0.99, 0),
            _instance(self.size, (20, 15, 85, 70), 0.72, 1),
            _instance(self.size, (80, 60, 100, 80), 0.90, 2),
        )

        result = self._processor(instances).process(self.image)

        self.assertEqual(result.selected_instance, 1)
        self.assertIn("3 hojas elegibles", result.warnings[-1])
        self.assertEqual(len(result.selection_traces), 3)

    def test_complex_background_is_removed_without_touching_leaf_pixels(self) -> None:
        image = Image.new("RGB", self.size, (180, 80, 20))
        ImageDraw.Draw(image).rectangle((25, 15, 74, 64), fill=(10, 200, 50))
        processor = self._processor((_instance(self.size, (25, 15, 75, 65), 0.9, 0),))

        result = processor.process(image)

        self.assertEqual(result.masked_image.getpixel((10, 10)), (0, 0, 0))  # type: ignore[union-attr]
        self.assertEqual(result.masked_image.getpixel((50, 40)), (10, 200, 50))  # type: ignore[union-attr]

    def test_partially_visible_leaf_touching_border_is_accepted(self) -> None:
        result = self._processor(
            (_instance(self.size, (0, 20, 70, 75), 0.88, 0),)
        ).process(self.image)

        self.assertFalse(result.fallback_used)
        self.assertEqual(result.bbox, (0, 20, 70, 75))

    def test_small_leaf_uses_original_fallback(self) -> None:
        result = self._processor(
            (_instance(self.size, (0, 0, 5, 5), 0.99, 0),),
            min_mask_area_ratio=0.01,
        ).process(self.image)

        self.assertTrue(result.fallback_used)
        self.assertIn("sin instancias elegibles", result.fallback_reason or "")
        self.assertEqual(result.processed_image.tobytes(), self.image.tobytes())  # type: ignore[union-attr]

    def test_no_leaf_uses_traced_fallback(self) -> None:
        result = self._processor(()).process(self.image)

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.number_of_instances, 0)
        self.assertEqual(result.fallback_reason, "sin instancias")

    def test_low_confidence_leaf_uses_fallback(self) -> None:
        result = self._processor(
            (_instance(self.size, (10, 10, 90, 70), 0.49, 0),)
        ).process(self.image)

        self.assertTrue(result.fallback_used)
        self.assertIn("confidence", result.selection_traces[0].reason or "")

    def test_empty_corrupt_mask_does_not_crash_pipeline(self) -> None:
        empty = LeafInstance(Image.new("L", self.size, 0), 0.99, (0, 0, 1, 1), 0)

        result = self._processor((empty,)).process(self.image)

        self.assertTrue(result.fallback_used)
        self.assertIn("corrupta", result.selection_traces[0].reason or "")

    def test_exact_full_mask_is_rejected_as_degenerate(self) -> None:
        result = self._processor(
            (_instance(self.size, (0, 0, 100, 80), 0.99, 0),)
        ).process(self.image)

        self.assertTrue(result.fallback_used)
        self.assertIn("exactamente", result.selection_traces[0].reason or "")

    def test_near_full_mask_is_accepted_with_warning(self) -> None:
        result = self._processor(
            (_instance(self.size, (0, 0, 99, 80), 0.99, 0),)
        ).process(self.image)

        self.assertFalse(result.fallback_used)
        self.assertTrue(any("casi a cuadro completo" in item for item in result.warnings))

    def test_horizontal_and_vertical_images_preserve_original_resolution(self) -> None:
        for size in ((160, 60), (60, 160)):
            image = Image.new("RGB", size, (4, 5, 6))
            instance = _instance(size, (5, 5, size[0] - 5, size[1] - 5), 0.9, 0)

            result = self._processor((instance,)).process(image)

            self.assertEqual(result.mask.size, size)  # type: ignore[union-attr]
            self.assertEqual(result.processed_image.size, size)  # type: ignore[union-attr]

    def test_bbox_crop_and_crop_letterbox_are_explicit_profiles(self) -> None:
        instance = _instance(self.size, (20, 10, 80, 70), 0.95, 0)
        cropped = self._processor(
            (instance,), processing_profile=BBOX_CROP
        ).process(self.image)
        letterboxed = self._processor(
            (instance,),
            processing_profile=CROP_MASK_LETTERBOX,
            target_size=(120, 200),
        ).process(self.image)

        self.assertEqual(cropped.processed_image.size, (60, 60))  # type: ignore[union-attr]
        self.assertEqual(letterboxed.processed_image.size, (200, 120))  # type: ignore[union-attr]

    def test_reject_fallback_returns_no_processed_image(self) -> None:
        result = self._processor((), fallback=FALLBACK_REJECT).process(self.image)

        self.assertTrue(result.fallback_used)
        self.assertIsNone(result.processed_image)

    def test_debug_bundle_contains_visuals_and_trace_without_overwrite(self) -> None:
        processor = self._processor((_instance(self.size, (20, 10, 80, 70), 0.95, 0),))
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "image_001"

            result = processor.process(self.image, source_image="leaf.jpg", debug_dir=output)

            expected = {
                "original.jpg",
                "mask.png",
                "overlay.jpg",
                "masked_black.png",
                "crop.png",
                "comparison.jpg",
                "metadata.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["selected_instance"], 0)
            self.assertEqual(metadata["segmenter_checkpoint"], "fixture.pt")
            self.assertEqual(metadata, result.to_metadata())
            with self.assertRaises(FileExistsError):
                processor.process(self.image, debug_dir=output)

    def test_original_image_is_not_modified(self) -> None:
        before = self.image.tobytes()
        processor = self._processor((_instance(self.size, (20, 10, 80, 70), 0.95, 0),))

        processor.process(self.image)

        self.assertEqual(self.image.size, self.size)
        self.assertEqual(self.image.tobytes(), before)
        self.assertEqual(MASK_BLACK, processor.config.processing_profile)

    def test_square_crop_profile_centers_leaf_without_black_background(self) -> None:
        # Imagen 100x80 con fondo (25, 120, 45)
        instance = _instance(self.size, (30, 20, 70, 60), 0.95, 0)
        processor = self._processor(
            (instance,),
            processing_profile=SQUARE_CROP,
            target_size=(224, 224),
        )
        result = processor.process(self.image)

        self.assertFalse(result.fallback_used)
        self.assertIsNotNone(result.processed_image)
        self.assertEqual(result.processed_image.size, (224, 224))  # type: ignore[union-attr]
        # Verificar que la esquina superior izquierda tiene el fondo original
        # (25, 120, 45), no negro
        corner_pixel = result.processed_image.getpixel((0, 0))  # type: ignore[union-attr]
        self.assertEqual(corner_pixel, (25, 120, 45))

