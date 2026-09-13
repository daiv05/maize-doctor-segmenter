"""Tests for aspect-ratio-preserving image output."""

from unittest import TestCase

from PIL import Image

from src.preprocessing.letterbox import letterbox_image


class LetterboxTests(TestCase):
    def test_horizontal_image_to_square_target(self) -> None:
        result = letterbox_image(Image.new("RGB", (200, 100), "green"), (224, 224))

        self.assertEqual(result.image.size, (224, 224))
        self.assertEqual(result.resized_size_width_height, (224, 112))
        self.assertEqual(result.padding, (0, 56, 0, 56))

    def test_vertical_image_to_square_target(self) -> None:
        result = letterbox_image(Image.new("RGB", (100, 200), "green"), (224, 224))

        self.assertEqual(result.resized_size_width_height, (112, 224))
        self.assertEqual(result.padding, (56, 0, 56, 0))

    def test_square_image_has_no_padding(self) -> None:
        result = letterbox_image(Image.new("RGB", (80, 80)), (224, 224))

        self.assertEqual(result.resized_size_width_height, (224, 224))
        self.assertEqual(result.padding, (0, 0, 0, 0))

    def test_rectangular_target_uses_height_width_convention(self) -> None:
        result = letterbox_image(Image.new("RGB", (200, 100)), (224, 320))

        self.assertEqual(result.image.size, (320, 224))
        self.assertEqual(result.resized_size_width_height, (320, 160))
        self.assertEqual(result.padding, (0, 32, 0, 32))

    def test_aspect_ratio_is_conserved(self) -> None:
        result = letterbox_image(Image.new("RGB", (700, 300)), (224, 224))

        source_ratio = 700 / 300
        resized_ratio = result.resized_size_width_height[0] / result.resized_size_width_height[1]
        self.assertAlmostEqual(resized_ratio, source_ratio, delta=0.02)

    def test_rgb_padding_value_is_applied(self) -> None:
        result = letterbox_image(
            Image.new("RGB", (200, 100), (20, 30, 40)),
            (100, 100),
            padding_value=(1, 2, 3),
        )

        self.assertEqual(result.image.getpixel((0, 0)), (1, 2, 3))
        self.assertEqual(result.image.getpixel((50, 50)), (20, 30, 40))

    def test_original_image_is_unchanged(self) -> None:
        image = Image.new("RGB", (31, 17), (7, 8, 9))
        before = image.tobytes()

        letterbox_image(image, (80, 120))

        self.assertEqual(image.size, (31, 17))
        self.assertEqual(image.tobytes(), before)

    def test_small_valid_target_never_produces_zero_resize(self) -> None:
        result = letterbox_image(Image.new("RGB", (1000, 1)), (1, 1))

        self.assertEqual(result.image.size, (1, 1))
        self.assertEqual(result.resized_size_width_height, (1, 1))

    def test_zero_target_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mayores que cero"):
            letterbox_image(Image.new("RGB", (20, 20)), (0, 224))

    def test_invalid_rgb_padding_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tres canales"):
            letterbox_image(Image.new("RGB", (20, 20)), (224, 224), padding_value=(1, 2))
