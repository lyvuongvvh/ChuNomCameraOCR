"""Unit tests for api/pipeline/detection.py - crop math on fixtures, blla.segment (via
segment_page) mocked so these run without kraken/torch actually loaded.

    python -m unittest api.tests.test_pipeline_detection -v
"""
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from PIL import Image

from api.pipeline.detection import crop_region, detect_lines


class TestCropRegion(unittest.TestCase):
    def test_axis_aligned_bbox_of_a_rectangle(self):
        img = Image.new("RGB", (100, 100))
        crop = crop_region(img, [(10, 20), (50, 20), (50, 80), (10, 80)])
        self.assertEqual(crop.size, (40, 60))

    def test_bbox_of_a_non_rectangular_polygon(self):
        """kraken's boundary polygons aren't guaranteed to be exactly 4 points - crop_region
        must take the bounding box over however many points there are, not assume 4 corners."""
        img = Image.new("RGB", (100, 100))
        crop = crop_region(img, [(10, 10), (30, 5), (40, 50), (15, 60), (5, 30)])
        self.assertEqual(crop.size, (35, 55))  # bbox: x in [5,40], y in [5,60]


@dataclass
class _FakeQuad:
    points: tuple


class TestDetectLines(unittest.TestCase):
    @patch("api.pipeline.detection.segment_page")
    def test_returns_point_lists_from_segment_page_quads(self, mock_segment_page):
        mock_segment_page.return_value = [
            _FakeQuad(points=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))),
            _FakeQuad(points=((5.0, 6.0), (7.0, 6.0), (7.0, 8.0), (5.0, 8.0))),
        ]
        result = detect_lines("fake_page.jpg", model_path="fake_model.mlmodel")
        self.assertEqual(result, [
            [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
            [(5.0, 6.0), (7.0, 6.0), (7.0, 8.0), (5.0, 8.0)],
        ])
        mock_segment_page.assert_called_once_with("fake_page.jpg", model_path="fake_model.mlmodel")

    @patch("api.pipeline.detection.segment_page")
    def test_no_lines_detected_returns_empty_list(self, mock_segment_page):
        mock_segment_page.return_value = []
        self.assertEqual(detect_lines("fake_page.jpg", model_path="fake_model.mlmodel"), [])


if __name__ == "__main__":
    unittest.main()
