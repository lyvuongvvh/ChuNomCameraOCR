"""Unit tests for detect_lib/geometry.py - pure Python + shapely, no kraken/TF needed:

    python -m unittest tests.test_geometry -v
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from detect_lib.geometry import Quad, greedy_match, parse_gts_file, parse_gts_line, polygon_iou  # noqa: E402


def rect(x0, y0, x1, y1, transcript="") -> Quad:
    """A simple axis-aligned rectangle, matching gts/*.txt's own point order
    (top-left, top-right, bottom-right, bottom-left)."""
    return Quad(points=((x0, y0), (x1, y0), (x1, y1), (x0, y1)), transcript=transcript)


class TestParseGtsLine(unittest.TestCase):
    def test_parses_real_format_line(self):
        # a real line from experiments/NomNaOCR/Pages/DVSKTT-1 Quyen thu/gts/DVSKTT_thu_III_1a.txt
        line = "259.0,44.0,280.0,44.0,280.0,206.0,259.0,206.0,大越史記外紀全書"
        quad = parse_gts_line(line)
        self.assertEqual(quad.points, ((259.0, 44.0), (280.0, 44.0), (280.0, 206.0), (259.0, 206.0)))
        self.assertEqual(quad.transcript, "大越史記外紀全書")

    def test_raises_on_wrong_field_count(self):
        with self.assertRaises(ValueError):
            parse_gts_line("1.0,2.0,3.0,4.0,5.0,6.0,7.0,not_enough_fields")

    def test_handles_trailing_newline(self):
        quad = parse_gts_line("0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,x\n")
        self.assertEqual(quad.transcript, "x")


class TestParseGtsFile(unittest.TestCase):
    def test_parses_real_file(self):
        path = pathlib.Path(__file__).resolve().parent.parent.parent / "NomNaOCR" / "Pages" / "DVSKTT-1 Quyen thu" / "gts" / "DVSKTT_thu_III_1a.txt"
        if not path.exists():
            self.skipTest("NomNaOCR dataset not present in this environment")
        quads = parse_gts_file(str(path))
        self.assertGreater(len(quads), 0)
        self.assertTrue(all(q.transcript for q in quads))

    def test_skips_blank_lines(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,a\n\n1.0,1.0,2.0,1.0,2.0,2.0,1.0,2.0,b\n")
            path = f.name
        try:
            quads = parse_gts_file(path)
            self.assertEqual(len(quads), 2)
        finally:
            pathlib.Path(path).unlink()


class TestPolygonIou(unittest.TestCase):
    def test_identical_quads_score_one(self):
        a = rect(0, 0, 10, 10)
        self.assertAlmostEqual(polygon_iou(a, a), 1.0)

    def test_non_overlapping_quads_score_zero(self):
        a = rect(0, 0, 10, 10)
        b = rect(100, 100, 110, 110)
        self.assertEqual(polygon_iou(a, b), 0.0)

    def test_known_partial_overlap(self):
        # two 10x10 squares overlapping in a 5x10 strip: intersection=50, union=150
        a = rect(0, 0, 10, 10)
        b = rect(5, 0, 15, 10)
        self.assertAlmostEqual(polygon_iou(a, b), 50 / 150)


class TestGreedyMatch(unittest.TestCase):
    def test_perfect_match_scores_one_and_one(self):
        gts = [rect(0, 0, 10, 10, "a"), rect(20, 0, 30, 10, "b")]
        preds = [rect(0, 0, 10, 10), rect(20, 0, 30, 10)]
        result = greedy_match(gts, preds)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)

    def test_over_detection_hurts_precision_not_recall(self):
        """Regression test: mirrors the real Phase 0 finding (26 detected vs ~9 expected columns)
        - if the true columns are all found but with extra spurious detections alongside, recall
        should stay high while precision drops, not the other way around. This is exactly the
        "good enough to proceed on recall, needs work on precision" outcome the plan's bars are
        designed to distinguish."""
        gts = [rect(0, 0, 10, 10, "a"), rect(20, 0, 30, 10, "b")]
        preds = [rect(0, 0, 10, 10), rect(20, 0, 30, 10), rect(100, 100, 110, 110)]  # one spurious extra
        result = greedy_match(gts, preds)
        self.assertEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.precision, 2 / 3)

    def test_missed_detection_hurts_recall(self):
        gts = [rect(0, 0, 10, 10, "a"), rect(20, 0, 30, 10, "b")]
        preds = [rect(0, 0, 10, 10)]  # missed the second column entirely
        result = greedy_match(gts, preds)
        self.assertAlmostEqual(result.recall, 0.5)
        self.assertEqual(result.precision, 1.0)

    def test_below_threshold_iou_does_not_count_as_a_match(self):
        gts = [rect(0, 0, 10, 10, "a")]
        preds = [rect(8, 0, 18, 10)]  # small sliver overlap, IoU well under 0.5
        result = greedy_match(gts, preds, iou_threshold=0.5)
        self.assertEqual(result.recall, 0.0)

    def test_empty_gt_and_empty_pred_handled_without_error(self):
        self.assertEqual(greedy_match([], []).recall, 0.0)
        self.assertEqual(greedy_match([rect(0, 0, 1, 1)], []).recall, 0.0)
        self.assertEqual(greedy_match([], [rect(0, 0, 1, 1)]).precision, 0.0)

    def test_one_to_one_not_many_to_one(self):
        """A single large predicted box overlapping two separate gt columns should match at most
        one of them, not double-count - otherwise a detector that just returns one giant box
        covering the whole page would score a perfect, meaningless recall."""
        gts = [rect(0, 0, 10, 10, "a"), rect(20, 0, 30, 10, "b")]
        preds = [rect(0, 0, 30, 10)]  # one big box spanning both, and then some
        result = greedy_match(gts, preds, iou_threshold=0.3)
        self.assertLessEqual(result.matched_gt, 1)


if __name__ == "__main__":
    unittest.main()
