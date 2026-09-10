"""Unit tests for translate_lib/reading.py, run on the host (stdlib only):

    python -m unittest tests.test_reading -v
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from translate_lib.reading import apply_reading_dict, coverage  # noqa: E402

TOY_DICT = {
    "使": {"readings": ["sứ", "sử"], "sources": ["unihan"]},
    "通": {"readings": ["thông"], "sources": ["unihan"]},
    "好": {"readings": [], "sources": []},  # entry exists but empty readings - treated as uncovered
}


class TestApplyReadingDict(unittest.TestCase):
    def test_known_characters_use_first_reading(self):
        self.assertEqual(apply_reading_dict("使通", TOY_DICT), "sứ thông")

    def test_unknown_character_passed_through_bracketed(self):
        self.assertEqual(apply_reading_dict("使X", TOY_DICT), "sứ [X]")

    def test_entry_with_empty_readings_treated_as_unknown(self):
        self.assertEqual(apply_reading_dict("好", TOY_DICT), "[好]")

    def test_empty_string(self):
        self.assertEqual(apply_reading_dict("", TOY_DICT), "")


class TestCoverage(unittest.TestCase):
    def test_partial_coverage(self):
        self.assertEqual(coverage("使通X", TOY_DICT), (2, 3))

    def test_full_coverage(self):
        self.assertEqual(coverage("使通", TOY_DICT), (2, 2))

    def test_zero_coverage(self):
        self.assertEqual(coverage("XYZ", TOY_DICT), (0, 3))


if __name__ == "__main__":
    unittest.main()
