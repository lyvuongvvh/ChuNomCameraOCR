"""Unit tests for eval_lib/scoring.py - no network, no real Anthropic API, no Docker/TF:

    python -m unittest tests.test_scoring -v
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval_lib.scoring import (  # noqa: E402
    content_words,
    edit_similarity,
    strip_low_confidence_note,
    word_jaccard,
    word_recall,
)


class TestStripLowConfidenceNote(unittest.TestCase):
    def test_strips_trailing_note(self):
        raw = 'Ta sai sứ giả sang thông hiếu.\n\n[LOW CONFIDENCE: câu bị cắt ngang ở cuối]'
        self.assertEqual(strip_low_confidence_note(raw), "Ta sai sứ giả sang thông hiếu.")

    def test_leaves_clean_translation_unchanged(self):
        raw = "Trăm năm trong cõi người ta,"
        self.assertEqual(strip_low_confidence_note(raw), raw)

    def test_strips_note_regardless_of_newline_count(self):
        """Regression test: real data has the note glued on with 0, 1, or 2 newlines before it
        (2,738 lines with 2, 39 with 1, 6 with 0, out of 2,783 total low-confidence lines) - the
        model's own formatting isn't perfectly consistent despite the system prompt asking for
        "a new line" (singular). An earlier version of this regex required exactly "\\n\\n" and
        silently left the note attached (leaking into scoring) on the other 45 lines."""
        one_newline = 'Sau hiên.\n[LOW CONFIDENCE: khó xác định]'
        zero_newline = 'Sau hiên.[LOW CONFIDENCE: khó xác định]'
        self.assertEqual(strip_low_confidence_note(one_newline), "Sau hiên.")
        self.assertEqual(strip_low_confidence_note(zero_newline), "Sau hiên.")


class TestContentWords(unittest.TestCase):
    def test_excludes_stopwords(self):
        words = content_words("Đây là một câu có nhiều từ")
        self.assertNotIn("la", words)
        self.assertNotIn("mot", words)
        self.assertNotIn("co", words)

    def test_diacritics_stripped_for_comparison(self):
        self.assertEqual(content_words("Sứ giả"), content_words("su gia"))

    def test_keeps_real_content_words(self):
        words = content_words("Vân Tiên ngồi lược")
        self.assertIn("van", words)
        self.assertIn("tien", words)
        self.assertIn("luoc", words)


class TestEditSimilarity(unittest.TestCase):
    def test_identical_strings_score_one(self):
        self.assertEqual(edit_similarity("Trăm năm", "Trăm năm"), 1.0)

    def test_completely_different_strings_score_low(self):
        self.assertLess(edit_similarity("Trăm năm trong cõi người ta", "xyz"), 0.3)

    def test_both_empty_scores_one(self):
        self.assertEqual(edit_similarity("", ""), 1.0)


class TestWordJaccard(unittest.TestCase):
    def test_identical_sentences_score_one(self):
        self.assertEqual(word_jaccard("Vân Tiên ngồi lược", "Vân Tiên ngồi lược"), 1.0)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(word_jaccard("Vân Tiên ngồi lược", "xyz abc def"), 0.0)

    def test_partial_overlap_scores_between(self):
        score = word_jaccard("Vân Tiên ngồi lược qua nhà", "Vân Tiên đứng dậy đi ra")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_translation_scores_zero_not_error(self):
        self.assertEqual(word_jaccard("", "Vân Tiên ngồi lược"), 0.0)


class TestWordRecall(unittest.TestCase):
    def test_all_translation_words_found_in_larger_reference(self):
        """DVSKTT's real use case: the reference is a whole leaf (many sentences), the
        translation is one short line - recall should still be high if the translation's own
        words genuinely appear somewhere in that larger text."""
        translation = "Vua sai sứ giả"
        reference = "Mùa xuân, vua sai sứ giả sang nhà Minh tạ ơn. Tháng sau, có việc khác xảy ra."
        self.assertEqual(word_recall(translation, reference), 1.0)

    def test_missing_words_reduce_recall(self):
        translation = "Vua sai sứ giả sang thông hiếu"
        reference = "Vua sai sứ giả đi."
        score = word_recall(translation, reference)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_translation_scores_zero_not_error(self):
        self.assertEqual(word_recall("", "some reference text"), 0.0)


if __name__ == "__main__":
    unittest.main()
