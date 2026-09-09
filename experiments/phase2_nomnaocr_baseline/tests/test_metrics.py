"""Unit tests for eval_lib/metrics.py, run on the host (stdlib only, no TF/Docker needed):

    python -m unittest experiments.phase2_nomnaocr_baseline.tests.test_metrics -v
or, from this directory:
    python -m unittest tests.test_metrics -v
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from eval_lib.metrics import (  # noqa: E402
    aggregate,
    aggregate_cer_macro,
    aggregate_cer_micro,
    cer_macro_term,
    cer_micro_terms,
    character_accuracy_counts,
    edit_distance,
    sequence_correct,
    SampleScore,
)


class TestEditDistance(unittest.TestCase):
    def test_textbook_example(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_identical(self):
        self.assertEqual(edit_distance("abc", "abc"), 0)

    def test_empty(self):
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertEqual(edit_distance("abc", ""), 3)
        self.assertEqual(edit_distance("", ""), 0)


class TestSequenceCorrect(unittest.TestCase):
    def test_identical_strings_correct(self):
        self.assertTrue(sequence_correct("abc", "abc"))

    def test_single_char_difference_wrong(self):
        self.assertFalse(sequence_correct("abd", "abc"))

    def test_length_difference_wrong(self):
        self.assertFalse(sequence_correct("ab", "abc"))


class TestCharacterAccuracyCounts(unittest.TestCase):
    def test_perfect_match(self):
        correct, total = character_accuracy_counts("abc", "abc", max_length=10)
        self.assertEqual((correct, total), (3, 3))

    def test_leading_insertion_tanks_score(self):
        """Documents the known positional (non-alignment) weakness: pred="xabc" contains every
        character of gt="abc", just shifted right by one insertion - a real edit-distance-aware
        metric would barely penalize this, but NomNaOCR's positional Character Accuracy scores
        it 0/3, since every position after the insertion is misaligned."""
        correct, total = character_accuracy_counts("xabc", "abc", max_length=10)
        self.assertEqual((correct, total), (0, 3))

    def test_trailing_extra_chars_in_pred_not_penalized(self):
        """gt is shorter than pred: only positions where gt is non-pad count towards `total`, so
        spurious trailing predicted characters beyond gt's length are never compared."""
        correct, total = character_accuracy_counts("abx", "ab", max_length=10)
        self.assertEqual((correct, total), (2, 2))

    def test_partial_match(self):
        correct, total = character_accuracy_counts("abx", "abc", max_length=10)
        self.assertEqual((correct, total), (2, 3))

    def test_truncated_at_max_length(self):
        correct, total = character_accuracy_counts("abcde", "abcde", max_length=3)
        self.assertEqual((correct, total), (3, 3))


class TestCER(unittest.TestCase):
    def test_macro_term_matches_normalized_edit_distance(self):
        self.assertAlmostEqual(cer_macro_term("abx", "abc"), 1 / 3)

    def test_macro_term_empty_gt_is_zero(self):
        self.assertEqual(cer_macro_term("abc", ""), 0.0)

    def test_micro_terms(self):
        self.assertEqual(cer_micro_terms("abx", "abc"), (1, 3))

    def test_macro_vs_micro_differ_on_unequal_length_samples(self):
        """Worked example showing NomNaOCR upstream's own comment (warp_cer_metric's result "is
        the same as" LevenshteinDistance(normalize=True)) does not hold in general: two samples
        with very different ground-truth lengths give different macro vs. micro CER."""
        pairs = [("ay", "ab"), ("abcdefghix", "abcdefghij")]  # (pred, gt), 1 error each, lengths 2 and 10
        macro = aggregate_cer_macro([cer_macro_term(p, g) for p, g in pairs])
        micro = aggregate_cer_micro([cer_micro_terms(p, g) for p, g in pairs])
        self.assertAlmostEqual(macro, (1 / 2 + 1 / 10) / 2)  # 0.3
        self.assertAlmostEqual(micro, 2 / 12)  # ~0.1667
        self.assertNotAlmostEqual(macro, micro)


class TestAggregate(unittest.TestCase):
    def test_overall_and_subset_rollup(self):
        scores = [
            SampleScore.compute("a.jpg", "work1", pred="abc", gt="abc", max_length=10, subsets=["poem"]),
            SampleScore.compute("b.jpg", "work1", pred="abx", gt="abc", max_length=10, subsets=["prose"]),
        ]
        result = aggregate(scores)
        self.assertEqual(result["overall"]["n"], 2)
        self.assertAlmostEqual(result["overall"]["sequence_accuracy"], 0.5)
        self.assertAlmostEqual(result["overall"]["character_accuracy"], 5 / 6)
        self.assertEqual(result["poem"]["n"], 1)
        self.assertEqual(result["prose"]["n"], 1)
        self.assertAlmostEqual(result["poem"]["sequence_accuracy"], 1.0)
        self.assertAlmostEqual(result["prose"]["sequence_accuracy"], 0.0)

    def test_empty_input(self):
        result = aggregate([])
        self.assertEqual(result["overall"], {"n": 0})


if __name__ == "__main__":
    unittest.main()
