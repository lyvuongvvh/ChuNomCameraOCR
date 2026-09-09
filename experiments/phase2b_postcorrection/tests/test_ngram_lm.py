"""Unit tests for lm/ngram_lm.py, run on the host (stdlib only, no TF/Docker needed):

    python -m unittest tests.test_ngram_lm -v
"""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lm.ngram_lm import CharNgramLM, START  # noqa: E402


class TestTraining(unittest.TestCase):
    def test_start_sentinel_excluded_from_unigrams(self):
        lm = CharNgramLM.train(["abc"], order=3)
        self.assertNotIn(START, lm.counts[1])

    def test_higher_order_start_context_kept(self):
        """Bigrams/trigrams that *contain* START as leading context (e.g. "this is the first
        character of a line") are meaningful and should be counted, unlike the pure unigram."""
        lm = CharNgramLM.train(["abc"], order=3)
        self.assertIn(START + "a", lm.counts[2])
        self.assertIn(START * 2 + "a", lm.counts[3])

    def test_unigram_counts(self):
        lm = CharNgramLM.train(["aab"], order=1)
        self.assertEqual(lm.counts[1]["a"], 2)
        self.assertEqual(lm.counts[1]["b"], 1)


class TestScoring(unittest.TestCase):
    def test_prefers_seen_pattern_over_unseen(self):
        texts = ["abcabc", "abcabc", "abcabd"] * 20
        lm = CharNgramLM.train(texts, order=3)
        self.assertGreater(lm.score("abcabc"), lm.score("xyzxyz"))

    def test_unseen_character_does_not_crash(self):
        lm = CharNgramLM.train(["abc"], order=3)
        score = lm.score("zzz")
        self.assertTrue(score < 0)  # log of a small positive floor, not an error/NaN

    def test_score_is_finite_for_empty_text(self):
        lm = CharNgramLM.train(["abc"], order=3)
        self.assertEqual(lm.score(""), 0.0)

    def test_longer_text_has_lower_total_score(self):
        """Total (summed, not length-normalized) log score - a longer text accumulates more
        per-character terms, so it's expected to be more negative even if equally "natural". Uses
        a corpus with genuine next-character uncertainty (each line is 'a' followed by one of
        b/c/d) so per-character scores are meaningfully below 1.0, not near-deterministic."""
        texts = ["ab", "ac", "ad"] * 20
        lm = CharNgramLM.train(texts, order=2)
        self.assertLess(lm.score("abababab"), lm.score("ab"))


class TestSaveLoad(unittest.TestCase):
    def test_round_trip_preserves_scores(self):
        lm = CharNgramLM.train(["abcabc", "abcabd"], order=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "lm.json")
            lm.save(path)
            loaded = CharNgramLM.load(path)
        self.assertEqual(lm.order, loaded.order)
        self.assertAlmostEqual(lm.score("abcabc"), loaded.score("abcabc"))

    def test_saved_file_is_valid_json(self):
        lm = CharNgramLM.train(["abc"], order=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "lm.json")
            lm.save(path)
            data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data["order"], 2)


if __name__ == "__main__":
    unittest.main()
