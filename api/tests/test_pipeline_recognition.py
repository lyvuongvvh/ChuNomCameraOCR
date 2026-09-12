"""Unit tests for api/pipeline/recognition.py - a mocked CRNNRecognizer (no TF model actually
built/run) + a real, tiny CharNgramLM trained in-memory (not the full 12MB char_lm.json).

    python -m unittest api.tests.test_pipeline_recognition -v
"""
import unittest
from unittest.mock import MagicMock

from PIL import Image

import api._pipeline_paths  # noqa: F401
from lm.ngram_lm import CharNgramLM

from api.pipeline.recognition import recognize_crop


class TestRecognizeCrop(unittest.TestCase):
    def setUp(self):
        self.crop = Image.new("RGB", (48, 432))
        self.lm = CharNgramLM.train(["hello world", "hello there"], order=2)

    def test_lam_zero_picks_the_top_ctc_beam(self):
        """With lam=0, rescore_patch's score reduces to ctc_logp alone, so the highest-logp beam
        must win regardless of what the LM thinks of it."""
        recognizer = MagicMock()
        recognizer.predict_beams.return_value = [("xyz", -5.0), ("hello", -10.0)]
        text, confidence = recognize_crop(self.crop, recognizer, self.lm, lam=0.0, beam_width=5)
        self.assertEqual(text, "xyz")
        self.assertEqual(confidence, -5.0)

    def test_lam_nonzero_can_prefer_a_lower_ctc_beam_the_lm_likes_more(self):
        recognizer = MagicMock()
        recognizer.predict_beams.return_value = [("zzzzz", -5.0), ("hello", -5.5)]
        text, confidence = recognize_crop(self.crop, recognizer, self.lm, lam=1.0, beam_width=5)
        self.assertEqual(text, "hello")
        self.assertEqual(confidence, -5.5)

    def test_predict_beams_called_with_a_real_file_path_and_configured_beam_width(self):
        recognizer = MagicMock()
        recognizer.predict_beams.return_value = [("a", -1.0)]
        recognize_crop(self.crop, recognizer, self.lm, lam=0.0, beam_width=7)
        (img_path,), kwargs = recognizer.predict_beams.call_args
        self.assertTrue(img_path.endswith(".jpg"))
        self.assertEqual(kwargs["beam_width"], 7)


if __name__ == "__main__":
    unittest.main()
