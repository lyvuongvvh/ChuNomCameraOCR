"""Wraps CRNNRecognizer.predict_beams (Phase 0/2b) + rescore_patch (Phase 2b) - the same
baseline-recognizer + post-correction combination Phase 3's translation pipeline was validated
against (not any Phase 2c fine-tuned checkpoint, whose gains were within noise and plateaued by
epoch 1 - see the Phase 4 plan's §3).
"""
from __future__ import annotations

import pathlib
import tempfile

import api._pipeline_paths  # noqa: F401,E402

from lm.rescoring import rescore_patch  # noqa: E402
from nomnaocr_lib.model import CRNNRecognizer  # noqa: E402
from lm.ngram_lm import CharNgramLM  # noqa: E402

from PIL import Image  # noqa: E402


def recognize_crop(
    crop: Image.Image,
    recognizer: CRNNRecognizer,
    lm: CharNgramLM,
    lam: float,
    beam_width: int,
) -> tuple[str, float]:
    """Returns (corrected_text, confidence) for one detected line's cropped image.
    `predict_beams` only accepts a file path (no in-memory-image overload), so the crop is
    written to a per-call temp file rather than modifying the already-validated recognizer.
    `confidence` is that winning text's own ctc_logp, looked back up out of the beams list -
    rescore_patch itself only returns the winning string, not a score, and modifying it isn't
    warranted just to expose one."""
    # A TemporaryDirectory + explicit filename, not tempfile.NamedTemporaryFile: on Windows,
    # NamedTemporaryFile's own open handle exclusively locks the file, so a second open (here,
    # predict_beams' own tf.io.read_file) would fail while the first handle is still held.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = str(pathlib.Path(tmp_dir) / "crop.jpg")
        crop.convert("RGB").save(tmp_path)
        beams = recognizer.predict_beams(tmp_path, beam_width=beam_width)

    text = rescore_patch(beams, lm, lam)
    confidence = next((logp for candidate, logp in beams if candidate == text), None)
    return text, confidence
