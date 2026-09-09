"""Phase 2b (post-correction): combine each patch's CTC beam-search hypotheses (model confidence)
with the character n-gram LM's score (corpus plausibility) - classic shallow fusion, picking the
candidate maximizing `ctc_logp + lambda * lm_logp` for each line independently.

Both terms are raw summed log values over the whole line (neither length-normalized), so this
implicitly compares candidates of similar length fairly (they're alternate transcriptions of the
SAME line) without needing a separate length-normalization scheme for either term.

Ceiling, worth restating here since it shapes what results to expect: this can only ever pick
among the `beam_width` hypotheses the acoustic model already considered plausible enough to keep
in its beam - see ../README.md's beam-search probe findings for cases where the correct character
never appears in any beam at all, which no amount of LM rescoring here can fix.
"""
from __future__ import annotations

from .ngram_lm import CharNgramLM


def rescore_patch(beams: list, lm: CharNgramLM, lam: float) -> str:
    """beams: list of [text, ctc_logp] pairs (as loaded from beams.json). Returns the text
    maximizing ctc_logp + lam * lm.score(text); lam=0 recovers the model's own top beam."""
    best_text, best_score = None, float("-inf")
    for text, ctc_logp in beams:
        score = ctc_logp + lam * lm.score(text)
        if score > best_score:
            best_text, best_score = text, score
    return best_text


def rescore_all(beams_by_img: dict, lm: CharNgramLM, lam: float) -> dict:
    return {img_name: rescore_patch(beams, lm, lam) for img_name, beams in beams_by_img.items()}
