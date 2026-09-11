"""Scores Phase 3's `translation` field against the real ground truth built for all three
source works (Kieu/Luc Van Tien: eval_lib/wikisource_alignment.py; DVSKTT: eval_lib/
dvsktt_ground_truth.py) - see results.md for why the two ground-truth kinds need different
metrics, not the same one applied uniformly.

Kieu/Luc Van Tien: the reference is a single verse - a real cross-language(-ish) comparison is
possible directly. Two metrics, not one, because they fail differently:
  - edit_similarity: normalized Levenshtein similarity (reuses eval_lib.wikisource_alignment's
    normalize()/edit_distance()) - sensitive to word ORDER and exact spelling, so it punishes a
    faithful translation that merely reorders clauses or picks a synonym.
    - word_jaccard: content-word set overlap (order-independent, tolerant of paraphrase) - but
    blind to word order and can be fooled by two unrelated sentences that happen to share a few
    common words.
Reporting both, not picking one, is deliberate: agreement between them is a stronger signal than
either alone.

DVSKTT: the reference is a whole leaf's translated text (multiple sentences), not one verse -
edit_similarity against the whole leaf would be dominated by the length mismatch and wouldn't
mean anything. Only word_jaccard-style RECALL is meaningful here: what fraction of the
translation's own content words appear somewhere in the matched leaf's real translation. This is
a coarse "is this translation's vocabulary consistent with the real one for this passage" check,
not a fluency or exactness score.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib.wikisource_alignment import edit_distance, normalize  # noqa: E402

LOW_CONFIDENCE_RE = re.compile(r"\n*\[LOW CONFIDENCE:.*\]\s*$", re.DOTALL)

# Common Vietnamese function words/particles - excluded from word_jaccard so the score reflects
# real content overlap, not just "both sentences use 'là' and 'của' like every Vietnamese
# sentence does." Not exhaustive (no full POS tagger available here) - a pragmatic, documented
# simplification, not a claim of linguistic completeness.
STOPWORDS = frozenset("""
la cua va co duoc nhung cac mot nay do cho de ma thi nen cung khi da se dang khong rat o tai voi
hay hoac neu vi do boi nhung tuy du chi con lai ra vao len xuong di ve toi ta minh nguoi ay no
chung nhu the nao sao vay a oi u nhi nhe roi lai vua cung deu tu den tren duoi trong ngoai giua
sau truoc theo boi vi neu ma thi
""".split())


def strip_low_confidence_note(translation: str) -> str:
    """Removes translate_lib.llm_translate's own "[LOW CONFIDENCE: ...]" tail (see its
    SYSTEM_PROMPT) so scoring compares only the actual translated content, not the model's own
    self-assessment note appended to it."""
    return LOW_CONFIDENCE_RE.sub("", translation).strip()


def content_words(text: str) -> set[str]:
    """Word-level tokens (whitespace/punctuation-split, diacritics stripped via normalize() -
    same normalization Kieu/Luc Van Tien's alignment uses, for the same reason: tone-mark
    differences between two otherwise-identical words shouldn't count as a real mismatch here),
    with common function words (STOPWORDS) excluded."""
    words = re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)
    normalized = {normalize(w) for w in words}
    return {w for w in normalized if w and w not in STOPWORDS}


def edit_similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 1.0
    return 1.0 - edit_distance(na, nb) / max(len(na), len(nb), 1)


def word_jaccard(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def word_recall(translation: str, reference: str) -> float:
    """What fraction of the TRANSLATION's own content words appear anywhere in `reference` -
    used for DVSKTT, where `reference` is a whole leaf's text (many sentences) rather than one
    matching sentence, so precision/Jaccard would be meaningless (the reference necessarily has
    far more distinct words than any single translated line)."""
    wt = content_words(translation)
    if not wt:
        return 0.0
    wr = content_words(reference)
    return len(wt & wr) / len(wt)
