"""Phase 3, Stage 1: apply the combined reading dictionary (scripts/build_reading_dict.py) to
recognized Han-Nom text, producing a best-effort space-separated Sino-Vietnamese/Nom reading
string. This is NOT a translation - it's a per-character phonetic gloss, roughly analogous to
giving each character its "sound" without resolving grammar, word order, or word-sense
ambiguity. Only 56% of this project's character vocabulary has a known reading (see
README.md) - characters with none are passed through unchanged in brackets, not guessed, so
Stage 2 can tell resolved readings apart from gaps it needs to fill from context.
"""
from __future__ import annotations


def apply_reading_dict(text: str, reading_dict: dict) -> str:
    """text: recognized Han-Nom string. reading_dict: {char: {"readings": [...], ...}} as
    written by build_reading_dict.py. Returns a space-separated string, one token per input
    character: the first known reading if covered, or "[char]" if not."""
    tokens = []
    for ch in text:
        entry = reading_dict.get(ch)
        if entry and entry["readings"]:
            tokens.append(entry["readings"][0])
        else:
            tokens.append(f"[{ch}]")
    return " ".join(tokens)


def coverage(text: str, reading_dict: dict) -> tuple[int, int]:
    """Returns (covered_chars, total_chars) for `text` against `reading_dict` - useful for
    reporting per-line or aggregate coverage without re-running apply_reading_dict just to count."""
    total = len(text)
    covered = sum(1 for ch in text if ch in reading_dict and reading_dict[ch]["readings"])
    return covered, total
