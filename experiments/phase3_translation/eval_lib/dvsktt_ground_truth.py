"""Ground-truth alignment for Dai Viet Su Ky Toan Thu (DVSKTT) - structurally different from
Kieu/Luc Van Tien (eval_lib/wikisource_alignment.py) in a way that changes the whole approach:

- Kieu/Luc Van Tien were composed IN Vietnamese verse, so Stage 1's phonetic reading of the Nom
  text is already close to the modern spelling - text-similarity alignment against a same-
  language reference works directly.
- DVSKTT is genuine Literary Chinese prose. Stage 1's reading is a Sino-Vietnamese phonetic
  rendering of Han characters (e.g. "su thong hao chap su me..."), which bears no textual
  resemblance to how a real Vietnamese TRANSLATION of that sentence reads (e.g. "Ta sai su gia
  sang thong hieu..."). Text-similarity alignment is not applicable here at all - what's needed
  is a real Han->Viet translation, aligned by physical page structure instead of text content.

Source of the reference text: the real 1993 published translation (Vien Khoa Hoc Xa Hoi Viet
Nam / Nha xuat ban Khoa Hoc Xa Hoi), via its Internet Archive OCR derivative (item "IViTSKTonTh").
Verified as the genuine published translation (not a paraphrase) by exact content match against
dvsktt.com's own citations. Public domain / no known rights holder objection for a 1993
government-institute publication used for non-commercial research here.

Alignment mechanism: the translation's own OCR text carries inline leaf markers like "[1a]",
"[1b]", "[2a]"... marking where each ORIGINAL Han woodblock leaf begins/ends within the
translated prose - the exact same (quyen, leaf, side) identifiers NomNaOCR's own manifest
filenames use (e.g. "DVSKTT_ban_toan_V_30a_9.jpg" = Ban Ky Toan Thu, quyen V, leaf 30 side a,
patch 9). This is a direct structural key lookup, not a fuzzy text match - far more reliable than
Kieu/Luc Van Tien's n-gram-seeded DP, but coarser: it recovers "the translated text of the leaf
this line was written on," not a per-line exact correspondence (Han prose doesn't segment into
one-line-per-idea the way Nom lucbat verse does; multiple OCR patches share one leaf's text).
"""
from __future__ import annotations

import re

# Manifest filename prefix -> the 1993 translation's own section name for that part of the book.
# "thu" (Quyen Thu, front-matter: prefaces/genealogical tables) has NO corresponding section in
# this translation at all - checked, not assumed: the translation's raw text has exactly two top-
# level sections, "Ngoai Ky" (Ngoai Ky Toan Thu) and "Ban Ky" (unified, not split into Toan Thu/
# Thuc Luc/Tuc Bien the way the manifest's filenames are - see QUYEN_OFFSET below for how the
# manifest's split maps onto the translation's single continuous Ban Ky quyen numbering).
SECTION_MAP = {
    "ngoai": "Ngoại Kỷ",
    "ban_toan": "Bản Kỷ",
    "ban_thuc": "Bản Kỷ",
    "ban_tuc": "Bản Kỷ",
}

IMG_NAME_RE = re.compile(r"DVSKTT_(thu|ngoai|ban_toan|ban_thuc|ban_tuc)_([IVXLC]+)_(\d+)([ab])_\d+\.jpg$")
HEADER_RE = re.compile(r"Đại Việt Sử Ký Toàn Thư\s*-\s*(Ngoại Kỷ|Bản Kỷ)\s*-\s*Quyển\s+([IVXLC]+)")
# Leaf markers are OCR'd inconsistently - "1" gets misread as "l"/"i"/"ì"/"Ì" depending on font
# context (verified: the SAME leaf 11 appears as both "[lia]"->"[11a]" and "[llb]"->"[11b]"
# elsewhere in the same document), and "0" sometimes as "O". Restricted to this specific
# confusable-character whitelist (not e.g. "[a-zA-Z0-9]+") so real bracketed annotations that
# aren't leaf markers at all (e.g. "[Vu]", found in the raw text, presumably an abbreviation or
# citation marker) don't get misparsed as leaf numbers.
LEAF_MARKER_RE = re.compile(r"\[([0-9lLiIìÌOo]{1,3})([ab])\]")
_LEAF_DIGIT_MAP = str.maketrans({"l": "1", "L": "1", "i": "1", "I": "1", "ì": "1", "Ì": "1", "O": "0", "o": "0"})


def parse_img_key(img_name: str) -> tuple[str, str, int, str]:
    """Parses a DVSKTT manifest img_name into (section_prefix, quyen_roman, leaf_number, side) -
    e.g. "DVSKTT_ban_toan_V_30a_9.jpg" -> ("ban_toan", "V", 30, "a")."""
    m = IMG_NAME_RE.search(img_name)
    if not m:
        raise ValueError(f"unrecognized DVSKTT image name format: {img_name}")
    prefix, quyen, leaf, side = m.groups()
    return (prefix, quyen, int(leaf), side)


def parse_translation_pages(raw_text: str) -> dict[tuple[str, str, int, str], str]:
    """Parses the 1993 translation's raw OCR text into {(section, quyen_roman, leaf_number,
    side): translated_text_for_that_leaf_side}. section is "Ngoại Kỷ" or "Bản Kỷ" (matching
    SECTION_MAP's values, not the manifest's finer-grained prefixes).

    Tracks the current (section, quyen) via the book's own running header line, which repeats on
    every physical page (harmless - re-matching an unchanged header is a no-op) and updates
    whenever a genuinely new Quyển begins. Text between one leaf marker and the next belongs to
    the leaf/side named by the FIRST of that pair (matching the printed convention that a leaf
    marker announces the start of that leaf's content). Bare-digit lines (the 1993 book's own
    printed page numbers, distinct from the original woodblock leaf numbers) are dropped as noise
    rather than accumulated into any leaf's text.
    """
    pages: dict[tuple[str, str, int, str], list[str]] = {}
    section = quyen = None
    current_key = None
    buffer: list[str] = []

    def flush():
        if current_key is not None and buffer:
            pages.setdefault(current_key, []).extend(buffer)
        buffer.clear()

    for line in raw_text.split("\n"):
        header = HEADER_RE.search(line)
        if header:
            section, quyen = header.groups()
            continue
        if line.strip().isdigit():
            continue  # the 1993 book's own printed page number, not a leaf marker
        # a line can contain a leaf marker followed by real content on the same line
        last_end = 0
        for m in LEAF_MARKER_RE.finditer(line):
            pre_text = line[last_end:m.start()]
            if pre_text.strip():
                buffer.append(pre_text)
            flush()
            leaf_digits = m.group(1).translate(_LEAF_DIGIT_MAP)
            if not leaf_digits.isdigit():
                current_key = None  # unparseable marker (e.g. a stray OCR artifact) - drop until the next one
            elif section is None:
                current_key = None  # marker appeared before any section header was seen
            else:
                current_key = (section, quyen, int(leaf_digits), m.group(2))
            last_end = m.end()
        remainder = line[last_end:]
        if remainder.strip():
            buffer.append(remainder)
    flush()

    return {key: " ".join(" ".join(chunks).split()) for key, chunks in pages.items()}
