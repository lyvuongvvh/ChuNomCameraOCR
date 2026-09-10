"""Builds a genuine parallel corpus for Luc Van Tien: aligns NomNaOCR's single digitized edition
against the real, complete poem (Vietnamese Wikisource), the same approach used for Truyen Kieu
(scripts/build_kieu_ground_truth.py) - see eval_lib/wikisource_alignment.py for why this needs
alignment, not just manifest order, and results.md for background.

Luc Van Tien's Wikisource text is split across 4 sub-pages (I-IV, ~2082 verses total) rather than
Kieu's single page, so each part is parsed with start_number continuing from the previous part's
last verse - see eval_lib.wikisource_alignment.parse_wikisource_poem's docstring for why this
uses pure sequential counting rather than trusting the source's own {{so|N}} markers (they have
real transcription errors in this specific work).

Usage:
    python scripts/build_lvt_ground_truth.py \
        --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --translations data/translations_full.json \
        --wikisource-raw-glob "data/lvt_wikisource_raw_{part}.txt" \
        --out data/lvt_ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib.wikisource_alignment import (  # noqa: E402
    align_edition,
    parse_img_sort_key,
    parse_wikisource_poem,
)

WORK_NAME = "Luc Van Tien"
PARTS = ("I", "II", "III", "IV")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Phase 2 manifest.json (img_name/work/ground_truth)")
    ap.add_argument("--translations", required=True, help="Phase 3 translations JSON (img_name -> {reading, ...})")
    ap.add_argument("--wikisource-raw-glob", required=True,
                     help='path template with a "{part}" placeholder, e.g. "data/lvt_wikisource_raw_{part}.txt"')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.translations, encoding="utf-8") as f:
        translations = json.load(f)

    verses = []
    next_start = 1
    for part in PARTS:
        path = args.wikisource_raw_glob.format(part=part)
        with open(path, encoding="utf-8") as f:
            raw_wikitext = f.read()
        part_verses = parse_wikisource_poem(raw_wikitext, start_number=next_start)
        print(f"Parsed {len(part_verses)} verses from part {part} ({path})")
        verses.extend(part_verses)
        next_start = part_verses[-1][0] + 1

    verse_nums = [v[0] for v in verses]
    assert verse_nums == sorted(verse_nums), "verse numbers went backwards - should be impossible with pure sequential numbering"
    print(f"Total {len(verses)} verses parsed (expect ~2082)")

    rows = []
    for row in manifest:
        if row["work"] != WORK_NAME:
            continue
        translation = translations.get(row["img_name"])
        if translation is None:
            continue  # not yet translated - can't align without a Stage-1 reading
        rows.append({"img_name": row["img_name"], "reading": translation["reading"]})
    rows.sort(key=lambda r: parse_img_sort_key(r["img_name"]))

    print(f"\n{WORK_NAME}: aligning {len(rows)} lines against {len(verses)} verses...")
    aligned = align_edition(rows, verses)

    # Similarity (not a flat distance cutoff), same reasoning as build_kieu_ground_truth.py:
    # verse length varies and a genuinely correct alignment can still carry real distance from
    # Stage 1 picking an archaic/dialectal reading over Wikisource's modern spelling.
    high = sum(1 for a in aligned if a.similarity >= 0.7)
    mid = sum(1 for a in aligned if 0.4 <= a.similarity < 0.7)
    low = sum(1 for a in aligned if a.similarity < 0.4)
    avg_sim = sum(a.similarity for a in aligned) / len(aligned) if aligned else 0.0
    print(f"  similarity >=0.7: {high}  0.4-0.7: {mid}  <0.4: {low}  avg: {avg_sim:.3f}")

    result_verse_nums = [a.verse_number for a in aligned]
    assert result_verse_nums == sorted(result_verse_nums), "alignment produced non-monotonic verse numbers"

    out = {
        a.img_name: {
            "work": WORK_NAME,
            "verse_number": a.verse_number,
            "reference_text": a.reference_text,
            "ocr_reading": a.ocr_reading,
            "distance": a.distance,
            "similarity": round(a.similarity, 4),
        }
        for a in aligned
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(out)} aligned lines to {args.out}")


if __name__ == "__main__":
    main()
