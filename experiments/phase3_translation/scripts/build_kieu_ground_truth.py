"""Builds a genuine parallel corpus for Truyen Kieu: aligns each of NomNaOCR's three digitized
editions (1866/1871/1872) against the real, complete 3254-verse poem (Vietnamese Wikisource),
so Phase 3's translations can finally be scored against real ground truth instead of judged
qualitatively (see results.md and eval_lib/kieu_ground_truth.py for why this needs alignment,
not just manifest order).

Usage:
    python scripts/build_kieu_ground_truth.py \
        --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --translations data/translations_full.json \
        --wikisource-raw data/kieu_wikisource_raw.txt \
        --out data/kieu_ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib.wikisource_alignment import (  # noqa: E402
    align_edition,
    parse_img_sort_key,
    parse_wikisource_poem,
)

KIEU_WORKS = ("Tale of Kieu 1866", "Tale of Kieu 1871", "Tale of Kieu 1872")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Phase 2 manifest.json (img_name/work/ground_truth)")
    ap.add_argument("--translations", required=True, help="Phase 3 translations JSON (img_name -> {reading, ...})")
    ap.add_argument("--wikisource-raw", required=True, help="raw wikitext of vi.wikisource.org/wiki/Truyen_Kieu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.translations, encoding="utf-8") as f:
        translations = json.load(f)
    with open(args.wikisource_raw, encoding="utf-8") as f:
        raw_wikitext = f.read()

    verses = parse_wikisource_poem(raw_wikitext)
    print(f"Parsed {len(verses)} verses from Wikisource (expect 3254)")

    by_work = defaultdict(list)
    for row in manifest:
        if row["work"] not in KIEU_WORKS:
            continue
        translation = translations.get(row["img_name"])
        if translation is None:
            continue  # not yet translated - can't align without a Stage-1 reading
        by_work[row["work"]].append({"img_name": row["img_name"], "reading": translation["reading"]})

    all_aligned = {}
    for work in KIEU_WORKS:
        rows = by_work[work]
        rows.sort(key=lambda r: parse_img_sort_key(r["img_name"]))
        print(f"\n{work}: aligning {len(rows)} lines against {len(verses)} verses...")
        aligned = align_edition(rows, verses)

        # Similarity (distance / verse length), not a flat distance cutoff: verses range 13-36
        # normalized characters, and even a genuinely correct alignment can carry real distance
        # from Stage 1 picking an archaic/dialectal reading over the modern Wikisource spelling
        # (e.g. "gioi" vs "troi" for "sky" - different letters, same word) - see results.md.
        high = sum(1 for a in aligned if a.similarity >= 0.7)
        mid = sum(1 for a in aligned if 0.4 <= a.similarity < 0.7)
        low = sum(1 for a in aligned if a.similarity < 0.4)
        avg_sim = sum(a.similarity for a in aligned) / len(aligned) if aligned else 0.0
        print(f"  similarity >=0.7: {high}  0.4-0.7: {mid}  <0.4: {low}  avg: {avg_sim:.3f}")

        # order should be non-decreasing verse numbers within an edition (monotonic alignment
        # invariant) - a broken invariant here would mean a DP bug, not a data quality issue
        verse_nums = [a.verse_number for a in aligned]
        assert verse_nums == sorted(verse_nums), f"{work}: alignment produced non-monotonic verse numbers"

        for a in aligned:
            all_aligned[a.img_name] = {
                "work": work,
                "verse_number": a.verse_number,
                "reference_text": a.reference_text,
                "ocr_reading": a.ocr_reading,
                "distance": a.distance,
                "similarity": round(a.similarity, 4),
            }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_aligned, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(all_aligned)} aligned lines to {args.out}")


if __name__ == "__main__":
    main()
