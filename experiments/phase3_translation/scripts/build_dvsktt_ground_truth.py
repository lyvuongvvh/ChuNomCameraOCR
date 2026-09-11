"""Builds a genuine parallel corpus for Dai Viet Su Ky Toan Thu (DVSKTT): matches each manifest
line to the real 1993 published translation's text for the same original woodblock leaf, via
structural (quyen, leaf, side) key lookup rather than text-similarity alignment - see
eval_lib/dvsktt_ground_truth.py for why DVSKTT (genuine Literary Chinese prose) needs a completely
different approach from Kieu/Luc Van Tien (Vietnamese verse, aligned by eval_lib/
wikisource_alignment.py).

Coarser than the Kieu/Luc Van Tien ground truth: this recovers "the translated text of the leaf
this line was written on," not a per-line exact correspondence - Han prose doesn't segment into
one-line-per-idea the way Nom luc-bat verse does, so multiple OCR patches on the same leaf/side
legitimately share one reference_text.

Usage:
    python scripts/build_dvsktt_ground_truth.py \
        --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --translation-raw data/dvsktt_1993_translation_raw.txt \
        --out data/dvsktt_ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib.dvsktt_ground_truth import (  # noqa: E402
    SECTION_MAP,
    parse_img_key,
    parse_translation_pages,
)

DVSKTT_WORKS = (
    "DVSKTT-1 Quyen thu",
    "DVSKTT-2 Ngoai ky toan thu",
    "DVSKTT-3 Ban ky toan thu",
    "DVSKTT-4 Ban ky thuc luc",
    "DVSKTT-5 Ban ky tuc bien",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Phase 2 manifest.json (img_name/work/ground_truth)")
    ap.add_argument("--translation-raw", required=True, help="raw OCR text of the 1993 translation (Internet Archive)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.translation_raw, encoding="utf-8") as f:
        raw_text = f.read()

    pages = parse_translation_pages(raw_text)
    print(f"Parsed {len(pages)} leaf-side text chunks from the 1993 translation")

    out = {}
    for work in DVSKTT_WORKS:
        rows = [r for r in manifest if r["work"] == work]
        hit = 0
        for row in rows:
            prefix, quyen, leaf, side = parse_img_key(row["img_name"])
            section = SECTION_MAP.get(prefix)
            key = (section, quyen, leaf, side)
            reference_text = pages.get(key)
            if reference_text is None:
                continue
            hit += 1
            out[row["img_name"]] = {
                "work": work,
                "quyen": quyen,
                "leaf": leaf,
                "side": side,
                "han_text": row["ground_truth"],
                "reference_text": reference_text,
            }
        pct = f"{100 * hit / len(rows):.1f}%" if rows else "n/a"
        print(f"{work}: {hit}/{len(rows)} lines matched to a translated leaf ({pct})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(out)} lines with ground truth to {args.out}")


if __name__ == "__main__":
    main()
