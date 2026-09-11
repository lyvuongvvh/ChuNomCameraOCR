"""Scores Phase 3's `translation` field against the real ground truth built for all three
source works - see eval_lib/scoring.py for why Kieu/Luc Van Tien and DVSKTT need different
metrics, and results.md for the full write-up.

Usage:
    python scripts/score_translations.py \
        --translations data/translations_full.json \
        --kieu-ground-truth data/kieu_ground_truth.json \
        --lvt-ground-truth data/lvt_ground_truth.json \
        --dvsktt-ground-truth data/dvsktt_ground_truth.json \
        --out data/translation_scores.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib.scoring import (  # noqa: E402
    edit_similarity,
    strip_low_confidence_note,
    word_jaccard,
    word_recall,
)

# Below this alignment similarity, the ground truth itself is unreliable (the DP was likely
# forced into a wrong verse) - scoring a translation against a probably-wrong reference would
# just add noise, not signal. See results.md's "alignment quality is reported, not guaranteed."
MIN_ALIGNMENT_SIMILARITY = 0.7


def score_verse_work(translations: dict, ground_truth: dict, work_label: str) -> list[dict]:
    scored = []
    for img, gt in ground_truth.items():
        if gt["similarity"] < MIN_ALIGNMENT_SIMILARITY:
            continue
        t = translations.get(img)
        if t is None:
            continue
        translation = strip_low_confidence_note(t["translation"])
        reference = gt["reference_text"]
        scored.append({
            "img_name": img,
            "work": work_label,
            "low_confidence": "LOW CONFIDENCE" in t["translation"],
            "edit_similarity": round(edit_similarity(translation, reference), 4),
            "word_jaccard": round(word_jaccard(translation, reference), 4),
            "translation": translation,
            "reference_text": reference,
        })
    return scored


def score_dvsktt(translations: dict, ground_truth: dict) -> list[dict]:
    scored = []
    for img, gt in ground_truth.items():
        t = translations.get(img)
        if t is None:
            continue
        translation = strip_low_confidence_note(t["translation"])
        reference = gt["reference_text"]
        scored.append({
            "img_name": img,
            "work": gt["work"],
            "low_confidence": "LOW CONFIDENCE" in t["translation"],
            "word_recall": round(word_recall(translation, reference), 4),
            "translation": translation,
            "reference_text": reference,
        })
    return scored


def summarize(rows: list[dict], score_key: str, label: str):
    if not rows:
        print(f"  {label}: no scored lines")
        return
    scores = [r[score_key] for r in rows]
    hi = [r for r in rows if not r["low_confidence"]]
    lo = [r for r in rows if r["low_confidence"]]
    print(f"  {label}: n={len(rows)}  mean={statistics.mean(scores):.3f}  median={statistics.median(scores):.3f}")
    if hi and lo:
        print(f"    self-confident (n={len(hi)}): mean={statistics.mean(r[score_key] for r in hi):.3f}"
              f"   self-flagged LOW CONFIDENCE (n={len(lo)}): mean={statistics.mean(r[score_key] for r in lo):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--translations", required=True)
    ap.add_argument("--kieu-ground-truth", required=True)
    ap.add_argument("--lvt-ground-truth", required=True)
    ap.add_argument("--dvsktt-ground-truth", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.translations, encoding="utf-8") as f:
        translations = json.load(f)
    with open(args.kieu_ground_truth, encoding="utf-8") as f:
        kieu_gt = json.load(f)
    with open(args.lvt_ground_truth, encoding="utf-8") as f:
        lvt_gt = json.load(f)
    with open(args.dvsktt_ground_truth, encoding="utf-8") as f:
        dvsktt_gt = json.load(f)

    kieu_scored = score_verse_work(translations, kieu_gt, "Kieu")
    lvt_scored = score_verse_work(translations, lvt_gt, "Luc Van Tien")
    dvsktt_scored = score_dvsktt(translations, dvsktt_gt)

    print(f"Scored against alignment-confident (similarity >= {MIN_ALIGNMENT_SIMILARITY}) ground truth:")
    summarize(kieu_scored, "edit_similarity", "Kieu (edit_similarity)")
    summarize(kieu_scored, "word_jaccard", "Kieu (word_jaccard)")
    summarize(lvt_scored, "edit_similarity", "Luc Van Tien (edit_similarity)")
    summarize(lvt_scored, "word_jaccard", "Luc Van Tien (word_jaccard)")
    print()
    print("Scored against DVSKTT's leaf-level ground truth (word_recall):")
    summarize(dvsktt_scored, "word_recall", "DVSKTT")

    out = {
        "kieu": kieu_scored,
        "luc_van_tien": lvt_scored,
        "dvsktt": dvsktt_scored,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(kieu_scored) + len(lvt_scored) + len(dvsktt_scored)} scored lines to {args.out}")


if __name__ == "__main__":
    main()
