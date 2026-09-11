"""Root-causes WHY specific low-scoring lines (scripts/score_translations.py) score badly, not
just which ones do - distinguishing four candidate causes per line via evidence already on hand
(no new Anthropic API calls):

    1. OCR error - Phase 2b's predicted text doesn't match the true manifest label, so
       everything downstream (reading, translation) was built on a wrong character.
    2. Dictionary gap - the predicted text is correct, but Stage 1 couldn't resolve one or more
       characters (bracketed "[X]" in the reading), so Stage 2 had less to work with.
    3. Likely valid paraphrase (metric harshness) - Kieu/Luc Van Tien only: word_jaccard is
       decent while edit_similarity is low, the classic signature of a correct translation using
       different word order or a synonym (see results.md's "gio sam" vs "phong loi" example).
    4. Unexplained - none of the above - OCR was right, dictionary resolved fine, and (for verse)
       the words don't overlap much either. Candidate genuine Stage 2 translation mistakes -
       flagged for manual review, not auto-classified further (that needs actual reading, not a
       heuristic).

"Low-scoring" = bottom 25% by the work's primary metric (edit_similarity for Kieu/Luc Van Tien,
word_recall for DVSKTT) - a data-driven cutoff, not an arbitrary absolute threshold.

Usage:
    python scripts/diagnose_low_scores.py \
        --scores data/translation_scores.json \
        --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --translations data/translations_full.json \
        --out data/low_score_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_lib.wikisource_alignment import edit_distance  # noqa: E402

UNRESOLVED_RE = re.compile(r"\[.\]")
PARAPHRASE_JACCARD_MIN = 0.4  # decent word overlap despite a low edit_similarity


def classify(row: dict, manifest_row: dict, translation_row: dict) -> str:
    pred_text = translation_row["text"]
    true_text = manifest_row["ground_truth"]
    if edit_distance(pred_text, true_text) > 0:
        return "ocr_error"
    if UNRESOLVED_RE.search(translation_row["reading"]):
        return "dictionary_gap"
    if "word_jaccard" in row and row["word_jaccard"] >= PARAPHRASE_JACCARD_MIN:
        return "likely_valid_paraphrase"
    return "unexplained"


def bottom_quartile(rows: list[dict], key: str) -> list[dict]:
    if not rows:
        return []
    scores = sorted(r[key] for r in rows)
    cutoff = scores[len(scores) // 4]
    return [r for r in rows if r[key] <= cutoff]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--translations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.scores, encoding="utf-8") as f:
        scores = json.load(f)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = {r["img_name"]: r for r in json.load(f)}
    with open(args.translations, encoding="utf-8") as f:
        translations = json.load(f)

    groups = [
        ("kieu", "edit_similarity"),
        ("luc_van_tien", "edit_similarity"),
        ("dvsktt", "word_recall"),
    ]

    diagnosis = {}
    for group_key, metric in groups:
        rows = scores[group_key]
        low = bottom_quartile(rows, metric)
        print(f"\n{group_key}: {len(low)} of {len(rows)} lines in the bottom quartile by {metric}")

        classified = []
        for row in low:
            m_row = manifest[row["img_name"]]
            t_row = translations[row["img_name"]]
            cause = classify(row, m_row, t_row)
            classified.append({**row, "cause": cause, "predicted_han": t_row["text"], "true_han": m_row["ground_truth"]})

        counts = Counter(r["cause"] for r in classified)
        total = len(classified)
        for cause, n in counts.most_common():
            print(f"  {cause}: {n} ({100 * n / total:.1f}%)")

        diagnosis[group_key] = classified

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, ensure_ascii=False, indent=2)
    print(f"\nWrote diagnosis for {sum(len(v) for v in diagnosis.values())} lines to {args.out}")


if __name__ == "__main__":
    main()
