"""Phase 2: turn score.py's output into results.md - NomNaOCR's own pretrained CRNNxCTC model's
Sequence Accuracy, Character Accuracy, and CER on its full held-out validation split, the
numbers that decide whether it's solid enough to carry forward as the recognizer for later
phases (translation, API, app).

Usage:
    python scripts/report.py --scores data/scores.json --out results.md
"""
import argparse
import json
import pathlib

SUBSET_LABELS = {
    "poem": "Poem lines",
    "prose": "Prose lines",
    "gt10": "Lines > 10 chars",
    "lte10": "Lines <= 10 chars",
}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="data/scores.json", type=pathlib.Path)
    parser.add_argument("--out", default="results.md", type=pathlib.Path)
    args = parser.parse_args()

    data = json.loads(args.scores.read_text(encoding="utf-8"))
    agg = data["aggregate"]
    overall = agg["overall"]

    lines = [
        "# Phase 2: NomNaOCR Baseline Evaluation",
        "",
        "NomNaOCR's own pretrained CRNNxCTC model, run as-is (no fine-tuning), scored against "
        "its own full held-out validation split (`Patches/Validate.txt`) using its own reported "
        "metrics - Sequence Accuracy, Character Accuracy, and Character Error Rate (CER). This "
        "supersedes Phase 0's page-level bag-of-characters proxy (Hán-only, 15 pages) with the "
        "real per-line metrics over the complete held-out set.",
        "",
        "## Aggregate",
        "",
        f"n = {overall['n']} held-out patches",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Sequence Accuracy | {pct(overall['sequence_accuracy'])} |",
        f"| Character Accuracy | {pct(overall['character_accuracy'])} ({overall['char_correct']}/{overall['char_total']} chars) |",
        f"| CER (macro - mean of per-line edit-distance/length) | {overall['cer_macro']:.4f} |",
        f"| CER (micro - total edit distance / total chars) | {overall['cer_micro']:.4f} |",
        "",
        "## By subset",
        "",
        "| Subset | n | Sequence Accuracy | Character Accuracy | CER (macro) | CER (micro) |",
        "|---|---|---|---|---|---|",
    ]
    for tag, label in SUBSET_LABELS.items():
        if tag not in agg:
            continue
        s = agg[tag]
        lines.append(
            f"| {label} | {s['n']} | {pct(s['sequence_accuracy'])} | "
            f"{pct(s['character_accuracy'])} | {s['cer_macro']:.4f} | {s['cer_micro']:.4f} |"
        )

    lines += [
        "",
        "## Reading this result",
        "",
        "- **Character Accuracy is positional, not alignment-based** (per NomNaOCR's own "
        "`CharacterAccuracy` metric) - a single leading insertion or deletion in a prediction "
        "misaligns every character after it, undercounting an otherwise-good prediction. This is "
        "a faithful replication of NomNaOCR's own metric, not a bug in this scoring code - see "
        "`eval_lib/metrics.py`'s docstring and `tests/test_metrics.py` for a worked example.",
        "- **CER is reported two ways** because NomNaOCR's own upstream code defines it two ways "
        "with different aggregation (macro: mean of per-line ratios; micro: total edit distance "
        "over total characters) - despite an upstream code comment claiming they're the same, "
        "they only coincide when every line has equal length. Neither is silently preferred here.",
        "- **Ground truth** comes from `nomnaocr_lib.vocab.load_labels()`'s lowercased, "
        "`is_clean_text`-filtered text (same filtering the model's vocabulary is built from), not "
        "raw `Validate.txt` lines - assumed to mirror NomNaOCR's own evaluation, not "
        "independently verified since their eval script isn't vendored in this repo.",
        "- **Compare against Phase 0's 86.7%** (bag-of-characters, Hán-only, 15 pages, "
        "`experiments/phase0_validation/results.md`): expect a plausible but different number "
        "here, not an exact match - this evaluation covers all characters (not just Hán), scores "
        "per line (not per page), and uses positional/edit-distance metrics (not an "
        "order-insensitive multiset intersection). An exact 86.7% match would suggest an "
        "accidental logic collapse between the two scoring methods; a near-0% result would "
        "suggest a padding or decoding bug in this pipeline.",
    ]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Sequence Accuracy: {pct(overall['sequence_accuracy'])}")
    print(f"Character Accuracy: {pct(overall['character_accuracy'])}")
    print(f"CER (macro / micro): {overall['cer_macro']:.4f} / {overall['cer_micro']:.4f}")


if __name__ == "__main__":
    main()
