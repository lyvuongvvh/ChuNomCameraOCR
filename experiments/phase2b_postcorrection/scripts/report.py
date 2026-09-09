"""Phase 2b (post-correction): compare Phase 2's original greedy-decode predictions against the
beam-search + LM-rescored predictions, on the "final" split only (held out from lambda tuning),
using the same Sequence/Character Accuracy/CER metrics as Phase 2 (eval_lib/metrics.py, reused
unchanged). Writes results.md.

Usage:
    python scripts/report.py \
        --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --baseline-predictions ../phase2_nomnaocr_baseline/data/predictions.json \
        --corrected-predictions data/corrected_predictions.json \
        --split data/split.json \
        --all-labels ../NomNaOCR/Patches/All.txt \
        --best-lambda data/best_lambda.json \
        --out results.md
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase2_nomnaocr_baseline"))

from nomnaocr_lib.vocab import max_label_length  # noqa: E402
from eval_lib.metrics import SampleScore, aggregate  # noqa: E402


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def score_split(manifest: dict, predictions: dict, names: list, max_length: int) -> dict:
    scores = [
        SampleScore.compute(
            img_name=name, work=name.split("/")[0],
            pred=predictions[name], gt=manifest[name], max_length=max_length,
        )
        for name in names
    ]
    return aggregate(scores)["overall"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-predictions", required=True, type=pathlib.Path)
    parser.add_argument("--corrected-predictions", required=True, type=pathlib.Path)
    parser.add_argument("--split", required=True, type=pathlib.Path)
    parser.add_argument("--all-labels", required=True, type=pathlib.Path)
    parser.add_argument("--best-lambda", required=True, type=pathlib.Path)
    parser.add_argument("--out", default="results.md", type=pathlib.Path)
    args = parser.parse_args()

    manifest = {e["img_name"]: e["ground_truth"] for e in json.loads(args.manifest.read_text(encoding="utf-8"))}
    baseline_preds = json.loads(args.baseline_predictions.read_text(encoding="utf-8"))
    corrected_preds = json.loads(args.corrected_predictions.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    max_length = max_label_length(str(args.all_labels), min_length=1)
    best_lambda = json.loads(args.best_lambda.read_text(encoding="utf-8"))

    final_names = split["final"]
    baseline = score_split(manifest, baseline_preds, final_names, max_length)
    corrected = score_split(manifest, corrected_preds, final_names, max_length)

    delta_seq = corrected["sequence_accuracy"] - baseline["sequence_accuracy"]
    delta_char = corrected["character_accuracy"] - baseline["character_accuracy"]

    lines = [
        "# Phase 2b: Post-Correction (Beam Search + Character N-gram LM)",
        "",
        "Rescores NomNaOCR's own CTC beam-search hypotheses with a character-level n-gram "
        "language model trained on `Patches/Train.txt`, instead of retraining the recognizer "
        "itself. See `README.md` for the approach and its known ceiling (it can only pick among "
        "candidates the acoustic model already considered plausible - see the beam-search probe "
        "findings there for cases where the correct character never appears in any beam at all).",
        "",
        f"Lambda (LM weight) = **{best_lambda['lambda']}**, chosen by grid search on a held-out "
        f"10% tune slice (`data/split.json`) never used in the numbers below.",
        "",
        "## Before vs. after (final split, held out from tuning)",
        "",
        f"n = {baseline['n']} patches",
        "",
        "| Metric | Baseline (greedy) | Corrected (beam + LM) | Delta |",
        "|---|---|---|---|",
        f"| Sequence Accuracy | {pct(baseline['sequence_accuracy'])} | {pct(corrected['sequence_accuracy'])} | {delta_seq:+.1%} |",
        f"| Character Accuracy | {pct(baseline['character_accuracy'])} | {pct(corrected['character_accuracy'])} | {delta_char:+.1%} |",
        f"| CER (macro) | {baseline['cer_macro']:.4f} | {corrected['cer_macro']:.4f} | {corrected['cer_macro'] - baseline['cer_macro']:+.4f} |",
        f"| CER (micro) | {baseline['cer_micro']:.4f} | {corrected['cer_micro']:.4f} | {corrected['cer_micro'] - baseline['cer_micro']:+.4f} |",
        "",
        "## Reading this result",
        "",
        "- **Baseline here is Phase 2's own greedy-decode predictions**, restricted to the same "
        "\"final\" patches used for the corrected column - an apples-to-apples subset comparison, "
        "not the full-set numbers in `experiments/phase2_nomnaocr_baseline/results.md`.",
        "- **Lambda was chosen on a separate 10% tune slice**, never on these \"final\" patches, "
        "so this delta isn't inflated by tuning against the same data it's measured on.",
        "- If the delta is small or negative, that's consistent with the beam-search probe finding "
        "in `README.md`: a meaningful share of this recognizer's held-out errors are cases where "
        "the correct character isn't in the beam at all, which no amount of LM rescoring can fix - "
        "closing that gap further would need retraining the recognizer, not just correcting its output.",
    ]

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Sequence Accuracy: {pct(baseline['sequence_accuracy'])} -> {pct(corrected['sequence_accuracy'])} ({delta_seq:+.1%})")
    print(f"Character Accuracy: {pct(baseline['character_accuracy'])} -> {pct(corrected['character_accuracy'])} ({delta_char:+.1%})")


if __name__ == "__main__":
    main()
