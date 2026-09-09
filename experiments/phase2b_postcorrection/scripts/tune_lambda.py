"""Phase 2b (post-correction): grid-search the LM rescoring weight lambda on the "tune" slice
(scripts/split_tune_final.py), never touching the "final" slice used for the headline
before/after numbers in results.md. No re-inference needed here - beams are already generated
(scripts/run_beams.py); this just rescoring already-computed candidates for each candidate
lambda, so the whole grid search is cheap (pure Python, no TF).

Usage:
    python scripts/tune_lambda.py --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --beams data/beams.json --split data/split.json --lm data/char_lm.json \
        --all-labels ../NomNaOCR/Patches/All.txt --out data/best_lambda.json
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase2_nomnaocr_baseline"))

from nomnaocr_lib.vocab import max_label_length  # noqa: E402
from lm.ngram_lm import CharNgramLM  # noqa: E402
from lm.rescoring import rescore_patch  # noqa: E402
from eval_lib.metrics import sequence_correct, character_accuracy_counts  # noqa: E402

DEFAULT_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--beams", required=True, type=pathlib.Path)
    parser.add_argument("--split", required=True, type=pathlib.Path)
    parser.add_argument("--lm", required=True, type=pathlib.Path)
    parser.add_argument("--all-labels", required=True, type=pathlib.Path)
    parser.add_argument("--lambdas", default=None,
                         help="Comma-separated lambda grid; default: " + ",".join(map(str, DEFAULT_LAMBDAS)))
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else DEFAULT_LAMBDAS

    manifest = {e["img_name"]: e["ground_truth"] for e in json.loads(args.manifest.read_text(encoding="utf-8"))}
    beams = json.loads(args.beams.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    lm = CharNgramLM.load(str(args.lm))
    max_length = max_label_length(str(args.all_labels), min_length=1)

    tune_names = split["tune"]
    print(f"Tuning on {len(tune_names)} patches (held out from the final comparison)")

    print(f"{'lambda':>8} | {'seq_acc':>8} | {'char_acc':>8}")
    best = {"lambda": None, "seq_acc": -1.0, "char_acc": -1.0}
    for lam in lambdas:
        n_correct_seq = 0
        char_correct_total, char_total_total = 0, 0
        for img_name in tune_names:
            gt = manifest[img_name]
            pred = rescore_patch(beams[img_name], lm, lam)
            n_correct_seq += sequence_correct(pred, gt)
            c, t = character_accuracy_counts(pred, gt, max_length)
            char_correct_total += c
            char_total_total += t
        seq_acc = n_correct_seq / len(tune_names)
        char_acc = char_correct_total / char_total_total if char_total_total else 0.0
        print(f"{lam:>8.2f} | {seq_acc:>8.1%} | {char_acc:>8.1%}")
        if (seq_acc, char_acc) > (best["seq_acc"], best["char_acc"]):
            best = {"lambda": lam, "seq_acc": seq_acc, "char_acc": char_acc}

    print(f"\nBest lambda: {best['lambda']} (tune seq_acc={best['seq_acc']:.1%}, char_acc={best['char_acc']:.1%})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
