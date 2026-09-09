"""Phase 2b (post-correction): apply the tuned lambda (scripts/tune_lambda.py) to every patch's
beams, producing the LM-corrected prediction for each - used for the headline "final" split
comparison against Phase 2's original greedy predictions in report.py.

Usage:
    python scripts/rescore.py --beams data/beams.json --lm data/char_lm.json \
        --best-lambda data/best_lambda.json --out data/corrected_predictions.json
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from lm.ngram_lm import CharNgramLM  # noqa: E402
from lm.rescoring import rescore_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beams", required=True, type=pathlib.Path)
    parser.add_argument("--lm", required=True, type=pathlib.Path)
    parser.add_argument("--best-lambda", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    beams = json.loads(args.beams.read_text(encoding="utf-8"))
    lm = CharNgramLM.load(str(args.lm))
    lam = json.loads(args.best_lambda.read_text(encoding="utf-8"))["lambda"]
    print(f"Rescoring {len(beams)} patches with lambda={lam}")

    corrected = rescore_all(beams, lm, lam)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
