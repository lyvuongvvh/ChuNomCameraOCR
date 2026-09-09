"""Phase 2b (post-correction): train the character n-gram LM (lm/ngram_lm.py) on NomNaOCR's own
`Patches/Train.txt` ground truth - the same split the recognizer itself was trained on, so this
introduces no new data and no leakage into `Patches/Validate.txt` (the held-out set used for the
final before/after comparison).

Usage:
    python scripts/build_lm.py --dataset-root ../NomNaOCR --order 4 --out data/char_lm.json
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nomnaocr_lib.vocab import load_labels  # noqa: E402
from lm.ngram_lm import CharNgramLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path,
                         help="Path to the unzipped NomNaOCR dataset (contains Patches/)")
    parser.add_argument("--split", default="Patches/Train.txt",
                         help="Label file to train the LM on, relative to --dataset-root")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    split_path = args.dataset_root / args.split
    assert split_path.exists(), f"missing {split_path}"

    pairs = load_labels(str(split_path), min_length=1)
    texts = [text for _, text in pairs]
    print(f"Training order-{args.order} char LM on {len(texts)} lines from {split_path}")

    lm = CharNgramLM.train(texts, order=args.order)
    for n in range(1, args.order + 1):
        print(f"  {n}-gram vocabulary: {len(lm.counts[n])} distinct n-grams")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lm.save(str(args.out))
    print(f"Wrote LM to {args.out}")


if __name__ == "__main__":
    main()
