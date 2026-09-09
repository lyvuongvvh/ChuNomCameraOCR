"""Phase 2b (post-correction): carve Phase 2's held-out manifest into a small "tune" slice (for
picking the LM rescoring weight lambda) and a "final" slice (held out from all tuning, used for
the headline before/after comparison in results.md).

Why a split *within* Validate.txt rather than tuning on Train.txt: the recognizer was fit to
Train.txt, so its greedy/beam error patterns there are unrepresentatively good (memorization) -
tuning lambda against those errors wouldn't reflect what it needs to fix on real held-out data.
Carving a small tune slice out of Validate.txt itself (standard practice when no dedicated dev
split exists) means lambda is chosen against the *same kind* of errors the final numbers measure,
without ever letting the final 90% influence that choice.

Usage:
    python scripts/split_tune_final.py --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
        --out data/split.json --tune-fraction 0.1 --seed 0
"""
import argparse
import json
import pathlib
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--tune-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    img_names = [entry["img_name"] for entry in manifest]

    rng = random.Random(args.seed)
    shuffled = img_names[:]
    rng.shuffle(shuffled)

    n_tune = round(len(shuffled) * args.tune_fraction)
    tune = sorted(shuffled[:n_tune])
    final = sorted(shuffled[n_tune:])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"tune": tune, "final": final}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"tune: {len(tune)} patches, final: {len(final)} patches -> {args.out}")


if __name__ == "__main__":
    main()
