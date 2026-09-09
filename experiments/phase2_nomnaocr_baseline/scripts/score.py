"""Phase 2: score NomNaOCR's predictions (run_eval.py) against the eval manifest
(build_eval_manifest.py), using Sequence Accuracy, Character Accuracy, and CER
(eval_lib/metrics.py) - NomNaOCR's own reported metrics, not Phase 0's bag-of-characters proxy.

Usage:
    python scripts/score.py --manifest data/manifest.json --predictions data/predictions.json \
        --all-labels /path/to/NomNaOCR/Patches/All.txt --out data/scores.json
"""
import argparse
import json
import pathlib
import sys
from dataclasses import asdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from nomnaocr_lib.vocab import max_label_length  # noqa: E402
from eval_lib.metrics import SampleScore, aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.json", type=pathlib.Path)
    parser.add_argument("--predictions", default="data/predictions.json", type=pathlib.Path)
    parser.add_argument("--all-labels", required=True, type=pathlib.Path,
                         help="NomNaOCR's full All.txt, to recompute the same max_length "
                              "run_eval.py used for CTC decoding (see nomnaocr_lib/vocab.py)")
    parser.add_argument("--out", default="data/scores.json", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    max_length = max_label_length(str(args.all_labels), min_length=1)

    missing = [e["img_name"] for e in manifest if e["img_name"] not in predictions]
    assert not missing, (
        f"{len(missing)} manifest patches have no prediction (e.g. {missing[:5]}) - "
        f"re-run run_eval.py, possibly with --resume"
    )

    scores = [
        SampleScore.compute(
            img_name=entry["img_name"],
            work=entry["work"],
            pred=predictions[entry["img_name"]],
            gt=entry["ground_truth"],
            max_length=max_length,
            subsets=entry["subsets"],
        )
        for entry in manifest
    ]

    aggregated = aggregate(scores)
    output = {
        "aggregate": aggregated,
        "per_patch": [asdict(s) for s in scores],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = aggregated["overall"]
    print(f"Wrote scores for {len(scores)} patches to {args.out}")
    print(f"Sequence Accuracy: {overall['sequence_accuracy']:.1%}")
    print(f"Character Accuracy: {overall['character_accuracy']:.1%}")
    print(f"CER (macro): {overall['cer_macro']:.4f}")
    print(f"CER (micro): {overall['cer_micro']:.4f}")


if __name__ == "__main__":
    main()
