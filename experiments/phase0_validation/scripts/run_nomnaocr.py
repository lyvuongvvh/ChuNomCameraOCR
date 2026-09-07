"""Phase 0 validation: run NomNaOCR's pretrained CRNNxCTC model on the same sample patches
used for the CHAT comparison, as-is (no fine-tuning changes).

Usage (inside the nomnaocr docker image, /workspace mounted to experiments/phase0_validation):
    python scripts/run_nomnaocr.py --patches data/sample_patches \
        --all-labels data/nomnaocr_dataset/Patches/All.txt \
        --weights data/nomnaocr_weights/NomNaOCR_CRNNxCTC.h5 \
        --out data/predictions_nomnaocr.json

--all-labels must point at NomNaOCR's *full* label file (not just the sample subset): the
pretrained weights' output indices are tied to the character vocabulary built from the entire
training set, so the vocab has to be reconstructed the same way to decode predictions
correctly. See nomnaocr_lib/vocab.py for details/caveats.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True, type=pathlib.Path, help="Dir of sample patch images")
    parser.add_argument("--all-labels", required=True, type=pathlib.Path, help="Path to NomNaOCR's full All.txt")
    parser.add_argument("--weights", required=True, type=pathlib.Path, help="Path to NomNaOCR_CRNNxCTC.h5")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="Output JSON path")
    args = parser.parse_args()

    from nomnaocr_lib.vocab import build_vocab, max_label_length
    from nomnaocr_lib.model import CRNNRecognizer

    assert args.all_labels.exists(), f"missing {args.all_labels}"
    assert args.weights.exists(), f"missing {args.weights}"

    print("Rebuilding training-time character vocabulary from", args.all_labels)
    vocab = build_vocab(str(args.all_labels), min_length=1)
    max_length = max_label_length(str(args.all_labels), min_length=1)
    print(f"Vocab size: {len(vocab)}, max label length: {max_length}")

    recognizer = CRNNRecognizer(vocab, max_length, str(args.weights))

    patch_paths = sorted(p for p in args.patches.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not patch_paths:
        raise SystemExit(f"no images found in {args.patches}")

    results = {}
    for patch_path in patch_paths:
        print(f"[nomnaocr] {patch_path.name}")
        results[patch_path.name] = recognizer.predict_text(str(patch_path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} patch predictions to {args.out}")


if __name__ == "__main__":
    main()
