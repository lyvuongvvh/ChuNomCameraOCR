"""Phase 2: run NomNaOCR's pretrained CRNNxCTC model over the full evaluation manifest
(scripts/build_eval_manifest.py), as-is - no fine-tuning, same `nomnaocr_lib` inference code
Phase 0's `run_nomnaocr.py` uses (unchanged), just scaled from 62 sample patches to the full
~7,663-patch held-out split.

Usage (inside phase0_validation/docker/nomnaocr's image, with the whole experiments/ folder
mounted so this script can reach ../phase0_validation/nomnaocr_lib):
    docker run --rm -v "$PWD/experiments:/workspace" -w /workspace --entrypoint python \\
        phase0-nomnaocr phase2_nomnaocr_baseline/scripts/run_eval.py \\
        --dataset-root NomNaOCR \\
        --all-labels NomNaOCR/Patches/All.txt \\
        --weights NomNaOCR_H5/NomNaOCR_CRNNxCTC.h5 \\
        --manifest phase2_nomnaocr_baseline/data/manifest.json \\
        --out phase2_nomnaocr_baseline/data/predictions.json --resume

~7,663 sequential single-image CPU inferences run long enough that an interruption shouldn't
mean starting over: progress is checkpointed to --out every --checkpoint-every patches, and
--resume skips img_names already present in an existing --out file.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path,
                         help="Path to the unzipped NomNaOCR dataset (contains Patches/)")
    parser.add_argument("--all-labels", required=True, type=pathlib.Path,
                         help="Path to NomNaOCR's full All.txt - see nomnaocr_lib/vocab.py for "
                              "why the *full* label file is needed, not just the eval manifest")
    parser.add_argument("--weights", required=True, type=pathlib.Path,
                         help="Path to NomNaOCR_CRNNxCTC.h5")
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path, help="Output predictions.json")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true",
                         help="Skip img_names already present in --out")
    args = parser.parse_args()

    from nomnaocr_lib.vocab import build_vocab, max_label_length
    from nomnaocr_lib.model import CRNNRecognizer

    assert args.all_labels.exists(), f"missing {args.all_labels}"
    assert args.weights.exists(), f"missing {args.weights}"

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(f"Loaded manifest: {len(manifest)} patches")

    results = {}
    if args.resume and args.out.exists():
        results = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"--resume: {len(results)} predictions already in {args.out}, skipping those")

    todo = [entry for entry in manifest if entry["img_name"] not in results]
    if not todo:
        print("Nothing to do - all manifest patches already predicted.")
        return

    print("Rebuilding training-time character vocabulary from", args.all_labels)
    vocab = build_vocab(str(args.all_labels), min_length=1)
    max_length = max_label_length(str(args.all_labels), min_length=1)
    print(f"Vocab size: {len(vocab)}, max label length: {max_length}")

    recognizer = CRNNRecognizer(vocab, max_length, str(args.weights))

    patches_dir = args.dataset_root / "Patches"
    start = time.monotonic()
    for i, entry in enumerate(todo, start=1):
        img_path = patches_dir / entry["img_name"]
        results[entry["img_name"]] = recognizer.predict_text(str(img_path))

        if i % args.checkpoint_every == 0 or i == len(todo):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - i) / rate if rate > 0 else float("inf")
            print(f"[{i}/{len(todo)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(results)} predictions to {args.out}")


if __name__ == "__main__":
    main()
