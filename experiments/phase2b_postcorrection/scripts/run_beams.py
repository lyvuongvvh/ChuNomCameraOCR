"""Phase 2b (post-correction): run NomNaOCR's pretrained CRNNxCTC model over Phase 2's evaluation
manifest, keeping the top-`beam_width` CTC beam-search hypotheses (+ their log-probabilities) per
patch instead of just the single greedy decode Phase 2's `run_eval.py` kept - the raw material
LM rescoring needs (scripts/rescore.py). Reuses Phase 2's manifest
(`../phase2_nomnaocr_baseline/data/manifest.json`) directly rather than rebuilding it.

Usage (inside phase0_validation/docker/nomnaocr's image, whole experiments/ mounted):
    docker run --rm -v "$PWD/experiments:/workspace" -w /workspace --entrypoint python \\
        phase0-nomnaocr phase2b_postcorrection/scripts/run_beams.py \\
        --dataset-root NomNaOCR \\
        --all-labels NomNaOCR/Patches/All.txt \\
        --weights NomNaOCR_H5/NomNaOCR_CRNNxCTC.h5 \\
        --manifest phase2_nomnaocr_baseline/data/manifest.json \\
        --out data/beams.json --resume
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
                         help="Path to NomNaOCR's full All.txt (vocab reconstruction)")
    parser.add_argument("--weights", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path,
                         help="Phase 2's manifest.json (reused, not rebuilt)")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
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
        print(f"--resume: {len(results)} beam sets already in {args.out}, skipping those")

    todo = [entry for entry in manifest if entry["img_name"] not in results]
    if not todo:
        print("Nothing to do - all manifest patches already have beams.")
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
        results[entry["img_name"]] = recognizer.predict_beams(str(img_path), beam_width=args.beam_width)

        if i % args.checkpoint_every == 0 or i == len(todo):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - i) / rate if rate > 0 else float("inf")
            print(f"[{i}/{len(todo)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote beams for {len(results)} patches to {args.out}")


if __name__ == "__main__":
    main()
