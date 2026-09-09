"""Phase 2: build the evaluation manifest for NomNaOCR's pretrained CRNNxCTC model, over its
own held-out split (`Patches/Validate.txt`), for Sequence/Character Accuracy and CER scoring.

Unlike Phase 0's `build_manifest.py` (which grouped patches by page for a page-level
bag-of-characters comparison), this operates per-patch: Sequence/Character Accuracy/CER are
line-level metrics, and `Patches/Validate.txt` is already NomNaOCR's authoritative held-out
split - there's no train/test contamination to re-derive by picking pages, unlike Phase 0's
page-sampling approach.

Ground-truth text is loaded via `nomnaocr_lib.vocab.load_labels()` (unchanged, reused from
Phase 0) rather than a hand-rolled parser, so the same lowercasing + `is_clean_text` + min_length
filtering that the model's vocabulary/decode targets assume is applied consistently here too.

Usage:
    python scripts/build_eval_manifest.py --dataset-root experiments/NomNaOCR \
        --out data/manifest.json
"""
import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))

from nomnaocr_lib.vocab import load_labels  # noqa: E402

SUBSET_FILES = {
    "gt10": "Validate_gt10.txt",
    "lte10": "Validate_lte10.txt",
    "poem": "Validate_poem.txt",
    "prose": "Validate_prose.txt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path,
                         help="Path to the unzipped NomNaOCR dataset (contains Patches/)")
    parser.add_argument("--split", default="Patches/Validate.txt",
                         help="Held-out split file, relative to --dataset-root")
    parser.add_argument("--train-split", default="Patches/Train.txt",
                         help="Training split file, relative to --dataset-root, for the "
                              "contamination guard")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--sample-fraction", type=float, default=None,
                         help="Dev-loop only: randomly keep this fraction of patches. Never use "
                              "for final reported numbers.")
    parser.add_argument("--max-patches", type=int, default=None,
                         help="Dev-loop only: cap the manifest at this many patches (applied "
                              "after --sample-fraction). Never use for final reported numbers.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_path = args.dataset_root / args.split
    train_split_path = args.dataset_root / args.train_split
    patches_dir = args.dataset_root / "Patches"
    assert split_path.exists(), f"missing {split_path}"
    assert train_split_path.exists(), f"missing {train_split_path}"

    pairs = load_labels(str(split_path), min_length=1)
    print(f"{split_path}: {len(pairs)} labeled patches after is_clean_text/min_length filtering")

    present, missing = [], []
    for img_name, text in pairs:
        if (patches_dir / img_name).exists():
            present.append((img_name, text))
        else:
            missing.append(img_name)
    if missing:
        print(f"WARNING: {len(missing)} patches listed in {args.split} are missing on disk "
              f"(e.g. {missing[:5]}) - excluded from the manifest")

    train_keys = {img_name for img_name, _ in load_labels(str(train_split_path), min_length=1)}
    contaminated = [img_name for img_name, _ in present if img_name in train_keys]
    assert not contaminated, (
        f"{len(contaminated)} manifest patches also appear in {args.train_split} - "
        f"held-out/train contamination, refusing to build a manifest on this split"
    )

    subset_members = {}
    for tag, filename in SUBSET_FILES.items():
        subset_path = args.dataset_root / "Patches" / filename
        if not subset_path.exists():
            print(f"WARNING: subset file {subset_path} not found, skipping '{tag}' tagging")
            continue
        subset_members[tag] = {img_name for img_name, _ in load_labels(str(subset_path), min_length=1)}

    subset_total = len(subset_members.get("poem", set())) + len(subset_members.get("prose", set()))
    if subset_total and subset_total != len(pairs):
        print(f"NOTE: poem+prose subset files sum to {subset_total} lines vs. {len(pairs)} in "
              f"{args.split} - a known small discrepancy in NomNaOCR's released split files, "
              f"not a bug in this script.")

    entries = present
    if args.sample_fraction is not None:
        rng = random.Random(args.seed)
        k = round(len(entries) * args.sample_fraction)
        entries = rng.sample(entries, k)
        print(f"--sample-fraction {args.sample_fraction}: kept {len(entries)} of {len(present)} "
              f"(DEV-LOOP ONLY - do not use for final reported numbers)")
    if args.max_patches is not None and len(entries) > args.max_patches:
        rng = random.Random(args.seed)
        entries = rng.sample(entries, args.max_patches)
        print(f"--max-patches {args.max_patches}: kept {len(entries)} "
              f"(DEV-LOOP ONLY - do not use for final reported numbers)")

    manifest = []
    for img_name, text in entries:
        work = img_name.split("/")[0]
        subsets = [tag for tag, members in subset_members.items() if img_name in members]
        manifest.append({
            "img_name": img_name,
            "work": work,
            "ground_truth": text,
            "subsets": subsets,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} patches to {args.out}")


if __name__ == "__main__":
    main()
