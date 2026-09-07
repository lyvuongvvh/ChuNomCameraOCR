"""Phase 0 validation: build data/manifest.json + data/sample_pages/ + data/sample_patches/
directly from an extracted NomNaOCR dataset folder (downloaded from Kaggle per README.md).

Confirmed dataset conventions (inspected directly against the real downloaded dataset - this
replaces the README's earlier "check the naming convention once downloaded" placeholder):
  - Pages/<work>/imgs/<page>.jpg is a full page scan; Pages/<work>/gts/<page>.txt has one line
    per column-patch (`x1,y1,...,x4,y4,text`), already in reading order.
  - Patches/<work>/<page>_<i>.jpg is the i-th line of that page's gts file (0-indexed, same
    reading order) - verified against Luc Van Tien/nlvnpf-0059-002 (14 gts lines, patches
    _0.._13, texts match Patches/All.txt exactly).
  - Patches/All.txt (`work/patchfile.jpg<TAB>text`) is the only part of Patches/ actually
    needed here - nomnaocr_lib/vocab.py only reads label text from it, never patch images, so
    copying the full ~38K-image Patches/ folder (as the README originally suggested) is
    unnecessary; run_nomnaocr.py's --all-labels can point straight at the downloaded All.txt.

Usage (run on the host, no Docker needed):
    python scripts/build_manifest.py --dataset-root /path/to/NomNaOCR --num-pages 15

IMPORTANT: patches are restricted to NomNaOCR's own held-out split (Patches/Validate.txt), not
just any patch belonging to a chosen page. An earlier version of this script sampled patches
without regard to train/validate membership; checking afterwards showed 144/186 (77%) of the
sampled patches were actually in NomNaOCR's *training* set (Patches/Train.txt), which would
have made NomNaOCR's Phase 0 accuracy mostly a measure of memorization, not generalization -
exactly what CLAUDE.md's "always check held-out test performance" rule warns about. The dataset
splits patches, not whole pages, into train/validate (a single page can have some patches in
each), so filtering to Validate.txt necessarily makes "pages" here mean "the held-out patches
belonging to that page," not the full page transcription - see build_manifest.py's docstring
on `ground_truth` below for what this means for scoring.
"""
import argparse
import json
import pathlib
import re
import shutil

IMAGE_EXT = ".jpg"


def load_labels(labels_path: pathlib.Path) -> dict[str, str]:
    """Parses a NomNaOCR `work/patchfile.jpg<TAB>text` label file (used for both All.txt and
    the Train.txt/Validate.txt split files, which share this format)."""
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            img_name, text = line.split("\t")
            labels[img_name] = text
    return labels


def sanitize(work: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", work)


def find_candidate_pages(dataset_root: pathlib.Path, validate_keys: set[str], min_held_out: int = 3):
    """Yield (work, page_stem, page_img_path, [held-out patch_paths in reading order]) for
    every page with a gts file, a page image, and at least `min_held_out` patches that are (a)
    present on disk and (b) in NomNaOCR's own Validate.txt split - i.e. patches CRNNxCTC was
    NOT trained on, so scoring it against these is a fair generalization test rather than a
    memorization check."""
    pages_dir = dataset_root / "Pages"
    patches_dir = dataset_root / "Patches"
    candidates = []
    for work_dir in sorted(p for p in pages_dir.iterdir() if p.is_dir()):
        work = work_dir.name
        gts_dir = work_dir / "gts"
        imgs_dir = work_dir / "imgs"
        if not gts_dir.is_dir() or not imgs_dir.is_dir():
            continue
        for gts_file in sorted(gts_dir.glob("*.txt")):
            page_stem = gts_file.stem
            page_img = imgs_dir / f"{page_stem}{IMAGE_EXT}"
            if not page_img.exists():
                continue
            lines = [line for line in gts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue
            held_out_patches = []
            for i in range(len(lines)):
                patch_path = patches_dir / work / f"{page_stem}_{i}{IMAGE_EXT}"
                if patch_path.exists() and f"{work}/{patch_path.name}" in validate_keys:
                    held_out_patches.append(patch_path)
            if len(held_out_patches) >= min_held_out:
                candidates.append((work, page_stem, page_img, held_out_patches))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path,
                         help="Path to the unzipped NomNaOCR dataset (contains Pages/, Patches/)")
    parser.add_argument("--num-pages", type=int, default=15)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data"))
    args = parser.parse_args()

    all_txt = args.dataset_root / "Patches" / "All.txt"
    validate_txt = args.dataset_root / "Patches" / "Validate.txt"
    assert all_txt.exists(), f"missing {all_txt}"
    assert validate_txt.exists(), f"missing {validate_txt}"
    labels = load_labels(all_txt)
    validate_keys = set(load_labels(validate_txt))

    candidates = find_candidate_pages(args.dataset_root, validate_keys)
    if not candidates:
        raise SystemExit(f"no usable pages found under {args.dataset_root}")

    # Evenly spaced sample across the sorted (work, page) list, so the sample spans multiple
    # works instead of clustering in whichever work happens to sort first.
    step = max(1, len(candidates) // args.num_pages)
    chosen = candidates[::step][: args.num_pages]

    sample_pages_dir = args.out_dir / "sample_pages"
    sample_patches_dir = args.out_dir / "sample_patches"
    sample_pages_dir.mkdir(parents=True, exist_ok=True)
    sample_patches_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for work, page_stem, page_img, patch_paths in chosen:
        key = f"{sanitize(work)}_{page_stem}"
        out_page_name = f"{key}{IMAGE_EXT}"
        shutil.copy(page_img, sample_pages_dir / out_page_name)

        patch_names = []
        gt_parts = []
        for i, patch_path in enumerate(patch_paths):
            out_patch_name = f"{key}_{i}{IMAGE_EXT}"
            shutil.copy(patch_path, sample_patches_dir / out_patch_name)
            patch_names.append(out_patch_name)
            gt_parts.append(labels.get(f"{work}/{patch_path.name}", ""))

        manifest[key] = {
            "page_image": out_page_name,
            "patches": patch_names,
            "ground_truth": "".join(gt_parts),
        }

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} pages to {manifest_path}")
    print(f"Copied page images to {sample_pages_dir}")
    print(f"Copied {sum(len(v['patches']) for v in manifest.values())} patches to {sample_patches_dir}")

    # run_nomnaocr.py needs the *full* All.txt (for vocab reconstruction), copied alongside the
    # rest of data/ so a single -v data:/workspace/data Docker mount covers everything it needs.
    all_labels_out = args.out_dir / "all_labels.txt"
    shutil.copy(all_txt, all_labels_out)
    print(f"Copied full label file to {all_labels_out}")


if __name__ == "__main__":
    main()
