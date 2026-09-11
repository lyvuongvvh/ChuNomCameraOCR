"""Phase 4 detection: builds a manifest of (page image, ground-truth quads) pairs from
NomNaOCR's real, previously-unused detection labels (`experiments/NomNaOCR/Pages/*/gts/*.txt`),
for scripts/run_detection.py and scripts/score_detection.py to consume.

Verified before writing this: `gts/*.txt` and `imgs/*.jpg` pair 1:1 by filename stem across all
9 works (2,953 of each, zero missing pairs either direction).

Usage:
    python scripts/build_detection_manifest.py \
        --pages-root ../NomNaOCR/Pages \
        --out data/manifest.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from detect_lib.geometry import parse_gts_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages-root", required=True, type=pathlib.Path, help="experiments/NomNaOCR/Pages")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--sample-fraction", type=float, default=None,
                     help="For fast dev iteration only - e.g. 0.05 for a ~5% sample. Final "
                          "validation numbers should always use the full manifest, not a sample.")
    args = ap.parse_args()

    entries = []
    skipped_no_img = 0
    for work_dir in sorted(args.pages_root.iterdir()):
        if not work_dir.is_dir():
            continue
        gts_dir = work_dir / "gts"
        imgs_dir = work_dir / "imgs"
        if not gts_dir.is_dir():
            continue
        for gts_path in sorted(gts_dir.glob("*.txt")):
            img_path = imgs_dir / f"{gts_path.stem}.jpg"
            if not img_path.exists():
                skipped_no_img += 1
                continue
            quads = parse_gts_file(str(gts_path))
            entries.append({
                "work": work_dir.name,
                "page_name": gts_path.stem,
                "image_path": str(img_path),
                "gt_quads": [{"points": q.points, "transcript": q.transcript} for q in quads],
            })

    if skipped_no_img:
        print(f"WARNING: {skipped_no_img} gts files had no matching image - skipped")

    if args.sample_fraction is not None:
        import random
        random.seed(0)
        k = max(1, round(len(entries) * args.sample_fraction))
        entries = random.sample(entries, k)
        print(f"Sampled {k} of the full page set (--sample-fraction {args.sample_fraction})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    total_quads = sum(len(e["gt_quads"]) for e in entries)
    print(f"Wrote {len(entries)} pages ({total_quads} ground-truth lines total) to {args.out}")


if __name__ == "__main__":
    main()
