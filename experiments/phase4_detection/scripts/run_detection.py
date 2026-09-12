"""Phase 4 detection: runs kraken's segmenter - the generic bundled default (Rung 0) or a Rung 1
fine-tuned checkpoint via --model - optionally followed by an x-position merge post-process, over
a page sample, writing predictions in the format scripts/score_detection.py expects. See
detect_lib/kraken_segment.py and README.md for why Rung 0 needs the merge step - measured
directly, not assumed: kraken's generic bundled segmenter fragments every real column into many
small vertically-stacked pieces sharing the true column's x-position, not noise mixed with good
detections. A Rung 1 fine-tuned checkpoint is expected to need this less or not at all - pass
neither --gap-threshold nor --gap-fraction to score its raw output directly.

Usage:
    python scripts/run_detection.py \
        --manifest data/manifest.json \
        --limit 15 \
        --gap-threshold 8 \
        --out data/predictions_rung0.json

    python scripts/run_detection.py \
        --manifest data/manifest_sample15.json \
        --model data/rung1_model/segtrain_finetuned_best.mlmodel \
        --out data/predictions_rung1_best.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from detect_lib.kraken_segment import merge_by_xposition, segment_page  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N manifest pages (fast dev iteration)")
    ap.add_argument("--gap-threshold", type=float, default=None,
                     help="Fixed x-center gap (px) - mutually exclusive with --gap-fraction. Measured to work "
                          "on one 290px-wide page (README.md) but confirmed NOT to generalize across works "
                          "with different scan widths - prefer --gap-fraction unless you have a reason not to.")
    ap.add_argument("--gap-fraction", type=float, default=None,
                     help="Gap threshold as a fraction of page width, scale-independent across works with "
                          "different scan resolutions (see README.md's scale-mismatch finding). "
                          "8px / 290px-wide tuning page = ~0.0276.")
    ap.add_argument("--upscale", action="store_true")
    ap.add_argument("--model", type=str, default=None,
                     help="Path to a fine-tuned segmentation checkpoint (Rung 1). Omit to use "
                          "kraken's generic bundled default (Rung 0).")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.gap_threshold is not None and args.gap_fraction is not None:
        raise SystemExit("pass at most one of --gap-threshold or --gap-fraction")
    do_merge = args.gap_threshold is not None or args.gap_fraction is not None

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[:args.limit]

    predictions = {}
    if args.resume and args.out.exists():
        predictions = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"--resume: {len(predictions)} pages already in {args.out}, skipping those")

    todo = [e for e in manifest if e["page_name"] not in predictions]
    if do_merge:
        mode = f"gap_threshold={args.gap_threshold}" if args.gap_threshold is not None else f"gap_fraction={args.gap_fraction}"
    else:
        mode = "no merge - raw model output"
    model_desc = args.model if args.model else "generic bundled default (Rung 0)"
    print(f"Running detection on {len(todo)} pages (model={model_desc}, {mode}, upscale={args.upscale})")

    start = time.monotonic()
    for i, entry in enumerate(todo, 1):
        page_name = entry["page_name"]
        try:
            quads = segment_page(entry["image_path"], upscale=args.upscale, model_path=args.model)
            if do_merge:
                from PIL import Image
                page_width = Image.open(entry["image_path"]).size[0]
                gap = args.gap_threshold if args.gap_threshold is not None else args.gap_fraction * page_width
                merged = merge_by_xposition(quads, gap_threshold=gap)
            else:
                merged = quads
        except Exception as e:
            print(f"  ERROR on {page_name}: {e!r}")
            merged = []
        predictions[page_name] = [list(q.points) for q in merged]

        if i % 5 == 0 or i == len(todo):
            elapsed = time.monotonic() - start
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [{i}/{len(todo)}] {elapsed:.0f}s elapsed")

    print(f"Wrote {len(predictions)} page predictions to {args.out}")


if __name__ == "__main__":
    main()
