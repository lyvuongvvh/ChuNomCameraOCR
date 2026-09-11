"""Phase 4 detection: scores a candidate detector's predictions against real ground truth via
IoU-based precision/recall (detect_lib/geometry.py) - the validate-before-wiring-in gate described
in results.md. Bars (see README.md): recall >= 0.90 AND precision >= 0.70, averaged over the
scored page sample, to consider a detection candidate good enough to promote into api/.

Usage:
    python scripts/score_detection.py \
        --manifest data/manifest.json \
        --predictions data/predictions_rung0.json \
        --out data/scores_rung0.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from detect_lib.geometry import Quad, greedy_match  # noqa: E402

GOOD_ENOUGH_RECALL = 0.90
GOOD_ENOUGH_PRECISION = 0.70


def _quads_from_points(points_list: list) -> list[Quad]:
    return [Quad(points=tuple(tuple(p) for p in points)) for points in points_list]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--predictions", required=True, type=pathlib.Path,
                     help='JSON: {page_name: [[[x,y],[x,y],[x,y],[x,y]], ...]} - one entry per '
                          "predicted quad's 4 points, no transcript needed for scoring")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))

    per_page = []
    missing_predictions = 0
    for entry in manifest:
        page_name = entry["page_name"]
        gt_quads = [Quad(points=tuple(tuple(p) for p in q["points"])) for q in entry["gt_quads"]]
        if page_name not in predictions:
            missing_predictions += 1
            continue
        pred_quads = _quads_from_points(predictions[page_name])
        result = greedy_match(gt_quads, pred_quads, iou_threshold=args.iou_threshold)
        per_page.append({
            "page_name": page_name, "work": entry["work"],
            "recall": result.recall, "precision": result.precision,
            "n_gt": result.total_gt, "n_pred": result.total_pred,
        })

    if missing_predictions:
        print(f"WARNING: {missing_predictions} manifest pages had no prediction entry - excluded from scoring")
    if not per_page:
        raise SystemExit("no pages scored - check --predictions keys match manifest page_names")

    mean_recall = statistics.mean(p["recall"] for p in per_page)
    mean_precision = statistics.mean(p["precision"] for p in per_page)
    passes_bar = mean_recall >= GOOD_ENOUGH_RECALL and mean_precision >= GOOD_ENOUGH_PRECISION

    print(f"Scored {len(per_page)} pages at IoU >= {args.iou_threshold}")
    print(f"  mean recall:    {mean_recall:.3f}  (bar: >= {GOOD_ENOUGH_RECALL})")
    print(f"  mean precision: {mean_precision:.3f}  (bar: >= {GOOD_ENOUGH_PRECISION})")
    print(f"  {'PASSES' if passes_bar else 'DOES NOT PASS'} the good-enough-to-proceed bar")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "iou_threshold": args.iou_threshold,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "passes_bar": passes_bar,
        "n_pages_scored": len(per_page),
        "per_page": per_page,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote per-page scores to {args.out}")


if __name__ == "__main__":
    main()
