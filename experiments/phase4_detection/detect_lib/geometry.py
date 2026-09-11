"""Phase 4 detection: parses NomNaOCR's real, previously-unused detection ground truth
(`experiments/NomNaOCR/Pages/*/gts/*.txt`) and scores a candidate detector against it via
IoU-based precision/recall - the validate-before-wiring-in step described in
`experiments/phase4_detection/README.md`.

Ground truth format (confirmed consistent across all 9 works, 2,953 pages): one line per text
column, `x1,y1,x2,y2,x3,y3,x4,y4,transcript` - a 4-point quad (p0=top-left, p1=top-right,
p2=bottom-right, p3=bottom-left) followed by the recognized text for that column. No embedded
commas found in any transcript (verified: every line has exactly 9 comma-separated fields), so a
plain split is safe - no special-casing needed for a transcript containing a literal comma.
"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

Point = tuple[float, float]


@dataclass
class Quad:
    """Despite the name, `points` isn't restricted to exactly 4 - ground-truth quads always are,
    but a detector's predicted region boundary (e.g. kraken's blla.segment output) can be an
    arbitrary polygon with many more vertices. polygon_iou works on both via shapely, which
    doesn't care how many points a Polygon has."""
    points: tuple[Point, ...]
    transcript: str = ""

    def polygon(self) -> Polygon:
        return Polygon(self.points)


def parse_gts_line(line: str) -> Quad:
    parts = line.rstrip("\n").split(",", maxsplit=8)
    if len(parts) != 9:
        raise ValueError(f"expected 8 coords + transcript (9 comma-separated fields), got {len(parts)}: {line!r}")
    coords = [float(p) for p in parts[:8]]
    points = tuple(zip(coords[0::2], coords[1::2]))
    return Quad(points=points, transcript=parts[8])


def parse_gts_file(path: str) -> list[Quad]:
    with open(path, encoding="utf-8") as f:
        return [parse_gts_line(line) for line in f if line.strip()]


def polygon_iou(a: Quad, b: Quad) -> float:
    """Real polygon IoU (not axis-aligned bounding-box IoU) via shapely - the ground-truth quads
    are near-rectangular in practice, but a detector's predicted quads aren't guaranteed to be
    (kraken's segmenter follows a baseline, which can be slightly non-rectangular), so this
    doesn't assume axis-alignment."""
    pa, pb = a.polygon(), b.polygon()
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    intersection = pa.intersection(pb).area
    if intersection == 0.0:
        return 0.0
    union = pa.union(pb).area
    return intersection / union if union > 0 else 0.0


@dataclass
class MatchResult:
    matched_gt: int
    total_gt: int
    matched_pred: int
    total_pred: int

    @property
    def recall(self) -> float:
        return self.matched_gt / self.total_gt if self.total_gt else 0.0

    @property
    def precision(self) -> float:
        return self.matched_pred / self.total_pred if self.total_pred else 0.0


def greedy_match(gt_quads: list[Quad], pred_quads: list[Quad], iou_threshold: float = 0.5) -> MatchResult:
    """Standard ICDAR/DetEval-style greedy one-to-one matching: rank every (gt, pred) pair by
    IoU descending, and assign greedily as long as both sides are still unmatched and IoU clears
    the threshold. A pred quad that doesn't match any gt (e.g. a spurious over-detection) simply
    lowers precision; a gt quad no pred quad reaches lowers recall - matching counts, not the
    pairs themselves, is all a precision/recall rollup needs.
    """
    if not gt_quads or not pred_quads:
        return MatchResult(matched_gt=0, total_gt=len(gt_quads), matched_pred=0, total_pred=len(pred_quads))

    pairs = []
    for gi, gt in enumerate(gt_quads):
        for pi, pred in enumerate(pred_quads):
            iou = polygon_iou(gt, pred)
            if iou >= iou_threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda t: t[0], reverse=True)

    matched_gt_idx: set[int] = set()
    matched_pred_idx: set[int] = set()
    for _iou, gi, pi in pairs:
        if gi in matched_gt_idx or pi in matched_pred_idx:
            continue
        matched_gt_idx.add(gi)
        matched_pred_idx.add(pi)

    return MatchResult(
        matched_gt=len(matched_gt_idx), total_gt=len(gt_quads),
        matched_pred=len(matched_pred_idx), total_pred=len(pred_quads),
    )
