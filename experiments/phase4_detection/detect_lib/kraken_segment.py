"""Phase 4 detection, Rung 0: kraken's own generic bundled segmenter (`model=None`), not CHAT's
trained segmentation model - genuinely untested in this project before now (Phase 0 only tested
CHAT's own `chat_seg.mlmodel`). See README.md for why that's a real, meaningful distinction, not
just a rerun of the same prior negative result.

Image preprocessing mirrors experiments/phase0_validation/scripts/run_chat.py's already-debugged
approach (mode "L" not "1" - PIL's mode "1" is bit-packed and numpy reads it back as bool,
silently breaking kraken's normalization).
"""
from __future__ import annotations

# Must happen before any other import (including detect_lib.geometry, which pulls in shapely/
# GEOS) - confirmed by testing: a script that imports shapely before torch touches it hits a
# reproducible SIGSEGV in blla.segment ("std::system_error: random_device could not be read"),
# even though the SAME torch.set_num_threads(1) call fixes it fine when nothing else was
# imported first. Import order, not just call order, matters here - set it at module load time
# so any caller of this module gets it regardless of what else they import afterward. Same root
# cause experiments/phase0_validation/scripts/run_chat.py already works around for CHAT's
# segmenter.
import torch  # noqa: E402

torch.set_num_threads(1)

from PIL import Image  # noqa: E402

from detect_lib.geometry import Quad  # noqa: E402

DEFAULT_MIN_LONG_SIDE = 1600  # same target run_chat.py uses for CHAT, for a fair comparison


def upscale_if_small(img: Image.Image, target_long_side: int = DEFAULT_MIN_LONG_SIDE) -> Image.Image:
    long_side = max(img.size)
    if long_side >= target_long_side:
        return img
    scale = target_long_side / long_side
    new_size = (round(img.width * scale), round(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


def segment_page(img_path: str, upscale: bool = False, model_path: str | None = None) -> list[Quad]:
    """Runs kraken's segmenter on one page - the generic bundled default (model=None) unless
    model_path points at a Rung 1 fine-tuned checkpoint (see scripts/build_segtrain_data.py,
    README.md) - and returns the raw (unfiltered) detected regions as Quads using each line's
    'boundary' polygon - not the 'baseline' polyline, which is a curve through the line's center,
    not its extent, and would need to be inflated into a region by some assumed width. kraken
    already computes a real boundary polygon per line (confirmed by direct inspection - see
    README.md); use that directly rather than re-deriving something worse from the baseline.
    """
    from kraken import blla
    from kraken.lib import vgsl

    img = Image.open(img_path).convert("L")
    if upscale:
        img = upscale_if_small(img)

    model = vgsl.TorchVGSLModel.load_model(model_path) if model_path else None
    seg = blla.segment(img, text_direction="vertical-rl", model=model)
    lines = seg["lines"] if isinstance(seg, dict) else seg.lines

    quads = []
    for line in lines:
        boundary = line["boundary"] if isinstance(line, dict) else line.boundary
        if boundary is None or len(boundary) < 3:
            continue  # not a real polygon - skip rather than let shapely choke on it
        quads.append(Quad(points=tuple((float(x), float(y)) for x, y in boundary)))
    return quads


def filter_by_area(quads: list[Quad], min_area: float) -> list[Quad]:
    """Drops degenerate/tiny detected regions. Kept for completeness, but measured as
    ineffective alone on this domain (see README.md): kraken's generic segmenter doesn't mix a
    few good detections in with noise, it fragments every real column into many small vertically-
    stacked pieces, ALL of them smaller than the smallest real column - no area threshold
    separates "real" from "fragment" because none of the raw detections are real-sized. Use
    merge_by_xposition for this domain instead."""
    return [q for q in quads if q.polygon().area >= min_area]


def _xcenter(q: Quad) -> float:
    xs = [p[0] for p in q.points]
    return (min(xs) + max(xs)) / 2


def merge_by_xposition(quads: list[Quad], gap_threshold: float) -> list[Quad]:
    """Merges kraken's per-fragment detections back into whole columns. Measured directly (see
    README.md): kraken's generic segmenter correctly finds each true column's x-position, but
    returns 5-14 small vertically-stacked fragments per column instead of one tall region -
    confirmed by comparing predicted fragment x-centers against real gts/*.txt column x-ranges
    on a real page, where fragments cluster tightly (within ~10px) around each of the true
    columns' centers, with a much larger gap (~20px+) between columns. This is a 1D clustering
    problem, not a general 2D one, given the domain's near-uniform column pitch: sort by
    x-center, greedily start a new cluster whenever the gap to the next fragment's x-center
    exceeds gap_threshold, then take each cluster's convex hull as the merged column region.
    `gap_threshold` needs to be smaller than the real inter-column gap but larger than the
    within-column fragment spread - both measured, not assumed, when tuning this per page sample.
    """
    if not quads:
        return []
    ordered = sorted(quads, key=_xcenter)
    clusters: list[list[Quad]] = [[ordered[0]]]
    for prev, curr in zip(ordered, ordered[1:]):
        if _xcenter(curr) - _xcenter(prev) <= gap_threshold:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])

    merged = []
    for cluster in clusters:
        all_points = [p for q in cluster for p in q.points]
        hull = Quad(points=tuple(all_points)).polygon().convex_hull
        merged.append(Quad(points=tuple(hull.exterior.coords)))
    return merged
