"""Wraps the validated Rung 1 detection model (experiments/phase4_detection/README.md:
mean recall=0.902, mean precision=0.887 on real held-out ground truth, raw output, no merge
post-processing needed - unlike Rung 0's generic segmenter). `segment_page` itself is reused from
experiments/phase4_detection/detect_lib/kraken_segment.py (not duplicated), including its
`torch.set_num_threads(1)`-before-shapely import-order fix for a real SIGSEGV that fix depends on
running first - importing it here, at module load time, keeps that ordering guaranteed regardless
of what else this process imports afterward.
"""
from __future__ import annotations

import api._pipeline_paths  # noqa: F401,E402

from detect_lib.kraken_segment import segment_page  # noqa: E402

from PIL import Image  # noqa: E402

Point = tuple[float, float]


def detect_lines(img_path: str, model_path: str) -> list[list[Point]]:
    """Returns one polygon (list of (x, y) points) per detected text line, in the coordinate
    space of the original image at img_path."""
    quads = segment_page(img_path, model_path=model_path)
    return [list(q.points) for q in quads]


def crop_region(image: Image.Image, points: list[Point]) -> Image.Image:
    """Crops a detected line's polygon out of the full page image. A plain axis-aligned bounding
    box, not a kraken-style baseline dewarp - confirmed by comparing NomNaOCR's own real
    Patches/*.jpg files against their source gts/*.txt quads (pixel-identical to a straight bbox
    crop, no perspective warp), and safe regardless since CRNNRecognizer.predict_beams internally
    resizes/pads whatever it's given to a fixed (432, 48)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return image.crop((min(xs), min(ys), max(xs), max(ys)))
