"""Centralizes the sys.path.insert shims every experiments/*/scripts/*.py already does
individually (e.g. experiments/phase3_translation/scripts/translate.py), so api/ can import
nomnaocr_lib, translate_lib, and phase2b's lm package without duplicating them.

Import this module (for its side effect) before importing from nomnaocr_lib/translate_lib/lm
anywhere in api/ - e.g. `import api._pipeline_paths  # noqa: F401`.
"""
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EXPERIMENTS = _REPO_ROOT / "experiments"

for _path in (
    _EXPERIMENTS / "phase0_validation",   # nomnaocr_lib
    _EXPERIMENTS / "phase2b_postcorrection",  # lm
    _EXPERIMENTS / "phase3_translation",  # translate_lib
    # detect_lib stays in experiments/phase4_detection (not physically moved into api/) since its
    # own build_detection_manifest.py/run_detection.py/score_detection.py/tests still depend on
    # it there - reused via the same sys.path pattern as the three libraries above, consistent
    # with how they're handled, rather than a one-off duplicate copy of segment_page's logic.
    _EXPERIMENTS / "phase4_detection",  # detect_lib
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
