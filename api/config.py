"""Env-var-configurable settings for the recognizer/rescoring/translation/detection pipelines.
Defaults point at the existing experiments/*/data locations these files already live in - they
are reused in place, not copied, so existing experiment scripts/tests keep working unchanged.
"""
from __future__ import annotations

import os
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EXPERIMENTS = _REPO_ROOT / "experiments"


def _env_path(name: str, default: pathlib.Path) -> pathlib.Path:
    value = os.environ.get(name)
    return pathlib.Path(value) if value else default


class Settings:
    def __init__(self) -> None:
        # Recognition (Phase 0/2)
        self.weights_path = _env_path(
            "WEIGHTS_PATH", _EXPERIMENTS / "NomNaOCR_H5" / "NomNaOCR_CRNNxCTC.h5"
        )
        self.all_labels_path = _env_path(
            "ALL_LABELS_PATH", _EXPERIMENTS / "NomNaOCR" / "Patches" / "All.txt"
        )
        # beam_width: 5, not predict_beams' own default of 10 - best_lambda.json's tuning was
        # fit against the real beam_width=5 run (see experiments/phase2b_postcorrection/README.md
        # and the Phase 4 plan's Context section). Using 10 here would feed rescore_patch a
        # differently-distributed candidate set than lambda was calibrated on.
        self.beam_width = int(os.environ.get("BEAM_WIDTH", "5"))

        # Post-correction (Phase 2b)
        self.char_lm_path = _env_path(
            "CHAR_LM_PATH", _EXPERIMENTS / "phase2b_postcorrection" / "data" / "char_lm.json"
        )
        self.best_lambda_path = _env_path(
            "BEST_LAMBDA_PATH", _EXPERIMENTS / "phase2b_postcorrection" / "data" / "best_lambda.json"
        )

        # Translation (Phase 3)
        self.reading_dict_path = _env_path(
            "READING_DICT_PATH", _EXPERIMENTS / "phase3_translation" / "data" / "reading_dict.json"
        )
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.translate_model = os.environ.get("TRANSLATE_MODEL", "claude-sonnet-5")
        self.translate_workers = int(os.environ.get("TRANSLATE_WORKERS", "8"))
        self.max_lines_to_translate_default = int(os.environ.get("MAX_LINES_TO_TRANSLATE", "50"))

        # Detection (Phase 4, Rung 1 - validated, see experiments/phase4_detection/README.md)
        self.detection_model_path = _env_path(
            "DETECTION_MODEL_PATH",
            _EXPERIMENTS / "phase4_detection" / "data" / "rung1_model" / "segtrain_finetuned_best.mlmodel",
        )


settings = Settings()
