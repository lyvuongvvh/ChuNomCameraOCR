"""Constructs every heavy/expensive dependency exactly once. Called from main.py's lifespan
startup and stored on app.state - never reconstructed per-request.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from fastapi import Request

import api._pipeline_paths  # noqa: F401,E402

from nomnaocr_lib.model import CRNNRecognizer  # noqa: E402
from nomnaocr_lib.vocab import build_vocab, max_label_length  # noqa: E402
from lm.ngram_lm import CharNgramLM  # noqa: E402

from api.config import Settings  # noqa: E402


@dataclass
class Dependencies:
    recognizer: CRNNRecognizer
    lm: CharNgramLM
    best_lambda: float
    reading_dict: dict
    anthropic_client: object | None  # None if ANTHROPIC_API_KEY isn't set - translate=True fails fast, not lazily
    translate_pool: ThreadPoolExecutor
    settings: Settings


def _require_file(path, what: str):
    if not path.exists():
        raise RuntimeError(f"Missing {what}: {path} - fail fast at startup, not on first request.")
    return path


def build_dependencies(settings: Settings) -> Dependencies:
    _require_file(settings.weights_path, "recognizer weights")
    _require_file(settings.all_labels_path, "recognizer vocab source (All.txt)")
    _require_file(settings.char_lm_path, "char n-gram LM")
    _require_file(settings.best_lambda_path, "best_lambda.json")
    _require_file(settings.reading_dict_path, "reading_dict.json")
    _require_file(settings.detection_model_path, "detection model checkpoint")

    vocab = build_vocab(str(settings.all_labels_path), min_length=1)
    max_length = max_label_length(str(settings.all_labels_path), min_length=1)
    recognizer = CRNNRecognizer(vocab, max_length, str(settings.weights_path))

    lm = CharNgramLM.load(str(settings.char_lm_path))
    best_lambda = json.loads(settings.best_lambda_path.read_text(encoding="utf-8"))["lambda"]
    reading_dict = json.loads(settings.reading_dict_path.read_text(encoding="utf-8"))

    anthropic_client = None
    if settings.anthropic_api_key:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # One shared, bounded pool for the whole process, not one per request - matches
    # experiments/phase3_translation/scripts/translate.py's ThreadPoolExecutor pattern.
    translate_pool = ThreadPoolExecutor(max_workers=settings.translate_workers)

    return Dependencies(
        recognizer=recognizer,
        lm=lm,
        best_lambda=best_lambda,
        reading_dict=reading_dict,
        anthropic_client=anthropic_client,
        translate_pool=translate_pool,
        settings=settings,
    )


def get_deps(request: Request) -> Dependencies:
    """FastAPI dependency - routers take `deps: Dependencies = Depends(get_deps)`. Tests override
    this via `app.dependency_overrides[get_deps] = lambda: fake_deps` (FastAPI TestClient +
    dependency_overrides, per api/tests/test_ocr_router.py) rather than reconstructing real
    heavy objects."""
    return request.app.state.deps
