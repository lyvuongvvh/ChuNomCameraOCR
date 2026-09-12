"""Wraps apply_reading_dict (Phase 3, Stage 1) + translate_line (Phase 3, Stage 2).

translate_line itself only retries internally on an *empty* response - it never catches
exceptions from the Anthropic call (network errors, rate limits, API errors all propagate
uncaught). Catching those here, per-line, is this module's job, so one failed line's exception
doesn't take down the whole request.
"""
from __future__ import annotations

import api._pipeline_paths  # noqa: F401,E402

from translate_lib.reading import apply_reading_dict  # noqa: E402
from translate_lib.llm_translate import translate_line  # noqa: E402


def translate_text(
    client,
    text: str,
    reading_dict: dict,
    model: str,
) -> tuple[str, str, str]:
    """Returns (reading, translation, status) where status is "ok", "empty" (translate_line's
    own already-tested non-raising outcome - its internal retry ran and still got nothing), or
    "error" (the Anthropic call raised - caught here, not propagated)."""
    reading = apply_reading_dict(text, reading_dict)
    try:
        translation, _usage = translate_line(client, text, reading, model=model)
    except Exception:
        return reading, "", "error"
    return reading, translation, ("ok" if translation else "empty")
