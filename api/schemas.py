"""Pydantic request/response models for POST /v1/ocr."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LineResult(BaseModel):
    line_id: int
    polygon: list[tuple[float, float]]
    text: str | None = None
    recognition_error: str | None = None
    confidence: float | None = None
    reading: str | None = None
    translation: str | None = None
    translation_status: Literal["ok", "empty", "error", "skipped"] = "skipped"


class OcrResponse(BaseModel):
    request_id: str
    detection_status: Literal["ok", "no_lines_found"]
    lines: list[LineResult]
    line_count: int
    translated_count: int
    translation_truncated: bool
    warnings: list[str] = []
