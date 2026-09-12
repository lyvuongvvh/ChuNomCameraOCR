"""POST /v1/ocr - detection -> recognition (+ post-correction) -> optional translation.

Error-handling table (Phase 4 plan §4):
  0 lines detected            -> 200, detection_status="no_lines_found", lines=[]
  detection throws            -> 422 (bad upload) or 500 (kraken error on a valid image)
  one line fails recognition  -> that line gets recognition_error set, others proceed, 200
  translate_line raises       -> per-line translation_status="error", others proceed, 200
  translate=True, no API key  -> 503 before any per-line attempt
"""
from __future__ import annotations

import io
import pathlib
import tempfile
import uuid
from concurrent.futures import as_completed

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.deps import Dependencies, get_deps
from api.pipeline.detection import crop_region, detect_lines
from api.pipeline.recognition import recognize_crop
from api.pipeline.translation import translate_text
from api.schemas import LineResult, OcrResponse

router = APIRouter()


@router.post("/v1/ocr", response_model=OcrResponse)
def ocr(
    file: UploadFile = File(...),
    rescore: bool = Form(True),
    translate: bool = Form(False),
    translate_line_ids: str | None = Form(None),  # comma-separated ints, multipart has no native list
    max_lines_to_translate: int | None = Form(None),
    deps: Dependencies = Depends(get_deps),
) -> OcrResponse:
    request_id = str(uuid.uuid4())
    warnings: list[str] = []

    raw = file.file.read()
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="Uploaded file is not a decodable image.")

    if translate and deps.anthropic_client is None:
        raise HTTPException(
            status_code=503,
            detail="translate=True but ANTHROPIC_API_KEY is not configured on this server.",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        page_path = str(pathlib.Path(tmp_dir) / "page.jpg")
        image.convert("RGB").save(page_path)
        try:
            polygons = detect_lines(page_path, str(deps.settings.detection_model_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Detection failed: {e!r}")

    if not polygons:
        return OcrResponse(
            request_id=request_id,
            detection_status="no_lines_found",
            lines=[],
            line_count=0,
            translated_count=0,
            translation_truncated=False,
            warnings=warnings,
        )

    lam = deps.best_lambda if rescore else 0.0
    lines: list[LineResult] = []
    for line_id, points in enumerate(polygons):
        result = LineResult(line_id=line_id, polygon=points)
        try:
            crop = crop_region(image, points)
            text, confidence = recognize_crop(
                crop, deps.recognizer, deps.lm, lam, deps.settings.beam_width
            )
            result.text = text
            result.confidence = confidence
        except Exception as e:
            result.recognition_error = repr(e)
            warnings.append(f"line {line_id}: recognition failed: {e!r}")
        lines.append(result)

    translated_count = 0
    translation_truncated = False
    if translate:
        cap = max_lines_to_translate if max_lines_to_translate is not None else deps.settings.max_lines_to_translate_default
        if translate_line_ids is not None:
            requested_ids = [int(s) for s in translate_line_ids.split(",") if s.strip()]
        else:
            requested_ids = [l.line_id for l in lines if l.text is not None]

        eligible_ids = [i for i in requested_ids if 0 <= i < len(lines) and lines[i].text is not None]
        target_ids = eligible_ids[:cap]
        translation_truncated = len(eligible_ids) > len(target_ids)

        futures = {
            deps.translate_pool.submit(
                translate_text, deps.anthropic_client, lines[i].text, deps.reading_dict,
                deps.settings.translate_model,
            ): i
            for i in target_ids
        }
        for future in as_completed(futures):
            i = futures[future]
            reading, translation, status = future.result()
            lines[i].reading = reading
            lines[i].translation = translation if translation else None
            lines[i].translation_status = status
            translated_count += 1
            if status == "empty":
                warnings.append(f"line {i}: translation returned empty after retry")
            elif status == "error":
                warnings.append(f"line {i}: translation request failed")

    return OcrResponse(
        request_id=request_id,
        detection_status="ok",
        lines=lines,
        line_count=len(lines),
        translated_count=translated_count,
        translation_truncated=translation_truncated,
        warnings=warnings,
    )
