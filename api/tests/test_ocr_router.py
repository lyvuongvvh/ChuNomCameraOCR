"""FastAPI TestClient + dependency_overrides tests for POST /v1/ocr. Builds a small standalone
app (same routers as api.main.app) instead of importing api.main.app directly, so tests never
trigger the real lifespan's build_dependencies() (which requires real weights/LM/reading-dict
files on disk) - covers: full success, 0-lines, partial recognition failure,
translate-opt-in-default-off, translate-cap-truncation, bad-upload-422.

    python -m unittest api.tests.test_ocr_router -v
"""
import io
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.deps import Dependencies, get_deps
from api.routers import health, ocr


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(ocr.router)
    return app


def make_fake_deps(anthropic_client=None) -> Dependencies:
    settings = MagicMock()
    settings.detection_model_path = "fake_model.mlmodel"
    settings.beam_width = 5
    settings.translate_model = "claude-sonnet-5"
    settings.max_lines_to_translate_default = 50
    return Dependencies(
        recognizer=MagicMock(),
        lm=MagicMock(),
        best_lambda=1.0,
        reading_dict={},
        anthropic_client=anthropic_client,
        translate_pool=_InlineExecutor(),
        settings=settings,
    )


class _InlineExecutor:
    """Runs submit() synchronously in-thread, wrapping the result in a real
    concurrent.futures.Future (not a bare MagicMock) - api/routers/ocr.py's as_completed(futures)
    inspects real Future-only internals (_condition, _state) that a MagicMock doesn't actually
    implement, and blocks forever waiting on them otherwise."""
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)
        return future


def make_upload_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 300), color="white").save(buf, format="JPEG")
    return buf.getvalue()


class TestOcrRouter(unittest.TestCase):
    def setUp(self):
        self.app = make_app()
        self.client = TestClient(self.app)

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_bad_upload_422(self):
        self.app.dependency_overrides[get_deps] = lambda: make_fake_deps()
        resp = self.client.post(
            "/v1/ocr", files={"file": ("not_an_image.txt", b"not an image", "text/plain")}
        )
        self.assertEqual(resp.status_code, 422)

    @patch("api.routers.ocr.detect_lines", return_value=[])
    def test_zero_lines_detected(self, _mock_detect):
        self.app.dependency_overrides[get_deps] = lambda: make_fake_deps()
        resp = self.client.post("/v1/ocr", files={"file": ("page.jpg", make_upload_jpeg(), "image/jpeg")})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["detection_status"], "no_lines_found")
        self.assertEqual(body["lines"], [])

    @patch("api.routers.ocr.recognize_crop")
    @patch("api.routers.ocr.detect_lines")
    def test_full_success_translate_opt_in_default_off(self, mock_detect, mock_recognize):
        mock_detect.return_value = [[(0, 0), (10, 0), (10, 10), (0, 10)]]
        mock_recognize.return_value = ("使通", -3.5)
        self.app.dependency_overrides[get_deps] = lambda: make_fake_deps(anthropic_client=MagicMock())

        resp = self.client.post("/v1/ocr", files={"file": ("page.jpg", make_upload_jpeg(), "image/jpeg")})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["detection_status"], "ok")
        self.assertEqual(body["line_count"], 1)
        self.assertEqual(body["lines"][0]["text"], "使通")
        # translate defaults to False - no translation attempted, no Anthropic call
        self.assertEqual(body["translated_count"], 0)
        self.assertIsNone(body["lines"][0]["translation"])
        self.assertEqual(body["lines"][0]["translation_status"], "skipped")

    @patch("api.routers.ocr.recognize_crop")
    @patch("api.routers.ocr.detect_lines")
    def test_partial_recognition_failure_other_lines_proceed(self, mock_detect, mock_recognize):
        mock_detect.return_value = [
            [(0, 0), (10, 0), (10, 10), (0, 10)],
            [(20, 0), (30, 0), (30, 10), (20, 10)],
        ]
        mock_recognize.side_effect = [RuntimeError("bad crop"), ("ok text", -1.0)]
        self.app.dependency_overrides[get_deps] = lambda: make_fake_deps()

        resp = self.client.post("/v1/ocr", files={"file": ("page.jpg", make_upload_jpeg(), "image/jpeg")})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["line_count"], 2)
        self.assertIsNotNone(body["lines"][0]["recognition_error"])
        self.assertIsNone(body["lines"][0]["text"])
        self.assertEqual(body["lines"][1]["text"], "ok text")
        self.assertTrue(any("recognition failed" in w for w in body["warnings"]))

    @patch("api.routers.ocr.translate_text")
    @patch("api.routers.ocr.recognize_crop")
    @patch("api.routers.ocr.detect_lines")
    def test_translate_cap_truncation(self, mock_detect, mock_recognize, mock_translate):
        mock_detect.return_value = [[(i, 0), (i + 5, 0), (i + 5, 5), (i, 5)] for i in range(3)]
        mock_recognize.return_value = ("text", -1.0)
        mock_translate.return_value = ("reading", "translation", "ok")
        self.app.dependency_overrides[get_deps] = lambda: make_fake_deps(anthropic_client=MagicMock())

        resp = self.client.post(
            "/v1/ocr",
            files={"file": ("page.jpg", make_upload_jpeg(), "image/jpeg")},
            data={"translate": "true", "max_lines_to_translate": "2"},
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["translated_count"], 2)
        self.assertTrue(body["translation_truncated"])

    def test_translate_true_without_api_key_configured_fails_fast_503(self):
        self.app.dependency_overrides[get_deps] = lambda: make_fake_deps(anthropic_client=None)
        resp = self.client.post(
            "/v1/ocr",
            files={"file": ("page.jpg", make_upload_jpeg(), "image/jpeg")},
            data={"translate": "true"},
        )
        self.assertEqual(resp.status_code, 503)

    def test_healthz(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
