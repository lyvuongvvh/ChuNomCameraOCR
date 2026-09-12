"""Unit tests for api/pipeline/translation.py - a MagicMock Anthropic client + SimpleNamespace
responses, same pattern as experiments/phase3_translation/tests/test_llm_translate.py.

    python -m unittest api.tests.test_pipeline_translation -v
"""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from api.pipeline.translation import translate_text

READING_DICT = {"使": {"readings": ["sứ"]}, "通": {"readings": ["thông"]}}


def make_mock_client(response_text: str, input_tokens: int = 42, output_tokens: int = 7):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=response_text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return client


class TestTranslateText(unittest.TestCase):
    def test_success_returns_ok_status(self):
        client = make_mock_client("Đây là bản dịch.")
        reading, translation, status = translate_text(client, "使通", READING_DICT, model="claude-sonnet-5")
        self.assertEqual(reading, "sứ thông")
        self.assertEqual(translation, "Đây là bản dịch.")
        self.assertEqual(status, "ok")

    def test_still_empty_after_translate_lines_own_retry_is_empty_status_not_error(self):
        client = make_mock_client("")  # translate_line retries once internally, still empty
        reading, translation, status = translate_text(client, "使通", READING_DICT, model="claude-sonnet-5")
        self.assertEqual(translation, "")
        self.assertEqual(status, "empty")
        self.assertEqual(client.messages.create.call_count, 2)

    def test_raised_exception_is_caught_as_error_status_not_propagated(self):
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("rate limited")
        reading, translation, status = translate_text(client, "使通", READING_DICT, model="claude-sonnet-5")
        self.assertEqual(reading, "sứ thông")  # reading is computed before the API call, unaffected
        self.assertEqual(translation, "")
        self.assertEqual(status, "error")

    def test_unresolved_characters_pass_through_in_brackets(self):
        client = make_mock_client("output")
        reading, _translation, _status = translate_text(client, "使X", READING_DICT, model="claude-sonnet-5")
        self.assertEqual(reading, "sứ [X]")


if __name__ == "__main__":
    unittest.main()
