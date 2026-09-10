"""Unit tests for translate_lib/llm_translate.py using a mocked Anthropic client - no real API
calls, no API key needed:

    python -m unittest tests.test_llm_translate -v
"""
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from translate_lib.llm_translate import translate_line, SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402


def make_mock_client(response_text: str, input_tokens: int = 42, output_tokens: int = 7):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=response_text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return client


class TestTranslateLine(unittest.TestCase):
    def test_returns_translation_and_usage(self):
        client = make_mock_client("Đây là bản dịch.")
        translation, usage = translate_line(client, "使通好", "sứ thông háo", model="claude-sonnet-5")
        self.assertEqual(translation, "Đây là bản dịch.")
        self.assertEqual(usage, {"input_tokens": 42, "output_tokens": 7})

    def test_strips_whitespace_from_response(self):
        client = make_mock_client("  Bản dịch có khoảng trắng.  \n")
        translation, _ = translate_line(client, "text", "reading")
        self.assertEqual(translation, "Bản dịch có khoảng trắng.")

    def test_concatenates_multiple_text_blocks(self):
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Phần một. "),
                SimpleNamespace(type="text", text="Phần hai."),
            ],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        translation, _ = translate_line(client, "text", "reading")
        self.assertEqual(translation, "Phần một. Phần hai.")

    def test_prompt_includes_text_and_reading(self):
        client = make_mock_client("output")
        translate_line(client, "使通好", "sứ thông háo", model="claude-sonnet-5")
        call_kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "claude-sonnet-5")
        self.assertEqual(call_kwargs["system"], SYSTEM_PROMPT)
        user_content = call_kwargs["messages"][0]["content"]
        self.assertIn("使通好", user_content)
        self.assertIn("sứ thông háo", user_content)
        self.assertEqual(user_content, USER_TEMPLATE.format(text="使通好", reading="sứ thông háo"))


if __name__ == "__main__":
    unittest.main()
