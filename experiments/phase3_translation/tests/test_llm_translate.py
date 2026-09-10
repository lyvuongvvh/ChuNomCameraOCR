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

    def test_thinking_disabled(self):
        """Regression test: a real API call on a genuinely garbled test line once returned a
        response consisting ONLY of a `thinking` block and zero text content (stop_reason=
        "end_turn", not a max_tokens truncation) - a real empty translation with no error raised.
        Disabling thinking fixed that case in live testing; this just guards the parameter stays
        set so a future edit can't silently drop it."""
        client = make_mock_client("output")
        translate_line(client, "text", "reading")
        call_kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["thinking"], {"type": "disabled"})

    def test_prompt_explains_nom_verse_grammar(self):
        """Regression test: the low-confidence trigger originally checked only for "coherent
        Classical Chinese/Han-Nom," which fits DVSKTT's prose but not Truyen Kieu/Luc Van Tien's
        Nom poetry (vernacular Vietnamese verse, never meant to parse as Classical Chinese) -
        real-API testing found this made poetry ~2x as likely to be flagged low-confidence as
        prose, even on fully dictionary-resolved lines (see results.md). The fix explains Nom
        verse's grammar explicitly; this guards that explanation stays in the prompt."""
        self.assertIn("VIETNAMESE verse", SYSTEM_PROMPT)
        self.assertIn("NOT by itself a sign of an OCR error", SYSTEM_PROMPT)

    def test_retries_once_on_empty_translation(self):
        client = MagicMock()
        client.messages.create.side_effect = [
            SimpleNamespace(
                content=[SimpleNamespace(type="thinking", text="")],  # no text block at all
                usage=SimpleNamespace(input_tokens=50, output_tokens=60),
            ),
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Bản dịch sau khi thử lại.")],
                usage=SimpleNamespace(input_tokens=50, output_tokens=10),
            ),
        ]
        translation, usage = translate_line(client, "text", "reading")
        self.assertEqual(translation, "Bản dịch sau khi thử lại.")
        self.assertEqual(client.messages.create.call_count, 2)
        # usage is summed across both attempts, so cost tracking isn't silently undercounted
        self.assertEqual(usage, {"input_tokens": 100, "output_tokens": 70})

    def test_does_not_retry_when_first_attempt_succeeds(self):
        client = make_mock_client("Bản dịch thành công ngay lần đầu.")
        translate_line(client, "text", "reading")
        self.assertEqual(client.messages.create.call_count, 1)

    def test_still_empty_after_retry_returns_empty_not_an_error(self):
        client = make_mock_client("")  # every call returns empty
        translation, usage = translate_line(client, "text", "reading")
        self.assertEqual(translation, "")
        self.assertEqual(client.messages.create.call_count, 2)
        self.assertEqual(usage, {"input_tokens": 84, "output_tokens": 14})


if __name__ == "__main__":
    unittest.main()
