"""Phase 3, Stage 2: LLM fluency pass - turns recognized Han-Nom text + Stage 1's partial
reading into fluent modern Vietnamese, via the real Anthropic API (not the manual in-conversation
translation used for the earlier qualitative demos in README.md/results.md).

Requires the `anthropic` package and an ANTHROPIC_API_KEY environment variable (a real Anthropic
Console API key - separate from and not obtainable through a Claude Code session's own OAuth
session). Get one at https://console.anthropic.com/settings/keys; this is billed separately from
any Claude Code subscription.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a historian and translator specializing in Vietnamese Han-Nom texts \
(historical Vietnamese written using Chinese characters and Vietnamese-invented Nom characters, \
mixed together). You will be given one line of recognized Han-Nom text - the output of an OCR \
model, so it may contain recognition errors - along with a best-effort, partial per-character \
Sino-Vietnamese/Nom phonetic reading. Characters the reading dictionary could not resolve are \
shown in the original script, in brackets (e.g. [以]) - use the character itself plus \
surrounding context to work out its meaning.

Produce a single fluent modern Vietnamese (Quoc Ngu) translation of the line. If a character \
seems implausible given context (a likely OCR error), use your best judgment about the probable \
intended character and meaning rather than translating a nonsensical reading literally - but \
do not invent content the text does not support.

Respond with ONLY the Vietnamese translation. No preamble, no explanation, no quotes around it."""

USER_TEMPLATE = """Original Han-Nom text: {text}
Partial phonetic reading: {reading}

Translation:"""


def translate_line(client, text: str, reading: str, model: str = "claude-sonnet-5") -> tuple[str, dict]:
    """Returns (translation, usage_dict) - usage_dict has input_tokens/output_tokens for cost
    tracking (see scripts/translate.py)."""
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(text=text, reading=reading)}],
    )
    translation = "".join(block.text for block in response.content if block.type == "text").strip()
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return translation, usage
