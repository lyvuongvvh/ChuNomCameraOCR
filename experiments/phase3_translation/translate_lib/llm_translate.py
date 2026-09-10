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
mixed together). These lines come from two kinds of source text with very different grammar:
- Historical prose (e.g. the chronicle Dai Viet Su Ky Toan Thu), written in genuine Literary \
Chinese - classical grammar, classical particles, Chinese word order.
- Nom poetry (e.g. Truyen Kieu, Luc Van Tien), which encodes spoken VIETNAMESE verse, not \
Chinese - Vietnamese word order, compressed and sometimes inverted for 6-8 syllable meter, and \
characters used purely for their Vietnamese sound rather than their Chinese meaning. A poetic \
line reading oddly as "Chinese" is completely normal and NOT by itself a sign of an OCR error -
judge it as Vietnamese verse, not against a Classical Chinese grammar bar.

You will be given one line of recognized Han-Nom text - the output of an OCR model, so it may \
contain recognition errors - along with a best-effort, partial per-character Sino-Vietnamese/Nom \
phonetic reading. Characters the reading dictionary could not resolve are shown in the original \
script, in brackets (e.g. [以]) - use the character itself plus surrounding context to work out \
its meaning.

Produce a single fluent modern Vietnamese (Quoc Ngu) translation of the line. If a character \
seems implausible given context (a likely OCR error), use your best judgment about the probable \
intended character and meaning rather than translating a nonsensical reading literally - but \
do not invent content the text does not support.

Only flag low confidence when you genuinely cannot produce a sensible translation - e.g. the \
characters/readings don't combine into any plausible meaning even as compressed Vietnamese verse, \
or there is an obvious truncation or garbled fragment. Do NOT flag low confidence merely because \
a poetic line does not read like grammatical Classical Chinese - that is expected for Nom verse, \
not a defect. If you do need to flag it, still give your best-effort translation of whatever you \
can confidently make out, then add on a new line: "[LOW CONFIDENCE: <brief reason>]". Never \
respond with nothing.

Respond with ONLY the Vietnamese translation (and, if needed, the low-confidence note). No \
preamble, no other explanation, no quotes around it."""

USER_TEMPLATE = """Original Han-Nom text: {text}
Partial phonetic reading: {reading}

Translation:"""


def _call(client, text: str, reading: str, model: str) -> tuple[str, dict]:
    response = client.messages.create(
        model=model,
        max_tokens=500,
        # Disabled, not just unrequested: on at least one genuinely garbled test line, this model
        # spent its whole turn on an internal `thinking` block and produced ZERO text content
        # (stop_reason="end_turn" with no text block at all - not a max_tokens truncation, a real
        # empty response). Disabling thinking outright fixed that exact case in testing and isn't
        # needed for a short, direct translation task anyway.
        thinking={"type": "disabled"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(text=text, reading=reading)}],
    )
    translation = "".join(block.text for block in response.content if block.type == "text").strip()
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return translation, usage


def translate_line(client, text: str, reading: str, model: str = "claude-sonnet-5") -> tuple[str, dict]:
    """Returns (translation, usage_dict) - usage_dict has input_tokens/output_tokens for cost
    tracking (see scripts/translate.py), summed across attempts if a retry happened.

    One retry on an empty translation (belt-and-suspenders on top of disabling thinking above -
    an empty response was observed even with a clear "never respond with nothing" system prompt
    instruction, so this only trusts the fix, not the prompt wording, to prevent silent empty
    results)."""
    translation, usage = _call(client, text, reading, model)
    if not translation:
        retry_translation, retry_usage = _call(client, text, reading, model)
        translation = retry_translation
        usage = {k: usage[k] + retry_usage[k] for k in usage}
    return translation, usage
