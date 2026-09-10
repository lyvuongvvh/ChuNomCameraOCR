# Phase 3: Translation Against Real OCR Predictions — Results

Ran the Stage 1 (dictionary reading) + Stage 2 (LLM fluency pass) pipeline against actual
recognizer output - not ground truth - to see how OCR errors propagate into translation, and
whether Phase 2b (post-correction) and Phase 2c (fine-tuning)'s modest accuracy gains actually
matter for translation quality specifically.

## Method

1. **Qualitative (3 hand-picked lines):** picked real held-out lines where the baseline
   recognizer made a visible error, ran Stage 1 + Stage 2 on baseline/Phase 2b/Phase 2c's
   predictions for the same lines, and translated each version.
2. **Quantitative (full held-out set):** for every character position where the baseline
   recognizer was wrong, checked whether Phase 2b's post-corrected output or Phase 2c's
   fine-tuned (epoch 8) output fixed it - a hard, automated, no-judgment-needed count.
   Restricted to the 6,994 of 7,577 lines where baseline/epoch8/corrected/ground-truth text are
   all the same length (so character positions align unambiguously - a real limitation, see
   "Known simplifications" below).
3. **Qualitative at scale (40 sampled fixes, 20 per model):** randomly sampled fixed positions
   from each model and judged whether the character swap was meaning-changing or merely
   cosmetic (a glyph variant of the same word), to check whether "character accuracy" gains
   translate into real translation-relevant fixes, not noise.

## Qualitative: 3 real lines, full pipeline

| Line | Ground truth (fragment) | Baseline error | Fine-tuned (epoch 8) | Post-corrected |
|---|---|---|---|---|
| L1 | 使養之... "[he] **ordered** [someone] to tend to it" | 候養之 "attended to and nursed it" (wrong) | 從察之 "followed and inspected it" (wrong, *worse*) | 假養之 "falsely nursed it" (wrong, differently) |
| L4 | ...親問以... "personally **asked** about..." | 親聞以 "personally **heard/learned of**..." (wrong) | 親間以 barely parses (wrong) | 親問以 - **exact match, correct** |
| L6 | ...木印帖子... "**wood-block** seals" | 太印帖子 "**great/official** seals" (wrong) | 太印帖子 same error (wrong) | 木印帖子 - **exact match, correct** |

L1: none of the three interventions recovered this character - an honest miss all around, and
fine-tuning's guess was arguably further from the truth than baseline's. L4 and L6: post-
correction recovered a semantically load-bearing character (問 "asked" vs 聞 "heard"; 木 "wood"
vs 太 "great") that both baseline and fine-tuning missed - these aren't cosmetic OCR noise, they
change what the sentence says.

## Quantitative: does this hold at scale?

Across all 11,215 baseline-error character positions in the same-length-restricted set:

| | Positions | % of baseline errors |
|---|---|---|
| Fixed by post-correction only | 1,090 | **9.7%** |
| Fixed by fine-tuning only | 441 | 3.9% |
| Fixed by both | 411 | 3.7% |
| Fixed by neither | 9,273 | 82.7% |

**Post-correction uniquely fixes ~2.5x as many baseline error positions as fine-tuning does.**
The large "fixed by neither" majority (82.7%) is consistent with Phase 2b's beam-search probe
finding: many errors are the recognizer being confidently wrong, with the correct character
never appearing in any beam candidate and never recoverable by a light fine-tune either.

**Regression check** (positions baseline got *right* - does the intervention ever break them?):
of 70,697 baseline-correct positions, post-correction breaks 732 (**1.04%**) and fine-tuning
breaks 595 (**0.84%**). Both are small relative to their fix rates, but post-correction's
regression rate is proportionally the larger of the two, alongside its larger fix rate.

## Qualitative at scale: are the fixes meaning-changing, or cosmetic?

Reviewed 20 randomly sampled "post-correction fixed, fine-tuning didn't" positions and 20
"fine-tuning fixed, post-correction didn't" positions. **Nearly all (≈38 of 40) were genuine
meaning-changing substitutions**, not cosmetic glyph variants - including several with real
stakes for a historical chronicle: a wrong calendar stem character (戊 "5th stem" misread as 戌
"11th stem" - a different year designation), a number misread as an unrelated character (千
"thousand" vs 干 "dry/stem"), pronouns swapped for unrelated words (宜 "should" vs 吾 "I/my"), and
a negation particle misread as a noun (未 "not yet" vs 末 "end"). Only one sampled case (䏻/能,
both meaning "able") was a near-pure glyph variant with no real meaning difference.

**Conclusion: this is not noise.** When either intervention fixes a character, it is overwhelmingly
a real, translation-relevant fix, not a cosmetic one - confirming the 3-line qualitative finding
generalizes. Combined with the quantitative fix-rate gap, **post-correction (Phase 2b) is the
stronger lever for downstream translation quality specifically**, consistent with (and now backed
by much more evidence than) the phase-level Sequence Accuracy comparison (+3.0pp vs +0.5pp).

## Automated pipeline at scale: 2,000-line spot-check

Ran the scripted Stage 2 pipeline (`scripts/translate.py`, real Anthropic API) over 2,000 of
Phase 2b's 7,577 corrected-prediction lines - see `README.md` for the run's cost ($4.02, 0
errors). Spot-checked 8 randomly sampled translations plus scanned all 2,000 outputs for
self-reported confidence:

- **0 of 2,000 empty translations** - the `thinking={"type": "disabled"}` fix holds at scale, not
  just on the single line it was diagnosed against.
- **535 of 2,000 (26.8%) carry the system prompt's `[LOW CONFIDENCE: ...]` tag.** Spot-checking 5
  of these confirms honest, not spurious, flagging - genuinely garbled/truncated OCR lines where
  the model still attempts a translation but explicitly signals low trust rather than presenting
  a guess as fact (e.g. "câu bị cắt ngang ở cuối," "nhiều khả năng lỗi OCR"). This is a real,
  larger-than-expected data point on downstream usability: roughly 1 in 4 lines in this
  unfiltered 2,000-line sample has OCR damage severe enough that the model itself doesn't trust
  its own translation.
- **8 randomly sampled translations were all fluent and contextually correct**, including cases
  requiring real inference beyond Stage 1's reading: `登庸` correctly read as the proper name
  "Đăng Dung" (not translated literally), and dictionary-unresolved characters like `[徃]`
  (→ "qua lại") and `[侍]` in `[侍]御史` (→ correctly reconstructing the official title "Thị ngự
  sử") resolved correctly from surrounding context alone.

## Automated pipeline at full scale: all 7,577 lines, prose vs. poetry

Completed the run: all 7,577 lines (previously only the 2,000-line DVSKTT-only sample above) -
**$14.00 total real cost** ($4.02 + $9.96 for the remaining 5,577 lines), **0 errors**. Output in
`data/translations_full.json`.

**The 2,000-line sample above was entirely Đại Việt Sử Ký Toàn Thư (DVSKTT) prose - 0 lines from
either poetic work (Truyện Kiều's 3 editions, Lục Vân Tiên), because of manifest ordering.** That
matters: completing the full set surfaced a large, genre-driven gap the prose-only sample
completely missed.

| Work | Lines | Low-confidence rate |
|---|---|---|
| DVSKTT (5 volumes, prose) | 5,347 | 23.9%–29.3% |
| Truyện Kiều (3 editions, poetry) | 1,823 | 56.0%–66.7% |
| Lục Vân Tiên (poetry) | 407 | 61.9% |
| **Overall** | **7,577** | **36.7%** |

Poetry's low-confidence rate is roughly **double** prose's. Two obvious explanations don't hold up:
- **Not worse OCR** - Phase 2's own subset breakdown found poem and prose Character Accuracy
  nearly identical (84.8% vs 84.7%) and CER comparable, on the recognizer's output directly.
- **Not more dictionary gaps** - poetry lines actually have *fewer* Stage 1 dictionary gaps per
  line (0.22–0.35) than prose (0.93–1.34). Of the gaps poetry *does* have, though, 73–85% are
  Private-Use-Area codepoints (project/font-specific glyphs with no standard Unicode assignment,
  so no external dictionary - including all four Stage 1 sources - could ever resolve them,
  structurally, not just by chance), versus only 1.4–6.2% for DVSKTT.

Neither fully explains a 2x gap this size. The remaining, untested hypothesis: poetic register
itself (compressed grammar, allusion, elision for meter) is genuinely harder for the model to
render confidently, and/or the system prompt's low-confidence trigger ("does not parse into
coherent Classical Chinese/Han-Nom") is implicitly prose-shaped - verse isn't supposed to read
like classical Chinese prose, so the model may flag normal poetic structure more readily than
warranted. **Not resolved here** - flagged as a real, specific open question rather than guessed
at further.

Spot-checked 5 low-confidence Kiều lines: all cite a genuine unresolved character (usually a
Private-Use-Area glyph) as the reason, not spurious flagging - consistent with the earlier
DVSKTT-only spot-check finding honest flags, just at a higher rate for this genre.

**Cross-edition consistency check (Kiều-specific, needs no ground truth):** found 60 lines where
the post-corrected Han-Nôm text is byte-identical across 2-3 of the poem's 3 digitized editions
(1866/1871/1872) - a real check Kiều's multi-edition structure uniquely allows. Sampled 3: two
produced translations matching in meaning despite independent model calls (e.g. "計之仍浽育塘" →
both editions' translations agree on "kể ra vẫn còn nuôi dưỡng lòng mong muốn ấy," down to the
same low-confidence phrasing); one genuinely hard line ("牢𫽄別意思之") produced two translations
that diverge more (treating 牢 as "prison" vs. as an untranslated filler) - both flagged low
confidence by the model in both editions, consistent with the flag correctly predicting
instability. Only a 3-line spot-check, not a full audit of all 60.

## Known simplifications (flagged, not silently assumed)

- **Same-length restriction excludes 583 of 7,577 lines (7.7%)** where baseline/epoch8/
  corrected/ground-truth aren't all equal length - exactly the cases where a single insertion/
  deletion could misalign everything after it (Phase 2's own documented Character Accuracy
  caveat). This is why this analysis's net position counts (more fixed than broken, for both
  models) don't fully reconcile with Phase 2b's reported aggregate Character Accuracy *decline*
  (84.7%→83.6%) - that number reflects the full set including length-changing lines, which this
  position-wise breakdown doesn't cover. Flagged as a real gap in this analysis, not resolved.
- **"Meaning-changing" judgments are this session's own assessment**, not verified against a
  second human reviewer or a real Vietnamese-language expert.
- **Stage 2 (LLM translation) in the 3-line demo was done manually in-conversation**, not via a
  scripted pipeline - still true, see `README.md`'s "What's not built yet."
