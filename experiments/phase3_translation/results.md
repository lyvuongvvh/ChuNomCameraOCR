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
