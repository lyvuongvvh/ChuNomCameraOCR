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

Neither fully explains a 2x gap this size.

### Follow-up investigation: it's the prompt's coherence check, not the pipeline

Two further tests isolate the real cause:

**1. The gap survives even on fully dictionary-resolved lines.** Restricting to lines where Stage
1 resolved *every* character (no `[bracket]` gaps at all - nothing missing to blame):

| | Zero-gap lines | Low-confidence rate |
|---|---|---|
| Poetry | 1,669 | **55.4%** |
| Prose | 2,197 | **20.3%** |

Almost identical to each genre's overall rate. If missing dictionary readings were the driver,
fully-resolved lines should look similar across genres - they don't.

**2. The gap survives controlling for line length.** Poetry's lines are much shorter on average
(6.9 chars vs. prose's 13.8 - expected, given 6-8 syllable Lục bát/song thất lục bát meter), so
length is a real confound. Comparing genres at the *same* length:

| Length (chars) | Poetry low-conf | Prose low-conf |
|---|---|---|
| 6 | 57.7% (n=1,129) | 28.2% (n=177) |
| 8 | 67.3% (n=1,052) | 32.0% (n=150) |

At matched length, poetry is still roughly double prose. Not a length artifact either.

**3. The smoking gun - what the model itself says when it flags a zero-gap line.** Of poetry's
925 zero-gap low-confidence lines, **82.7% explicitly blame "OCR error"/"nhận dạng"** as the
reason - despite every character already having a valid Stage 1 reading, i.e. there is no actual
unresolved character to blame. Prose does this too, but less (61.5% of its 447 zero-gap
low-confidence lines). Spot-checking these poetry lines directly shows why: several are
already fluent, coherent Vietnamese output that a human reader wouldn't flag at all -
e.g. `群身乙吏填培固欺` → "Bản thân mỗi người làm quan lại đều bồi đắp cho nền móng vững chắc"
(grammatical, sensible) and `㳥箋昔越扵𢬣` → "Buông tờ tiên xưa vượt khỏi tay" (fluent, evocative) -
both flagged `[LOW CONFIDENCE: ...lỗi OCR...]` anyway. Others in the same sample are genuinely
awkward, so this isn't uniformly spurious - but the rate is clearly inflated by something beyond
real garbling.

**Root cause: the system prompt's trigger is calibrated for Classical Chinese, not Nôm verse.**
The exact wording (`translate_lib/llm_translate.py`): flag low confidence when a line "does not
parse into coherent **Classical Chinese/Han-Nom** even after accounting for likely **OCR
mistakes**." DVSKTT is written in genuine Literary Chinese, so that bar fits it. Truyện Kiều and
Lục Vân Tiên are chữ Nôm transcriptions of **spoken Vietnamese** verse - different word order,
syntax compressed and inverted for meter, sound-borrowed characters - and were never meant to
parse as Classical Chinese prose to begin with. The prompt gives the model exactly one
explanation to reach for when something doesn't match that bar ("likely OCR mistakes"), so
well-formed poetic Vietnamese gets misdiagnosed as garbled recognition output at a much higher
rate than prose does. **This is a prompt-calibration issue, not evidence that OCR, Stage 1
coverage, or the translations themselves are actually worse for poetry** - confirmed independently
by points 1 and 2 above ruling out coverage and length.

### Fix, validated against the real API

Rewrote `SYSTEM_PROMPT` in `translate_lib/llm_translate.py`: added an explicit explanation that
Nôm poetry encodes spoken Vietnamese verse (not Classical Chinese), so unusual-looking word order
is expected and not itself a sign of OCR error, and tightened the low-confidence trigger to
require a genuinely unparseable line rather than merely "doesn't read like Classical Chinese."

Tested on a 156-line poetry sample (the 6 lines spot-checked above, plus 150 random poetry
lines) - real API, cost **$0.33**:

| | Old prompt | New prompt |
|---|---|---|
| Low-confidence rate on this sample | **66.0%** (103/156) | **21.2%** (33/156) |

72 lines flipped from flagged to confident; only 2 flipped the other way, both on genuinely
ambiguous lines (e.g. one where the model reasonably second-guessed a terse original translation)
- not a clear regression pattern. Of the 6 reference lines: 5 flipped to confident with equally
fluent translations (e.g. `花春公羕𣈜春群𨱽` → "Hoa xuân rỡ ràng, ngày xuân còn dài", no longer
second-guessed as OCR garbage); the 6th (`𦝄花限時𠬠牟默粧`) **stayed flagged**, but now for a more
specific, genuine-sounding reason ("khó ghép thành nghĩa mạch lạc" - hard to form coherent
meaning) - evidence the fix suppresses the spurious "not Classical Chinese" flags specifically,
not flags in general.

**21.2% is now below prose's own baseline rate (24-29%)** on this sample - arguably a slight
overcorrection, though the sample is small (156 lines, not the full 1,823+407 poetry lines).

**Not yet applied to the shipped corpus** - `data/translations_full.json` still reflects the old
prompt (budget spent on the validation test, not a full re-run). Re-running the full poetry
subset (~2,230 lines) with the fixed prompt would cost roughly $4-5 more; the full 7,577-line
corpus, roughly $14 again. Neither has been run - this section documents the fix and its
validation, not a corrected corpus.

**Cross-edition consistency check (Kiều-specific, needs no ground truth):** found 60 lines where
the post-corrected Han-Nôm text is byte-identical across 2-3 of the poem's 3 digitized editions
(1866/1871/1872) - a real check Kiều's multi-edition structure uniquely allows. Sampled 3: two
produced translations matching in meaning despite independent model calls (e.g. "計之仍浽育塘" →
both editions' translations agree on "kể ra vẫn còn nuôi dưỡng lòng mong muốn ấy," down to the
same low-confidence phrasing); one genuinely hard line ("牢𫽄別意思之") produced two translations
that diverge more (treating 牢 as "prison" vs. as an untranslated filler) - both flagged low
confidence by the model in both editions, consistent with the flag correctly predicting
instability. Only a 3-line spot-check, not a full audit of all 60.

## A real parallel corpus for Kiều and Lục Vân Tiên - ground-truth alignment against Wikisource

The one gap called out everywhere above - "no modern-Vietnamese ground truth to score
translations against" - has a fix for Truyện Kiều and Lục Vân Tiên specifically: unlike DVSKTT (a
Han source needing an actual translation), both poems' source text is ALREADY in Vietnamese - each
was composed in Nôm verse, so a clean modern-spelling edition is itself the ground truth, no
translation step needed. Vietnamese Wikisource hosts both complete poems: Kiều as a single
numbered 3,254-verse page (public domain; https://vi.wikisource.org/wiki/Truyện_Kiều), Lục Vân
Tiên as 4 sub-pages totaling ~2,082 verses (https://vi.wikisource.org/wiki/Lục_Vân_Tiên_(bản_Quốc_ngữ_2082_câu)),
matching the same works NomNaOCR's editions digitized (Kiều: 1866/1871/1872; Lục Vân Tiên: one
edition).

**Why this needs real alignment, not just manifest order:** each edition's manifest only covers a
few hundred spot-digitized verses out of 3,254 (482/639/702 for 1866/1871/1872) - OCR line N is
not verse N, it could be any verse, as long as order is preserved (pages are scanned in reading
order, never shuffled). This is a subsequence-alignment problem: find the best strictly-increasing
mapping of each edition's OCR lines onto the full verse list, skipping undigitized verses.

**Approach (`eval_lib/wikisource_alignment.py`, `scripts/build_kieu_ground_truth.py`):** a standard
weighted subsequence-alignment DP scored by real Levenshtein distance between each line's Stage-1
phonetic reading (already computed, not the LLM's fluent `translation` - Nôm poetry's reading is
close to the verse's actual wording, since the characters are chosen for their Vietnamese sound)
and each verse's normalized text (diacritics/case/punctuation stripped, and Stage 1's own "[X]"
unresolved-character brackets removed, since Python counts CJK characters as alphanumeric and
they'd otherwise get compared directly against Latin text for no good reason).

Two cheaper cost functions were tried and rejected before landing on real edit distance,
documented as a concrete lesson rather than silently fixed:
- **Multiset ("bag") distance** (O(1) per pair via Counter overlap, to avoid O(n×M) calls to an
  O(L²) function): WRONG, not just approximate - on a small ~22-letter alphabet, any two
  same-length Vietnamese strings share a large fraction of their multiset by chance, so it
  systematically favored similar-*length* wrong verses over correct-*content* right ones. Found
  on real data: a line's true verse scored real edit distance 12, but bag distance ranked a wrong
  verse's 8 ahead of it.
- **Real edit distance gated by a normalized-length window** (only compute the O(L²) call when
  |len_a - len_b| ≤ 6, else skip it): also wrong on lines with multiple unresolved characters -
  removing 2 bracket-wrapped characters can shorten a reading by 10+ characters relative to its
  true (fully-spelled) verse, pushing the correct verse outside any reasonable length window. A
  widened window doesn't fix it either: Kiều's normalized verse lengths are so densely clustered
  (measured: hundreds of the 3,254 verses land on any single length value 20-28) that a "clearly
  narrower than the whole poem" window barely filters anything, blowing per-edition alignment
  time to ~210s.

**What actually worked:** real edit distance, kept affordable via n-gram-seeded candidate
generation instead of a length filter - index every verse's 3-character substrings once, then for
each OCR line only evaluate real edit distance against the ~40 verses sharing the most 3-grams
with its reading (falling back to a tight length window only when a reading shares zero 3-grams
with anything, e.g. almost entirely unresolved brackets). This is robust to Stage 1's dictionary
gaps in a way length never was, because it only needs a handful of correctly-read characters
anywhere in the line, not overall length agreement. Runtime: ~10s for the largest edition (702
lines), versus ~210s for the naive full O(n×M) real-edit-distance version.

**Results, run over all three editions' Stage-1 readings (1,823 lines total):**

| Edition | Lines aligned | Similarity ≥0.7 | 0.4-0.7 | <0.4 | Avg similarity |
|---|---|---|---|---|---|
| 1866 | 482 | 358 (74.3%) | 116 | 8 | 0.780 |
| 1871 | 639 | 420 (65.7%) | 195 | 24 | 0.748 |
| 1872 | 702 | 410 (58.4%) | 240 | 52 | 0.703 |
| **All three** | **1,823** | **1,188 (65.2%)** | 551 (30.2%) | 84 (4.6%) | 0.739 |

Similarity (not a flat distance cutoff) is reported because verse length varies 13-36 normalized
characters, and even a genuinely *correct* alignment can carry real distance from Stage 1 picking
an archaic/dialectal spelling over Wikisource's modern one (e.g. "giời" vs "trời" for "sky" -
different letters, same word) - a fixed distance threshold would misjudge these as bad matches.

Spot-checked examples across the similarity range:

| Similarity | img_name | Verse | Reference (Wikisource) | Stage-1 reading |
|---|---|---|---|---|
| 0.96 | `Tale of Kieu 1866/page037b_15.jpg` | 1768 | Tiểu thư phải buổi mới về ninh gia. | tiểu thư phải buổi mãi về ninh gia |
| 0.83 | `Tale of Kieu 1872/page07b_5.jpg` | 226 | Màu hoa lê hãy dầm dề giọt mưa? | mào hoa lê hãy dào đề dột mưa |
| 0.68 | `Tale of Kieu 1872/page61b_1.jpg` | 2380 | Cửa viên lại dắt một dây dẫn vào, | cửa [轅] lại dứt một dai dụng vào |
| 0.0 | `Tale of Kieu 1872/page03a_9.jpg` | 44 (likely wrong) | Lễ là tảo mộ, hội là đạp Thanh. | nhôi hoàng phấn khuyến hôi tiền chỉ bi |

The last row is an honest failure case, not a hidden one: this specific line's Stage-1 reading
shares essentially no real content with its true verse (checked manually - only a single 3-gram
overlap exists anywhere in the poem for this reading, an information-theoretic dead end no
text-matching method can recover from), so the DP was forced to assign *some* verse, but the
resulting near-zero similarity honestly flags it as unreliable. **This is why every aligned line
carries its own `distance`/`similarity`, not just a verse number** - any downstream scoring should
filter on it (e.g. keep only ≥0.7) rather than trusting every assignment equally.

Output: `data/kieu_ground_truth.json` (gitignored, like other `data/` outputs), keyed by
`img_name`, each entry giving `work`, `verse_number`, `reference_text`, `ocr_reading`, `distance`,
`similarity`.

### Extending to Lục Vân Tiên: real transcription quirks, not just a rerun

Same engine (`scripts/build_lvt_ground_truth.py`), reused unchanged for the alignment DP itself -
but Lục Vân Tiên's Wikisource text surfaced two new real quirks Kiều's didn't have, both found by
checking the *parsed output* against the source rather than assuming a second work would behave
like the first:

- **The `{{số|N}}` verse markers have genuine transcription errors.** An earlier version of
  `parse_wikisource_poem` trusted each marker's number as authoritative (correct for Kiều - its
  markers are self-consistent and match sequential line count exactly). Lục Vân Tiên's part I
  reads `...20, 35, 30, 45, 40, 45, 50...` where the surrounding strict every-5-lines pattern
  makes it obvious two adjacent marker pairs got transposed in the source (should read `...20,
  25, 30, 35, 40, 45, 50...`). Trusting the marker value there produced verse numbers jumping
  *backwards* (39 → 30), violating the alignment DP's core assumption that verse order is
  monotonic. Fixed by switching to pure sequential counting - markers are stripped as noise but
  their digit is no longer used for anything - which only assumes actual poem lines were never
  reordered, not that every marker digit was transcribed correctly. This didn't change Kiều's
  output (its markers already agreed with sequential count).
- **Inline `<ref>` footnotes and two flavors of editorial template.** Lục Vân Tiên's wikitext has
  footnotes attached mid-verse (`<ref>...</ref>`, stripped whole), a `{{khác|A|B}}` "variant
  reading" template appearing as the first word of one verse (resolved to its primary form `A`,
  not dropped - that word is part of the verse), and a bare `{{ba sao}}` section-break marker on
  its own line (dropped entirely - it isn't a verse, and leaving it in would have shifted every
  subsequent verse's number by one). None of these appear in Kiều's cleaner Wikisource page.

Parsed verse count after these fixes: exactly **2,082** across the 4 sub-pages, matching the
page's own title ("bản Quốc ngữ 2082 câu").

**Results (407 lines, NomNaOCR's single digitized edition):**

| Similarity ≥0.7 | 0.4-0.7 | <0.4 | Avg similarity |
|---|---|---|---|
| 158 (38.8%) | 173 (42.5%) | 76 (18.7%) | 0.593 |

Markedly lower confidence than Kiều's 65.2%/0.739 average.

### Root cause: mostly upstream OCR quality, not the alignment method

Checked, not assumed. Comparing Phase 2b's corrected predictions against the true manifest labels
per work (Sequence Accuracy, CER, same metrics as Phase 2 - see
[`experiments/phase2_nomnaocr_baseline/results.md`](../phase2_nomnaocr_baseline/results.md)):

| Work | Sequence Accuracy | CER (micro) | Alignment similarity ≥0.7 |
|---|---|---|---|
| Kiều 1866 | 56.6% | 9.1% | 74.3% |
| Kiều 1871 | 49.1% | 11.8% | 65.7% |
| Kiều 1872 | 36.0% | 17.0% | 58.4% |
| **Lục Vân Tiên** | **30.5%** | **17.1%** | **38.8%** |

The ranking is identical across both independent OCR-accuracy metrics and alignment confidence -
Lục Vân Tiên has the worst underlying recognition of the four editions, by the largest margin.
Confirmed within Lục Vân Tiên's own data too, not just across editions: lines where Phase 2b's
prediction exactly matches the true Han/Nôm label average **0.681** similarity (54.8%
high-confidence), versus **0.554** (31.8%) for lines with any OCR error at all - a real, direct
effect, not a cross-work artifact.

**Why Stage 1's dictionary "coverage" stat didn't already show this**: coverage (95-97% across
all four works, actually *best* for Lục Vân Tiên at 96.9%) only measures whether a character got
*some* reading, not whether OCR recognized the *right* character. A misrecognized character still
gets confidently looked up and produces a fluent-looking but wrong reading - coverage can't see
that, so it looked like dictionary gaps weren't the problem when the real problem was one layer
further upstream.

**Likely reason recognition is worse for Lục Vân Tiên**: Kiều's three digitized editions combine
to 7,063 training patches (`Patches/Train.txt`) all reinforcing the *same* underlying vocabulary,
versus Lục Vân Tiên's single edition at only 1,646 - even though train/validation split ratios are
similar (~80%) across all four works, Lục Vân Tiên's model exposure to its own specific vocabulary
is roughly 4x thinner.

**A residual gap OCR alone doesn't explain**: even Lục Vân Tiên lines with *perfect* OCR only
reach 54.8% high-confidence, still below Kiều's 65.2% overall average (which includes its own
OCR-error lines). Spot-checked one such case (`nlvnpf-0059-018_18.jpg`, edit distance 0 against
the true label, reading "song thân nghe nói lòng bôi") - searched the full 2,082-verse Wikisource
text for anything close and found nothing (`Song thân dạy bảo vừa xong,` at verse 331 is the
closest by shared opening words, and it isn't close). Two plausible, unconfirmed explanations:
genuine textual variance between NomNaOCR's specific print edition and Wikisource's chosen edition
(Lục Vân Tiên has a less standardized transmission history than Kiều's near-canonical text), or
Stage 1's already-documented "first reading only" limitation (see README's "Known
simplifications") landing harder on Lục Vân Tiên-specific vocabulary if the underlying
dictionaries (BTCN, Digitizing Vietnam) were curated with more Kiều examples. Flagged as a
plausible factor, not independently verified further.

Output: `data/lvt_ground_truth.json` (gitignored), same schema as Kiều's.

**Not yet done:** using either ground truth to actually score Phase 3's `translation` field (as
opposed to the `reading` field used for alignment) - the `translation` field is the LLM's fluent
paraphrase, which for Nôm poetry often already reads close to the modern verse itself (not a
cross-language translation in the usual sense), so a real scoring pass would need to decide what
"correct" means for a paraphrase rather than an exact transcription. Also not done: confirming
which of the two residual-gap hypotheses above actually applies, or sourcing/aligning DVSKTT's
real 1993 published translation (a genuine Han→Việt translation, not a same-language spelling
normalization, so a different and harder kind of ground truth).

## Known simplifications (flagged, not silently assumed)

- **Kiều/Lục Vân Tiên alignment quality is reported, not guaranteed** - 4.6% of Kiều's and 18.7%
  of Lục Vân Tiên's aligned lines score similarity <0.4, meaning the DP was forced to pick *some*
  verse but likely picked the wrong one (see "A real parallel corpus" above). Any consumer of
  `kieu_ground_truth.json`/`lvt_ground_truth.json` must filter on `similarity` rather than
  trusting every `verse_number` equally.
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
