# Phase 3: Translation (Hán-Nôm → Modern Vietnamese)

Per `CLAUDE.md`'s working order, Phase 3 wires up translation on top of the recognized text from
Phase 2/2b/2c. **First finding: `NomNaNMT` (referenced in this project's own README as the
translation component) does not exist as a usable tool.** Checked directly: the
`ds4v/NomNaNMT` GitHub repo is an empty one-line placeholder, and NomNaOCR's own README lists it
as a TODO item ("Dịch các phiên âm Hán-Nôm sang chữ Quốc Ngữ: Đã được triển khai bởi HCMUS" - i.e.
someone else did this, unpublished), not a finished, reusable project. There is nothing to "wire
up."

## Approach: two-stage, not a trained NMT model

1. **Stage 1 - deterministic character reading** (`translate_lib/reading.py`): look up each
   recognized Han-Nôm character's Sino-Vietnamese/Nôm reading in a combined dictionary
   (`scripts/build_reading_dict.py`), producing a per-character phonetic gloss - not a
   translation, just each character's "sound."
2. **Stage 2 - LLM fluency pass**: feed the original recognized text *and* Stage 1's partial
   reading to an LLM, which resolves word order, grammar, and word-sense ambiguity, and fills in
   characters Stage 1 couldn't cover using context - producing fluent modern Vietnamese prose.

This mirrors NomNaOCR's own two-step framing (Hán-Nôm → Quốc Ngữ reading, then Quốc Ngữ → fluent
Vietnamese), except Stage 1 here is a lookup table instead of the unpublished HCMUS tool, and
Stage 2 leans on a general-purpose LLM instead of a from-scratch NMT model - training a real NMT
model would need a real parallel Hán-Nôm/modern-Vietnamese corpus, which doesn't obviously exist
in usable form (NomNaOCR's own dataset has OCR transcription labels, not translations).

## Stage 1: dictionary coverage - a real, honest limitation

Combines four sources (`scripts/build_reading_dict.py`):
- **Unicode's Unihan database** (`kVietnamese` field) - authoritative, permissively licensed
  (Unicode Character Database terms), but covers only **50.2%** of this project's own character
  vocabulary (7,479 characters, from `Patches/All.txt`) - and notably incomplete even for common
  classical Hán characters (以, 為, 而, 者, 北, 何, 亦, 父 all have no `kVietnamese` entry), not
  just genuinely Nôm-invented ones.
- **`pearapple123/rime-chunom`'s `chu_nom.dict.yaml`** (single-character section only) - a small
  (~920-entry), community-curated Rime IME dictionary sourced from chunom.org, covering
  high-frequency Nôm-invented function words Unihan has no concept of at all (e.g. 𧵑 "của", 㐌
  "đã", 吧 "và"). **No explicit license found on that repository** - used here for research/
  prototyping only; flagged as an open question before any production use.
- **VNPF's "Bảng tra chữ Nôm"** (Hồ Lê, 1976), queried live at nomfoundation.org for exactly the
  characters the two sources above don't cover - resolved **356 of 3,645** of them (e.g. 以→"dĩ",
  北→"bắc"/"bác"/"bấc"/"bậc"/"bước"). Empirically *not* a superset of Unihan (it doesn't resolve
  為 or 何, both very basic characters) - it catalogs characters as used to write Nôm, not a
  general Hán-Việt dictionary, so it's a genuine complement rather than a bigger version of the
  same thing. Used under nomfoundation.org's terms of use (non-commercial/research use with
  attribution; only commercial redistribution requires permission).
- **Digitizing Vietnam's "Unified Hán-Nôm Lookup"** (Columbia University Vietnamese Studies
  Program) - searches Nguyễn Quang Hồng's **"Tự Điển Chữ Nôm Dẫn Giải"** (~10,000 entries, the
  authoritative academic dictionary - not otherwise available as usable data; VNPF's own site
  only has a catalog page for the printed 2-volume edition) together with **"Nguyễn Trãi Quốc Âm
  Từ Điển"**, queried live for the characters the three sources above still don't cover. Resolved
  **757 of 3,289** of them (e.g. 為→10 readings including "vì"/"vay"/"vài", none of which any
  other source found). Only accepts one character per request (unlike BTCN's batching, confirmed
  empirically - a 2-character query is a literal compound search, not two lookups), so this took
  ~3,289 individual, rate-limited requests (~70 min) rather than BTCN's ~46 batched ones. No
  explicit terms of use found; used here as a rate-limited, identified, non-commercial research
  query against a university project that says it wants to share its materials, not a bulk
  redistribution.

**Combined coverage: 66.1%** (4,947 of 7,479 characters) - up from 56.0% with the first three
sources, a real **+10.1pp** gain from Digitizing Vietnam alone. **A third of this project's
character vocabulary still has no dictionary reading at all.** Uncovered characters are passed
through unchanged (bracketed, e.g. `[父]` - still uncovered by all four sources), not guessed -
Stage 2 is expected to carry the
majority of the real translation work using the original text and context, not just patch small
gaps in an otherwise-complete gloss.

## Stage 2: LLM fluency pass - demonstrated, not yet automated

Manually demonstrated (in-conversation, not yet a scripted pipeline) on three real
`Patches/Validate.txt` ground-truth lines from Đại Việt Sử Ký Toàn Thư, mixing Stage 1's partial
reading with the original Hán-Nôm text as context:

| Original | Stage 1 reading (partial) | Stage 2 (fluent Vietnamese) |
|---|---|---|
| 使通好執事迷而不反我是以有徃年之師帝遣 | sứ thông háo chấp sự mê `[而]` bất phản ngã thị `[以]` hữu `[徃]` nên chi sư đấy khiến | "Ta sai sứ giả sang thông hiếu, nhưng viên quan phụ trách mê muội không trở về. Vì vậy ta mới có cuộc chinh phạt năm trước đó, do hoàng đế phái đi." |
| 不得棄本遂末并托以販賣技術游足游手其有 | bất đắc `[棄]` bản toại mạt `[并]` thác `[以]` `[販]` mại kĩ thuật `[游]` túc `[游]` thủ kì hữu | "Không được bỏ gốc theo ngọn, dựa vào việc buôn bán làm nghề mà lêu lổng, rong chơi; có những kẻ..." |
| 池有大蛇入見二十日上以砲旗皷制之賜月 | trì hữu đại xà nhập kiến nhì thập nhật thượng `[以]` bác cờ `[皷]` chế chi `[賜]` nguyệt | "Trong ao có con rắn lớn xuất hiện, hai mươi ngày sau, vua dùng súng, cờ, trống để trấn áp nó..." |

Even with ~50% dictionary gaps in these examples, the fluent output stayed plausible - the LLM
stage reads the surrounding original characters and historical context, not just Stage 1's gloss.
**This has not been independently verified against real modern-Vietnamese ground truth** (none
exists in this dataset - NomNaOCR provides OCR transcription labels only, not translations), so
this is a qualitative plausibility check, not a scored evaluation the way Phase 2's Sequence
Accuracy is.

**Tested against real (not ground-truth) OCR predictions, and confirmed at scale in
[`results.md`](results.md):** ran the same pipeline on baseline/Phase 2b/Phase 2c predictions
for real held-out lines with actual recognition errors. Finding: post-correction (Phase 2b)
recovers about **2.5x as many baseline error positions** as fine-tuning (Phase 2c) does uniquely
(9.7% vs 3.9% of all baseline errors, across 11,215 error positions), and when sampled at scale
(40 fixed positions reviewed), **nearly all fixes were genuine meaning-changing corrections**
(wrong calendar dates, numbers, pronouns, negation particles), not cosmetic OCR noise - including
cases like 問 "asked" being recognized as 聞 "heard", which post-correction fixed and fine-tuning
didn't. This makes Phase 2b's modest aggregate accuracy gain a stronger practical signal for
translation quality than the percentage alone suggests.

## Stage 2, automated: `translate_lib/llm_translate.py` + `scripts/translate.py`

A real, callable pipeline (not the manual in-conversation translation used for the demos above)
using the Anthropic API directly. Defaults to Phase 2b's post-corrected predictions as input
(see `results.md` for why: it recovers ~2.5x more translation-relevant errors than fine-tuning).

Requires `pip install anthropic` and your own `ANTHROPIC_API_KEY` (from
https://console.anthropic.com/settings/keys - a separate, billed credential, not obtainable from
a Claude Code session's own authentication). Defaults to `--max-lines 10` so a first run is cheap;
`--all` runs the complete input set (real cost - not a default to reach for casually). Tracks
token usage and prints a rough running cost estimate (verify against current pricing before
trusting it for real budgeting).

```bash
export ANTHROPIC_API_KEY=...
pip install anthropic
python scripts/translate.py \
    --predictions ../phase2b_postcorrection/data/corrected_predictions.json \
    --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
    --reading-dict data/reading_dict.json \
    --out data/translations.json --max-lines 10
```

Unit-tested with a mocked API client (`tests/test_llm_translate.py`) and **verified against the
real API**: a 10-line test batch (Phase 2b's corrected predictions) cost **$0.0211** (5,582 input
+ 992 output tokens) and produced 10/10 fluent, plausible translations.

**Confirmed at larger scale (2,000 lines, budget-capped):** real run over the first 2,000 lines
of Phase 2b's corrected predictions, 8 concurrent workers, cost **$4.02** (1,115,415 input +
179,128 output tokens, ~$0.0020/line - consistent with the 10-line rate), **0 errors**, 901s
wall-clock. Output in `data/translations_2000.json`.

**Completed at full scale: all 7,577 lines.** The 2,000-line sample above turned out to be 100%
Đại Việt Sử Ký Toàn Thư prose (a manifest-ordering artifact, not intentional) - the remaining
5,577 lines include both poetic works (Truyện Kiều's 3 editions, Lục Vân Tiên), which the 2,000
line sample had zero coverage of. Completing it cost **$9.96** more (**$14.00 total**), **0
errors**. Output in `data/translations_full.json`. Finding: poetry's self-flagged low-confidence
rate (56-67%) is roughly double prose's (24-29%), not explained by worse OCR or more dictionary
gaps - see `results.md` for the full breakdown and a Kiều-specific cross-edition consistency
check.

**Bug found and fixed during that first real run**: on one genuinely hard line (three compounding
OCR errors producing an ungrammatical tail), the model's response consisted of *only* an internal
`thinking` block and zero visible text - `stop_reason: "end_turn"`, not a token-limit truncation,
just a real empty translation with no error raised. A stronger "never respond with nothing" system
prompt instruction did not fix this on its own. **Disabling extended thinking outright
(`thinking={"type": "disabled"}`) did** - re-tested and the same line now translates correctly,
with a bonus: the "low confidence" fallback added to the prompt is now the two-layer defense
(thinking disabled first, low-confidence flag second, one automatic retry third - see
`translate_line`'s docstring) instead of relying on prompt wording alone.

## Ground truth for all three works: alignment against Wikisource and a real translation

All three of NomNaOCR's source works now have real, non-LLM ground truth to check translations
against - no more "there's nothing to score against" (see `results.md` for full detail on each):

- **Kiều and Lục Vân Tiên** (`eval_lib/wikisource_alignment.py`) - both composed directly in
  Vietnamese verse, so a clean modern-spelling edition (Vietnamese Wikisource's complete,
  verse-numbered text for each, public domain) IS the ground truth, no translation step needed.
  Aligns each digitized edition (Kiều: 1866/1871/1872, via `scripts/build_kieu_ground_truth.py`;
  Lục Vân Tiên: one edition, via `scripts/build_lvt_ground_truth.py`) - only a few hundred
  spot-digitized verses each, not the whole poem, so this is a real subsequence-alignment
  problem, not just "manifest order = verse order" - via a Levenshtein-scored DP over Stage 1's
  phonetic reading, made affordable with n-gram-seeded candidate generation instead of a naive
  full search (two cheaper approaches were tried and gave wrong answers first). Result: 1,823
  Kiều lines aligned (65.2% at similarity ≥0.7) and 407 Lục Vân Tiên lines (38.8%, root-caused to
  genuinely worse upstream OCR plus a confirmed residual of real cross-edition textual variance -
  not a dictionary or alignment-method artifact).
- **DVSKTT** (`eval_lib/dvsktt_ground_truth.py`) - genuine Literary Chinese prose, so the
  same-language trick above doesn't apply: a Sino-Vietnamese phonetic reading bears no
  resemblance to how a real translation reads. Instead aligns by physical page structure: the
  real 1993 published translation (fetched from Internet Archive) carries inline leaf markers
  (`[1a]`, `[1b]`...) matching NomNaOCR's own filename convention exactly, so this is a direct
  structural key lookup rather than a fuzzy text match - coarser (leaf-level, not line-level, so
  many OCR patches share one leaf's translated text) but far more reliable when it hits. Result:
  4,976 of 5,347 lines matched (93.1%) via `scripts/build_dvsktt_ground_truth.py` - Quyển Thủ
  (front matter, 188 lines) has zero coverage since this specific translation omits it entirely.
  Spot-checked against Phase 3's own independently-generated translations and found close,
  sometimes near-verbatim agreement.

Output: `data/kieu_ground_truth.json` / `data/lvt_ground_truth.json` / `data/dvsktt_ground_truth.json`
(all gitignored).

## Scoring translations against all three ground truths

`scripts/score_translations.py` (`eval_lib/scoring.py`) scores Phase 3's `translation` field for
real, using different metrics for the two ground-truth kinds (Kiều/Lục Vân Tiên: `edit_similarity`
+ `word_jaccard` against a single matching verse; DVSKTT: `word_recall` against a whole leaf's
text, since edit distance against a multi-sentence page would be meaningless) - see `results.md`
for why. **The headline finding**: in every single work, on every metric, lines the model
self-flagged `[LOW CONFIDENCE: ...]` score measurably lower against real ground truth than
self-confident ones (e.g. Kiều edit_similarity 0.695 confident vs. 0.531 low-confidence) - real
evidence the confidence flag tracks actual translation quality, not just noise. Two real caveats
found and documented, not glossed over: `edit_similarity`/`word_jaccard` can score a genuinely
good paraphrase badly (verse translation is a fluency pass, not a transcription), and DVSKTT's
`word_recall` is noisy on very short lines (a single missing word swings the score by 0.5+) -
confirmed with real numbers (0.559 avg for ≤2-content-word lines vs. 0.708 for ≥6).

## Root-causing low-scoring lines

`scripts/diagnose_low_scores.py` classifies each bottom-quartile-scoring line into OCR error,
Stage 1 dictionary gap, likely valid paraphrase (verse only), or unexplained. DVSKTT's low scores
are mostly (81.3%) explained by upstream OCR error alone. The "unexplained" bucket (~30-35% of
Kiều/Lục Vân Tiên's low scorers) turned out to mostly be **the same root cause already found for
poetry's low-confidence rate**, just showing up as a real accuracy problem instead of a false
flag: the old prompt translates a Nôm character by its literal Chinese meaning instead of
trusting Stage 1's already-correct phonetic reading. Confirmed for free using the earlier
prompt-fix validation sample (no new API spend): 9 lines that overlap both datasets show the
fixed prompt nearly doubling `word_jaccard` (0.215→0.518) and substantially improving
`edit_similarity` (0.383→0.666) - one hits a word-for-word exact match. **Confirmed at the full
115-line bucket scale, still no new API spend**, via a cheap proxy: `word_jaccard` between each
line's OLD translation and its own Stage 1 reading ("reading-fidelity") - 65.2% score below 0.3,
and the 9 confirmed lines average right in line with the whole bucket (0.243 vs 0.236), so they're
representative, not a lucky sample. The remaining ~35% turned out to be a **different, separate
problem**: proper names/titles getting genericized (e.g. "Hồ công" → "Ông") and at least one
apparent outright fabrication. **Both fixes were then combined and validated together against the
real API** (156-line sample, $0.4176): mean `edit_similarity` against real ground truth 0.610→0.773,
`word_jaccard` 0.446→0.581, and the one known proper-noun case that landed in this sample confirmed
directly ("Ông" → "Hầu công", matching the real "Hồ công"). This is a stronger check than the
earlier free proxy - 99 real ground-truth-scorable lines from a fresh run, not an extrapolation.
Both fixes are now confirmed, not just theorized, to improve real translation accuracy. Checked
(not assumed) whether DVSKTT's own "unexplained" bucket needed the same fixes, since they're
worded poetry-only - it doesn't: 77.4% of it is terse year/reign-title/particle fragments
(≤4 Han characters) that are already translated correctly, just scored against a leaf-level
reference with different phrasing - the known granularity limitation, not a translation bug. See
`results.md`.

## What's not built yet

- **The full corpus re-run itself.** Both prompt fixes are validated (above) - Nôm verse grammar
  ($0.33, first pass) and proper-noun/anti-fabrication (combined validation, $0.4176) - but
  `data/translations_full.json` still reflects the *original* old prompt. Re-running the full
  poetry subset (~$4-5) or the whole 7,577-line corpus (~$14, ~47 min) to regenerate it hasn't
  been done yet (deliberately deferred, not forgotten).
- **Classifying the ~40 proper-noun/fabrication-bucket lines** beyond the handful manually
  inspected and the one directly re-validated above.

## Running Stage 1

```bash
python scripts/build_reading_dict.py --vocab-labels ../NomNaOCR/Patches/All.txt \
    --out data/reading_dict.json
python -m unittest tests.test_reading tests.test_build_reading_dict -v
```

The build queries VNPF's and Digitizing Vietnam's live dictionary tools for uncovered characters
(see above) - pass `--skip-btcn`/`--skip-dvn` for a fast, network-light rerun during dev iteration
(Unihan+rime-chunom only, 51.3% coverage instead of 66.1%). `--skip-dvn` is worth using on its own
too: the Digitizing Vietnam pass alone takes ~70 minutes (one request per character, no
batching), and checkpoints to `data/dvn_query_cache.json` so an interrupted run resumes instead of
re-querying from scratch.

## Known simplifications (flagged, not silently assumed)

- **Dictionary coverage is 66.1%, not complete** - see above. This is a hard limitation of
  publicly available data, not a bug to fix by trying harder at the same sources.
- **`rime-chunom`'s license is unclear** - used for research/prototyping; would need resolving
  before any production/public-facing use.
- **BTCN (VNPF) and Digitizing Vietnam are scraped, not downloadable datasets** - queried live
  per run, batched/rate-limited (BTCN) or rate-limited with checkpointing (Digitizing Vietnam) to
  stay polite to their servers; used for non-commercial research (BTCN explicitly permits this in
  its terms of use; Digitizing Vietnam has none published, but is itself a non-commercial
  academic project).
- **First reading only** - `apply_reading_dict` uses each character's first listed reading;
  the underlying sources sometimes list multiple candidates (e.g. homographs with different
  readings by meaning, or BTCN/Digitizing Vietnam's several unrelated readings for one character
  like 北 or 為), and no attempt is made to disambiguate by context at the dictionary stage - that
  disambiguation is left entirely to Stage 2.
