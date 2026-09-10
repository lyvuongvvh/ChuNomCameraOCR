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

Combines two sources (`scripts/build_reading_dict.py`):
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

**Combined coverage: 51.3%** (3,834 of 7,479 characters) - barely better than Unihan alone, since
rime-chunom's small vocabulary mostly doesn't overlap with what Unihan is already missing.
Roughly **half of this project's character vocabulary has no dictionary reading at all.**
Uncovered characters are passed through unchanged (bracketed, e.g. `[以]`), not guessed - Stage 2
is expected to carry the majority of the real translation work using the original text and
context, not just patch small gaps in an otherwise-complete gloss.

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

## What's not built yet

- **A scripted, automated Stage 2** - the demonstration above was done manually in conversation
  (this being an LLM itself), not via a callable API integration. Productionizing this needs an
  actual API key (Anthropic or another provider) and real cost/latency considerations - a
  decision for the project owner before building that integration, not assumed here.
- **End-to-end wiring to Phase 2/2b/2c's recognizer output** - this experiment used ground-truth
  text for the Stage 2 demonstration, not actual OCR predictions (which have their own ~15-30%
  error rate per Phase 2's results). Running the full pipeline on real recognizer output, with
  its errors, is the real end-to-end test and hasn't been done yet.
- **Any evaluation methodology** - unlike Phase 2's clean Sequence/Character Accuracy against
  known-correct OCR labels, there's no modern-Vietnamese ground truth to score against
  automatically. Any real evaluation here would need either human review or sourcing a genuine
  parallel corpus (e.g. from a published bilingual edition of one of NomNaOCR's three source
  works), which is a separate, nontrivial effort.

## Running Stage 1

```bash
python scripts/build_reading_dict.py --vocab-labels ../NomNaOCR/Patches/All.txt \
    --out data/reading_dict.json
python -m unittest tests.test_reading -v
```

## Known simplifications (flagged, not silently assumed)

- **Dictionary coverage is ~51%, not complete** - see above. This is a hard limitation of
  publicly available data, not a bug to fix by trying harder at the same sources.
- **`rime-chunom`'s license is unclear** - used for research/prototyping; would need resolving
  before any production/public-facing use.
- **First reading only** - `apply_reading_dict` uses each character's first listed reading;
  Unihan/rime-chunom sometimes list multiple candidates (e.g. homographs with different
  readings by meaning), and no attempt is made to disambiguate by context at the dictionary
  stage - that disambiguation is left entirely to Stage 2.
