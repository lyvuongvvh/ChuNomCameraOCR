# Chữ Nôm Camera OCR

Camera-based recognition and translation of Chữ Nôm (𡨸喃) text on mobile devices, built by fine-tuning an existing large-vocabulary historical Chinese OCR model on Vietnamese Nôm data.

> **Status: Experimental / Proof of Concept.** This project proposes an approach that, to our knowledge, has not yet been tried or validated by others. It combines two existing open-source projects in a new way. Expect to do real research and debugging, not just wire up a finished pipeline.

---

## 1. Motivation

Chữ Nôm is a historical Vietnamese script, built on and around Chinese characters, used for roughly 10 centuries until it was displaced by the Latin-based Quốc Ngữ alphabet. Today, fewer than 100 people worldwide can read it fluently — putting a large share of Vietnam's pre-20th-century literature, historical records, and inscriptions effectively out of reach for the general public.

No mainstream translation tool (Google Translate, Apple Translate, etc.) supports Chữ Nôm. This project aims to close part of that gap with a mobile app: point a phone camera at Nôm text, and get back a best-effort transcription and translation into modern Vietnamese.

## 2. The Core Problem

A typical Chữ Nôm document is a **mix of two character types**:

| Character type | Description | Approx. share of a typical text |
|---|---|---|
| **Chữ Hán** | Unmodified Chinese characters, read with Sino-Vietnamese pronunciation | ~40–60% |
| **Chữ Nôm proper** | Characters invented by the Vietnamese; do not exist anywhere in the Chinese script tradition | ~40–60% |

This means:
- A **Chinese OCR model alone** will recognize the Hán portion reasonably well (especially a model trained on a large, classical-era character set) but will completely fail on true Nôm-only characters — those glyphs simply never appeared in its training data.
- A **Nôm-specific OCR model** (like NomNaOCR, see below) is trained only on ~2,953 pages from three literary works, giving it a much smaller vocabulary of Chinese-derived characters than a dedicated Chinese historical OCR model would have.

**Hypothesis of this project:** combining the two — using a large-vocabulary historical Chinese OCR model as a pretrained base, then fine-tuning it on Nôm-specific data — should outperform either approach used alone.

## 3. Proposed Architecture

```
                     ┌─────────────────────────┐
                     │   Mobile Camera Capture  │
                     │   (iOS / Android app)    │
                     └────────────┬─────────────┘
                                  │ image
                                  ▼
                     ┌─────────────────────────┐
                     │   Text Detection Stage   │
                     │  (DBNet / PaddleOCR)     │
                     │  locates character boxes │
                     └────────────┬─────────────┘
                                  │ cropped regions
                                  ▼
                     ┌─────────────────────────┐
                     │  Text Recognition Stage  │
                     │  Base: CHAT (Kraken)     │
                     │  Fine-tuned on: NomNaOCR │
                     └────────────┬─────────────┘
                                  │ Hán-Nôm string
                                  ▼
                     ┌─────────────────────────┐
                     │  Nôm → Quốc Ngữ          │
                     │  translation step        │
                     │  (see NomNaNMT)          │
                     └────────────┬─────────────┘
                                  │ modern Vietnamese
                                  ▼
                     ┌─────────────────────────┐
                     │   Result shown in app    │
                     └─────────────────────────┘
```

## 4. Building Blocks (existing open-source projects this builds on)

| Project | Role in this pipeline | Link |
|---|---|---|
| **NomNaOCR** | Source of labeled Nôm training data (~38K patches) and the original detection/recognition pipeline this project builds on | https://github.com/ds4v/NomNaOCR |
| **CHAT_models** | Pretrained historical Chinese OCR model (Kraken engine), 16,000+ characters, 99%+ accuracy, trained on 1.7M lines spanning the 10th–20th century | https://github.com/colibrisson/CHAT_models |
| **NomNaSite** | Reference web app showing an existing (Nôm-only) recognition pipeline in production | https://github.com/ds4v/NomNaSite |
| **NomNaNMT** | ~~Translation of recognized Hán-Nôm text into modern Quốc Ngữ~~ **Does not exist as a usable tool** (empty placeholder repo; listed as NomNaOCR's own TODO, not a finished project) - Phase 3 uses a different approach instead, see below | https://github.com/ds4v/NomNaNMT |
| **Kraken** | OCR engine used by CHAT; supports fine-tuning on new character sets, which this project relies on | https://kraken.re |

## 5. Roadmap

- [x] **Phase 0 — Validate the hypothesis.** Confirm that CHAT's pretrained model, run as-is, actually recognizes a meaningfully higher number of the Hán-character portion of Nôm sample pages than NomNaOCR's own models do. This should be checked *before* investing in the fine-tuning pipeline.
  **Result: it doesn't.** On 15 held-out sample pages (679 Chữ Hán characters), CHAT's pretrained model scored 22.2% vs. NomNaOCR's own pretrained CRNNxCTC at 86.7% — the opposite of what the hybrid hypothesis needs. A follow-up quick fine-tuning trial (10 epochs on a Kaggle GPU) was run to check whether this was a fixable pipeline issue rather than a hard capability gap — it made things *worse* (4.1%, likely catastrophic forgetting from fine-tuning on a very small, narrow slice of text), reinforcing rather than undermining this result. Full writeup, methodology, and caveats in [`experiments/phase0_validation/README.md`](experiments/phase0_validation/README.md).
- [ ] **Phase 1 — Fine-tuning pipeline.** Adapt Kraken's fine-tuning tooling to continue training CHAT's model on NomNaOCR's labeled dataset. **Not recommended based on Phase 0's result** — see the note above; NomNaOCR's own pretrained model remains the best available baseline for Chữ Hán recognition on this data by a wide margin.
- [x] **Phase 2 — Evaluation (retargeted).** Originally scoped to evaluate a fine-tuned hybrid model against NomNaOCR's originals; since Phase 1 is rejected, this instead rigorously evaluates NomNaOCR's own pretrained model as the adopted baseline recognizer, using its own reported metrics over its full held-out validation split (7,577 patches, vs. Phase 0's 15-page proxy sample).
  **Result:** Character Accuracy **84.7%** (in line with Phase 0's 86.7% Hán-only proxy figure, as expected), CER 0.14–0.15, but Sequence Accuracy (exact full-line match) only **29.4%** — most lines have at least one character error, even though per-character accuracy is high. Full breakdown (by poem/prose, line length) and caveats in [`experiments/phase2_nomnaocr_baseline/results.md`](experiments/phase2_nomnaocr_baseline/results.md).
- [x] **Phase 2b — Post-correction (addendum to Phase 2).** Before considering a full retraining effort, tried the cheaper option: rescore the recognizer's own CTC beam-search hypotheses with a character n-gram language model, without touching its weights.
  **Result: a modest, real gain, not a fix.** Sequence Accuracy 29.4% → 32.4% (+3.0pp) on a held-out slice never used for tuning, at a small Character Accuracy cost (84.8% → 83.6%). Consistent with a ceiling found before building the pipeline: correction can only recover errors where the right character exists in *some* beam candidate, not ones the model never considered at all. Full writeup, the beam-search probe finding, and caveats in [`experiments/phase2b_postcorrection/results.md`](experiments/phase2b_postcorrection/results.md).
- [x] **Phase 2c — Fine-tuning NomNaOCR's own model (addendum to Phase 2).** Continued training the pretrained CRNNxCTC model on its own `Train.txt` for 8 epochs (Kaggle GPU), evaluating every checkpoint against the same held-out set as Phase 2/2b.
  **Result: a modest gain that plateaus almost immediately, not a breakthrough.** All 8 epoch checkpoints were evaluated against the held-out set (not just the first/last): Sequence Accuracy moves from 29.4% to a 29.6–30.0% band and Character Accuracy from 84.7% to 84.8–84.9%, both reached essentially by epoch 1 and flat (within noise) thereafter — more epochs of this same setup would not be expected to help further. Notable methodological finding: the checkpoint with the *worst* in-training dev loss (epoch 8, apparent overfitting) tied for *best* on the real held-out set — that dev signal (necessarily carved from `Train.txt`, since `Validate.txt` is never uploaded to Kaggle) was not a reliable stand-in for real evaluation here. Full per-epoch numbers and the loss-vs-accuracy discrepancy in [`experiments/phase2c_finetune/results.md`](experiments/phase2c_finetune/results.md).
- [~] **Phase 3 — Translation integration.** `NomNaNMT` turned out not to exist as a usable tool (empty repo, listed as NomNaOCR's own TODO) — pivoted to a two-stage approach: a deterministic Hán-Nôm character reading dictionary (Stage 1), then an LLM fluency pass (Stage 2) to produce modern Vietnamese.
  **In progress.** Stage 1 built: combines Unicode's Unihan (`kVietnamese`), a small community Nôm IME dictionary, VNPF's "Bảng tra chữ Nôm," and Digitizing Vietnam's (Columbia University) "Tự Điển Chữ Nôm Dẫn Giải"/"Nguyễn Trãi Quốc Âm Từ Điển" lookup (the latter two scraped live for gaps), covering **66.1%** of this project's character vocabulary — a real, hard limitation of available data, not a bug. Stage 2 is a scripted, parallelized pipeline against the real Anthropic API, run against **all 7,577 lines** for real: **$14.00 total, 0 errors**. Post-correction (Phase 2b) recovers ~2.5x as many recognition errors as fine-tuning (Phase 2c) does uniquely, and the recovered characters are overwhelmingly meaning-changing, not cosmetic — the case for Phase 2b's output as the translation input. New finding from the full run: poetry (Truyện Kiều, Lục Vân Tiên) self-flags low translation confidence roughly 2x as often as prose, not explained by OCR accuracy or dictionary coverage — an open question. Still no ground-truth modern-Vietnamese corpus to score translations against automatically. See [`experiments/phase3_translation/results.md`](experiments/phase3_translation/results.md).
- [ ] **Phase 4 — Backend API.** Wrap the full pipeline (detection → recognition → translation) in a lightweight API (e.g. FastAPI) that a mobile app can call.
- [ ] **Phase 5 — Mobile app.** Build the camera capture + result display app (iOS/Android or cross-platform), calling the backend API above.

## 6. Known Limitations (be upfront about these)

- Both NomNaOCR's and CHAT's training data lean heavily on **printed / woodblock text**. Accuracy on handwriting, worn inscriptions, or damaged documents is expected to be considerably lower and is untested.
- NomNaOCR's dataset comes from only **three literary works** (Truyện Kiều, Lục Vân Tiên, Đại Việt Sử Ký Toàn Thư). A model fine-tuned on this data may not generalize well to Nôm text with different regional or period-specific character variants.
- The Nôm-to-modern-Vietnamese translation step is inherently ambiguous: a single Nôm/Hán character can represent different Vietnamese words depending on context, similar to multiple-reading ambiguity in Japanese Kanji.
- This is a research/hobby-stage project. There is no guarantee the core hypothesis (Chinese-OCR-as-pretraining helps) holds up in practice — that's exactly what Phase 0 above is meant to test.

## 7. Contributing

This project is at an early, exploratory stage. If you have experience with:
- Kraken fine-tuning workflows,
- Han-Nôm paleography, or
- Vietnamese historical linguistics,

your input would be especially valuable — please open an issue to discuss before submitting large PRs, since the overall approach is still being validated.

## 8. License & Attribution

This project builds directly on the datasets and models linked above; please review and respect each upstream project's own license before reusing their weights or data commercially. See `LICENSE` in this repo for the license of code original to this project.

## 9. Acknowledgements

This project would not be possible without the foundational work of the [ds4v/NomNaOCR](https://github.com/ds4v/NomNaOCR) team and the [Vietnamese Nôm Preservation Foundation (VNPF)](http://www.nomfoundation.org), whose digitization efforts made a dataset like this possible in the first place, as well as the [colibrisson/CHAT_models](https://github.com/colibrisson/CHAT_models) project for open-sourcing a strong historical Chinese OCR baseline.
