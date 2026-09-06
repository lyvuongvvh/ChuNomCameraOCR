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
| **NomNaNMT** | Translation of recognized Hán-Nôm text into modern Quốc Ngữ | referenced in NomNaOCR's roadmap |
| **Kraken** | OCR engine used by CHAT; supports fine-tuning on new character sets, which this project relies on | https://kraken.re |

## 5. Roadmap

- [ ] **Phase 0 — Validate the hypothesis.** Confirm that CHAT's pretrained model, run as-is, actually recognizes a meaningfully higher number of the Hán-character portion of Nôm sample pages than NomNaOCR's own models do. This should be checked *before* investing in the fine-tuning pipeline.
- [ ] **Phase 1 — Fine-tuning pipeline.** Adapt Kraken's fine-tuning tooling to continue training CHAT's model on NomNaOCR's labeled dataset.
- [ ] **Phase 2 — Evaluation.** Compare the fine-tuned hybrid model against NomNaOCR's original models on a held-out test set, using the same metrics NomNaOCR used (Sequence Accuracy, Character Accuracy, Character Error Rate).
- [ ] **Phase 3 — Translation integration.** Wire up a Nôm-to-Quốc-Ngữ translation step (via NomNaNMT or an equivalent) on top of recognized text.
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
