# Project Context for Claude Code

This file is read automatically by Claude Code at the start of every session in this repo. It summarizes the project so you don't need to re-explain it each time — see `README.md` for full background.

## What this project is

Camera-based OCR + translation for Chữ Nôm (historical Vietnamese script). Core hypothesis: fine-tune a large-vocabulary historical **Chinese** OCR model (CHAT, via Kraken) on the **Nôm-specific** NomNaOCR dataset, producing a hybrid recognizer that's better than either alone.

**This hypothesis is unvalidated.** Do not skip straight to building the full pipeline — see "Working order" below.

## Key upstream references

- Nôm training data + original pipeline: https://github.com/ds4v/NomNaOCR
- Pretrained historical Chinese OCR model: https://github.com/colibrisson/CHAT_models
- Reference web app (Nôm-only, for comparison): https://github.com/ds4v/NomNaSite
- OCR engine used by CHAT (supports fine-tuning): https://kraken.re

## Working order — do not skip ahead

1. **Validate first.** Before writing any fine-tuning code, help me pull a handful of sample Nôm pages (from NomNaOCR's dataset or NomNaSite's demo) and run CHAT's pretrained model on them as-is. Count how many Chinese-derived (Chữ Hán) characters it correctly recognizes vs. NomNaOCR's own model on the same pages. This determines whether the whole hybrid approach is worth pursuing.
   **Done — result is negative.** CHAT scored 22.2% vs. NomNaOCR's own pretrained model at 86.7% (15 held-out pages, 679 Chữ Hán chars); a follow-up CHAT fine-tuning trial made it worse (4.1%). See `experiments/phase0_validation/results.md`.
2. ~~Only if step 1 shows a real improvement, proceed to building the Kraken fine-tuning pipeline on NomNaOCR's dataset.~~ **Not pursued** — Phase 0's result doesn't support it. NomNaOCR's own pretrained model is adopted as the baseline recognizer instead.
3. **Evaluate** (retargeted from evaluating a fine-tuned hybrid to evaluating NomNaOCR's own pretrained model, since step 2 was skipped) against NomNaOCR's original reported metrics (Sequence Accuracy, Character Accuracy, Character Error Rate), on its full held-out validation split.
   **Done.** Character Accuracy 84.7%, CER 0.14–0.15, but Sequence Accuracy (exact line match) only 29.4% — most lines have at least one character wrong despite high per-character accuracy. This is worth weighing before Phase 4/5: downstream translation will usually see near-correct, not exact, transcriptions. See `experiments/phase2_nomnaocr_baseline/results.md`.
   **Addendum (Phase 2b, post-correction) — done.** Before considering a full retraining pass, tried rescoring the recognizer's own beam-search output with a character n-gram LM, no weight changes. Result: modest real gain (Sequence Accuracy 29.4%→32.4% on held-out data never used for tuning), not a fix — a beam-search probe found some errors are confidently-wrong characters that never appear in any beam candidate, which no rescoring can recover. See `experiments/phase2b_postcorrection/results.md`.
   **Addendum (Phase 2c, fine-tuning) — done.** Actually continued training NomNaOCR's own model on `Train.txt` (8 epochs, Kaggle GPU) — unlike the CHAT trial, not cross-domain transfer, so no catastrophic forgetting. Result: modest gain, same order of magnitude as Phase 2b (Sequence Accuracy 29.4%→29.6-30.0%, Character Accuracy 84.7%→84.8-84.9%) that **plateaus almost immediately** — all 8 epoch checkpoints were evaluated, and essentially all the improvement happens by epoch 1, with epochs 2-8 flat within noise. More epochs of this same setup (fixed lr, no schedule/augmentation) would not be expected to help further; a genuinely different setup would be needed to test for a higher ceiling. Notable lesson: the in-training dev loss (necessarily carved from `Train.txt`, not `Validate.txt`) was actively misleading — the checkpoint it flagged as most overfit tied for *best* on the real held-out set. Always verify against the actual Phase 2 held-out pipeline, not a training loop's own loss signal. See `experiments/phase2c_finetune/results.md`.
4. **Only after a working, evaluated recognizer exists**, build the translation step (Nôm → modern Vietnamese) and the API/mobile app layers.
   **In progress.** `NomNaNMT` (this file's own reference, above) turned out not to exist as a usable tool — empty placeholder repo, listed as NomNaOCR's own TODO. Pivoted to a two-stage approach instead: Stage 1 is a deterministic Hán-Nôm character reading dictionary (Unihan `kVietnamese` + a small community Nôm IME dictionary), Stage 2 is an LLM fluency pass. Stage 1 covers only 51.3% of this project's vocabulary — a real limitation of available dictionary data. Stage 2 demonstrated manually on real ground-truth lines with plausible results, but not yet a scripted pipeline (needs an API key decision) and has no automatic ground truth to score against (no modern-Vietnamese parallel corpus exists in this dataset). See `experiments/phase3_translation/README.md`.

## Conventions for this repo

- Python for all model/pipeline code (matches upstream projects' language, makes reusing their code straightforward).
- Keep experiment scripts (Phase 0 validation, evaluation runs) in an `experiments/` folder, separate from any production pipeline code — this is research-stage work and the two shouldn't be conflated.
- Every model change should be evaluated against the same held-out test set for fair comparison — don't report accuracy numbers that used different test data between runs.
- Be explicit in commit messages and code comments about which phase (1-4 above) a piece of work belongs to.
- Flag clearly in code/PRs when something is untested assumption vs. verified result — this project's core premise is a hypothesis, not a given.

## What NOT to do

- Don't build the mobile app UI or backend API before the recognition pipeline is validated — that's Phase 4/5 work and premature before Phase 1-3.
- Don't assume Chữ Nôm-specific characters will be recognized by CHAT alone — only the Chữ Hán (pure Chinese-derived) subset is expected to benefit from it.
- Don't report a model as "working" based on the training/sample data alone — always check held-out test performance.
