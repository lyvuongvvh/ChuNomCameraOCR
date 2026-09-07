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
2. **Only if step 1 shows a real improvement**, proceed to building the Kraken fine-tuning pipeline on NomNaOCR's dataset.
3. **Evaluate** the fine-tuned model against NomNaOCR's original models using the same metrics NomNaOCR reports (Sequence Accuracy, Character Accuracy, Character Error Rate).
4. **Only after a working, evaluated recognizer exists**, build the translation step (Nôm → modern Vietnamese) and the API/mobile app layers.

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
