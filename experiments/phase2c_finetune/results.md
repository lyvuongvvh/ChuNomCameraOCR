# Phase 2c: Fine-Tuning NomNaOCR's Own Model — Results

Continued training NomNaOCR's pretrained CRNNxCTC model on its own `Patches/Train.txt` for 8
epochs (Kaggle, Tesla P100, lr=1e-5, batch=32), then evaluated every checkpoint against the
*same* full held-out `Patches/Validate.txt` split (7,577 patches) and metrics Phase 2 and Phase
2b use, for a direct comparison.

## Training curve (in-training, carved from Train.txt only)

| Epoch | Train loss | Dev loss (Train.txt slice) |
|---|---|---|
| 1 | 0.0610 | **0.0415** (best) |
| 2 | 0.0558 | 0.0446 |
| 3 | 0.0533 | 0.0486 |
| 4 | 0.0504 | 0.0534 |
| 5 | 0.0492 | 0.0587 |
| 6 | 0.0472 | 0.0657 |
| 7 | 0.0462 | 0.0720 |
| 8 | 0.0447 | 0.0791 |

Read in isolation, this looks like textbook overfitting: train loss falls monotonically while
dev loss nearly doubles. Epoch 1 looks like the correct early-stopping point.

## Real held-out results (full Validate.txt, 7,577 patches) — the actual test

| Metric | Baseline (pretrained) | Epoch 1 | Epoch 8 |
|---|---|---|---|
| Sequence Accuracy | 29.4% | 29.6% (+0.2pp) | **29.9% (+0.5pp)** |
| Character Accuracy | 84.7% | 84.9% (+0.2pp) | 84.9% (+0.2pp) |
| CER (macro) | 0.1509 | 0.1487 | **0.1475** |
| CER (micro) | 0.1384 | 0.1362 | **0.1353** |

**Epoch 8 - the checkpoint the in-training dev loss said was most overfit - actually scores best
on every real held-out metric.** The in-training "dev loss" (carved from `Train.txt`, since
`Validate.txt`'s images are deliberately never uploaded to Kaggle - see `README.md`) was not a
reliable proxy for real held-out performance in this setup: CTC loss and downstream greedy-decode
accuracy are not tightly coupled, and/or that dev slice doesn't represent `Validate.txt`'s
distribution well enough to use for early stopping. **Flagged as a methodological lesson**: a
future attempt at this should validate against a real Phase-2-style evaluation periodically
rather than trusting the training loop's own loss-based dev signal.

## Comparison against Phase 2b (post-correction)

| Metric | Baseline | Phase 2b (beam+LM) | Phase 2c (epoch 8) |
|---|---|---|---|
| Sequence Accuracy | 29.4% | **32.4% (+3.0pp)** | 29.9% (+0.5pp) |
| Character Accuracy | 84.7% | 83.6% (-1.2pp) | **84.9% (+0.2pp)** |
| CER (macro/micro) | 0.1509/0.1384 | 0.1420/0.1309 | 0.1475/0.1353 |

**The cheap post-correction approach (Phase 2b, no training, ~90 min of CPU beam search) beat
this expensive fine-tuning approach (Kaggle GPU, ~65 min training + iteration time) on Sequence
Accuracy**, though fine-tuning improved Character Accuracy slightly (post-correction cost some
Character Accuracy to gain Sequence Accuracy; fine-tuning improved both, just by less). Neither
approach is a large win on its own - both land in the same modest range (roughly +0.2 to +3.0pp
on the metric each favors).

## What this does and doesn't tell us

- **This isn't proof fine-tuning can't help** - only that *this specific* setup (8 epochs,
  fixed lr=1e-5, no learning-rate schedule, no data augmentation, no regularization beyond the
  low learning rate itself) yields marginal gains. A more careful attempt (frozen backbone
  layers, learning-rate schedule, more epochs with real held-out-based early stopping instead of
  the misleading in-training dev loss, or more training data) might do better - this is a
  materially larger effort than this run was scoped for, consistent with the same caveat noted
  for Phase 0's CHAT fine-tuning trial.
- **Combining approaches is untested** - post-correction (Phase 2b) could in principle be applied
  on top of this fine-tuned model's beam search output too, potentially compounding both gains.
  Not attempted here.
- **Real technical outcome, independent of the accuracy numbers**: this experiment fully solved
  the Keras 3 (Kaggle) vs. Keras 2 (this project's local pin) compatibility problem end to end -
  `nomnaocr_lib` now works under both TensorFlow versions, and `kaggle_train.py` produces
  checkpoints directly loadable by this project's existing eval pipeline. That infrastructure is
  reusable for any future fine-tuning attempt without repeating this session's iterative
  debugging (`vocab_size` -> `vocabulary_size`, `get_shape()` -> `.shape`, and the Keras-3 native
  weights format needing a from-scratch Keras-2-legacy writer).

## Recommendation

Given the modest, roughly-comparable gains from both the cheap (Phase 2b) and expensive
(Phase 2c) levers tried so far, and that Phase 2c's real result only emerged by evaluating every
checkpoint on the actual held-out set (not by trusting the training loop's own signal) - the
reasonable next decision is not "which one to productionize" so much as "is either gain, or a
combination, worth carrying into Phase 3 (translation) at all, versus accepting the 84.7%/29.4%
baseline and letting translation-step tolerance absorb the remaining recognition noise." That's a
judgment call for the project owner, not something this experiment resolves unilaterally.
