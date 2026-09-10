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

Every one of the 8 epoch checkpoints was evaluated against the full held-out set (not just
epoch 1 and 8) to see the real trend, not extrapolate from two endpoints:

| Epoch | Sequence Accuracy | Character Accuracy | CER (macro) | CER (micro) |
|---|---|---|---|---|
| Baseline | 29.4% | 84.7% | 0.1509 | 0.1384 |
| 1 | 29.6% | 84.9% | 0.1487 | 0.1362 |
| 2 | 29.9% | 84.8% | 0.1487 | 0.1361 |
| 3 | 29.6% | 84.8% | 0.1487 | 0.1362 |
| 4 | 29.8% | 84.9% | 0.1480 | 0.1354 |
| 5 | 29.9% | 84.9% | 0.1480 | 0.1355 |
| 6 | **30.0%** | 84.9% | 0.1476 | 0.1354 |
| 7 | 29.9% | 84.9% | 0.1475 | 0.1355 |
| 8 | 29.9% | 84.9% | **0.1475** | **0.1353** |

**This is a plateau, not a trend.** Nearly all the real improvement happens between baseline and
epoch 1; Sequence Accuracy then oscillates in a 29.6-30.0% band for the remaining 7 epochs with
no further epochs clearly ahead of epoch 1-2 (differences between epochs are smaller than the
~0.5pp sampling noise floor at n=7,577). Character Accuracy is fully flat at 84.8-84.9% from
epoch 1 onward. CER is the one metric with a real, if tiny, continuing signal - a smoothly
decelerating curve converging toward ~0.147-0.148 by epoch 6, not accelerating. **Conclusion: more
epochs of this same setup (fixed lr=1e-5, no schedule, no augmentation) would not meaningfully
improve results further** - the model reaches essentially its ceiling for this configuration
within the first epoch.

Separately, the checkpoint the in-training dev loss called "most overfit" (epoch 8, dev loss
nearly double epoch 1's) is tied for the best real Sequence Accuracy and CER. That in-training
signal (necessarily carved from `Train.txt`, since `Validate.txt`'s images are deliberately never
uploaded to Kaggle - see `README.md`) was not a reliable proxy for real held-out performance here:
CTC loss and downstream greedy-decode accuracy are not tightly coupled enough to trust for
early stopping in this setup, though in this instance it fortunately didn't matter since the real
metric had already plateaued anyway. **Flagged as a methodological lesson**: a future attempt at
this should validate against a real Phase-2-style evaluation periodically, not the training
loop's own loss signal.

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

- **This isn't proof fine-tuning can't help - but it is evidence that "just run more epochs" of
  this exact setup won't.** The full 8-epoch trend above plateaus almost immediately, so simply
  extending training further (9, 20, 50 epochs at the same fixed lr=1e-5) is not expected to move
  the needle - the model has converged to this configuration's ceiling. A genuinely different
  setup (a learning-rate schedule/warmup, partial layer freezing, data augmentation, or more
  training data than `Train.txt`'s ~29K lines) would be needed to test whether a *higher* ceiling
  exists at all - this is a materially larger effort than this run was scoped for, consistent
  with the same caveat noted for Phase 0's CHAT fine-tuning trial.
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
