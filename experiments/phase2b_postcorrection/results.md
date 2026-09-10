# Phase 2b: Post-Correction (Beam Search + Character N-gram LM)

Rescores NomNaOCR's own CTC beam-search hypotheses with a character-level n-gram language model trained on `Patches/Train.txt`, instead of retraining the recognizer itself. See `README.md` for the approach and its known ceiling (it can only pick among candidates the acoustic model already considered plausible - see the beam-search probe findings there for cases where the correct character never appears in any beam at all).

Lambda (LM weight) = **1.0**, chosen by grid search on a held-out 10% tune slice (`data/split.json`) never used in the numbers below.

## Before vs. after (final split, held out from tuning)

n = 6819 patches

| Metric | Baseline (greedy) | Corrected (beam + LM) | Delta |
|---|---|---|---|
| Sequence Accuracy | 29.4% | 32.4% | +3.0% |
| Character Accuracy | 84.8% | 83.6% | -1.2% |
| CER (macro) | 0.1496 | 0.1420 | -0.0076 |
| CER (micro) | 0.1378 | 0.1309 | -0.0069 |

## Reading this result

- **Baseline here is Phase 2's own greedy-decode predictions**, restricted to the same "final" patches used for the corrected column - an apples-to-apples subset comparison, not the full-set numbers in `experiments/phase2_nomnaocr_baseline/results.md`.
- **Lambda was chosen on a separate 10% tune slice**, never on these "final" patches, so this delta isn't inflated by tuning against the same data it's measured on.
- If the delta is small or negative, that's consistent with the beam-search probe finding in `README.md`: a meaningful share of this recognizer's held-out errors are cases where the correct character isn't in the beam at all, which no amount of LM rescoring can fix - closing that gap further would need retraining the recognizer, not just correcting its output.