# Phase 2: NomNaOCR Baseline Evaluation

NomNaOCR's own pretrained CRNNxCTC model, run as-is (no fine-tuning), scored against its own full held-out validation split (`Patches/Validate.txt`) using its own reported metrics - Sequence Accuracy, Character Accuracy, and Character Error Rate (CER). This supersedes Phase 0's page-level bag-of-characters proxy (Hán-only, 15 pages) with the real per-line metrics over the complete held-out set.

## Aggregate

n = 7577 held-out patches

| Metric | Value |
|---|---|
| Sequence Accuracy | 29.4% |
| Character Accuracy | 84.7% (76054/89756 chars) |
| CER (macro - mean of per-line edit-distance/length) | 0.1509 |
| CER (micro - total edit distance / total chars) | 0.1384 |

## By subset

| Subset | n | Sequence Accuracy | Character Accuracy | CER (macro) | CER (micro) |
|---|---|---|---|---|---|
| Poem lines | 2230 | 38.2% | 84.8% | 0.1493 | 0.1503 |
| Prose lines | 5347 | 25.8% | 84.7% | 0.1515 | 0.1359 |
| Lines > 10 chars | 3859 | 15.0% | 85.2% | 0.1318 | 0.1314 |
| Lines <= 10 chars | 3718 | 44.4% | 83.3% | 0.1707 | 0.1600 |

## Reading this result

- **Character Accuracy is positional, not alignment-based** (per NomNaOCR's own `CharacterAccuracy` metric) - a single leading insertion or deletion in a prediction misaligns every character after it, undercounting an otherwise-good prediction. This is a faithful replication of NomNaOCR's own metric, not a bug in this scoring code - see `eval_lib/metrics.py`'s docstring and `tests/test_metrics.py` for a worked example.
- **CER is reported two ways** because NomNaOCR's own upstream code defines it two ways with different aggregation (macro: mean of per-line ratios; micro: total edit distance over total characters) - despite an upstream code comment claiming they're the same, they only coincide when every line has equal length. Neither is silently preferred here.
- **Ground truth** comes from `nomnaocr_lib.vocab.load_labels()`'s lowercased, `is_clean_text`-filtered text (same filtering the model's vocabulary is built from), not raw `Validate.txt` lines - assumed to mirror NomNaOCR's own evaluation, not independently verified since their eval script isn't vendored in this repo.
- **Compare against Phase 0's 86.7%** (bag-of-characters, Hán-only, 15 pages, `experiments/phase0_validation/results.md`): expect a plausible but different number here, not an exact match - this evaluation covers all characters (not just Hán), scores per line (not per page), and uses positional/edit-distance metrics (not an order-insensitive multiset intersection). An exact 86.7% match would suggest an accidental logic collapse between the two scoring methods; a near-0% result would suggest a padding or decoding bug in this pipeline.