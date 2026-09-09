# Phase 2: NomNaOCR Baseline Evaluation

Per `CLAUDE.md`'s working order, Phase 0 tested whether CHAT (a pretrained historical Chinese
OCR model) recognizes Chữ Hán characters in Nôm text better than NomNaOCR's own pretrained
CRNNxCTC model. It didn't (86.7% vs. 22.2%, and fine-tuning CHAT made it worse - see
`experiments/phase0_validation/results.md`). Phase 1 (the CHAT+NomNaOCR fine-tuning pipeline)
was rejected as a result.

This experiment retargets Phase 2 - originally scoped to evaluate a fine-tuned hybrid model -
at NomNaOCR's own pretrained model instead, since that's now the adopted baseline recognizer.
It replaces Phase 0's narrow proxy (Hán-only characters, bag-of-characters scoring, 15 pages)
with NomNaOCR's own reported metrics (Sequence Accuracy, Character Accuracy, CER) computed over
the *complete* held-out validation split (`Patches/Validate.txt`, 7,663 patches), covering all
characters, not just Hán ones.

If this evaluation shows the pretrained model performs well, it becomes the recognizer carried
forward into Phase 3 (translation) and beyond. If not, that's a separate discussion before any
further phases proceed.

## Status

- [x] Evaluation manifest built from the full `Patches/Validate.txt` split (7,577 of 7,663 lines
      survived `is_clean_text`/`min_length` filtering; zero missing images; zero overlap with
      `Patches/Train.txt`).
- [x] NomNaOCR's pretrained CRNNxCTC run over all 7,577 held-out patches (~16.5 min on CPU).
- [x] `results.md` generated:

  | Metric | Value |
  |---|---|
  | Sequence Accuracy | **29.4%** |
  | Character Accuracy | **84.7%** (in line with Phase 0's 86.7% Hán-only proxy, as expected) |
  | CER (macro / micro) | 0.1509 / 0.1384 |

  See `results.md` for the full subset breakdown (poem/prose, long/short lines).
- [x] **Result: the baseline is adopted, with a caveat.** Character Accuracy is solid, but
      Sequence Accuracy means most individual lines have at least one character wrong. Rather
      than jump straight to Phase 3 (translation) on top of that, or to a full retraining effort,
      [`experiments/phase2b_postcorrection/`](../phase2b_postcorrection/) tries a cheaper
      in-between step first: correcting the existing model's output via beam search + a
      language-model rescoring pass, without touching its weights.

## What this reuses vs. what's new

- `experiments/phase0_validation/nomnaocr_lib/` (`model.py`, `vocab.py`) - **unchanged**. The
  working `CRNNRecognizer` inference code and vocab-reconstruction logic from Phase 0 are reused
  directly via a `sys.path` insert, not duplicated.
- Everything else here (manifest building, metrics, scoring, reporting) is new, since Phase 0's
  `compare_hanzi.py`/`build_manifest.py` were purpose-built for a page-level, Hán-only,
  bag-of-characters comparison and don't apply to line-level Sequence/Character Accuracy/CER.

## Metrics

Reimplemented in `eval_lib/metrics.py` from the literal source of `Text recognition/metrics.py`
in ds4v/NomNaOCR (fetched from GitHub, not vendored in this repo; MIT licensed):

- **Sequence Accuracy** - exact match between predicted and ground-truth text.
- **Character Accuracy** - positional (not alignment-based) comparison, ignoring ground-truth
  padding positions. This is a faithful replication of NomNaOCR's own metric, including its
  known weakness: a single leading insertion/deletion in a prediction misaligns every character
  after it. See the module docstring and `tests/test_metrics.py` for a worked example.
- **CER**, reported two ways since NomNaOCR's own upstream code defines it two ways with
  different aggregation - macro (mean of per-line edit-distance/length ratios) and micro (total
  edit distance / total ground-truth characters). An upstream code comment claims these are
  equivalent; they aren't in general (only when every line has the same length), so both are
  reported here rather than picking one.

## Running it end to end

Requires the full NomNaOCR dataset already downloaded (see
`experiments/phase0_validation/README.md`'s "Download NomNaOCR's dataset" section if not already
done - `experiments/NomNaOCR/` should contain `Patches/` with all `.jpg` files and the
`All.txt`/`Train.txt`/`Validate.txt`/`Validate_*.txt` label files) and the pretrained weights at
`experiments/NomNaOCR_H5/NomNaOCR_CRNNxCTC.h5`.

### 0. Unit tests (host, no Docker/dataset needed)

```bash
cd experiments/phase2_nomnaocr_baseline
python -m unittest tests.test_metrics -v
```

### 1. Build the evaluation manifest (host, no Docker needed)

```bash
python scripts/build_eval_manifest.py \
    --dataset-root ../NomNaOCR \
    --out data/manifest.json
```

Builds the full manifest from `Patches/Validate.txt` (~7,663 patches), verifying each patch
image exists, asserting zero overlap with `Patches/Train.txt`, and tagging each entry with
poem/prose and long/short-line subset membership. Use `--sample-fraction`/`--max-patches` only
for fast local dev iteration - never for final reported numbers (see the script's docstring).

### 2. Run NomNaOCR's pretrained model over the manifest (Docker, reuses phase0_validation's image)

```bash
docker build -t phase0-nomnaocr -f ../phase0_validation/docker/nomnaocr/Dockerfile ../phase0_validation
docker run --rm -v "$(cd .. && pwd):/workspace" -w /workspace --entrypoint python \
    phase0-nomnaocr phase2_nomnaocr_baseline/scripts/run_eval.py \
    --dataset-root NomNaOCR \
    --all-labels NomNaOCR/Patches/All.txt \
    --weights NomNaOCR_H5/NomNaOCR_CRNNxCTC.h5 \
    --manifest phase2_nomnaocr_baseline/data/manifest.json \
    --out phase2_nomnaocr_baseline/data/predictions.json \
    --resume
```

~7,663 sequential single-image CPU inferences take a while; progress is checkpointed to `--out`
periodically, and `--resume` picks up where a previous run left off if interrupted.

### 3. Score and report (host, no Docker needed)

```bash
python scripts/score.py --manifest data/manifest.json --predictions data/predictions.json \
    --all-labels ../NomNaOCR/Patches/All.txt --out data/scores.json
python scripts/report.py --scores data/scores.json --out results.md
```

Produces `results.md` with the headline Sequence Accuracy / Character Accuracy / CER numbers,
a breakdown by subset (poem/prose, long/short lines), and an explicit comparison against Phase
0's 86.7% bag-of-characters figure.

## Known simplifications (flagged, not silently assumed)

- **Character Accuracy is positional, not alignment-based** - a deliberate replication of
  NomNaOCR's own metric, not a scoring bug here.
- **Ground truth** comes from `nomnaocr_lib.vocab.load_labels()`'s filtered/lowercased text
  (same filtering the model's vocabulary assumes), not raw `Validate.txt` lines. Assumed to
  mirror NomNaOCR's own evaluation; not independently verified since their eval script isn't
  vendored in this repo.
- **CER macro vs. micro** are both reported because they measure meaningfully different things
  and NomNaOCR's own upstream comment conflating them doesn't hold in general - see
  `eval_lib/metrics.py`'s docstring.
- **Final reported numbers are always the full 7,663-patch run.** Any `--sample-fraction`/
  `--max-patches` dev run is clearly a dev artifact and must never be substituted into the
  headline `results.md` table.
