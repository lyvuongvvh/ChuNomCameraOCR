# Phase 2c: Fine-Tuning NomNaOCR's Own Model

Phase 2b's post-correction (beam search + LM rescoring) gave a modest gain (Sequence Accuracy
29.4%→32.4%) with a known ceiling: it can only recover errors where the right character exists
in some beam candidate, not ones the model never considered at all. This experiment tries the
larger option: actually continue training NomNaOCR's own pretrained CRNNxCTC model on its own
`Patches/Train.txt` data, to see if the underlying recognizer itself can be improved.

**Important distinction from Phase 0's CHAT fine-tuning trial (which made things worse):** that
was cross-domain transfer - a Chinese-OCR model forced onto Nôm data it never saw, which caused
catastrophic forgetting. This is continuing training the exact same architecture on the exact
same data distribution it was already trained on. The open question here isn't "will it forget
everything" (less likely to apply) but "will more training on data it's already fit to actually
move the needle, or just re-fit what it's effectively already learned."

## What's built and verified so far

- `train_lib/data.py` - tf.data pipeline. Reuses `nomnaocr_lib.model.CRNNRecognizer.process_image`
  unchanged (already matches training-time preprocessing); adds label tokenization/padding
  (`encode_label`), matching loader.py's `DataHandler.process_label` (no start/end tokens, since
  CRNNxCTC doesn't use them).
- `train_lib/ctc_loss.py` - CTC loss via `tf.keras.backend.ctc_batch_cost`. `input_length=25` is a
  **fixed constant, verified empirically** against the loaded pretrained model
  (`model.predict(...).shape == (1, 25, 7482)`), not assumed from reading the architecture code -
  every image resizes to a fixed 432x48 input, so the CNN always produces exactly 25 timesteps.
  `max_label_length` is 24, so the basic CTC feasibility condition (`input_length > label_length`)
  always holds; guards against the rarer case (labels with several consecutive repeated
  characters near the max length) producing a non-finite per-sample loss, so one bad sample can't
  turn a batch's gradient into NaN.
- `scripts/train.py` - a plain `GradientTape` training loop (not `model.fit`, since
  `ctc_batch_cost` needs per-sample label lengths alongside the usual `(x, y)` pair - a custom
  loop is simpler here than wrapping this in a Keras-`fit`-compatible signature). Includes a
  contamination guard (`--validate-split`, asserts zero overlap with `Patches/Train.txt`) and
  saves plain `.h5` checkpoints - **loadable directly by Phase 2's `run_eval.py --weights` with
  no changes**, confirmed by actually loading a checkpoint this way (see below).

**Verified via a local CPU smoke test** (40 lines, 1 epoch, batch size 4, ~8s total): the loop
runs, loss is finite throughout (0.35-0.87 range, no NaN/inf), zero non-finite samples skipped,
and the saved checkpoint loads back through `nomnaocr_lib.model.CRNNRecognizer` (Phase 2's own
inference code, unmodified) and produces a real prediction. This is a plumbing/shape check only -
too small and too few steps to say anything about whether real training helps.

## Known risks (flagged before running for real, not after)

- **Marginal-returns risk, not forgetting risk.** Since this model has already been fit to
  `Train.txt` once, further training on the same data may yield only small gains, or start
  overfitting further without moving held-out (`Validate.txt`) performance - unlike Phase 0's
  CHAT trial, we're not worried about wholesale collapse, but "no meaningful improvement" is a
  live, real possibility worth planning for.
- **CPU training is slow.** Measured from the smoke test: ~0.2s/sample (forward + backward) on
  this project's dev machine. A full epoch over 30,259 training lines would take roughly
  **1.5-2 hours on CPU**; multiple epochs would take most of a day. Unlike Phase 2b's beam-search
  bottleneck (a CPU-only TF op with no GPU kernel), *this* bottleneck is genuine gradient
  computation - dense matmuls that GPUs accelerate well - so Kaggle's free GPU tier is actually
  useful here, unlike for Phase 2b.
- **No learning-rate/epoch count has been validated yet.** `1e-5` and a handful of epochs are
  reasonable starting defaults for continued fine-tuning (small enough to avoid destabilizing
  already-good weights), not values tuned against this specific setup.

## Running it

Local CPU smoke test (verifies plumbing only, not real training - see "What's built" above):

```bash
docker run --rm -v "$PWD/experiments:/workspace" -w /workspace --entrypoint python \
    phase0-nomnaocr phase2c_finetune/scripts/train.py \
    --dataset-root NomNaOCR --all-labels NomNaOCR/Patches/All.txt \
    --train-split NomNaOCR/Patches/Train.txt --validate-split NomNaOCR/Patches/Validate.txt \
    --weights NomNaOCR_H5/NomNaOCR_CRNNxCTC.h5 \
    --epochs 1 --batch-size 4 --learning-rate 1e-5 --max-lines 40 --log-every 2 \
    --out-dir phase2c_finetune/data/smoke_checkpoints
```

A real training run (full `Train.txt`, several epochs) needs GPU to be practical - see "Known
risks" above. Not yet run - see the project owner's decision on local CPU vs. Kaggle GPU before
proceeding, since a Kaggle run means uploading training data/weights to their account and using
their compute quota.

Once a real checkpoint exists, evaluate it exactly like Phase 2's own baseline (no new eval code
needed):

```bash
docker run --rm -v "$PWD/experiments:/workspace" -w /workspace --entrypoint python \
    phase0-nomnaocr phase2_nomnaocr_baseline/scripts/run_eval.py \
    --dataset-root NomNaOCR --all-labels NomNaOCR/Patches/All.txt \
    --weights phase2c_finetune/data/checkpoints/finetuned_epoch3.h5 \
    --manifest phase2_nomnaocr_baseline/data/manifest.json \
    --out phase2c_finetune/data/finetuned_predictions.json --resume
# then phase2_nomnaocr_baseline/scripts/score.py + report.py, pointed at the new predictions,
# for a like-for-like comparison against the 29.4%/84.7% baseline.
```
