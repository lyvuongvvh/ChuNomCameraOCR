# Phase 2b: Post-Correction (Beam Search + Character N-gram LM)

Phase 2 found NomNaOCR's pretrained CRNNxCTC model has good Character Accuracy (84.7%) but poor
Sequence Accuracy (29.4% - most lines have at least one character wrong). Before considering a
full retraining effort (a real fine-tuning pass on NomNaOCR's own architecture/data, distinct
from the CHAT hybrid idea Phase 0 already rejected), this tries the cheaper option: correct the
existing model's output after the fact, without touching its weights at all.

## Approach

1. **CTC beam search** instead of greedy decoding, to get the model's top-K plausible
   transcriptions per line, not just its single best guess (`nomnaocr_lib/model.py`'s new
   `predict_beams()` - TF's own `ctc_decode(greedy=False)`, no custom beam search needed).
2. **A character-level n-gram language model** (`lm/ngram_lm.py`, stupid backoff per Brants et
   al. 2007), trained on NomNaOCR's own `Patches/Train.txt` ground truth - the same data the
   recognizer itself was trained on, so this introduces no new data source.
3. **Rescore**: for each line, pick the beam candidate maximizing
   `ctc_logp + lambda * lm_logp` (classic shallow fusion). `lambda` is tuned by grid search on a
   held-out 10% slice of the evaluation set, never on the 90% used for the headline numbers.

## Known ceiling (checked before building this, not after)

A quick probe (`predict_beams` with `beam_width=10, top_paths=5`, shown below) confirms beam
search surfaces real, plausible alternate readings - but also shows this approach's hard limit:

```
img: DVSKTT-3 Ban ky toan thu/DVSKTT_ban_toan_V_30a_9.jpg
gt:      使通好執事迷而不反我是以有徃年之師帝遣
beam[0]: 使通好執事遂而不反我是以有往年之帥帝遭  (greedy/top beam)
beam[1]: 使通好執事迷而不反我是以有往年之帥帝遭  (fixes "迷" correctly)
beam[3]: 使通好執事迷而不反我是以有往年之師帝遭  (fixes "迷" and "師")
```

None of the 5 beams ever produce "遣" (the model consistently prefers "遭" instead, across every
hypothesis it considers plausible) - a confidently-wrong character, not an uncertain one. **LM
rescoring can only pick among candidates the acoustic model already put in its beam; it cannot
invent a character the model assigns near-zero probability to.** So this approach can close part
of the Sequence Accuracy gap (cases like "迷" above, where the right answer exists in a
lower-ranked beam) but not all of it (cases like "遣" above, where it doesn't exist in any beam).
If the measured improvement below is modest, that's the expected reason, not a sign the approach
was implemented wrong - see `results.md`'s "Reading this result" section.

## Why tune on a slice of Validate.txt, not on Train.txt

The recognizer was fit to `Train.txt`, so its error patterns there are unrepresentatively good
(memorization) - tuning `lambda` against those errors wouldn't reflect what needs fixing on real
held-out data. Carving a small tune slice out of `Validate.txt` itself (standard practice when no
dedicated dev split exists upstream) means `lambda` is chosen against the same *kind* of errors
the final numbers measure, without letting the final 90% slice influence that choice - see
`scripts/split_tune_final.py`.

## Running it end to end

Requires Phase 2's manifest already built (`experiments/phase2_nomnaocr_baseline/data/manifest.json`)
and the same dataset/weights paths as Phase 2 - see
`experiments/phase2_nomnaocr_baseline/README.md` if not already set up.

```bash
# 1. Train the LM on NomNaOCR's own training split (host, no Docker needed)
python scripts/build_lm.py --dataset-root ../NomNaOCR --order 4 --out data/char_lm.json

# 2. Generate top-10 beam candidates for every patch in Phase 2's manifest (Docker, reuses
#    phase0_validation's image; similar runtime to Phase 2's run_eval.py)
docker build -t phase0-nomnaocr -f ../phase0_validation/docker/nomnaocr/Dockerfile ../phase0_validation
docker run --rm -v "$(cd .. && pwd):/workspace" -w /workspace --entrypoint python \
    phase0-nomnaocr phase2b_postcorrection/scripts/run_beams.py \
    --dataset-root NomNaOCR --all-labels NomNaOCR/Patches/All.txt \
    --weights NomNaOCR_H5/NomNaOCR_CRNNxCTC.h5 \
    --manifest phase2_nomnaocr_baseline/data/manifest.json \
    --out phase2b_postcorrection/data/beams.json --resume

# 3. Carve the tune/final split (host)
python scripts/split_tune_final.py --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
    --out data/split.json

# 4. Grid-search lambda on the tune slice only (host, fast - no re-inference)
python scripts/tune_lambda.py --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
    --beams data/beams.json --split data/split.json --lm data/char_lm.json \
    --all-labels ../NomNaOCR/Patches/All.txt --out data/best_lambda.json

# 5. Apply the tuned lambda to every patch, then report before/after on the final split
python scripts/rescore.py --beams data/beams.json --lm data/char_lm.json \
    --best-lambda data/best_lambda.json --out data/corrected_predictions.json
python scripts/report.py --manifest ../phase2_nomnaocr_baseline/data/manifest.json \
    --baseline-predictions ../phase2_nomnaocr_baseline/data/predictions.json \
    --corrected-predictions data/corrected_predictions.json --split data/split.json \
    --all-labels ../NomNaOCR/Patches/All.txt --best-lambda data/best_lambda.json --out results.md
```

Unit tests (`lm/ngram_lm.py`, stdlib only, no TF/Docker needed):

```bash
python -m unittest tests.test_ngram_lm -v
```

## Known simplifications (flagged, not silently assumed)

- **Stupid backoff, not a proper smoothed LM** - produces uncalibrated relative scores, not real
  probabilities. Fine for reranking a handful of candidates per line, not meant to be a
  general-purpose language model.
- **`lambda` is the only tuned hyperparameter** - `beam_width` (fixed at 10) and the LM order
  (fixed at 4) and backoff discount (fixed at 0.4, Brants et al.'s original value) are not swept,
  to keep the search space small given `run_beams.py` is the expensive step (one inference pass
  per `beam_width` choice, unlike `lambda` which is free to re-sweep from saved beams).
- **This is Phase 2b, not a new numbered CLAUDE.md phase** - it's an addendum to Phase 2
  (post-correction of the adopted baseline recognizer's output), not a substitute for Phase 1's
  fine-tuning question, which remains a separate, larger option if this doesn't close enough of
  the gap.
