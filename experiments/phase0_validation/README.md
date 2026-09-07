# Phase 0 Validation

Per `CLAUDE.md`'s "Working order," this compares CHAT's pretrained Kraken OCR model against
NomNaOCR's own pretrained CRNNxCTC model, **as-is, no fine-tuning**, on the same sample Nom
pages - counting how many Chu Han (Chinese-derived) characters each gets right. This decides
whether Phase 1 (the fine-tuning pipeline) is worth building. It is throwaway research code,
kept out of any future production pipeline.

## Status

- [x] CHAT's pretrained models downloaded (`data/chat_models/models/chat_{seg,rec}.mlmodel`,
      pulled directly from `colibrisson/CHAT_models` - no auth needed).
- [x] Both Docker images build, and the CHAT pipeline was verified end-to-end against a real
      classical-Chinese test page (from `CHAT_models`' own demo assets) - it correctly
      transcribed the page. The NomNaOCR/CRNNxCTC architecture and vocab-reconstruction code
      were verified against synthetic data (no real weights yet - see below). See "Bugs found
      during verification" below for what this shook out.
- [x] NomNaOCR dataset downloaded from Kaggle and `data/manifest.json` built - 15 sample pages
      across 7 works, via `scripts/build_manifest.py` (see "Build the manifest" below).
- [x] NomNaOCR pretrained CRNNxCTC weights downloaded and run against the 186 sample patches -
      predictions closely match ground truth (e.g. `史以記事也而` matched exactly on one page).
      This confirms the vocab-reconstruction approach (`nomnaocr_lib/vocab.py`) is correct.
- [x] `data/hanzi_charset.txt` built (44,348 characters, via `scripts/build_hanzi_charset.py`).
- [x] CHAT run against the 15 sample pages - but at native resolution the output is degenerate
      (near-constant repeated characters) on every page. Root-caused to a resolution mismatch,
      not a code bug - see "Finding: CHAT needs higher-resolution input" below. A preprocessing
      fix (upscaling) is confirmed to work on a single page and is being applied to the full set
      next.
- [ ] `results.md` generated - blocked on rerunning CHAT with the upscaling fix.

## Bugs found during verification

Smoke-testing against real CHAT weights (on a demo page from `CHAT_models` itself, not
NomNaOCR data) surfaced three issues, all fixed in the current code:

- **kraken version**: the latest kraken (7.x) produced degenerate, near-constant predictions
  ("八八一一一一一八八川川八..." repeating) from CHAT's `.mlmodel` weights. Pinning
  `docker/chat/Dockerfile` to `kraken==4.3.13` (current when CHAT_models was published, Sept
  2023 per its Zenodo DOI) fixed this - something about kraken's inference pipeline changed
  since then in a way that's incompatible with these older weights.
- **PIL binarization mode**: CHAT_models' original demo script binarizes to PIL mode `"1"`
  (bit-packed), which `numpy.array()` reads back as a `bool` array instead of a proper 0/255
  `uint8` array - silently breaking normalization downstream. Fixed in `run_chat.py` by
  binarizing to mode `"L"` instead (same 0/255 values, correct dtype).
- **kraken's segmentation return type differs by version**: 4.3.13 returns a plain dict
  (`baseline_seg["lines"]`), 7.x returns a `kraken.containers.Segmentation` dataclass
  (`baseline_seg.lines`). `check_line_direction()` in `run_chat.py` now handles both.

Separately, NomNaOCR's pretrained CRNNxCTC weights are `tensorflow==2.10.0` / Keras-2 format;
`docker/nomnaocr/Dockerfile` pins exactly that (plus `numpy<2`, which TF 2.10 requires) since
newer TensorFlow ships Keras 3, which changes enough (`KerasTensor.get_shape()`, `.h5` loading)
to risk silently mis-loading the pretrained weights.

`run_chat.py` was also parallelized across pages (one page per worker process, each still
single-threaded internally) after the original serial version took 27+ minutes without even
finishing the first of 15 pages. With 8 worker processes on an 8-core machine, all 15 pages
finished in under a minute. This is a throughput change only, not a correctness change - the
per-page inference logic is unmodified.

## Finding: CHAT needs higher-resolution input than NomNaOCR provides natively

Running CHAT on all 15 real NomNaOCR sample pages (at their native resolution) produced
degenerate, near-constant repeated-character output on **every single page** (e.g. `八一之不
一八○八八○孔一小以作以得以至大○八一之八十以中`) - the same failure signature as the
kraken-version bug above, which raised the concern that the parallelization change had
reintroduced it. A regression test against CHAT's own demo image (re-downloaded from
`CHAT_models`, run through the new parallelized code with `--workers 1`) came back mostly
clean and plausible, ruling that out.

The actual cause: NomNaOCR's page scans are only ~290x450px, meaning each of a page's ~9-24
text columns occupies roughly 15-30px of width - far smaller than CHAT's own demo image
(1011x1433px). Confirmed by upscaling one sample page 4x (Lanczos resampling) and rerunning
CHAT on it: the output changed from degenerate garbage to substantially correct, plausible
classical Chinese text closely matching the ground truth (e.g. `各有史如魯之` matched
exactly; `乃大限南比` vs. ground truth `乃天限南北` - a plausible character-level confusion,
not noise). Segmentation still over-detects lines even after upscaling (26 detected vs. ~9
expected columns on that page), but this doesn't invalidate the page-level bag-of-characters
scoring `compare_hanzi.py` uses, since that scoring works on full-page concatenated text, not
per-line alignment.

Given this, running CHAT "as-is" at NomNaOCR's native resolution would produce a badly biased,
near-zero result that doesn't reflect the model's real capability against Nom pages. Upscaling
before segmentation is a preprocessing step, not a model change, so applying it uniformly to
all sample pages is still consistent with Phase 0's "as-is, no fine-tuning" requirement - this
is the fix being rolled out next, before final Chu Han comparison numbers are generated.

## Why this needs your involvement

I can't complete the Kaggle or Google Drive downloads myself: Kaggle requires your own account
and API token, and Google Drive throttles large-file downloads behind a human
virus-scan-warning click that scripts can't get past reliably. Everything else (scaffolding,
inference scripts, comparison logic) is already written - the steps below are what's left.

## 1. Download NomNaOCR's dataset (Kaggle)

1. Create a Kaggle account if you don't have one, then go to
   https://www.kaggle.com/settings → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Check the dataset size first at https://www.kaggle.com/datasets/quandang/nomnaocr before
   downloading - Kaggle doesn't support partial/selective downloads, so this pulls the whole
   dataset (2,953 pages + 38,318 patches) even though we only need ~15 pages' worth for Phase 0.
3. Download it (either the "Download" button on the dataset page, or the Kaggle CLI:
   `pip install kaggle`, place `kaggle.json` per Kaggle's docs, then
   `kaggle datasets download -d quandang/nomnaocr`), and unzip it somewhere - e.g.
   `experiments/NomNaOCR/` (gitignored, alongside this folder).

The unzipped dataset has this layout (confirmed against the actual download, not guessed):

```
NomNaOCR/
  Pages/<work>/imgs/<page>.jpg   - full page scans
  Pages/<work>/gts/<page>.txt    - one line per column-patch: "x1,y1,...,x4,y4,text", in reading order
  Patches/<work>/<page>_<i>.jpg  - the i-th line of that page's gts file (0-indexed, same order)
  Patches/All.txt                - "work/patchfile.jpg<TAB>text" for the whole dataset
```

## 2. Build the manifest (run on the host, no Docker needed)

`scripts/build_manifest.py` picks sample pages, copies their page images and patches into
`data/sample_pages/` / `data/sample_patches/`, and writes `data/manifest.json` - all in one
step, using the confirmed layout above (previously this was a manual, error-prone step; the
prior version of this README asked you to inspect patch filenames by hand to work out reading
order, which turned out to be unnecessary once the actual convention was inspected):

```bash
python scripts/build_manifest.py --dataset-root experiments/NomNaOCR --num-pages 15
```

This only reads `Patches/All.txt` for ground-truth text - it does not copy or need the other
~38K patch images, since `nomnaocr_lib/vocab.py` (used later, for vocab reconstruction) only
reads label text, never image files.

## 3. Download NomNaOCR's pretrained CRNNxCTC weights (Google Drive)

**Done** - see the "Status" checklist above. Steps kept here for reproducibility:

1. Open the link from NomNaOCR's README: https://drive.google.com/file/d/1GDUM3gO5hDBaicCbESf3G07XH5I3sx4k/view
2. Download and unzip it. Look for a file named `NomNaOCR_CRNNxCTC.h5` (per
   `CRNNxCTC.ipynb`'s `reset_model.load_weights(f'NomNaOCR/NomNaOCR_{APPROACH_NAME}.h5')` -
   the exact folder layout inside the archive may differ; adjust the path below to match).
3. Copy it to `data/nomnaocr_weights/NomNaOCR_CRNNxCTC.h5`.

## 4. Build the Chu Han charset (run on the host, no Docker needed)

```bash
python scripts/build_hanzi_charset.py --out data/hanzi_charset.txt
```

This is a Unihan-`kMandarin`-based heuristic for "is this a real Chinese character," not
verified ground truth for Chu Han vs. Chu Nom-proper - see the script's docstring.

## 5. Run CHAT on the sample pages (Docker)

```bash
docker build -t phase0-chat -f docker/chat/Dockerfile .
docker run --rm -v "$PWD:/workspace" phase0-chat \
    --images data/sample_pages --models data/chat_models/models --out data/predictions_chat.json
```

**Note:** at NomNaOCR's native page resolution this produces degenerate output - see "Finding:
CHAT needs higher-resolution input" above. An upscaling preprocessing step is being added to
this script; once in place this section will document the flag needed to enable it.

## 6. Run NomNaOCR on the sample patches (Docker)

```bash
docker build -t phase0-nomnaocr -f docker/nomnaocr/Dockerfile .
docker run --rm -v "$PWD:/workspace" phase0-nomnaocr \
    --patches data/sample_patches \
    --all-labels data/all_labels.txt \
    --weights data/nomnaocr_weights/NomNaOCR_CRNNxCTC.h5 \
    --out data/predictions_nomnaocr.json
```

## 7. Compare and report (run on the host)

```bash
python scripts/compare_hanzi.py
python scripts/report.py
```

Produces `results.md` with the CHAT-vs-NomNaOCR Chu Han accuracy numbers CLAUDE.md's Phase 0
step needs.

## Known simplifications (flagged, not verified)

- **Chu Han vs. Chu Nom-proper classification** is a Unihan `kMandarin` heuristic
  (`build_hanzi_charset.py`), not linguistically verified ground truth.
- **Scoring** is a per-page bag-of-characters (multiset) intersection, not NomNaOCR's own
  Sequence Accuracy / Character Accuracy / CER - chosen to sidestep reading-order mismatches
  between the two models (CHAT's README notes a proper reading-order model isn't published
  yet). Phase 2 uses NomNaOCR's actual metrics (`Text recognition/metrics.py`) once there's a
  fine-tuned model to evaluate on a held-out test set.
- **NomNaOCR's character vocabulary** is reconstructed from `All.txt` text only
  (`nomnaocr_lib/vocab.py`), skipping the original `DataImporter`'s per-image
  `os.path.getsize` existence check (which would otherwise require the full 38K-image set on
  disk just to decode predictions). Expected to match the training-time vocab in practice; not
  independently verified against the original weights' training run.
