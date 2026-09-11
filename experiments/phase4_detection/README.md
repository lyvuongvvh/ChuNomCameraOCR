# Phase 4 (part 1): Text-Line Detection

Per `CLAUDE.md`'s roadmap, Phase 4 wraps the validated OCR+translation pipeline (Phases 0-3) into
a backend API. Investigating what exists first surfaced a real gap: **100% of Phases 0-3 operate
on already-cropped line images** (`experiments/NomNaOCR/Patches/*`) - no component has ever
located text lines within a full, uncropped page or photo. This experiment validates a detection
approach against NomNaOCR's own real (but previously unused) detection ground truth before
wiring anything into the API, mirroring Phase 0/2's own validate-before-build discipline.

## Ground truth: real, and previously unused

`experiments/NomNaOCR/Pages/*/gts/*.txt` - one file per page image, one line per text column:
`x1,y1,x2,y2,x3,y3,x4,y4,transcript` (a 4-point quad, p0=top-left/p1=top-right/p2=bottom-right/
p3=bottom-left, ICDAR-style). Verified before use: format is consistent across all 9 works,
`imgs/*.jpg` and `gts/*.txt` pair 1:1 by filename stem with zero missing pairs in either
direction (2,953 of each). `scripts/build_detection_manifest.py` builds `data/manifest.json`
(gitignored) from these - 2,953 pages, 38,613 ground-truth lines total.

## Approach: escalate in rungs, validate before wiring in

One real prior data point already exists: Phase 0 tested **CHAT's own trained segmentation
model** (`chat_seg.mlmodel`) via kraken's `blla.segment` on a real NomNaOCR page and measured
over-detection (26 detected vs. ~9 expected columns). That result is about CHAT's specific
segmentation model, not kraken's own generic bundled default (untested in this project) - a
distinction worth being precise about before writing it off.

**Rung 0 (try first, zero new dependencies):** kraken's generic bundled segmenter
(`blla.segment(img, text_direction="vertical-rl", model=None)`), paired with a cheap post-filter
(drop tiny boxes, merge near-adjacent ones) tuned to this domain's dense, roughly-uniform vertical
columns.

**Rung 1 (still zero new ML framework, only if Rung 0 misses the bar):** fine-tune kraken's own
segmentation model via `ketos segtrain` on the real `gts/*.txt` ground truth, reusing the same
quad→baseline geometry already proven in `experiments/phase0_validation/scripts/build_finetune_data.py`.

**Rung 2 (only if 0 and 1 both fail):** a dedicated pretrained detector (PaddleOCR-det or CRAFT) -
the only rung adding a new ML framework alongside the already-pinned TF 2.10.

**Metric and bars:** greedy one-to-one IoU matching (`detect_lib/geometry.py::greedy_match`,
ICDAR/DetEval-style) at IoU >= 0.5 against `gts/*.txt`, `scripts/score_detection.py` computing
mean recall/precision across the scored sample. **Good enough to proceed:** recall >= 90% AND
precision >= 70%. Recall is prioritized - a missed line silently drops content from the final
result; an extra spurious detection just wastes one downstream recognition+translation call on
garbage, which Phase 4's API-level error handling already needs to tolerate regardless.

## Environment: kraken + TensorFlow 2.10 coexistence, checked before assuming

Phase 0 ran kraken (Python 3.11) and TF 2.10 (Python <=3.10, a hard requirement) in separate
containers, and never tested them together. Checked directly before designing `api/`'s
architecture around an assumption: `pip install tensorflow-cpu==2.10.0 kraken==4.3.13` in one
Python 3.10 environment resolves cleanly (pip settles on `numpy==1.23.5`, satisfying both), and
both import and are usable in the same process. **This means Phase 4's API can be a single
process**, not a split detection-service + recognition-service architecture - confirmed, not
assumed. `docker/Dockerfile` pins this combination.

## Rung 0: results - does not pass, root cause identified

A real, non-obvious environment bug surfaced first: `blla.segment()` crashed with a SIGSEGV
(`std::system_error: random_device could not be read`) after a laptop suspend/resume cycle.
Ruled out `/dev/urandom` access, torch itself, and seccomp before finding the actual cause:
**import order**, not call order. `torch.set_num_threads(1)` must happen before shapely/GEOS is
imported anywhere in the process, or torch's native RNG init segfaults - calling it inside
`segment_page()` (after the calling script had already imported `detect_lib.geometry`, which
pulls in shapely) did not fix it; moving it to the top of `detect_lib/kraken_segment.py`, before
any other import, did. Fixed and verified across multiple subsequent runs with no recurrence.

**Raw output needs merging.** On a single tuning page (DVSKTT, 290px wide, 9 real columns),
kraken's generic segmenter returned 114 raw line fragments - not noise mixed with good
detections, but every real column split into many small vertically-stacked pieces sharing that
column's true x-position. `filter_by_area` (drop tiny boxes) doesn't help here: all the fragments
are smaller than any real column, so no area threshold separates "real" from "fragment."
`merge_by_xposition` (sort by x-center, greedily cluster on a gap threshold, convex-hull each
cluster) was built instead, and got recall=0.778/precision=0.778 on that one page at
`gap_threshold=8`.

**That threshold did not generalize.** Scored across a 15-page sample spanning all 4 works in the
sample (DVSKTT, Tale of Kieu 1866/1871/1872, Luc Van Tien): mean recall=0.430, mean
precision=0.479 - a large drop from the single-page result. Per-page breakdown showed DVSKTT
pages held up reasonably (0.2-0.8 recall range) while all 4 non-DVSKTT pages scored **exactly
0.000/0.000**. Image-dimension comparison showed a real scale difference (DVSKTT ~290px wide vs.
Kieu 420-435px vs. Luc Van Tien 495px), so `--gap-fraction` (threshold proportional to page width,
`8/290 = 0.0276`) was added as a scale-independent alternative to the fixed `--gap-threshold`.

**Gap-fraction did not fix it either**: mean recall=0.449, mean precision=0.477 - statistically
indistinguishable from the fixed threshold, and the same 4 pages still scored 0.000/0.000 (2-5
predicted regions against 20-24 ground-truth columns each). This ruled out "wrong threshold
value" as the explanation. Diagnosed directly on `Tale of Kieu 1871/page084.jpg` (423x513px, 24
GT columns at a clean ~31-32px pitch): kraken's raw output was 195 fragments (vs. 114/9 on the
DVSKTT tuning page - a similar ~8-13x over-fragmentation ratio), but unlike the DVSKTT case, the
fragments' x-centers form a near-continuous ladder across the page (typical gaps of 1-3px, only
occasional 6-10px gaps that don't line up with the true column boundaries) rather than clustering
tightly around each column's center with a clear larger gap between columns. **No single gap
threshold can recover the 24 real columns from this sequence**: small enough to avoid merging
across real column boundaries also fails to bridge many spurious within-column gaps, and large
enough to bridge those spurious gaps also merges multiple real columns together (which is what
was observed: 2-6 output regions instead of 24).

**Conclusion: Rung 0 does not pass the bar (recall >= 0.90, precision >= 0.70) and this is not a
threshold-tuning problem.** `merge_by_xposition`'s 1D-clustering-on-x-gap heuristic depends on an
assumption - raw fragments cluster tightly per real column with a clearly larger gap between
columns - that held on the DVSKTT tuning page but does not hold on the other 3 works' scan
characteristics (denser fragmentation with no clean inter-column gap signal at any threshold).
Per the plan, this is real, well-evidenced grounds to escalate to **Rung 1** (`ketos segtrain`
fine-tuning on the real `gts/*.txt` ground truth) rather than continuing to tune Rung 0's
post-processing heuristic.

## Rung 1: scoping - two environment bugs fixed, fine-tuning signal confirmed real

Before committing to a real training run, ran small CPU sanity trials to check the mechanics and
get real, not assumed, numbers to scope the full run from.

**Data prep**: `scripts/build_segtrain_data.py` converts `gts/*.txt` quads into PageXML at
whole-page granularity, reusing `build_finetune_data.py`'s synthetic-baseline geometry (top-edge
midpoint -> bottom-edge midpoint). It excludes any page in `--exclude-manifest` from the training
pool entirely - the same pages `score_detection.py` evaluates against must never be trained on,
mirroring Phase 0's own documented Train/Validate contamination fix.

**Two real environment bugs, found and fixed, not training-data issues:**
- `ketos segtrain` crashed at the very first callback (`IndexError: pop from empty list` in
  `rich.console.clear_live()`) regardless of data. Root cause: kraken's dependency pin leaves
  `rich` unbounded, so pip resolved `rich==15.0.0`, whose `Console.clear_live()` now raises
  instead of no-op'ing when `pytorch_lightning==2.0.9`'s `RichProgressBar` calls it before any
  `Live` was ever pushed - a real version incompatibility. Fixed by pinning `rich<14` in
  `docker/Dockerfile`.
- Once that was fixed, training crashed mid-epoch with `RuntimeError: DataLoader worker ... killed
  by signal: Bus error` - Docker's default `/dev/shm` (64MB) is too small for PyTorch's
  multiprocess DataLoader. Fixed by passing `--shm-size=2g` to `docker run`.

**From-scratch training doesn't work at this scale.** A first trial (`-N 2`, no `-i`, 34 training
pages, all from one work) trained a fresh randomly-initialized 1.3M-param net and finished with
`val_mean_iu: 0.0` on both epochs - high pixel accuracy (0.974-0.977) came entirely from the
dominant background class; the model learned to predict no baselines anywhere. Confirmed at
inference: 0 raw detections. Expected for this little data from random init, not a bug.

**Fine-tuning from kraken's own bundled weights works.** Kraken's generic default model (the same
`blla.mlmodel` Rung 0 used, resolved via `pkg_resources.resource_filename('kraken', 'blla.mlmodel')`)
was passed as `-i` with `--resize both`, same 34-page trial, `-N 3`. `val_mean_iu` climbed
0.003 -> 0.032 -> 0.08 across the 3 epochs (still rising, not plateaued) - a real, nonzero,
improving signal `blla.segment()` alone (Rung 0's actual weights) could never show on its own.
At inference on the known tuning page (`DVSKTT_thu_III_1a`, 9 real GT columns at x-centers
33.0/64.0/92.0/121.5/152.0/180.0/210.0/239.5/269.5), the 3-epoch checkpoint produced 20 raw
fragments clustering almost exactly onto those 9 positions (vs. 114 badly-fragmented raw
detections from the unmodified generic model, and 0 from the from-scratch attempt) - already
close to usable without even reaching for `merge_by_xposition`.

**Important caveat, not yet resolved:** all 40 trial pages came from a single work (`DVSKTT-1
Quyen thu`) - manifest.json's default ordering, not a deliberate choice. This result shows
fine-tuning *can* learn this domain fast, but says nothing about whether it generalizes across the
other works' different scan characteristics, which is the entire reason Rung 0 failed and Rung 1
exists. The real training set must be stratified across all works, not just alphabetically-first
pages - not yet built.

**Compute scoping, from measured trial timing:** ~10 minutes/epoch on CPU for 34 pages. That's
roughly linear in page count (whole-page forward/backward passes dominate), so a few-hundred-page
stratified training set would already push into hours/epoch, and the full ~2,938-page pool
(2,953 minus the 15 held out for evaluation) would be roughly 14 hours/epoch - not practical
locally. Per Phase 0/2c's own precedent (Kaggle GPU for the recognizer's fine-tuning), the real
Rung 1 run should move to Kaggle GPU, reusing `experiments/phase2c_finetune`'s
`kernel-metadata.json` + Kaggle Kernels API push/pull pattern rather than inventing a new one.

## Status

Rung 0 (kraken generic segmenter + x-position merge) closed out: does not pass, root cause is a
heuristic that doesn't generalize across this dataset's different scan sources, not a threshold
value.

Rung 1 (`ketos segtrain` fine-tuning) scoping in progress: pipeline mechanically validated,
fine-tuning-from-bundled-weights confirmed to produce a real, improving signal (as opposed to
training from scratch, which does not work at this scale) on one work. Not yet done: a
work-stratified training set, an early-stopping / epoch-budget policy (still climbing after only
3 epochs - unclear how many are actually needed), and moving the real run to Kaggle GPU.
