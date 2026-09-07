"""Phase 0 validation: run CHAT's pretrained Kraken model on sample Nom pages, as-is.

Adapted from colibrisson/CHAT_models' demo/chat_models_demo.py. No fine-tuning,
no changes to the model - this measures the pretrained model's out-of-the-box
performance on Nom pages, per CLAUDE.md's Phase 0 requirement.

Usage (inside the chat docker image, /workspace mounted to experiments/phase0_validation):
    python scripts/run_chat.py --images data/sample_pages --models data/chat_models/models \
        --out data/predictions_chat.json
"""
import argparse
import json
import multiprocessing
import pathlib

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

_seg_model = None
_rec_model = None


def _init_worker(seg_model_path: str, rec_model_path: str) -> None:
    """Runs once per worker process. Loads both models into process-local globals and pins
    this worker to a single CPU thread - parallelism comes from running one page per worker
    process instead of multi-threading a single page's inference (segmentation is the slow
    step and is embarrassingly parallel across pages, so this scales with core count instead
    of the poor intra-op scaling conv-heavy models tend to show on CPU)."""
    global _seg_model, _rec_model
    import torch
    from kraken.lib import vgsl, models

    torch.set_num_threads(1)
    _seg_model = vgsl.TorchVGSLModel.load_model(seg_model_path)
    _rec_model = models.load_any(rec_model_path)


def _process_one(img_path: pathlib.Path):
    from kraken import blla, rpred
    from PIL import Image

    print(f"[chat] {img_path.name}", flush=True)
    img = Image.open(img_path)
    img = img.convert("L")
    # Binarize but keep mode "L" (uint8, values 0/255), not mode "1": PIL's mode "1" is
    # bit-packed and numpy reads it back as a bool array, which silently breaks
    # normalization in kraken's current image-loading pipeline (confirmed by testing -
    # mode "1" here produced degenerate, near-constant predictions). CHAT_models' original
    # demo script used mode "1" against an older kraken version where this wasn't an issue.
    img = img.point(lambda x: 0 if x < 128 else 255, "L")

    baseline_seg = blla.segment(img, text_direction="vertical-rl", model=_seg_model)
    baseline_seg = check_line_direction(baseline_seg)

    lines = [str(record) for record in rpred.rpred(_rec_model, img, baseline_seg)]
    return img_path.name, {"lines": lines, "full_text": "".join(lines)}


def check_line_direction(baseline_seg):
    """Reverse baselines that don't run top-to-bottom (from CHAT_models' demo script).

    Handles both of kraken's blla.segment() return shapes: a plain dict
    (`baseline_seg["lines"]`, `line["baseline"]` - what kraken 4.3.13 returns, and what
    CHAT_models' original demo script was written against) and the newer
    `kraken.containers.Segmentation` dataclass (`.lines`, each `BaselineLine.baseline`).
    """
    lines = baseline_seg["lines"] if isinstance(baseline_seg, dict) else baseline_seg.lines
    for line in lines:
        baseline = line["baseline"] if isinstance(line, dict) else line.baseline
        if baseline[0][1] > baseline[-1][1]:
            baseline.reverse()
    return baseline_seg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=pathlib.Path, help="Dir of sample page images")
    parser.add_argument("--models", required=True, type=pathlib.Path, help="Dir containing chat_seg.mlmodel and chat_rec.mlmodel")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="Output JSON path")
    parser.add_argument("--workers", type=int, default=None,
                         help="Pages to process in parallel (default: min(page count, CPU count))")
    args = parser.parse_args()

    try:
        import kraken  # noqa: F401
    except ImportError:
        raise SystemExit("kraken is not installed. Run this script inside the chat docker image.")

    seg_model_path = args.models / "chat_seg.mlmodel"
    rec_model_path = args.models / "chat_rec.mlmodel"
    assert seg_model_path.exists(), f"missing {seg_model_path}"
    assert rec_model_path.exists(), f"missing {rec_model_path}"

    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"no images found in {args.images}")

    workers = args.workers or min(len(image_paths), multiprocessing.cpu_count())
    print(f"Processing {len(image_paths)} pages with {workers} worker process(es)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with multiprocessing.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(seg_model_path), str(rec_model_path)),
    ) as pool:
        for name, prediction in pool.imap_unordered(_process_one, image_paths):
            results[name] = prediction
            # Write after every page, not just at the end - a run over many large pages can
            # take a long time, so this keeps partial results on disk if it's interrupted.
            args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(results)} page predictions to {args.out}")


if __name__ == "__main__":
    main()
