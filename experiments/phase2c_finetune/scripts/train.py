"""Phase 2c: continue training NomNaOCR's own pretrained CRNNxCTC model on its own
`Patches/Train.txt` data (never touching `Patches/Validate.txt`), to see whether further training
moves the Sequence/Character Accuracy/CER numbers Phase 2/2b already measure on that same
held-out set.

Unlike Phase 0's CHAT fine-tuning trial, this is NOT cross-domain transfer - it continues
training the exact architecture on the exact data distribution it was already trained on, so
catastrophic forgetting (Phase 0's failure mode) isn't expected to apply the same way. The real
open question is whether it improves at all, or just re-fits data the model has effectively
already seen once (see ../README.md's "Known risks" section).

Produces plain `.h5` weight checkpoints, loadable directly by Phase 2's `run_eval.py --weights`
for a fair before/after comparison - no changes to the eval pipeline needed.

Usage (inside phase0_validation/docker/nomnaocr's image - same environment Phase 0/2/2b use):
    docker run --rm -v "$PWD/experiments:/workspace" -w /workspace --entrypoint python \\
        phase0-nomnaocr phase2c_finetune/scripts/train.py \\
        --dataset-root NomNaOCR --all-labels NomNaOCR/Patches/All.txt \\
        --train-split NomNaOCR/Patches/Train.txt \\
        --weights NomNaOCR_H5/NomNaOCR_CRNNxCTC.h5 \\
        --epochs 3 --batch-size 32 --learning-rate 1e-5 \\
        --out-dir phase2c_finetune/data/checkpoints

For a fast local smoke test before a real (Kaggle) run, add e.g. --max-lines 40 --epochs 1
--batch-size 4 - this should complete in well under a minute on CPU and is meant to catch shape/
plumbing bugs, not to produce a usable checkpoint.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "phase0_validation"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import tensorflow as tf  # noqa: E402

from nomnaocr_lib.vocab import build_vocab, max_label_length, load_labels  # noqa: E402
from nomnaocr_lib.model import CRNNRecognizer  # noqa: E402
from train_lib.data import build_dataset  # noqa: E402
from train_lib.ctc_loss import ctc_loss  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=pathlib.Path)
    parser.add_argument("--all-labels", required=True, type=pathlib.Path,
                         help="NomNaOCR's full All.txt (vocab reconstruction - must match the "
                              "vocab the starting --weights checkpoint was trained with)")
    parser.add_argument("--train-split", required=True, type=pathlib.Path)
    parser.add_argument("--validate-split", default=None, type=pathlib.Path,
                         help="If given, asserts zero overlap between --train-split and this "
                              "file before training (contamination guard, cheap safety net - "
                              "Train.txt/Validate.txt should already be disjoint by construction)")
    parser.add_argument("--weights", required=True, type=pathlib.Path,
                         help="Starting checkpoint - NomNaOCR's own pretrained weights")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-lines", type=int, default=None,
                         help="Dev-loop only: cap training data size for a fast smoke test")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    print("Rebuilding training-time character vocabulary from", args.all_labels)
    vocab = build_vocab(str(args.all_labels), min_length=1)
    max_length = max_label_length(str(args.all_labels), min_length=1)
    print(f"Vocab size: {len(vocab)}, max label length: {max_length}")

    recognizer = CRNNRecognizer(vocab, max_length, str(args.weights))
    model = recognizer.model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    pairs = load_labels(str(args.train_split), min_length=1)
    if args.validate_split:
        validate_keys = {img_name for img_name, _ in load_labels(str(args.validate_split), min_length=1)}
        contaminated = [img_name for img_name, _ in pairs if img_name in validate_keys]
        assert not contaminated, (
            f"{len(contaminated)} training patches also appear in {args.validate_split} - "
            f"refusing to train on this split"
        )

    patches_dir = args.dataset_root / "Patches"
    pairs = [(str(patches_dir / img_name), text) for img_name, text in pairs]
    if args.max_lines:
        pairs = pairs[:args.max_lines]
    print(f"Training on {len(pairs)} lines from {args.train_split}")

    dataset = build_dataset(pairs, recognizer, max_length, args.batch_size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        total_loss, n_batches, n_skipped = 0.0, 0, 0
        for images, labels, label_lengths in dataset:
            with tf.GradientTape() as tape:
                preds = model(images, training=True)
                loss, n_nonfinite = ctc_loss(labels, preds, label_lengths)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += float(loss)
            n_batches += 1
            n_skipped += int(n_nonfinite)
            if n_batches % args.log_every == 0:
                elapsed = time.monotonic() - start
                print(f"epoch {epoch} batch {n_batches}: avg_loss={total_loss / n_batches:.4f} "
                      f"({elapsed:.0f}s elapsed, {n_skipped} non-finite samples skipped so far)")

        avg_loss = total_loss / n_batches if n_batches else float("nan")
        print(f"epoch {epoch} done: avg_loss={avg_loss:.4f}, {n_batches} batches, "
              f"{n_skipped} non-finite samples skipped, {time.monotonic() - start:.0f}s")

        ckpt_path = args.out_dir / f"finetuned_epoch{epoch}.h5"
        model.save_weights(str(ckpt_path))
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
