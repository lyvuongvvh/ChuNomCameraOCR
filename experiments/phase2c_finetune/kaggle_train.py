"""Phase 2c: Kaggle training entry point - see experiments/phase2c_finetune/README.md for the
full experiment context. Run as a Kaggle *script* kernel (not a notebook): the diagnostic H5
compatibility check and the training run happen in one linear script, to minimize push/wait/
download cycles - Phase 0's CHAT fine-tuning trial needed 7 kernel iterations to work through
environment issues one at a time.

Expects a Kaggle Dataset attached as input, containing (see
experiments/phase2c_finetune/README.md's "Kaggle dataset layout"):
    Patches/<work>/*.jpg   - only Train.txt's images (Validate.txt's images are never uploaded,
                             so the held-out set is physically absent here, not just
                             code-guarded)
    Patches/All.txt, Train.txt, Validate.txt (text only)
    weights/NomNaOCR_CRNNxCTC.h5
    nomnaocr_lib/, train_lib/  - this repo's code, copied in at staging time
"""
import os
import sys
import time

import h5py

# A Train.txt sample, not Validate.txt: this Kaggle dataset never contains Validate.txt's images
# by construction (see the module docstring), so a Validate.txt sample here would always fail
# with a file-not-found error unrelated to H5/Keras compatibility - the bug that broke kernel v7.
EXPECTED_SAMPLE_IMAGE = "DVSKTT-4 Ban ky thuc luc/DVSKTT_ban_thuc_XII_7b_2.jpg"
EXPECTED_SAMPLE_TEXT = "不得棄本遂末并托以販賣技術游足游手其有"  # this project's own local pretrained-weights output for that image (exact match to ground truth, since it's a training example) - used to confirm the H5 weights loaded correctly under whichever TensorFlow this kernel ends up using, not as a training-quality signal


def find_dataset_root(marker="train.py", search_root="/kaggle/input"):
    # NOTE: "train.py" here refers to the copy staged inside kaggle_dataset/ (the *dataset*
    # input), not this script itself (kaggle_train.py, the *kernel* code, mounted separately at
    # /kaggle/src/script.py) - searching for this script's own filename was the bug that broke
    # kernel version 2 (FileNotFoundError, since kaggle_train.py is never part of /kaggle/input).
    for dirpath, _, filenames in os.walk(search_root):
        if marker in filenames:
            return dirpath
    raise FileNotFoundError(f"{marker} not found under {search_root} - dataset not mounted as expected")


def save_legacy_keras2_weights(model, filepath):
    """Write model weights in Keras 2's legacy HDF5 format directly (root 'layer_names' attr +
    per-layer named group with 'weight_names' attr + datasets), bypassing Keras 3's own
    save_weights() entirely - confirmed on kernel v8 that Keras 3's native weights-only H5 format
    uses a structurally different layout (generic auto-names like "conv2d_3" in a "layers/" tree,
    no root 'layer_names' attr at all) that this project's local Keras 2/TF 2.10
    Model.load_weights() cannot read ("Model expected 19 layers, found 0 saved layers").

    Uses model.layers/layer.get_weights() directly, so this preserves our actual layer names
    (e.g. "block1_conv1") natively - no need to reverse-engineer Keras 3's auto-naming scheme the
    way the one-off local converter script (used to validate kernel v8's checkpoint) had to.
    Model.load_weights() matches layers positionally (this project never uses by_name=True), so
    the exact weight_names strings only need to be internally self-consistent, not match any
    particular convention.
    """
    weighted_layers = [layer for layer in model.layers if layer.get_weights()]
    with h5py.File(filepath, "w") as f:
        f.attrs["layer_names"] = [layer.name.encode("utf8") for layer in weighted_layers]
        for layer in weighted_layers:
            g = f.create_group(layer.name)
            weights = layer.get_weights()
            weight_names = [f"{layer.name}_w{i}".encode("utf8") for i in range(len(weights))]
            g.attrs["weight_names"] = weight_names
            for wname, arr in zip(weight_names, weights):
                g.create_dataset(wname.decode("utf8"), data=arr)


def check_h5_compat(root):
    sys.path.insert(0, root)
    from nomnaocr_lib.vocab import build_vocab, max_label_length
    from nomnaocr_lib.model import CRNNRecognizer

    all_labels = f"{root}/Patches/All.txt"
    weights = f"{root}/weights/NomNaOCR_CRNNxCTC.h5"
    print(f"weights path: {weights}, exists={os.path.exists(weights)}, "
          f"size={os.path.getsize(weights) if os.path.exists(weights) else 'n/a'}")
    vocab = build_vocab(all_labels, min_length=1)
    max_length = max_label_length(all_labels, min_length=1)
    rec = CRNNRecognizer(vocab, max_length, weights)
    text = rec.predict_text(f"{root}/Patches/{EXPECTED_SAMPLE_IMAGE}")
    print(f"H5 compat check: predicted={text!r} expected={EXPECTED_SAMPLE_TEXT!r}")
    return text == EXPECTED_SAMPLE_TEXT


def main() -> None:
    root = find_dataset_root()
    print("dataset root:", root)

    import tensorflow as tf
    print("TensorFlow version (as shipped by Kaggle):", tf.__version__)
    print("GPUs visible:", tf.config.list_physical_devices("GPU"))

    # No TF-version-pinning fallback here: tensorflow==2.10.0 (this project's local pin) has no
    # wheel for Kaggle's Python 3.12 kernels at all, so pinning isn't an option on Kaggle even in
    # principle. The one real incompatibility found (Keras 3 removing StringLookup.vocab_size(),
    # fixed in nomnaocr_lib/model.py to use vocabulary_size() instead, which exists in both Keras
    # 2 and 3) is fixed at the source, so this check is now a plain confirmation, not a trigger
    # for a version swap.
    try:
        ok = check_h5_compat(root)
    except Exception:
        import traceback
        print("Failed to load/run the pretrained weights:")
        traceback.print_exc()
        ok = False

    if not ok:
        raise RuntimeError(
            "H5 compatibility check failed - the recognizer isn't producing the expected output "
            "under this Kaggle kernel's TensorFlow. Refusing to train against a recognizer that "
            "isn't verifiably correct, since the training forward pass would inherit the same bug."
        )

    print("H5 compatibility confirmed - proceeding to training.")
    if not tf.config.list_physical_devices("GPU"):
        print("WARNING: no GPU visible under the TensorFlow version in use. Training will "
              "proceed on CPU, which will be much slower than intended (see README.md's "
              "measured CPU rate) - check the kernel's GPU setting if this is unexpected.")

    sys.path.insert(0, root)
    from nomnaocr_lib.vocab import build_vocab, max_label_length, load_labels
    from nomnaocr_lib.model import CRNNRecognizer
    from train_lib.data import build_dataset
    from train_lib.ctc_loss import ctc_loss

    all_labels = f"{root}/Patches/All.txt"
    weights = f"{root}/weights/NomNaOCR_CRNNxCTC.h5"
    train_split = f"{root}/Patches/Train.txt"
    validate_split = f"{root}/Patches/Validate.txt"
    patches_dir = f"{root}/Patches"

    vocab = build_vocab(all_labels, min_length=1)
    max_length = max_label_length(all_labels, min_length=1)
    print(f"Vocab size: {len(vocab)}, max label length: {max_length}")

    recognizer = CRNNRecognizer(vocab, max_length, weights)
    model = recognizer.model

    # Kernel v8's 1-epoch validation run confirmed GPU training works (~7.7 min/epoch on a P100,
    # ~12x faster than this project's measured local CPU rate) and that a checkpoint round-trips
    # correctly into the local TF 2.10 eval pipeline (via save_legacy_keras2_weights above, once
    # converted - see README.md). That one epoch alone didn't visibly change any single greedy-
    # decoded example yet (avg_train_loss was already low, 0.061, before this run even started),
    # so a real attempt at moving Sequence/Character Accuracy needs more epochs - 8 is a
    # reasonable middle ground between giving training room to matter and session length
    # (~8 * 7.7min =~ 1 hour), adjustable via NOMNAOCR_EPOCHS.
    EPOCHS = int(os.environ.get("NOMNAOCR_EPOCHS", 8))
    BATCH_SIZE = int(os.environ.get("NOMNAOCR_BATCH_SIZE", 32))
    LEARNING_RATE = float(os.environ.get("NOMNAOCR_LR", 1e-5))
    DEV_FRACTION = 0.05  # small in-training monitoring slice, carved from Train.txt only -
    # Validate.txt's images are never present in this dataset at all, so this is purely a
    # convenience signal for watching over/under-fitting during the run, not a substitute for
    # Phase 2's held-out evaluation, which happens back on the dev machine after download.

    pairs = load_labels(train_split, min_length=1)
    validate_keys = {img_name for img_name, _ in load_labels(validate_split, min_length=1)}
    contaminated = [img_name for img_name, _ in pairs if img_name in validate_keys]
    assert not contaminated, f"{len(contaminated)} training patches also appear in Validate.txt"

    n_dev = round(len(pairs) * DEV_FRACTION)
    dev_pairs, train_pairs = pairs[:n_dev], pairs[n_dev:]
    print(f"{len(train_pairs)} training lines, {len(dev_pairs)} in-training dev lines "
          f"(carved from Train.txt, not Validate.txt)")

    train_pairs = [(f"{patches_dir}/{img_name}", text) for img_name, text in train_pairs]
    dev_pairs = [(f"{patches_dir}/{img_name}", text) for img_name, text in dev_pairs]

    train_ds = build_dataset(train_pairs, recognizer, max_length, BATCH_SIZE)
    dev_ds = build_dataset(dev_pairs, recognizer, max_length, BATCH_SIZE, shuffle=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    out_dir = "/kaggle/working"

    print(f"Training for {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    for epoch in range(1, EPOCHS + 1):
        start = time.monotonic()
        total_loss, n_batches, n_skipped = 0.0, 0, 0
        for images, labels, label_lengths in train_ds:
            with tf.GradientTape() as tape:
                preds = model(images, training=True)
                loss, n_nonfinite = ctc_loss(labels, preds, label_lengths)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += float(loss)
            n_batches += 1
            n_skipped += int(n_nonfinite)
            if n_batches % 100 == 0:
                elapsed = time.monotonic() - start
                print(f"epoch {epoch} batch {n_batches}: avg_train_loss={total_loss / n_batches:.4f} "
                      f"({elapsed:.0f}s elapsed)")

        dev_loss_total, dev_batches = 0.0, 0
        for images, labels, label_lengths in dev_ds:
            preds = model(images, training=False)
            loss, _ = ctc_loss(labels, preds, label_lengths)
            dev_loss_total += float(loss)
            dev_batches += 1
        dev_loss = dev_loss_total / dev_batches if dev_batches else float("nan")

        elapsed = time.monotonic() - start
        print(f"epoch {epoch} done: avg_train_loss={total_loss / n_batches:.4f}, "
              f"dev_loss={dev_loss:.4f}, {n_batches} batches, {n_skipped} non-finite skipped, "
              f"{elapsed:.0f}s ({elapsed / max(n_batches, 1):.3f}s/batch)")

        # save_legacy_keras2_weights, not model.save_weights(): confirmed on kernel v8 that
        # Keras 3's own save_weights() produces a file this project's local TF 2.10
        # Model.load_weights() cannot read at all ("Model expected 19 layers, found 0 saved
        # layers") - see that function's docstring above.
        ckpt_path = f"{out_dir}/finetuned_epoch{epoch}.h5"
        save_legacy_keras2_weights(model, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
