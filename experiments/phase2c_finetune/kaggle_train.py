"""Phase 2c: Kaggle training entry point - see experiments/phase2c_finetune/README.md for the
full experiment context. Run as a Kaggle *script* kernel (not a notebook): the whole diagnostic +
conditional TF-version fallback + training run happens in one linear script, to minimize
push/wait/download cycles - Phase 0's CHAT fine-tuning trial needed 7 kernel iterations to work
through environment issues one at a time; this front-loads that adaptation into a single run
where possible instead.

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
import subprocess
import sys
import time

EXPECTED_SAMPLE_IMAGE = "DVSKTT-3 Ban ky toan thu/DVSKTT_ban_toan_V_30a_9.jpg"
EXPECTED_SAMPLE_TEXT = "使通好執事遂而不反我是以有往年之帥帝遭"  # this project's own local pretrained-weights output for that image (nomnaocr_lib/model.py, unmodified) - used to confirm the H5 weights loaded correctly under whichever TensorFlow this kernel ends up using, not as a training-quality signal


def find_dataset_root(marker="train.py", search_root="/kaggle/input"):
    # NOTE: "train.py" here refers to the copy staged inside kaggle_dataset/ (the *dataset*
    # input), not this script itself (kaggle_train.py, the *kernel* code, mounted separately at
    # /kaggle/src/script.py) - searching for this script's own filename was the bug that broke
    # kernel version 2 (FileNotFoundError, since kaggle_train.py is never part of /kaggle/input).
    for dirpath, _, filenames in os.walk(search_root):
        if marker in filenames:
            return dirpath
    raise FileNotFoundError(f"{marker} not found under {search_root} - dataset not mounted as expected")


def check_h5_compat(root):
    sys.path.insert(0, root)
    from nomnaocr_lib.vocab import build_vocab, max_label_length
    from nomnaocr_lib.model import CRNNRecognizer

    all_labels = f"{root}/Patches/All.txt"
    weights = f"{root}/weights/NomNaOCR_CRNNxCTC.h5"
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

    ok = False
    try:
        ok = check_h5_compat(root)
    except Exception as e:
        print("Default TensorFlow failed to load/run the pretrained weights:", repr(e))

    if not ok and os.environ.get("NOMNAOCR_TF_PINNED") != "1":
        print("H5 compatibility check failed under Kaggle's default TensorFlow - falling back "
              "to the exact version this project's local Docker pipeline uses (tensorflow==2.10.0, "
              "matches nomnaocr_lib/model.py's own comments on why this pin matters for Keras "
              "3 vs. 2 .h5 loading behavior).")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "tensorflow==2.10.0"], check=True)
        # A fresh process is needed for `import tensorflow` to pick up the newly installed
        # version cleanly (the current process already has the old one cached in sys.modules).
        env = dict(os.environ, NOMNAOCR_TF_PINNED="1")
        result = subprocess.run([sys.executable, __file__], env=env)
        sys.exit(result.returncode)

    if not ok:
        raise RuntimeError(
            "H5 compatibility check still fails even after pinning tensorflow==2.10.0 - "
            "something other than the TF/Keras version is wrong (e.g. GPU-vs-CPU numerical "
            "difference producing a different greedy decode, or a genuinely corrupted upload). "
            "Refusing to train against a recognizer that isn't verifiably producing correct "
            "output, since the training forward pass would inherit the same bug."
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

    EPOCHS = int(os.environ.get("NOMNAOCR_EPOCHS", 5))
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

        ckpt_path = f"{out_dir}/finetuned_epoch{epoch}.h5"
        model.save_weights(ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
