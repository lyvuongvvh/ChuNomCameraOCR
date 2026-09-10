"""tf.data pipeline for continuing training on NomNaOCR's own CRNNxCTC architecture.

Reuses `nomnaocr_lib.model.CRNNRecognizer.process_image` unchanged (it already matches
training-time preprocessing exactly - see that method's own comments on
`distortion_free_resize(align_top=True)`). This module only adds label encoding and batching,
which inference never needed.
"""
import tensorflow as tf


def encode_label(text, char2num, max_length):
    """Right-pads to `max_length` with the [PAD] token (index 0), matching loader.py's
    DataHandler.process_label - no start/end tokens, since CRNNxCTC's own char2num usage never
    adds them (only NomNaOCR's attention-based models do). Returns (padded_tokens, real_length)."""
    chars = tf.strings.unicode_split(text, "UTF-8")
    tokens = tf.cast(char2num(chars), tf.int64)
    length = tf.shape(tokens)[0]
    padded = tf.pad(tokens, [[0, max_length - length]], constant_values=0)
    padded.set_shape([max_length])
    return padded, length


def build_dataset(pairs, recognizer, max_length, batch_size, shuffle=True, seed=0):
    """pairs: list of (img_path, text). Returns a batched tf.data.Dataset yielding
    (image, label_tokens, label_length) - everything train_lib.ctc_loss.ctc_loss needs, plus the
    image for the forward pass."""
    img_paths = [p for p, _ in pairs]
    texts = [t for _, t in pairs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, texts))
    if shuffle:
        ds = ds.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)

    def _load(img_path, text):
        image = recognizer.process_image(img_path)
        label, length = encode_label(text, recognizer.char2num, max_length)
        return image, label, length

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
