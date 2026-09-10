"""CTC loss for continuing training on NomNaOCR's CRNNxCTC architecture.

`input_length` is a fixed constant (25) for every sample here, not computed per-sample: every
image is resized to a fixed (432, 48) input (nomnaocr_lib/model.py's HEIGHT/WIDTH), so the CNN
always produces exactly 25 output timesteps regardless of image content - verified empirically
against the loaded pretrained model (`model.predict(...).shape == (1, 25, 7482)`), not assumed
from the architecture alone. `nomnaocr_lib/vocab.py`'s max_label_length is 24, so
`input_length (25) > label_length (<=24)` always holds for CTC's basic feasibility condition -
though a label with several consecutive repeated characters near the max length could still
violate the stricter `2*label_length - 1` requirement in rare cases (this is an existing property
of the original architecture/max_length choice, not something introduced here). `ctc_loss` guards
against the resulting non-finite per-sample loss rather than letting a single bad sample turn a
whole batch's gradient into NaN.
"""
import tensorflow as tf

TIME_STEPS = 25  # nomnaocr_lib/model.py's build_crnn() output timestep count, verified empirically


def ctc_loss(y_true, y_pred, label_length):
    """y_true: (batch, max_length) padded int label tokens. y_pred: (batch, TIME_STEPS, vocab+1)
    softmax output. label_length: (batch,) real (unpadded) label lengths.
    Returns (mean_loss_over_finite_samples, count_of_skipped_nonfinite_samples)."""
    batch_size = tf.shape(y_pred)[0]
    input_length = tf.fill([batch_size, 1], TIME_STEPS)
    label_length = tf.reshape(tf.cast(label_length, tf.int32), [-1, 1])
    per_sample_loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    per_sample_loss = tf.reshape(per_sample_loss, [-1])

    finite = tf.math.is_finite(per_sample_loss)
    safe_loss = tf.where(finite, per_sample_loss, tf.zeros_like(per_sample_loss))
    n_finite = tf.reduce_sum(tf.cast(finite, tf.float32))
    mean_loss = tf.math.divide_no_nan(tf.reduce_sum(safe_loss), n_finite)
    n_skipped = tf.reduce_sum(tf.cast(tf.logical_not(finite), tf.int32))
    return mean_loss, n_skipped
