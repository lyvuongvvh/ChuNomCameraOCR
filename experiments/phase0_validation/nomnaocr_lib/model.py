"""CRNNxCTC architecture and inference helpers, adapted from ds4v/NomNaOCR's
`Text recognition/CRNNxCTC/CRNNxCTC.ipynb`, `loader.py` and `utils.py` (MIT licensed).

Reproduces only what's needed to load the released `NomNaOCR_CRNNxCTC.h5` weights and run
inference on individual patch images - no training code. The `build_crnn` architecture must
exactly match the notebook's (`imagenet_model=None` branch) for `load_weights` to work, since
Keras matches weights to layers positionally/by-name.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, GRU

from .layers import custom_cnn, reshape_features

HEIGHT, WIDTH = 432, 48  # NomNaOCR/Text recognition/CRNNxCTC/CRNNxCTC.ipynb, cell 8
PADDING_CHAR = "[PAD]"


def build_crnn(vocab_size: int, name="CRNN") -> Model:
    """Matches CRNNxCTC.ipynb's build_crnn(imagenet_model=None, imagenet_output_layer=None)."""
    image_input = Input(shape=(HEIGHT, WIDTH, 3), dtype="float32", name="image")
    conv_blocks_config = {
        "block1": {"num_conv": 1, "filters": 64, "pool_size": (2, 2)},
        "block2": {"num_conv": 1, "filters": 128, "pool_size": (2, 2)},
        "block3": {"num_conv": 2, "filters": 256, "pool_size": (2, 2)},
        "block4": {"num_conv": 2, "filters": 512, "pool_size": (2, 2)},
        "block5": {"num_conv": 2, "filters": 512, "pool_size": None},
    }
    x = custom_cnn(conv_blocks_config, image_input)
    feature_maps = reshape_features(x, dim_to_keep=1, name="rnn_input")

    bigru1 = Bidirectional(GRU(256, return_sequences=True), name="bigru1")(feature_maps)
    bigru2 = Bidirectional(GRU(256, return_sequences=True), name="bigru2")(bigru1)

    y_pred = Dense(units=vocab_size + 1, activation="softmax", name="rnn_output")(bigru2)
    return Model(inputs=image_input, outputs=y_pred, name=name)


def ctc_decode(predictions, max_length):
    """From NomNaOCR's Text recognition/utils.py."""
    input_length = tf.ones(len(predictions)) * predictions.shape[1]
    preds_decoded = tf.keras.backend.ctc_decode(predictions, input_length=input_length, greedy=True)[0][0][:, :max_length]
    return tf.where(preds_decoded == tf.cast(1, tf.int64), tf.cast(-1, tf.int64), preds_decoded)


class CRNNRecognizer:
    """Slim stand-in for NomNaOCR's DataHandler, built from a vocabulary list instead of a
    DataImporter over the full image set (see nomnaocr_lib/vocab.py for why)."""

    def __init__(self, vocab: list[str], max_length: int, weights_path: str):
        self.char2num = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=PADDING_CHAR)
        self.num2char = tf.keras.layers.StringLookup(
            vocabulary=self.char2num.get_vocabulary(), mask_token=PADDING_CHAR, invert=True
        )
        # Matches DataHandler.max_length: longest label in the full training set (loader.py),
        # used to cap ctc_decode's output length.
        self.max_length = max_length
        # vocabulary_size(), not the deprecated vocab_size() alias: the latter was removed
        # entirely in Keras 3 (Kaggle's default TF as of this writing, 2.20.0) - vocabulary_size()
        # exists as the recommended replacement in both Keras 2 (this repo's pinned local
        # tensorflow==2.10.0 - it's the target of that version's own deprecation warning) and
        # Keras 3, so this one call is portable across both environments without needing a TF
        # version pin on Kaggle (which isn't even possible there - tensorflow==2.10.0 has no
        # wheel for Kaggle's Python 3.12).
        self.model = build_crnn(vocab_size=self.char2num.vocabulary_size())
        self.model.load_weights(weights_path)

    def process_image(self, img_path: str):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.resize(image, size=(HEIGHT, WIDTH), preserve_aspect_ratio=True)
        pad_height = HEIGHT - tf.shape(image)[0]
        pad_width = WIDTH - tf.shape(image)[1]
        image = tf.pad(
            image,
            # Height padding goes entirely at the bottom (top-aligned); width padding is split
            # with the extra pixel on the left when odd - matches loader.py's
            # distortion_free_resize(align_top=True) exactly.
            paddings=[[0, pad_height], [pad_width - pad_width // 2, pad_width // 2], [0, 0]],
            constant_values=255,
        )
        return tf.cast(image, tf.float32) / 255.0

    def predict_text(self, img_path: str) -> str:
        image = self.process_image(img_path)
        pred_tokens = self.model.predict(tf.expand_dims(image, axis=0), verbose=0)
        decoded = ctc_decode(pred_tokens, self.max_length)
        indices = tf.gather(decoded[0], tf.where(tf.logical_and(decoded[0] != 0, decoded[0] != -1)))
        text = tf.strings.reduce_join(self.num2char(indices))
        return text.numpy().decode("utf-8")

    def predict_beams(self, img_path: str, beam_width: int = 10) -> list[tuple[str, float]]:
        """Phase 2b (post-correction): top-`beam_width` CTC beam-search hypotheses and their
        log-probabilities, for LM rescoring - added on top of Phase 0/2's `predict_text` (which
        stays greedy-decode-only, unchanged) rather than replacing it. Uses TF's own
        `tf.nn.ctc_beam_search_decoder` (via `ctc_decode(greedy=False)`) - no custom beam search
        implementation needed. Returned in descending order of the model's own CTC log-probability
        (index 0 is what `predict_text` would return, modulo greedy/beam-search tie-breaking)."""
        image = self.process_image(img_path)
        pred_tokens = self.model.predict(tf.expand_dims(image, axis=0), verbose=0)
        input_length = tf.ones(1) * pred_tokens.shape[1]
        decoded, log_probs = tf.keras.backend.ctc_decode(
            pred_tokens, input_length=input_length,
            greedy=False, beam_width=beam_width, top_paths=beam_width,
        )
        beams = []
        for path_idx, path_tensor in enumerate(decoded):
            seq = path_tensor[0]
            indices = tf.gather(seq, tf.where(tf.logical_and(seq != 0, seq != -1)))
            text = tf.strings.reduce_join(self.num2char(indices)).numpy().decode("utf-8")
            beams.append((text, float(log_probs.numpy()[0][path_idx])))
        return beams
