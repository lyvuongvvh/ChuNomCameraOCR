"""Reproduces ds4v/NomNaOCR's DataImporter vocab-building logic (Text recognition/loader.py),
without requiring the full ~38K patch image set on disk.

The pretrained CRNNxCTC weights encode character indices via a `tf.keras.layers.StringLookup`
built from `Counter(''.join(labels)).most_common())` over NomNaOCR's *entire* training label
file (All.txt). To decode predictions correctly we must reconstruct that exact vocabulary
ordering. The original DataImporter also calls `os.path.getsize(img_path)` per row, which
requires every image file to exist - that's only needed for training (loading images), not
for recovering the frozen vocabulary of an already-trained model. This module replicates the
text-side filtering only (is_clean_text + min_length) and skips the image-existence check.

Known simplification (flagged per CLAUDE.md): if any row's image was missing/empty in the
original dataset, the original DataImporter would have excluded it from the vocab count and we
won't. This is expected to be rare-to-nonexistent in a clean released dataset and, even if it
occurs, is very unlikely to change the character frequency order given NomNaOCR's ~7.5K-entry
vocab. Not independently verified - see experiments/phase0_validation/README.md.
"""
import re
from collections import Counter
from string import printable

NOT_NOM_CHARS = (
    r'\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ'
)
_CLEAN_PATTERN = re.compile(f'[{NOT_NOM_CHARS}{re.escape(printable)}]')


def is_clean_text(text: str) -> bool:
    return not bool(_CLEAN_PATTERN.search(text.lower()))


def load_labels(labels_path: str, min_length: int = 1) -> list[tuple[str, str]]:
    """Parse a NomNaOCR label file (`img_name\\ttext` per line) into (img_name, text) pairs,
    applying the same min_length + is_clean_text filtering as DataImporter."""
    pairs = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            img_name, text = line.rstrip("\n").split("\t")
            text = text.strip().lower()
            if len(text) >= min_length and is_clean_text(text):
                pairs.append((img_name, text))
    return pairs


def build_vocab(labels_path: str, min_length: int = 1) -> list[str]:
    """Return the character vocabulary in the same order DataImporter would produce it
    (`Counter(''.join(labels)).most_common()`), for use as `tf.keras.layers.StringLookup`
    vocabulary when reconstructing the pretrained model's char2num/num2char mapping."""
    _, texts = zip(*load_labels(labels_path, min_length))
    vocabs = dict(Counter("".join(texts)).most_common())
    return list(vocabs)


def max_label_length(labels_path: str, min_length: int = 1) -> int:
    """Matches DataHandler.max_length (loader.py): longest label text, in characters, over
    the same filtered label set build_vocab() uses."""
    _, texts = zip(*load_labels(labels_path, min_length))
    return max(len(text) for text in texts)
