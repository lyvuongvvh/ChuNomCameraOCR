"""Phase 2: NomNaOCR's own evaluation metrics - Sequence Accuracy, Character Accuracy, and
Character Error Rate (CER) - reimplemented from the literal source of `Text recognition/
metrics.py` in ds4v/NomNaOCR (not vendored in this repo; fetched from GitHub and reproduced
here, MIT licensed):

    SequenceAccuracy.update_state: converts y_true/y_pred to dense [batch, max_length] token
        arrays (zero-padded); a sequence is wrong if `tf.reduce_any(y_true != y_pred, axis=1)`.
        result = correct sequences / total sequences.
    CharacterAccuracy.update_state: `num_errors = logical_and(y_true != y_pred, y_true != 0)`;
        result = matching non-padding chars / total non-padding chars (0 = the pad token).
    LevenshteinDistance(normalize=True): per-sample `tf.edit_distance(pred, true,
        normalize=True)` (edit distance / len(truth) per TF's own normalize semantics), then
        averaged across the batch - this is `cer_macro` below.
    warp_cer_metric: `sum(edit_distance) / sum(len(truth))` across the whole batch, i.e. total
        edit distance over total ground-truth characters - this is `cer_micro` below.

These operate on NomNaOCR's own token tensors during training/eval. Our pipeline instead has
`nomnaocr_lib.model.CRNNRecognizer.predict_text()`'s already CTC-decoded output text strings, so
these are reimplemented over plain Python strings rather than padded integer tensors.

Known upstream inaccuracy, flagged rather than silently trusted: `warp_cer_metric`'s own
docstring claims its result "is the same as... LevenshteinDistance... with normalize=True", but
the former is a micro-average (total distance / total length) and the latter is a macro-average
(mean of per-sample distance/length ratios) - these only coincide when every sample has the same
ground-truth length. We report both, clearly labeled, rather than assuming either is "the" CER.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (substitution/insertion/deletion each cost 1), matching
    `tf.edit_distance`'s default behavior."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def sequence_correct(pred: str, gt: str) -> bool:
    """Exact-match sequence accuracy.

    Equivalent to NomNaOCR's dense-array comparison, not merely an approximation of it: padding
    both `pred` and `gt` to any shared length >= max(len(pred), len(gt)) with the same pad value
    produces identical arrays iff the strings are identical, and differs in at least one position
    otherwise (including when only their lengths differ, since the shorter one gets padding where
    the longer one has a real, non-pad character). This assumes `max_length` (the shared padding
    width NomNaOCR pads to) is >= both sequences' lengths, which holds here since it's computed as
    the max label length over NomNaOCR's own full training label set and both pred and gt come
    from the same label distribution.
    """
    return pred == gt


def character_accuracy_counts(pred: str, gt: str, max_length: int) -> tuple[int, int]:
    """Positional (non-alignment-based) character accuracy: pad both `pred` and `gt` to
    `max_length`, compare position by position, and only count positions where `gt` is non-pad.

    Deliberately NOT edit-distance/alignment based - this faithfully replicates NomNaOCR's own
    metric, including its known weakness that a single leading insertion or deletion in `pred`
    misaligns every character after it, tanking the score even if every character is individually
    "correct but shifted". See tests/test_metrics.py for a worked example of this.
    """
    gt = gt[:max_length]
    pred = pred[:max_length]
    correct = sum(
        1 for i, g in enumerate(gt)
        if i < len(pred) and pred[i] == g
    )
    return correct, len(gt)


def cer_macro_term(pred: str, gt: str) -> float:
    """Per-sample term for macro-averaged CER (matches LevenshteinDistance(normalize=True)):
    edit distance normalized by ground-truth length. Undefined (0.0) for an empty ground truth,
    which shouldn't occur here since eval labels are filtered to min_length >= 1."""
    if not gt:
        return 0.0
    return edit_distance(pred, gt) / len(gt)


def cer_micro_terms(pred: str, gt: str) -> tuple[int, int]:
    """Per-sample (edit_distance, gt_length) pair for micro-averaged CER (matches
    warp_cer_metric): aggregate via aggregate_cer_micro, summing both components first."""
    return edit_distance(pred, gt), len(gt)


def aggregate_cer_macro(terms: list[float]) -> float:
    return sum(terms) / len(terms) if terms else 0.0


def aggregate_cer_micro(terms: list[tuple[int, int]]) -> float:
    total_distance = sum(d for d, _ in terms)
    total_length = sum(n for _, n in terms)
    return total_distance / total_length if total_length else 0.0


@dataclass
class SampleScore:
    img_name: str
    work: str
    pred: str
    gt: str
    seq_correct: bool
    char_correct: int
    char_total: int
    edit_dist: int
    subsets: list[str] = field(default_factory=list)

    @classmethod
    def compute(cls, img_name: str, work: str, pred: str, gt: str, max_length: int,
                subsets: list[str] | None = None) -> "SampleScore":
        char_correct, char_total = character_accuracy_counts(pred, gt, max_length)
        return cls(
            img_name=img_name,
            work=work,
            pred=pred,
            gt=gt,
            seq_correct=sequence_correct(pred, gt),
            char_correct=char_correct,
            char_total=char_total,
            edit_dist=edit_distance(pred, gt),
            subsets=subsets or [],
        )


def aggregate(scores: list[SampleScore]) -> dict:
    """Roll up SampleScores into overall + per-subset metrics."""

    def rollup(subset_scores: list[SampleScore]) -> dict:
        n = len(subset_scores)
        if n == 0:
            return {"n": 0}
        seq_acc = sum(s.seq_correct for s in subset_scores) / n
        char_correct = sum(s.char_correct for s in subset_scores)
        char_total = sum(s.char_total for s in subset_scores)
        char_acc = char_correct / char_total if char_total else 0.0
        cer_macro = aggregate_cer_macro([
            (s.edit_dist / len(s.gt)) if s.gt else 0.0 for s in subset_scores
        ])
        cer_micro = aggregate_cer_micro([(s.edit_dist, len(s.gt)) for s in subset_scores])
        return {
            "n": n,
            "sequence_accuracy": seq_acc,
            "character_accuracy": char_acc,
            "char_correct": char_correct,
            "char_total": char_total,
            "cer_macro": cer_macro,
            "cer_micro": cer_micro,
        }

    all_subsets = sorted({tag for s in scores for tag in s.subsets})
    result = {"overall": rollup(scores)}
    for tag in all_subsets:
        result[tag] = rollup([s for s in scores if tag in s.subsets])
    return result
