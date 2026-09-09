"""Character-level n-gram language model (stupid backoff, Brants et al. 2007), trained on
NomNaOCR's own `Patches/Train.txt` ground-truth text, used to rescore CTC beam-search candidates
in Phase 2b (post-correction) - see ../README.md for why this approach and its known ceiling.

Stupid backoff instead of a properly smoothed/normalized model (e.g. Kneser-Ney): it produces
uncalibrated scores, not real probabilities, but that's all reranking needs here (comparing a
handful of candidate transcriptions for the SAME line), and it needs no held-out perplexity
tuning of its own beyond the one fixed backoff discount - unlike a real smoothed LM, which would
need its own hyperparameter search on top of the rescoring weight this experiment already tunes.
"""
from __future__ import annotations

import json
import math
from collections import Counter

START = "\x02"  # sentinel start-of-line padding character; never appears in real text/predictions
ALPHA = 0.4     # Brants et al.'s stupid-backoff discount
FLOOR = 1e-12   # score floor for an entirely unseen character, to keep log() defined


class CharNgramLM:
    def __init__(self, order: int, counts: dict):
        self.order = order
        self.counts = counts  # {n: {ngram_string: count}}, n in 1..order

    @classmethod
    def train(cls, texts: list, order: int = 4) -> "CharNgramLM":
        counts = {n: Counter() for n in range(1, order + 1)}
        for text in texts:
            padded = START * (order - 1) + text
            for n in range(1, order + 1):
                for i in range(n - 1, len(padded)):
                    ngram = padded[i - n + 1:i + 1]
                    if n == 1 and ngram == START:
                        # Don't let (order-1) START-padding chars per line pollute the unigram
                        # base distribution - higher-order n-grams that merely *contain* START as
                        # leading context (e.g. "start of line, first char is X") are kept, since
                        # those are meaningful; only the pure single-START unigram count is not.
                        continue
                    counts[n][ngram] += 1
        return cls(order, {n: dict(c) for n, c in counts.items()})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"order": self.order, "counts": self.counts}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharNgramLM":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["order"], {int(n): c for n, c in data["counts"].items()})

    def _ngram_score(self, ngram: str) -> float:
        """Stupid-backoff score for the last character of `ngram` given its preceding n-1
        characters, recursing to shorter contexts when the full n-gram is unseen or its context
        never occurred."""
        n = len(ngram)
        if n == 1:
            total = sum(self.counts[1].values())
            count = self.counts[1].get(ngram, 0)
            return count / total if count > 0 and total > 0 else FLOOR
        count_ngram = self.counts.get(n, {}).get(ngram, 0)
        count_context = self.counts.get(n - 1, {}).get(ngram[:-1], 0)
        if count_ngram > 0 and count_context > 0:
            return count_ngram / count_context
        return ALPHA * self._ngram_score(ngram[1:])

    def score(self, text: str) -> float:
        """Total log score for `text`: sum of per-character stupid-backoff log scores. Higher
        means more consistent with the training corpus's character sequences. Not a normalized
        log-probability - only meaningful for comparing candidates of similar length for the
        same line, which is exactly how this is used (see scripts/rescore.py)."""
        padded = START * (self.order - 1) + text
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            ngram = padded[i - self.order + 1:i + 1]
            total += math.log(self._ngram_score(ngram))
        return total
