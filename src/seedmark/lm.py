"""A tiny transparent word-level language model for offline experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .core import normalize_word, tokenize

DEFAULT_CORPUS = """
research systems can reveal statistical patterns in generated text
research experiments compare marked text with an independent null baseline
language models assign probabilities to several plausible next tokens
language models generate useful text by sampling from probability distributions
scientific evaluation measures detection power and false positive behavior
scientific reports should separate demonstration assumptions from production claims
watermarking changes token sampling while preserving plausible language choices
watermarking can create a statistical signal across a long generated sequence
statistical tests aggregate weak evidence across many token positions
token probabilities describe what a language model considers plausible
secret keys make pseudorandom token preferences difficult to predict externally
pseudorandom scores can be reproduced by a detector that knows the secret key
detection does not need the original language model probability distribution
longer text usually provides more statistical evidence than very short text
robust experiments report uncertainty limitations and reproducible random seeds
transparent code helps researchers inspect assumptions and challenge conclusions
interactive visualizations make probability shifts easier to understand
generated sequences can be compared with unmarked sequences under matched settings
"""


@dataclass(slots=True)
class ToyBigramLM:
    """Smoothed bigram model with a unigram backoff component.

    It exists only to provide an inspectable base distribution p(v|h). SeedMark's
    detector intentionally does not depend on this object.
    """

    corpus: str = DEFAULT_CORPUS
    local_weight: float = 0.82
    alpha: float = 0.08
    _vocab: list[str] = field(init=False, repr=False)
    _unigram: Counter[str] = field(init=False, repr=False)
    _bigram: dict[str, Counter[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.local_weight <= 1.0):
            raise ValueError("local_weight must be between 0 and 1")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        lines = [tokenize(line) for line in self.corpus.splitlines() if tokenize(line)]
        self._unigram = Counter(token for line in lines for token in line)
        bigram: dict[str, Counter[str]] = defaultdict(Counter)
        for line in lines:
            for left, right in zip(line, line[1:]):
                bigram[left][right] += 1
        self._bigram = dict(bigram)
        self._vocab = sorted(self._unigram)
        if len(self._vocab) < 2:
            raise ValueError("corpus must contain at least two unique words")

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(self._vocab)

    def recommended_starts(self) -> tuple[str, ...]:
        preferred = ("research", "language", "scientific", "watermarking", "statistical", "generated")
        return tuple(word for word in preferred if word in self._unigram)

    def next_probabilities(self, previous_token: str) -> dict[str, float]:
        previous = normalize_word(previous_token)
        total_unigram = sum(self._unigram.values()) + self.alpha * len(self._vocab)
        unigram_prob = {
            token: (self._unigram[token] + self.alpha) / total_unigram
            for token in self._vocab
        }

        local_counts = self._bigram.get(previous)
        if not local_counts:
            return unigram_prob
        local_total = sum(local_counts.values()) + self.alpha * len(self._vocab)
        local_prob = {
            token: (local_counts[token] + self.alpha) / local_total
            for token in self._vocab
        }
        mixed = {
            token: self.local_weight * local_prob[token]
            + (1.0 - self.local_weight) * unigram_prob[token]
            for token in self._vocab
        }
        normalizer = sum(mixed.values())
        return {token: prob / normalizer for token, prob in mixed.items()}

    def probability_table(self, previous_token: str, tokens: Iterable[str]) -> dict[str, float]:
        distribution = self.next_probabilities(previous_token)
        return {token: distribution[token] for token in tokens if token in distribution}
