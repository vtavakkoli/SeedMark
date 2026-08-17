"""Core keyed pseudorandom watermark and detector.

This module deliberately implements a small, inspectable teaching algorithm. It is
NOT an implementation of Anthropic's production watermark or Google SynthID-Text.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import re
from typing import Mapping

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_TWO64 = float(1 << 64)


@dataclass(frozen=True, slots=True)
class WatermarkConfig:
    """Configuration shared by generation and detection."""

    secret_key: str = "seedmark-demo-key"
    strength: float = 1.5
    top_k: int = 8
    threshold_z: float = 3.0

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("strength must be >= 0")
        if self.top_k < 2:
            raise ValueError("top_k must be >= 2")
        if self.threshold_z <= 0:
            raise ValueError("threshold_z must be > 0")
        if not self.secret_key:
            raise ValueError("secret_key must not be empty")


@dataclass(frozen=True, slots=True)
class DetectionResult:
    n_scored_tokens: int
    mean_score: float
    z_score: float
    p_value_one_sided: float
    threshold_z: float
    detected: bool


def normalize_word(value: str) -> str:
    """Return the first normalized word in *value*."""

    match = _WORD_RE.search(value.lower())
    if not match:
        raise ValueError("a seed word is required")
    return match.group(0)


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer used consistently throughout the demonstration."""

    return _WORD_RE.findall(text.lower())


def first_word_seed(first_word: str) -> bytes:
    """Create the public 256-bit seed from the normalized first word.

    The first word is intentionally visible. Ownership/authenticity comes from the
    separate secret key used by the keyed pseudorandom function.
    """

    normalized = normalize_word(first_word)
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def seed_hex(first_word: str) -> str:
    return first_word_seed(first_word).hex()


def token_score(secret_key: str, first_word: str, position: int, token: str) -> float:
    """Deterministically map (first-word seed, position, token) to U[0, 1).

    HMAC-SHA256 is used as a pseudorandom function. The detector can reproduce this
    number from the final text, first word, position, and secret key; it does not
    require the language model's next-token probabilities.
    """

    if position < 1:
        raise ValueError("position must be >= 1; position 0 is the public seed word")
    seed = first_word_seed(first_word)
    normalized_token = normalize_word(token)
    message = seed + position.to_bytes(8, "big") + normalized_token.encode("utf-8")
    digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / _TWO64


def top_k_distribution(probabilities: Mapping[str, float], k: int) -> dict[str, float]:
    """Keep the k highest-probability tokens and renormalize."""

    if k < 2:
        raise ValueError("k must be >= 2")
    positive = [(token, float(prob)) for token, prob in probabilities.items() if prob > 0.0]
    if len(positive) < 2:
        raise ValueError("distribution must contain at least two positive-probability tokens")
    selected = sorted(positive, key=lambda item: (-item[1], item[0]))[: min(k, len(positive))]
    total = sum(prob for _, prob in selected)
    return {token: prob / total for token, prob in selected}


def reweight_distribution(
    base_probabilities: Mapping[str, float],
    *,
    secret_key: str,
    first_word: str,
    position: int,
    strength: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Exponentially tilt candidate probabilities toward high keyed PRF scores.

    q(v) ∝ p(v) exp(strength * (2*u(v)-1))

    This deliberately simple tilting rule is easy to inspect but is *distortionary*;
    unlike production watermark systems, it makes no non-distortion guarantee.
    """

    if strength < 0:
        raise ValueError("strength must be >= 0")
    scores = {
        token: token_score(secret_key, first_word, position, token)
        for token in base_probabilities
    }
    weights = {
        token: float(prob) * math.exp(strength * (2.0 * scores[token] - 1.0))
        for token, prob in base_probabilities.items()
    }
    normalizer = sum(weights.values())
    if normalizer <= 0:
        raise ValueError("invalid probability distribution")
    return ({token: weight / normalizer for token, weight in weights.items()}, scores)


def detect_tokens(
    tokens: list[str],
    *,
    secret_key: str,
    first_word: str | None = None,
    threshold_z: float = 3.0,
) -> DetectionResult:
    """Detect keyed correlation in a token sequence without model probabilities.

    Under the teaching null hypothesis, selected tokens are independent of the
    secret PRF scores, so each observed score is approximately U[0,1). Therefore
    E[u]=1/2 and Var[u]=1/12. The returned z-score tests for an excess of high
    keyed scores.
    """

    if threshold_z <= 0:
        raise ValueError("threshold_z must be > 0")
    if len(tokens) < 2:
        return DetectionResult(0, 0.5, 0.0, 0.5, threshold_z, False)

    seed_word = normalize_word(first_word if first_word is not None else tokens[0])
    observed = [
        token_score(secret_key, seed_word, position, token)
        for position, token in enumerate(tokens[1:], start=1)
    ]
    n = len(observed)
    mean_score = sum(observed) / n
    z_score = (sum(observed) - 0.5 * n) / math.sqrt(n / 12.0)
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return DetectionResult(
        n_scored_tokens=n,
        mean_score=mean_score,
        z_score=z_score,
        p_value_one_sided=p_value,
        threshold_z=threshold_z,
        detected=z_score >= threshold_z,
    )
