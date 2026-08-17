"""Generation and trace capture for SeedMark."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from .core import (
    DetectionResult,
    WatermarkConfig,
    detect_tokens,
    normalize_word,
    reweight_distribution,
    top_k_distribution,
)
from .lm import ToyBigramLM


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    token: str
    base_probability: float
    generation_probability: float
    watermark_score: float
    chosen: bool


@dataclass(frozen=True, slots=True)
class StepTrace:
    position: int
    context_token: str
    chosen_token: str
    chosen_base_probability: float
    chosen_generation_probability: float
    chosen_watermark_score: float
    cumulative_z: float
    candidates: tuple[CandidateTrace, ...]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    tokens: tuple[str, ...]
    text: str
    watermarked: bool
    first_word: str
    rng_seed: int
    detection: DetectionResult
    trace: tuple[StepTrace, ...]


def _sample(distribution: Mapping[str, float], rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    last = None
    for token, probability in distribution.items():
        cumulative += probability
        last = token
        if draw <= cumulative:
            return token
    assert last is not None
    return last


def generate_text(
    lm: ToyBigramLM,
    *,
    first_word: str = "research",
    length: int = 80,
    config: WatermarkConfig | None = None,
    watermarked: bool = True,
    rng_seed: int = 20260817,
) -> GenerationResult:
    """Generate a sequence and retain every probability/PRF decision."""

    if length < 2:
        raise ValueError("length must be >= 2")
    cfg = config or WatermarkConfig()
    seed_word = normalize_word(first_word)
    if seed_word not in lm.vocabulary:
        raise ValueError(f"first word {seed_word!r} is not in the toy model vocabulary")

    rng = random.Random(rng_seed)
    tokens = [seed_word]
    trace: list[StepTrace] = []
    selected_scores: list[float] = []

    for position in range(1, length):
        context_token = tokens[-1]
        full_base = lm.next_probabilities(context_token)
        base = top_k_distribution(full_base, cfg.top_k)
        marked, scores = reweight_distribution(
            base,
            secret_key=cfg.secret_key,
            first_word=seed_word,
            position=position,
            strength=cfg.strength,
        )
        generation_distribution = marked if watermarked else base
        chosen = _sample(generation_distribution, rng)
        chosen_score = scores[chosen]
        selected_scores.append(chosen_score)
        cumulative_n = len(selected_scores)
        cumulative_z = (
            sum(selected_scores) - 0.5 * cumulative_n
        ) / ((cumulative_n / 12.0) ** 0.5)

        ordered_candidates = sorted(base, key=lambda token: (-base[token], token))
        trace.append(
            StepTrace(
                position=position,
                context_token=context_token,
                chosen_token=chosen,
                chosen_base_probability=base[chosen],
                chosen_generation_probability=generation_distribution[chosen],
                chosen_watermark_score=chosen_score,
                cumulative_z=cumulative_z,
                candidates=tuple(
                    CandidateTrace(
                        token=token,
                        base_probability=base[token],
                        generation_probability=generation_distribution[token],
                        watermark_score=scores[token],
                        chosen=token == chosen,
                    )
                    for token in ordered_candidates
                ),
            )
        )
        tokens.append(chosen)

    detection = detect_tokens(
        tokens,
        secret_key=cfg.secret_key,
        first_word=seed_word,
        threshold_z=cfg.threshold_z,
    )
    return GenerationResult(
        tokens=tuple(tokens),
        text=" ".join(tokens),
        watermarked=watermarked,
        first_word=seed_word,
        rng_seed=rng_seed,
        detection=detection,
        trace=tuple(trace),
    )
