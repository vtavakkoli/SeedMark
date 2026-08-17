"""SeedMark: a transparent educational text-watermarking prototype."""

from .core import DetectionResult, WatermarkConfig, detect_tokens, first_word_seed, token_score
from .generation import GenerationResult, generate_text
from .lm import ToyBigramLM

__all__ = [
    "DetectionResult",
    "GenerationResult",
    "ToyBigramLM",
    "WatermarkConfig",
    "detect_tokens",
    "first_word_seed",
    "generate_text",
    "token_score",
]

__version__ = "0.1.0"
