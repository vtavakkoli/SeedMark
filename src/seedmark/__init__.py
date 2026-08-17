"""SeedMark: a transparent educational text-watermarking prototype."""

from ._version import __version__
from .core import DetectionResult, WatermarkConfig, detect_tokens, first_word_seed, token_score
from .generation import GenerationResult, generate_text
from .lm import ToyBigramLM

__all__ = [
    "__version__",
    "DetectionResult",
    "GenerationResult",
    "ToyBigramLM",
    "WatermarkConfig",
    "detect_tokens",
    "first_word_seed",
    "generate_text",
    "token_score",
]
