"""SeedMark: a transparent educational text-watermarking prototype."""

from ._version import __version__
from .core import DetectionResult, WatermarkConfig, detect_tokens, first_word_seed, token_score
from .generation import GenerationResult, generate_text
from .lm import ToyBigramLM
from .semantic import (
    DEFAULT_SEMANTIC_MODEL,
    HFMeanPoolingSemanticEncoder,
    SemanticBucketizer,
    SemanticContextTracker,
    SemanticKey,
    SemanticWatermarkConfig,
    detect_semantic_token_ids,
    semantic_token_score,
)

__all__ = [
    "__version__",
    "DEFAULT_SEMANTIC_MODEL",
    "DetectionResult",
    "GenerationResult",
    "HFMeanPoolingSemanticEncoder",
    "SemanticBucketizer",
    "SemanticContextTracker",
    "SemanticKey",
    "SemanticWatermarkConfig",
    "ToyBigramLM",
    "WatermarkConfig",
    "detect_semantic_token_ids",
    "detect_tokens",
    "first_word_seed",
    "generate_text",
    "semantic_token_score",
    "token_score",
]
