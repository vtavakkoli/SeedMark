"""SeedMark: a transparent educational text-watermarking prototype."""

from ._version import __version__
from .chat_llm import ChatLLMSeedMark
from .core import DetectionResult, WatermarkConfig, detect_tokens, first_word_seed, token_score
from .generation import GenerationResult, generate_text
from .hf_llm import LLMSeedMark
from .lm import ToyBigramLM
from .semantic import (
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SEMANTIC_SCOPE,
    SEMANTIC_SCOPES,
    HFMeanPoolingSemanticEncoder,
    SemanticBucketizer,
    SemanticContextTracker,
    SemanticKey,
    SemanticWatermarkConfig,
    detect_semantic_token_ids,
    is_paragraph_boundary,
    semantic_token_score,
)
from .semantic_llm import SemanticChatLLMSeedMark

__all__ = [
    "__version__",
    "DEFAULT_SEMANTIC_MODEL",
    "DEFAULT_SEMANTIC_SCOPE",
    "SEMANTIC_SCOPES",
    "ChatLLMSeedMark",
    "DetectionResult",
    "GenerationResult",
    "HFMeanPoolingSemanticEncoder",
    "LLMSeedMark",
    "SemanticBucketizer",
    "SemanticChatLLMSeedMark",
    "SemanticContextTracker",
    "SemanticKey",
    "SemanticWatermarkConfig",
    "ToyBigramLM",
    "WatermarkConfig",
    "detect_semantic_token_ids",
    "detect_tokens",
    "first_word_seed",
    "generate_text",
    "is_paragraph_boundary",
    "semantic_token_score",
    "token_score",
]
