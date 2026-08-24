"""Model-agnostic semantic chat API for Hugging Face SeedMark generation.

The implementation lives in :mod:`seedmark.semantic_chat` for backward
compatibility. This module exposes the preferred public class name without tying
the API to the repository's historical Qwen default model.
"""

from __future__ import annotations

from .semantic_chat import (
    SemanticChatQwenSeedMark,
    SemanticGenerationResult,
    SemanticCandidateTrace,
    SemanticStepTrace,
    detect_semantic_chat_text_with_tokenizer,
)


class SemanticChatLLMSeedMark(SemanticChatQwenSeedMark):
    """Generate semantic SeedMark chat answers with a Hugging Face LLM.

    Standard text checkpoints are loaded through ``AutoModelForCausalLM`` by the
    shared backend. Compatible multimodal checkpoints retain the existing loader
    fallback. The old ``SemanticChatQwenSeedMark`` name remains available for
    backward compatibility.
    """


__all__ = [
    "SemanticCandidateTrace",
    "SemanticChatLLMSeedMark",
    "SemanticGenerationResult",
    "SemanticStepTrace",
    "detect_semantic_chat_text_with_tokenizer",
]
