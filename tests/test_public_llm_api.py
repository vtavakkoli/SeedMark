"""Public API tests for model-agnostic Hugging Face class names."""

import unittest

from seedmark import ChatLLMSeedMark, LLMSeedMark, SemanticChatLLMSeedMark
from seedmark.chat_llm import ChatQwenSeedMark
from seedmark.hf_llm import QwenSeedMark
from seedmark.semantic_chat import SemanticChatQwenSeedMark


class PublicLLMApiTests(unittest.TestCase):
    def test_preferred_class_names_are_model_agnostic(self) -> None:
        self.assertEqual(LLMSeedMark.__name__, "LLMSeedMark")
        self.assertEqual(ChatLLMSeedMark.__name__, "ChatLLMSeedMark")
        self.assertEqual(SemanticChatLLMSeedMark.__name__, "SemanticChatLLMSeedMark")

    def test_legacy_qwen_names_remain_compatible(self) -> None:
        self.assertIs(QwenSeedMark, LLMSeedMark)
        self.assertIs(ChatQwenSeedMark, ChatLLMSeedMark)
        self.assertTrue(issubclass(SemanticChatLLMSeedMark, SemanticChatQwenSeedMark))


if __name__ == "__main__":
    unittest.main()
