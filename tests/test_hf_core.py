"""Dependency-free tests for the real-LLM watermark mathematics and loader policy."""

import unittest

from seedmark.hf_llm import (
    LLMSeedMark,
    QwenSeedMark,
    _load_hf_model,
    detect_token_ids,
    token_id_score,
)


class _CausalLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls.append((model_name, kwargs))
        return ("causal", model_name)


class _UnsupportedCausalLoader:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        raise ValueError("unsupported configuration")


class _MultimodalLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls.append((model_name, kwargs))
        return ("multimodal", model_name)


class TestHFTokenWatermark(unittest.TestCase):
    def test_token_id_score_is_reproducible_and_keyed(self) -> None:
        a = token_id_score("key-a", "Research", 3, 1234)
        self.assertEqual(a, token_id_score("key-a", "research", 3, 1234))
        self.assertNotEqual(a, token_id_score("key-b", "research", 3, 1234))
        self.assertNotEqual(a, token_id_score("key-a", "research", 4, 1234))

    def test_detector_requires_no_model_distribution(self) -> None:
        ids = list(range(10, 80))
        result = detect_token_ids(ids, secret_key="demo", first_word="research")
        self.assertEqual(result.n_scored_tokens, len(ids))
        self.assertTrue(0.0 <= result.mean_score < 1.0)

    def test_standard_causal_loader_is_preferred(self) -> None:
        loaded = _load_hf_model(
            _CausalLoader,
            _MultimodalLoader,
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype="auto",
        )
        self.assertEqual(loaded[0], "causal")

    def test_multimodal_loader_is_a_fallback(self) -> None:
        loaded = _load_hf_model(
            _UnsupportedCausalLoader,
            _MultimodalLoader,
            "Qwen/Qwen3.5-0.8B",
            torch_dtype="auto",
        )
        self.assertEqual(loaded[0], "multimodal")

    def test_qwen_class_name_remains_backward_compatible(self) -> None:
        self.assertIs(QwenSeedMark, LLMSeedMark)


if __name__ == "__main__":
    unittest.main()
