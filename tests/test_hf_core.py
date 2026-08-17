"""Dependency-free tests for the real-LLM watermark mathematics."""

import unittest

from seedmark.hf_llm import detect_token_ids, token_id_score


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


if __name__ == "__main__":
    unittest.main()
