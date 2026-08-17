from __future__ import annotations

import unittest

from seedmark import ToyBigramLM, WatermarkConfig, detect_tokens, first_word_seed, generate_text, token_score
from seedmark.core import reweight_distribution


class SeedMarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lm = ToyBigramLM()
        self.config = WatermarkConfig(secret_key="unit-test-key", strength=1.5, top_k=8, threshold_z=3.0)

    def test_first_word_seed_is_reproducible_and_word_sensitive(self) -> None:
        self.assertEqual(first_word_seed("Research"), first_word_seed("research"))
        self.assertNotEqual(first_word_seed("research"), first_word_seed("language"))

    def test_keyed_score_is_reproducible_and_bounded(self) -> None:
        a = token_score("k", "research", 7, "language")
        self.assertEqual(a, token_score("k", "research", 7, "language"))
        self.assertGreaterEqual(a, 0.0)
        self.assertLess(a, 1.0)
        self.assertNotEqual(a, token_score("other", "research", 7, "language"))

    def test_reweighting_normalizes_and_favors_scores(self) -> None:
        marked, scores = reweight_distribution({"alpha": 0.5, "beta": 0.5}, secret_key="k", first_word="research", position=1, strength=2.0)
        self.assertAlmostEqual(sum(marked.values()), 1.0, places=12)
        self.assertGreater(marked[max(scores, key=scores.get)], marked[min(scores, key=scores.get)])

    def test_detector_uses_no_language_model_distribution(self) -> None:
        marked = generate_text(self.lm, first_word="research", length=140, config=self.config, watermarked=True, rng_seed=42)
        result = detect_tokens(list(marked.tokens), secret_key=self.config.secret_key, first_word=marked.first_word, threshold_z=3.0)
        self.assertGreater(result.z_score, 3.0)
        self.assertTrue(result.detected)

    def test_marked_scores_above_matched_unmarked_control(self) -> None:
        marked = generate_text(self.lm, first_word="research", length=120, config=self.config, watermarked=True, rng_seed=123)
        unmarked = generate_text(self.lm, first_word="research", length=120, config=self.config, watermarked=False, rng_seed=123)
        self.assertGreater(marked.detection.z_score, unmarked.detection.z_score + 2.0)

    def test_trace_contains_probabilities_scores_and_one_choice(self) -> None:
        result = generate_text(self.lm, first_word="research", length=12, config=self.config, watermarked=True, rng_seed=9)
        self.assertEqual(len(result.trace), 11)
        for step in result.trace:
            self.assertAlmostEqual(sum(c.base_probability for c in step.candidates), 1.0, places=12)
            self.assertAlmostEqual(sum(c.generation_probability for c in step.candidates), 1.0, places=12)
            self.assertEqual(sum(c.chosen for c in step.candidates), 1)


if __name__ == "__main__":
    unittest.main()
