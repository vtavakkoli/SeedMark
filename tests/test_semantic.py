from __future__ import annotations

import unittest

from seedmark.semantic import (
    SemanticBucketizer,
    SemanticContextTracker,
    detect_semantic_token_ids,
    is_paragraph_boundary,
    semantic_token_score,
)


class TinySemanticEncoder:
    model_name = "tiny-test-semantic-encoder"

    def encode(self, text: str):
        text = text.lower()
        if "artificial intelligence" in text or " ai" in f" {text}":
            return (0.95, 0.10, 0.05, 0.01)
        if "benefit" in text or "help" in text or "assist" in text:
            return (0.10, 0.95, 0.03, 0.02)
        if "risk" in text or "harm" in text:
            return (0.04, 0.10, 0.95, 0.02)
        return (0.25, 0.25, 0.25, 0.25)


class FlatTokenizer:
    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        if len(token_ids) == 1 and int(token_ids[0]) == 999:
            return "\n\n"
        return "x" * len(token_ids)


class SemanticWatermarkTests(unittest.TestCase):
    def test_score_is_deterministic_and_domain_sensitive(self):
        first = semantic_token_score("secret", 3, 7, 101)
        self.assertEqual(first, semantic_token_score("secret", 3, 7, 101))
        self.assertNotEqual(first, semantic_token_score("secret", 4, 7, 101))
        self.assertNotEqual(first, semantic_token_score("secret", 3, 8, 101))
        self.assertNotEqual(first, semantic_token_score("secret", 3, 7, 102))

    def test_semantically_equivalent_phrases_share_bucket_with_test_encoder(self):
        encoder = TinySemanticEncoder()
        bucketizer = SemanticBucketizer(secret_key="semantic-test", bucket_count=16)
        left = bucketizer.key_for_text("AI helps people with routine work.", encoder)
        right = bucketizer.key_for_text(
            "Artificial intelligence assists people with routine work.", encoder
        )
        self.assertEqual(left.bucket, right.bucket)

    def test_tracker_rekeys_after_paragraph_not_sentence(self):
        encoder = TinySemanticEncoder()
        bucketizer = SemanticBucketizer(secret_key="semantic-test", bucket_count=16)
        tracker = SemanticContextTracker(
            encoder=encoder,
            bucketizer=bucketizer,
            bootstrap_text="What is AI?",
        )
        bootstrap_bucket = tracker.current_key.bucket
        self.assertFalse(tracker.observe_token(100, "Benefits can help people."))
        self.assertEqual(tracker.paragraph_index, 0)
        self.assertTrue(tracker.observe_token(999, "\n\n"))
        self.assertEqual(tracker.paragraph_index, 1)
        self.assertEqual(tracker.next_token_occurrence(100), 1)
        self.assertNotEqual(tracker.current_key.bucket, bootstrap_bucket)

    def test_paragraph_boundary_requires_blank_line(self):
        self.assertFalse(is_paragraph_boundary("This is a sentence."))
        self.assertFalse(is_paragraph_boundary("One line\n"))
        self.assertTrue(is_paragraph_boundary("Paragraph end\n\n"))
        self.assertTrue(is_paragraph_boundary("Paragraph end\r\n\r\n"))

    def test_answer_detector_recovers_strong_complete_answer_signal(self):
        secret = "semantic-test-key"
        encoder = TinySemanticEncoder()
        tokenizer = FlatTokenizer()
        bucketizer = SemanticBucketizer(secret_key=secret, bucket_count=16)
        bucket = bucketizer.key_for_text("x" * 64, encoder).bucket

        token_ids = []
        occurrences: dict[int, int] = {}
        candidates = range(100, 132)
        for _ in range(64):
            chosen = max(
                candidates,
                key=lambda token_id: semantic_token_score(
                    secret,
                    bucket,
                    occurrences.get(token_id, 0) + 1,
                    token_id,
                ),
            )
            occurrences[chosen] = occurrences.get(chosen, 0) + 1
            token_ids.append(chosen)

        result = detect_semantic_token_ids(
            token_ids,
            tokenizer=tokenizer,
            encoder=encoder,
            secret_key=secret,
            bootstrap_text="What is AI?",
            threshold_z=3.0,
            bucket_count=16,
            semantic_scope="answer",
        )
        self.assertEqual(result.n_scored_tokens, 64)
        self.assertTrue(result.detected)
        self.assertGreater(result.z_score, 3.0)

    def test_unrelated_token_does_not_change_occurrence_domain(self):
        secret = "semantic-test-key"
        bucket = 3
        score_before = semantic_token_score(secret, bucket, 2, 101)
        _ = semantic_token_score(secret, bucket, 1, 202)
        score_after = semantic_token_score(secret, bucket, 2, 101)
        self.assertEqual(score_before, score_after)


if __name__ == "__main__":
    unittest.main()
