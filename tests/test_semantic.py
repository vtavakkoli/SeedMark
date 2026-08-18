from __future__ import annotations

import unittest

from seedmark.semantic import (
    SemanticBucketizer,
    SemanticContextTracker,
    detect_semantic_token_ids,
    is_sentence_boundary,
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


class NoBoundaryTokenizer:
    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        return "x"


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

    def test_tracker_rekeys_after_sentence_boundary(self):
        encoder = TinySemanticEncoder()
        bucketizer = SemanticBucketizer(secret_key="semantic-test", bucket_count=16)
        tracker = SemanticContextTracker(
            encoder=encoder,
            bucketizer=bucketizer,
            bootstrap_text="What is AI?",
        )
        bootstrap_bucket = tracker.current_key.bucket
        self.assertFalse(tracker.observe_token_piece("Benefits can help"))
        self.assertTrue(tracker.observe_token_piece(" people."))
        self.assertEqual(tracker.sentence_index, 1)
        self.assertEqual(tracker.next_token_offset, 1)
        self.assertNotEqual(tracker.current_key.bucket, bootstrap_bucket)

    def test_sentence_boundary_recognizes_common_endings(self):
        self.assertTrue(is_sentence_boundary("This is a sentence."))
        self.assertTrue(is_sentence_boundary('Is this complete?"'))
        self.assertTrue(is_sentence_boundary("Paragraph end\n\n"))
        self.assertFalse(is_sentence_boundary("still generating"))

    def test_detector_recovers_strong_semantic_keyed_signal(self):
        secret = "semantic-test-key"
        encoder = TinySemanticEncoder()
        bucketizer = SemanticBucketizer(secret_key=secret, bucket_count=16)
        bucket = bucketizer.key_for_text("What is AI?", encoder).bucket

        token_ids = []
        candidates = range(100, 132)
        for offset in range(1, 65):
            chosen = max(
                candidates,
                key=lambda token_id: semantic_token_score(
                    secret, bucket, offset, token_id
                ),
            )
            token_ids.append(chosen)

        result = detect_semantic_token_ids(
            token_ids,
            tokenizer=NoBoundaryTokenizer(),
            encoder=encoder,
            secret_key=secret,
            bootstrap_text="What is AI?",
            threshold_z=3.0,
            bucket_count=16,
        )
        self.assertEqual(result.n_scored_tokens, 64)
        self.assertTrue(result.detected)
        self.assertGreater(result.z_score, 3.0)


if __name__ == "__main__":
    unittest.main()
