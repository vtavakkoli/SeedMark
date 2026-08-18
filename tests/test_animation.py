"""Tests for dependency-light animation context helpers."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from seedmark.animation import recent_sentence_parts, sentence_parts


class AnimationTests(unittest.TestCase):
    def test_sentence_parts_highlights_only_current_token(self) -> None:
        result = SimpleNamespace(
            prompt="Research is",
            trace=(
                SimpleNamespace(chosen_token_text=" useful"),
                SimpleNamespace(chosen_token_text=" because"),
                SimpleNamespace(chosen_token_text=" evidence"),
            ),
        )

        prefix, current = sentence_parts(result, 1)

        self.assertEqual(prefix, "Research is useful")
        self.assertEqual(current, " because")
        self.assertNotIn(" because", prefix)

    def test_recent_context_slides_and_keeps_current_token_separate(self) -> None:
        result = SimpleNamespace(
            prompt="Write a short article about artificial intelligence.",
            trace=tuple(
                SimpleNamespace(chosen_token_text=f" token-{index}")
                for index in range(30)
            ),
        )

        prefix, current = recent_sentence_parts(result, 29, max_chars=90)

        self.assertTrue(prefix.startswith("… "))
        self.assertLessEqual(len(prefix), 94)
        self.assertEqual(current, " token-29")
        self.assertNotIn("token-29", prefix)
        self.assertIn("token-28", prefix)

    def test_recent_context_collapses_historical_newlines(self) -> None:
        result = SimpleNamespace(
            prompt="Article:\n",
            trace=(
                SimpleNamespace(chosen_token_text="first line\n"),
                SimpleNamespace(chosen_token_text="second line\n"),
                SimpleNamespace(chosen_token_text="third"),
            ),
        )
        prefix, current = recent_sentence_parts(result, 2, max_chars=80)
        self.assertNotIn("\n", prefix)
        self.assertEqual(current, "third")

    def test_sentence_parts_rejects_invalid_step(self) -> None:
        result = SimpleNamespace(prompt="Research is", trace=())
        with self.assertRaises(IndexError):
            sentence_parts(result, 0)


if __name__ == "__main__":
    unittest.main()
