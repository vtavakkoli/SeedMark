"""Tests for the dependency-light animation helpers."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from seedmark.animation import sentence_parts


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

    def test_sentence_parts_rejects_invalid_step(self) -> None:
        result = SimpleNamespace(prompt="Research is", trace=())
        with self.assertRaises(IndexError):
            sentence_parts(result, 0)


if __name__ == "__main__":
    unittest.main()
