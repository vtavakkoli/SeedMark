"""CLI contract tests that do not load model weights."""

from __future__ import annotations

import unittest

from seedmark.cli import DEFAULT_QWEN_DEMO_PROMPT, build_parser


EXPECTED_ARTICLE_PROMPT = (
    "Write a short plain-language article answering: What is AI? Explain what AI is, "
    "where it is used, benefits, risks, and conclude briefly."
)


class CLIDefaultTests(unittest.TestCase):
    def test_qwen_demo_uses_article_prompt_and_article_length(self) -> None:
        args = build_parser().parse_args(["qwen-demo"])
        self.assertEqual(DEFAULT_QWEN_DEMO_PROMPT, EXPECTED_ARTICLE_PROMPT)
        self.assertEqual(args.prompt, EXPECTED_ARTICLE_PROMPT)
        self.assertEqual(args.max_new_tokens, 128)

    def test_qwen_detect_uses_same_default_prompt(self) -> None:
        args = build_parser().parse_args(["qwen-detect", "--text", "example"])
        self.assertEqual(args.prompt, EXPECTED_ARTICLE_PROMPT)


if __name__ == "__main__":
    unittest.main()
