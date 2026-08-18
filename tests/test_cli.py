"""CLI contract tests that do not load model weights."""

from __future__ import annotations

import unittest

from seedmark.chat_llm import DEFAULT_CHAT_QUESTION, DEFAULT_CHAT_SYSTEM_PROMPT
from seedmark.cli import DEFAULT_QWEN_DEMO_PROMPT, build_parser


class CLIDefaultTests(unittest.TestCase):
    def test_qwen_demo_uses_real_chat_question(self) -> None:
        args = build_parser().parse_args(["qwen-demo"])
        self.assertEqual(DEFAULT_QWEN_DEMO_PROMPT, "What is AI?")
        self.assertEqual(DEFAULT_CHAT_QUESTION, "What is AI?")
        self.assertEqual(args.question, "What is AI?")
        self.assertEqual(args.system_prompt, DEFAULT_CHAT_SYSTEM_PROMPT)
        self.assertEqual(args.max_new_tokens, 128)

    def test_prompt_alias_maps_to_user_question(self) -> None:
        args = build_parser().parse_args(["qwen-demo", "--prompt", "What is edge AI?"])
        self.assertEqual(args.question, "What is edge AI?")

    def test_qwen_detect_uses_same_chat_defaults(self) -> None:
        args = build_parser().parse_args(["qwen-detect", "--text", "example"])
        self.assertEqual(args.question, "What is AI?")
        self.assertEqual(args.system_prompt, DEFAULT_CHAT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
