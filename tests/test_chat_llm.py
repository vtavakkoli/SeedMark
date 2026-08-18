"""Dependency-light tests for the chat-first Qwen demo helpers."""

from __future__ import annotations

import unittest

from seedmark.chat_llm import (
    DEFAULT_CHAT_QUESTION,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    chat_messages,
    render_chat_prompt,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.last_messages = None
        self.last_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return "<system>article</system><user>What is AI?</user><assistant>"


class ChatPromptTests(unittest.TestCase):
    def test_default_conversation_is_system_plus_user_question(self) -> None:
        messages = chat_messages(DEFAULT_CHAT_QUESTION)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DEFAULT_CHAT_SYSTEM_PROMPT)
        self.assertEqual(messages[1], {"role": "user", "content": "What is AI?"})

    def test_native_chat_template_adds_assistant_generation_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        rendered = render_chat_prompt(tokenizer, "What is AI?")
        self.assertTrue(rendered.endswith("<assistant>"))
        self.assertEqual(tokenizer.last_messages[1]["content"], "What is AI?")
        self.assertFalse(tokenizer.last_kwargs["tokenize"])
        self.assertTrue(tokenizer.last_kwargs["add_generation_prompt"])
        self.assertFalse(tokenizer.last_kwargs["enable_thinking"])

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chat_messages("   ")


if __name__ == "__main__":
    unittest.main()
