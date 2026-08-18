"""Tests for the chat-specific report wrapper."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

from seedmark.chat_reporting import write_chat_report


def sample_result(*, detected: bool, watermarked: bool):
    detection = SimpleNamespace(
        n_scored_tokens=2,
        mean_score=0.7 if detected else 0.48,
        z_score=3.2 if detected else -0.1,
        p_value_one_sided=0.0007 if detected else 0.54,
        threshold_z=3.0,
        detected=detected,
    )
    step = SimpleNamespace(
        position=1,
        chosen_token_id=42,
        chosen_token_text=" Artificial",
        chosen_base_probability=0.30,
        chosen_generation_probability=0.42 if watermarked else 0.30,
        chosen_watermark_score=0.91 if detected else 0.48,
        cumulative_z=3.2 if detected else -0.1,
        candidates=(),
    )
    return SimpleNamespace(
        model_name="Qwen/test",
        prompt="User: What is AI?\nAssistant:",
        first_word="what",
        text="Artificial intelligence is technology that enables computers to perform useful intelligent tasks.",
        continuation="Artificial intelligence is technology that enables computers to perform useful intelligent tasks.",
        prompt_token_ids=(1, 2),
        generated_token_ids=(42, 43),
        watermarked=watermarked,
        temperature=1.0,
        top_k=20,
        strength=1.5,
        rng_seed=20260817,
        detection=detection,
        trace=(step,),
    )


class ChatReportTests(unittest.TestCase):
    def test_report_looks_like_user_to_assistant_chat(self) -> None:
        marked = sample_result(detected=True, watermarked=True)
        control = sample_result(detected=False, watermarked=False)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = write_chat_report(
                root,
                marked,
                control,
                question="What is AI?",
                system_prompt="Answer as a short article.",
                assets={},
            )
            document = path.read_text(encoding="utf-8")
            self.assertIn("One chat question. Two assistant answers", document)
            self.assertIn("User asks the AI", document)
            self.assertIn("What is AI?", document)
            self.assertIn("Assistant · watermarked", document)
            self.assertIn("Assistant · control", document)

            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode"], "chat")
            self.assertEqual(summary["chat"]["question"], "What is AI?")
            self.assertTrue(summary["chat"]["native_chat_template"])
            self.assertTrue(summary["chat"]["assistant_only_outputs"])


if __name__ == "__main__":
    unittest.main()
