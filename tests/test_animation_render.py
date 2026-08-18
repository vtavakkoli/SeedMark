"""Smoke tests for the Pillow-based scientific visual assets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import importlib.util
import unittest

from seedmark.animation import write_detection_gif, write_generation_gif, write_visual_assets


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
ARTICLE_PROMPT = (
    "Write a short plain-language article answering: What is AI? Explain what AI is, "
    "where it is used, benefits, risks, and conclude briefly."
)


def candidate(token_id: int, text: str, base: float, marked: float, score: float, chosen: bool = False):
    return SimpleNamespace(
        token_id=token_id,
        token_text=text,
        base_probability=base,
        generation_probability=marked,
        watermark_score=score,
        chosen=chosen,
    )


def sample_results():
    marked_steps = []
    control_steps = []
    marked_scores = (0.91, 0.82, 0.78, 0.84, 0.76)
    control_scores = (0.22, 0.63, 0.41, 0.52, 0.44)
    tokens = (" Artificial", " intelligence", " helps", " people", " today.")
    for position, (token, marked_u, control_u) in enumerate(zip(tokens, marked_scores, control_scores), start=1):
        candidates = (
            candidate(10 + position, token, 0.30, 0.43, marked_u, True),
            candidate(20 + position, " alternative", 0.27, 0.20, 0.22),
            candidate(30 + position, " option", 0.18, 0.19, 0.68),
        )
        m_scores = marked_scores[:position]
        c_scores = control_scores[:position]
        m_z = (sum(m_scores) - 0.5 * position) / (position / 12.0) ** 0.5
        c_z = (sum(c_scores) - 0.5 * position) / (position / 12.0) ** 0.5
        marked_steps.append(SimpleNamespace(
            position=position,
            chosen_token_id=10 + position,
            chosen_token_text=token,
            chosen_base_probability=0.30,
            chosen_generation_probability=0.43,
            chosen_watermark_score=marked_u,
            cumulative_z=m_z,
            candidates=candidates,
        ))
        control_steps.append(SimpleNamespace(
            position=position,
            chosen_token_id=10 + position,
            chosen_token_text=token,
            chosen_base_probability=0.30,
            chosen_generation_probability=0.30,
            chosen_watermark_score=control_u,
            cumulative_z=c_z,
            candidates=candidates,
        ))

    common = dict(model_name="Qwen/test", prompt=ARTICLE_PROMPT, top_k=3, strength=1.5)
    marked = SimpleNamespace(
        **common,
        trace=tuple(marked_steps),
        detection=SimpleNamespace(threshold_z=3.0),
    )
    control = SimpleNamespace(
        **common,
        trace=tuple(control_steps),
        detection=SimpleNamespace(threshold_z=3.0),
    )
    return marked, control


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is installed only for the animation smoke job")
class AnimationRenderTests(unittest.TestCase):
    def test_writes_both_multiframe_gifs_and_previews(self) -> None:
        from PIL import Image

        marked, control = sample_results()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            assets = write_visual_assets(root, marked, control, frame_ms=120, width=900, height=900)
            self.assertEqual(
                set(assets),
                {"generation_gif", "detection_gif", "generation_preview", "detection_preview"},
            )
            for key, name in assets.items():
                path = root / name
                self.assertTrue(path.exists(), key)
                self.assertGreater(path.stat().st_size, 1000, key)
                with Image.open(path) as image:
                    if name.endswith(".gif"):
                        self.assertEqual(image.format, "GIF")
                        self.assertEqual(getattr(image, "n_frames", 1), len(marked.trace))
                    else:
                        self.assertEqual(image.format, "PNG")

    def test_individual_gif_writers_are_valid_with_long_article_prompt(self) -> None:
        from PIL import Image

        marked, control = sample_results()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            generation = write_generation_gif(root / "generation.gif", marked, frame_ms=120, width=900, height=900)
            detection = write_detection_gif(root / "detection.gif", marked, control, frame_ms=120, width=900, height=900)
            for path in (generation, detection):
                with Image.open(path) as image:
                    self.assertGreaterEqual(image.n_frames, 2)
                    self.assertEqual(image.size, (900, 900))


if __name__ == "__main__":
    unittest.main()
