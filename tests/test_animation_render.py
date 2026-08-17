"""Render a tiny GIF without loading a language model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from seedmark.animation import write_generation_gif


class AnimationRenderTests(unittest.TestCase):
    def test_writes_multiframe_gif(self) -> None:
        from PIL import Image

        def candidate(token_id: int, text: str, base: float, marked: float, score: float, chosen: bool = False):
            return SimpleNamespace(
                token_id=token_id,
                token_text=text,
                base_probability=base,
                generation_probability=marked,
                watermark_score=score,
                chosen=chosen,
            )

        trace = (
            SimpleNamespace(
                position=1,
                chosen_token_id=11,
                chosen_token_text=" useful",
                chosen_base_probability=0.30,
                chosen_generation_probability=0.43,
                chosen_watermark_score=0.91,
                cumulative_z=1.42,
                candidates=(
                    candidate(11, " useful", 0.30, 0.43, 0.91, True),
                    candidate(12, " important", 0.27, 0.20, 0.22),
                    candidate(13, " complex", 0.18, 0.19, 0.61),
                ),
            ),
            SimpleNamespace(
                position=2,
                chosen_token_id=21,
                chosen_token_text=" because",
                chosen_base_probability=0.26,
                chosen_generation_probability=0.39,
                chosen_watermark_score=0.88,
                cumulative_z=2.17,
                candidates=(
                    candidate(21, " because", 0.26, 0.39, 0.88, True),
                    candidate(22, " when", 0.24, 0.18, 0.19),
                    candidate(23, " and", 0.17, 0.21, 0.70),
                ),
            ),
        )
        result = SimpleNamespace(
            model_name="Qwen/test",
            prompt="Research is",
            trace=trace,
            top_k=3,
            strength=1.5,
            detection=SimpleNamespace(threshold_z=3.0),
        )

        with TemporaryDirectory() as temp:
            path = write_generation_gif(Path(temp) / "generation.gif", result, frame_ms=120, width=900, height=620)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            with Image.open(path) as image:
                self.assertEqual(image.format, "GIF")
                self.assertGreaterEqual(getattr(image, "n_frames", 1), 2)


if __name__ == "__main__":
    unittest.main()
