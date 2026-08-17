"""Tests for the standalone real-LLM scientific report."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from seedmark.core import DetectionResult
from seedmark.hf_llm import HFCandidateTrace, HFGenerationResult, HFStepTrace
from seedmark.reporting import write_qwen_report


def sample_result(*, watermarked: bool, score: float, z: float) -> HFGenerationResult:
    candidate = HFCandidateTrace(
        token_id=42,
        token_text=" useful",
        base_probability=0.30,
        generation_probability=0.42 if watermarked else 0.30,
        watermark_score=score,
        chosen=True,
    )
    step = HFStepTrace(
        position=1,
        chosen_token_id=42,
        chosen_token_text=" useful",
        chosen_base_probability=0.30,
        chosen_generation_probability=0.42 if watermarked else 0.30,
        chosen_watermark_score=score,
        cumulative_z=z,
        candidates=(candidate,),
    )
    detection = DetectionResult(
        n_scored_tokens=1,
        mean_score=score,
        z_score=z,
        p_value_one_sided=0.001 if z >= 3 else 0.5,
        threshold_z=3.0,
        detected=z >= 3,
    )
    return HFGenerationResult(
        model_name="Qwen/test",
        prompt="Research is",
        first_word="research",
        text="Research is useful",
        continuation=" useful",
        prompt_token_ids=(1, 2),
        generated_token_ids=(42,),
        watermarked=watermarked,
        temperature=1.0,
        top_k=20,
        strength=1.5,
        rng_seed=20260817,
        detection=detection,
        trace=(step,),
    )


class ReportTests(unittest.TestCase):
    def test_report_embeds_visual_assets_and_scientific_caveat(self) -> None:
        marked = sample_result(watermarked=True, score=0.91, z=3.2)
        control = sample_result(watermarked=False, score=0.48, z=-0.1)
        assets = {
            "generation_gif": "generation.gif",
            "detection_gif": "detection.gif",
            "generation_preview": "generation-preview.png",
            "detection_preview": "detection-preview.png",
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            report = write_qwen_report(root, marked, control, assets=assets)
            html = report.read_text(encoding="utf-8")
            self.assertIn('src="generation.gif"', html)
            self.assertIn('src="detection.gif"', html)
            self.assertIn("Interactive token microscope", html)
            self.assertIn("not a posterior probability", html)
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["visual_assets"], assets)
            self.assertEqual(summary["decision_threshold"]["z"], 3.0)

    def test_report_without_gifs_has_clear_placeholder(self) -> None:
        marked = sample_result(watermarked=True, score=0.70, z=1.0)
        control = sample_result(watermarked=False, score=0.48, z=-0.1)
        with TemporaryDirectory() as temp:
            report = write_qwen_report(Path(temp), marked, control, assets={})
            html = report.read_text(encoding="utf-8")
            self.assertIn("Animation disabled for this run", html)


if __name__ == "__main__":
    unittest.main()
