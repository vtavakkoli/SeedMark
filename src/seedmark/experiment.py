"""Reproducible SeedMark benchmark and report generation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from .core import WatermarkConfig, seed_hex
from .generation import GenerationResult, generate_text
from .lm import ToyBigramLM
from .report import histogram_svg, roc_svg, write_interactive_report


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def empirical_auc(marked: list[float], null: list[float]) -> float:
    wins = 0.0
    for positive in marked:
        for negative in null:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(marked) * len(null))


def roc_points(marked: list[float], null: list[float]) -> list[dict[str, float]]:
    values = sorted(set(marked + null), reverse=True)
    thresholds = [float("inf"), *values, float("-inf")]
    points = []
    for threshold in thresholds:
        tpr = sum(value >= threshold for value in marked) / len(marked)
        fpr = sum(value >= threshold for value in null) / len(null)
        points.append({"threshold": threshold, "tpr": tpr, "fpr": fpr})
    return sorted(points, key=lambda item: (item["fpr"], item["tpr"]))


def generation_to_trace(result: GenerationResult) -> list[dict[str, object]]:
    return [
        {
            "position": step.position,
            "context_token": step.context_token,
            "chosen_token": step.chosen_token,
            "chosen_base_probability": step.chosen_base_probability,
            "chosen_generation_probability": step.chosen_generation_probability,
            "chosen_watermark_score": step.chosen_watermark_score,
            "cumulative_z": step.cumulative_z,
            "candidates": [asdict(candidate) for candidate in step.candidates],
        }
        for step in result.trace
    ]


def write_trace_csv(path: Path, result: GenerationResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "position",
            "context_token",
            "chosen_token",
            "base_probability",
            "generation_probability",
            "watermark_score",
            "cumulative_z",
        ])
        for step in result.trace:
            writer.writerow([
                step.position,
                step.context_token,
                step.chosen_token,
                f"{step.chosen_base_probability:.12f}",
                f"{step.chosen_generation_probability:.12f}",
                f"{step.chosen_watermark_score:.12f}",
                f"{step.cumulative_z:.12f}",
            ])


def run_experiment(
    output_dir: Path,
    *,
    trials: int = 300,
    length: int = 80,
    first_word: str = "research",
    experiment_seed: int = 20260817,
    config: WatermarkConfig | None = None,
) -> dict[str, object]:
    if trials < 20:
        raise ValueError("trials must be >= 20 for the benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or WatermarkConfig()
    lm = ToyBigramLM()

    marked_example = generate_text(
        lm,
        first_word=first_word,
        length=length,
        config=cfg,
        watermarked=True,
        rng_seed=experiment_seed,
    )
    null_example = generate_text(
        lm,
        first_word=first_word,
        length=length,
        config=cfg,
        watermarked=False,
        rng_seed=experiment_seed,
    )

    starts = lm.recommended_starts() or (first_word,)
    marked_scores: list[float] = []
    null_scores: list[float] = []
    trial_rows: list[tuple[int, str, float, float, bool, bool]] = []
    for trial in range(trials):
        start = starts[trial % len(starts)]
        rng_seed = experiment_seed + 1000 + trial
        marked = generate_text(
            lm,
            first_word=start,
            length=length,
            config=cfg,
            watermarked=True,
            rng_seed=rng_seed,
        )
        unmarked = generate_text(
            lm,
            first_word=start,
            length=length,
            config=cfg,
            watermarked=False,
            rng_seed=rng_seed,
        )
        marked_scores.append(marked.detection.z_score)
        null_scores.append(unmarked.detection.z_score)
        trial_rows.append((
            trial,
            start,
            marked.detection.z_score,
            unmarked.detection.z_score,
            marked.detection.detected,
            unmarked.detection.detected,
        ))

    tp = sum(score >= cfg.threshold_z for score in marked_scores)
    fp = sum(score >= cfg.threshold_z for score in null_scores)
    tpr = tp / trials
    fpr = fp / trials
    tpr_ci = wilson_interval(tp, trials)
    fpr_ci = wilson_interval(fp, trials)
    auc = empirical_auc(marked_scores, null_scores)
    points = roc_points(marked_scores, null_scores)

    summary: dict[str, object] = {
        "algorithm": "SeedMark first-word-seeded keyed PRF probability tilting",
        "scientific_scope": "educational prototype; not Anthropic/SynthID/C2PA compatible",
        "python": ">=3.13",
        "first_word": marked_example.first_word,
        "first_word_seed_sha256": seed_hex(marked_example.first_word),
        "key_fingerprint_sha256": hashlib.sha256(cfg.secret_key.encode()).hexdigest()[:16],
        "strength": cfg.strength,
        "top_k": cfg.top_k,
        "threshold_z": cfg.threshold_z,
        "length": length,
        "trials": trials,
        "experiment_seed": experiment_seed,
        "marked_example": asdict(marked_example.detection),
        "unmarked_example": asdict(null_example.detection),
        "marked_mean_z": statistics.fmean(marked_scores),
        "marked_std_z": statistics.stdev(marked_scores) if len(marked_scores) > 1 else 0.0,
        "unmarked_mean_z": statistics.fmean(null_scores),
        "unmarked_std_z": statistics.stdev(null_scores) if len(null_scores) > 1 else 0.0,
        "true_positive_rate": tpr,
        "false_positive_rate": fpr,
        "tpr_ci95": tpr_ci,
        "fpr_ci95": fpr_ci,
        "auc": auc,
    }

    (output_dir / "generated_watermarked.txt").write_text(marked_example.text + "\n", encoding="utf-8")
    (output_dir / "generated_unwatermarked.txt").write_text(null_example.text + "\n", encoding="utf-8")
    write_trace_csv(output_dir / "trace_watermarked.csv", marked_example)
    write_trace_csv(output_dir / "trace_unwatermarked.csv", null_example)

    with (output_dir / "trial_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "first_word", "watermarked_z", "unwatermarked_z", "watermarked_detected", "unwatermarked_detected"])
        writer.writerows(trial_rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "z_score_histogram.svg").write_text(
        histogram_svg(null_scores, marked_scores, cfg.threshold_z), encoding="utf-8"
    )
    (output_dir / "roc_curve.svg").write_text(roc_svg(points), encoding="utf-8")

    report_payload: dict[str, object] = {
        "summary": summary,
        "watermarked_text": marked_example.text,
        "unwatermarked_text": null_example.text,
        "watermarked_trace": generation_to_trace(marked_example),
        "unwatermarked_trace": generation_to_trace(null_example),
        "roc": points,
    }
    (output_dir / "report.json").write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    write_interactive_report(output_dir / "report.html", report_payload)
    return summary
