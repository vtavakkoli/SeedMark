"""Command-line interface for SeedMark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import WatermarkConfig, detect_tokens, tokenize
from .experiment import run_experiment
from .generation import generate_text
from .lm import ToyBigramLM


def _add_common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--secret-key", default="seedmark-demo-key")
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold-z", type=float, default=3.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedmark",
        description="First-word-seeded keyed pseudorandom text-watermark teaching lab",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate one inspectable toy sequence")
    generate.add_argument("--first-word", default="research")
    generate.add_argument("--length", type=int, default=80)
    generate.add_argument("--rng-seed", type=int, default=20260817)
    generate.add_argument("--unwatermarked", action="store_true")
    _add_common_config(generate)

    detect = sub.add_parser("detect", help="score supplied text without model probabilities")
    source = detect.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    detect.add_argument("--secret-key", default="seedmark-demo-key")
    detect.add_argument("--threshold-z", type=float, default=3.0)

    experiment = sub.add_parser("experiment", help="run Monte Carlo experiment and build report")
    experiment.add_argument("--output-dir", type=Path, default=Path("results/run"))
    experiment.add_argument("--trials", type=int, default=300)
    experiment.add_argument("--length", type=int, default=80)
    experiment.add_argument("--first-word", default="research")
    experiment.add_argument("--experiment-seed", type=int, default=20260817)
    _add_common_config(experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config = WatermarkConfig(args.secret_key, args.strength, args.top_k, args.threshold_z)
        result = generate_text(
            ToyBigramLM(),
            first_word=args.first_word,
            length=args.length,
            config=config,
            watermarked=not args.unwatermarked,
            rng_seed=args.rng_seed,
        )
        print(result.text)
        print(json.dumps({
            "watermarked": result.watermarked,
            "z_score": result.detection.z_score,
            "p_value_one_sided": result.detection.p_value_one_sided,
            "detected": result.detection.detected,
        }, indent=2))
        return 0

    if args.command == "detect":
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        tokens = tokenize(text)
        result = detect_tokens(tokens, secret_key=args.secret_key, threshold_z=args.threshold_z)
        print(json.dumps({
            "n_scored_tokens": result.n_scored_tokens,
            "mean_score": result.mean_score,
            "z_score": result.z_score,
            "p_value_one_sided": result.p_value_one_sided,
            "threshold_z": result.threshold_z,
            "detected": result.detected,
        }, indent=2))
        return 0

    config = WatermarkConfig(args.secret_key, args.strength, args.top_k, args.threshold_z)
    summary = run_experiment(
        args.output_dir,
        trials=args.trials,
        length=args.length,
        first_word=args.first_word,
        experiment_seed=args.experiment_seed,
        config=config,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nOpen {args.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
