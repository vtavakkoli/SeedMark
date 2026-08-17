"""Command-line interface for SeedMark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import WatermarkConfig, detect_tokens, tokenize
from .experiment import run_experiment
from .generation import generate_text
from .hf_llm import (
    DEFAULT_MODEL,
    QwenSeedMark,
    detect_text_with_tokenizer,
    write_qwen_report,
)
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

    detect = sub.add_parser("detect", help="score supplied toy text without model probabilities")
    source = detect.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    detect.add_argument("--secret-key", default="seedmark-demo-key")
    detect.add_argument("--threshold-z", type=float, default=3.0)

    experiment = sub.add_parser("experiment", help="run toy Monte Carlo experiment and build report")
    experiment.add_argument("--output-dir", type=Path, default=Path("results/run"))
    experiment.add_argument("--trials", type=int, default=300)
    experiment.add_argument("--length", type=int, default=80)
    experiment.add_argument("--first-word", default="research")
    experiment.add_argument("--experiment-seed", type=int, default=20260817)
    _add_common_config(experiment)

    qwen = sub.add_parser("qwen-demo", help="run a matched real-Qwen marked/control experiment")
    qwen.add_argument("--model", default=DEFAULT_MODEL)
    qwen.add_argument("--prompt", default="Research is")
    qwen.add_argument("--max-new-tokens", type=int, default=64)
    qwen.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    qwen.add_argument("--temperature", type=float, default=1.0)
    qwen.add_argument("--top-k", type=int, default=20)
    qwen.add_argument("--strength", type=float, default=1.5)
    qwen.add_argument("--threshold-z", type=float, default=3.0)
    qwen.add_argument("--secret-key", default="seedmark-demo-key")
    qwen.add_argument("--rng-seed", type=int, default=20260817)
    qwen.add_argument("--output-dir", type=Path, default=Path("results/qwen"))

    qdetect = sub.add_parser("qwen-detect", help="retokenize Qwen text and detect without loading model weights")
    qdetect.add_argument("--model", default=DEFAULT_MODEL)
    qdetect.add_argument("--prompt", default="Research is")
    qsource = qdetect.add_mutually_exclusive_group(required=True)
    qsource.add_argument("--text")
    qsource.add_argument("--text-file", type=Path)
    qdetect.add_argument("--secret-key", default="seedmark-demo-key")
    qdetect.add_argument("--threshold-z", type=float, default=3.0)
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

    if args.command == "qwen-demo":
        lab = QwenSeedMark(args.model, args.device)
        common = dict(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            secret_key=args.secret_key,
            strength=args.strength,
            top_k=args.top_k,
            temperature=args.temperature,
            threshold_z=args.threshold_z,
            rng_seed=args.rng_seed,
        )
        marked = lab.generate(**common, watermarked=True)
        control = lab.generate(**common, watermarked=False)
        write_qwen_report(args.output_dir, marked, control)
        print(json.dumps({
            "model": marked.model_name,
            "output_dir": str(args.output_dir),
            "watermarked_z": marked.detection.z_score,
            "watermarked_detected": marked.detection.detected,
            "control_z": control.detection.z_score,
            "control_detected": control.detection.detected,
            "generated_tokens": len(marked.generated_token_ids),
        }, indent=2))
        print(f"\nOpen {args.output_dir / 'report.html'}")
        return 0

    if args.command == "qwen-detect":
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        result = detect_text_with_tokenizer(
            model_name=args.model,
            text=text,
            prompt=args.prompt,
            secret_key=args.secret_key,
            threshold_z=args.threshold_z,
        )
        print(json.dumps({
            "model_loaded": False,
            "n_scored_tokens": result.n_scored_tokens,
            "mean_score": result.mean_score,
            "z_score": result.z_score,
            "p_value_one_sided": result.p_value_one_sided,
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
