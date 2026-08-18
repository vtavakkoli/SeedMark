"""Command-line interface for SeedMark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ._version import __version__
from .animation import write_visual_assets
from .chat_llm import (
    DEFAULT_CHAT_QUESTION,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    ChatQwenSeedMark,
    detect_chat_text_with_tokenizer,
)
from .chat_reporting import write_chat_report
from .core import WatermarkConfig, detect_tokens, tokenize
from .experiment import run_experiment
from .generation import generate_text
from .hf_llm import DEFAULT_MODEL
from .lm import ToyBigramLM
from .model_cache import cache_home, prefetch_model
from .semantic import DEFAULT_SEMANTIC_MODEL, DEFAULT_SEMANTIC_SCOPE, SEMANTIC_SCOPES
from .semantic_chat import (
    SemanticChatQwenSeedMark,
    detect_semantic_chat_text_with_tokenizer,
)

DEFAULT_QWEN_DEMO_PROMPT = DEFAULT_CHAT_QUESTION


def _add_common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--secret-key", default="seedmark-demo-key")
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold-z", type=float, default=3.0)


def _add_chat_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--question",
        "--prompt",
        dest="question",
        default=DEFAULT_CHAT_QUESTION,
        help=(
            "user question for the Qwen chat conversation; --prompt is retained as a "
            "backward-compatible alias"
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_CHAT_SYSTEM_PROMPT,
        help="system instruction used before the user question",
    )


def _add_real_generation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="maximum generated assistant tokens per marked/control answer (default: 128)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--strength", type=float, default=1.5)
    parser.add_argument("--threshold-z", type=float, default=3.0)
    parser.add_argument("--secret-key", default="seedmark-demo-key")
    parser.add_argument("--rng-seed", type=int, default=20260817)


def _add_semantic_scope_options(parser: argparse.ArgumentParser, *, generation: bool) -> None:
    parser.add_argument(
        "--semantic-scope",
        choices=SEMANTIC_SCOPES,
        default=DEFAULT_SEMANTIC_SCOPE,
        help=(
            "semantic key scope: 'answer' derives one key from the complete answer; "
            "'paragraph' re-keys only at blank-line paragraph boundaries"
        ),
    )
    parser.add_argument(
        "--context-paragraphs",
        type=int,
        default=1,
        help="number of completed paragraphs used to key the next paragraph",
    )
    if generation:
        parser.add_argument(
            "--max-answer-passes",
            type=int,
            default=4,
            help=(
                "maximum marked generations used to find a final answer in the same "
                "semantic bucket as the complete-answer draft"
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedmark",
        description="Inspectable token and semantic self-key text-watermark research lab",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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

    qcache = sub.add_parser("qwen-cache", help="prefetch a Qwen model into the persistent HF cache")
    qcache.add_argument("--model", default=DEFAULT_MODEL)
    qcache.add_argument("--revision", default=None)

    qwen = sub.add_parser(
        "qwen-demo",
        help="run a matched real-Qwen chat experiment with marked/control assistant answers",
    )
    qwen.add_argument("--model", default=DEFAULT_MODEL)
    _add_chat_options(qwen)
    _add_real_generation_options(qwen)
    qwen.add_argument("--output-dir", type=Path, default=Path("results/qwen"))
    qwen.add_argument("--gif-frame-ms", type=int, default=650, help="GIF frame duration in milliseconds")
    qwen.add_argument("--gif-width", type=int, default=1200, help="GIF/preview width in pixels")
    qwen.add_argument("--gif-height", type=int, default=900, help="GIF/preview height in pixels")
    qwen.add_argument(
        "--no-gif",
        action="store_true",
        help="skip generation/detection GIFs and static preview PNGs",
    )

    qdetect = sub.add_parser(
        "qwen-detect",
        help="retokenize a saved Qwen chat answer and detect without loading model weights",
    )
    qdetect.add_argument("--model", default=DEFAULT_MODEL)
    _add_chat_options(qdetect)
    qsource = qdetect.add_mutually_exclusive_group(required=True)
    qsource.add_argument("--text")
    qsource.add_argument("--text-file", type=Path)
    qdetect.add_argument("--secret-key", default="seedmark-demo-key")
    qdetect.add_argument("--threshold-z", type=float, default=3.0)

    semantic = sub.add_parser(
        "semantic-qwen-demo",
        help="run marked/control Qwen chat generation keyed by answer or paragraph semantics",
    )
    semantic.add_argument("--model", default=DEFAULT_MODEL)
    semantic.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    semantic.add_argument(
        "--semantic-device", default="cpu", choices=("auto", "cpu", "cuda", "mps")
    )
    _add_chat_options(semantic)
    _add_real_generation_options(semantic)
    semantic.add_argument("--bucket-count", type=int, default=32)
    _add_semantic_scope_options(semantic, generation=True)
    semantic.add_argument(
        "--output-dir", type=Path, default=Path("results/semantic-qwen")
    )

    semantic_detect = sub.add_parser(
        "semantic-qwen-detect",
        help=(
            "retokenize a saved semantic-watermarked Qwen answer and detect without "
            "loading generator model weights"
        ),
    )
    semantic_detect.add_argument("--model", default=DEFAULT_MODEL)
    semantic_detect.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    semantic_detect.add_argument(
        "--semantic-device", default="cpu", choices=("auto", "cpu", "cuda", "mps")
    )
    _add_chat_options(semantic_detect)
    semantic_source = semantic_detect.add_mutually_exclusive_group(required=True)
    semantic_source.add_argument("--text")
    semantic_source.add_argument("--text-file", type=Path)
    semantic_detect.add_argument("--secret-key", default="seedmark-demo-key")
    semantic_detect.add_argument("--threshold-z", type=float, default=3.0)
    semantic_detect.add_argument("--bucket-count", type=int, default=32)
    _add_semantic_scope_options(semantic_detect, generation=False)
    return parser


def _detection_dict(result) -> dict[str, object]:
    return {
        "n_scored_tokens": result.n_scored_tokens,
        "mean_score": result.mean_score,
        "z_score": result.z_score,
        "p_value_one_sided": result.p_value_one_sided,
        "threshold_z": result.threshold_z,
        "detected": result.detected,
    }


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
            "seedmark_version": __version__,
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
            "seedmark_version": __version__,
            **_detection_dict(result),
        }, indent=2))
        return 0

    if args.command == "qwen-cache":
        snapshot = prefetch_model(args.model, revision=args.revision)
        print(json.dumps({
            "seedmark_version": __version__,
            "model": args.model,
            "revision": args.revision or "latest",
            "cache_home": str(cache_home()),
            "snapshot_path": str(snapshot),
        }, indent=2))
        return 0

    if args.command == "qwen-demo":
        lab = ChatQwenSeedMark(args.model, args.device)
        common = dict(
            question=args.question,
            system_prompt=args.system_prompt,
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

        assets: dict[str, str] = {}
        if not args.no_gif:
            assets = write_visual_assets(
                args.output_dir,
                marked,
                control,
                frame_ms=args.gif_frame_ms,
                width=args.gif_width,
                height=args.gif_height,
            )
        report_path = write_chat_report(
            args.output_dir,
            marked,
            control,
            question=args.question,
            system_prompt=args.system_prompt,
            assets=assets,
        )

        print(json.dumps({
            "seedmark_version": __version__,
            "mode": "chat",
            "model": marked.model_name,
            "question": args.question,
            "system_prompt": args.system_prompt,
            "first_word_seed": marked.first_word,
            "cache_home": str(cache_home()),
            "output_dir": str(args.output_dir),
            "report": str(report_path),
            "visual_assets": assets,
            "watermarked_z": marked.detection.z_score,
            "watermarked_detected": marked.detection.detected,
            "control_z": control.detection.z_score,
            "control_detected": control.detection.detected,
            "comparison": (
                "watermarked output -> detected; control -> not detected"
                if marked.detection.detected and not control.detection.detected
                else "review this run: expected marked/control contrast was not achieved"
            ),
            "generated_tokens": len(marked.generated_token_ids),
        }, indent=2))
        print(f"\nOpen {report_path}")
        if assets:
            print(f"Generation animation: {args.output_dir / assets['generation_gif']}")
            print(f"Detection animation: {args.output_dir / assets['detection_gif']}")
        return 0

    if args.command == "qwen-detect":
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        result = detect_chat_text_with_tokenizer(
            model_name=args.model,
            text=text,
            question=args.question,
            system_prompt=args.system_prompt,
            secret_key=args.secret_key,
            threshold_z=args.threshold_z,
        )
        print(json.dumps({
            "seedmark_version": __version__,
            "mode": "chat",
            "model_loaded": False,
            "question": args.question,
            **_detection_dict(result),
        }, indent=2))
        return 0

    if args.command == "semantic-qwen-demo":
        lab = SemanticChatQwenSeedMark(
            args.model,
            args.device,
            semantic_model=args.semantic_model,
            semantic_device=args.semantic_device,
        )
        common = dict(
            question=args.question,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            secret_key=args.secret_key,
            strength=args.strength,
            top_k=args.top_k,
            temperature=args.temperature,
            threshold_z=args.threshold_z,
            rng_seed=args.rng_seed,
            bucket_count=args.bucket_count,
            semantic_scope=args.semantic_scope,
            context_paragraphs=args.context_paragraphs,
            max_answer_passes=args.max_answer_passes,
        )
        marked = lab.generate(**common, watermarked=True)
        control = lab.generate(**common, watermarked=False)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "generated_watermarked.txt").write_text(marked.text, encoding="utf-8")
        (args.output_dir / "generated_control.txt").write_text(control.text, encoding="utf-8")
        (args.output_dir / "watermarked-trace.json").write_text(
            json.dumps(asdict(marked), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (args.output_dir / "control-trace.json").write_text(
            json.dumps(asdict(control), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary = {
            "seedmark_version": __version__,
            "mode": "semantic-chat",
            "model": marked.model_name,
            "semantic_model": marked.semantic_model_name,
            "semantic_scope": args.semantic_scope,
            "question": args.question,
            "bucket_count": args.bucket_count,
            "context_paragraphs": args.context_paragraphs,
            "answer_key_attempts": marked.answer_key_attempts,
            "answer_key_stable": marked.answer_key_stable,
            "semantic_bucket": marked.semantic_bucket,
            "semantic_margin": marked.semantic_margin,
            "output_dir": str(args.output_dir),
            "watermarked": _detection_dict(marked.detection),
            "control": _detection_dict(control.detection),
            "comparison": (
                "watermarked output -> detected; control -> not detected"
                if marked.detection.detected and not control.detection.detected
                else "review this run: expected marked/control contrast was not achieved"
            ),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSemantic experiment written to {args.output_dir}")
        return 0

    if args.command == "semantic-qwen-detect":
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        result = detect_semantic_chat_text_with_tokenizer(
            model_name=args.model,
            semantic_model=args.semantic_model,
            semantic_device=args.semantic_device,
            text=text,
            question=args.question,
            system_prompt=args.system_prompt,
            secret_key=args.secret_key,
            threshold_z=args.threshold_z,
            bucket_count=args.bucket_count,
            semantic_scope=args.semantic_scope,
            context_paragraphs=args.context_paragraphs,
        )
        print(json.dumps({
            "seedmark_version": __version__,
            "mode": "semantic-chat",
            "semantic_scope": args.semantic_scope,
            "generator_model_loaded": False,
            "semantic_model": args.semantic_model,
            "question": args.question,
            **_detection_dict(result),
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
