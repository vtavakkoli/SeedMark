"""Real-LLM SeedMark adapter for Hugging Face Qwen models.

The watermark remains intentionally simple: the normalized first word supplies a
public seed, while HMAC-SHA256 over (seed, position, token-id) supplies a keyed
pseudorandom score. Generation tilts a real model's top-k probabilities toward
high-scoring token IDs. Detection needs the tokenizer and secret key, but never
the model logits or next-token probability distribution.

SeedMark is text-only. Qwen3.5 is a multimodal checkpoint, but we deliberately
load only its public tokenizer for text preprocessing. This avoids constructing
Qwen's image/video processors (and therefore avoids an unnecessary torchvision
dependency) while the multimodal model itself can still run with input_ids only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
from typing import Any

from .core import DetectionResult, first_word_seed, normalize_word

_TWO64 = float(1 << 64)
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
QUALITY_MODEL = "Qwen/Qwen3.5-2B"


@dataclass(frozen=True, slots=True)
class HFCandidateTrace:
    token_id: int
    token_text: str
    base_probability: float
    generation_probability: float
    watermark_score: float
    chosen: bool


@dataclass(frozen=True, slots=True)
class HFStepTrace:
    position: int
    chosen_token_id: int
    chosen_token_text: str
    chosen_base_probability: float
    chosen_generation_probability: float
    chosen_watermark_score: float
    cumulative_z: float
    candidates: tuple[HFCandidateTrace, ...]


@dataclass(frozen=True, slots=True)
class HFGenerationResult:
    model_name: str
    prompt: str
    first_word: str
    text: str
    continuation: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    watermarked: bool
    temperature: float
    top_k: int
    strength: float
    rng_seed: int
    detection: DetectionResult
    trace: tuple[HFStepTrace, ...]


def token_id_score(secret_key: str, first_word: str, position: int, token_id: int) -> float:
    """Map a model token id to a reproducible U[0,1) keyed score."""
    if not secret_key:
        raise ValueError("secret_key must not be empty")
    if position < 1:
        raise ValueError("position must be >= 1")
    if token_id < 0:
        raise ValueError("token_id must be >= 0")
    seed = first_word_seed(first_word)
    message = seed + position.to_bytes(8, "big") + token_id.to_bytes(8, "big")
    digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / _TWO64


def detect_token_ids(
    token_ids: list[int] | tuple[int, ...],
    *,
    secret_key: str,
    first_word: str,
    threshold_z: float = 3.0,
) -> DetectionResult:
    """Detect SeedMark correlation from observed token IDs; no logits are needed."""
    if threshold_z <= 0:
        raise ValueError("threshold_z must be > 0")
    if not token_ids:
        return DetectionResult(0, 0.5, 0.0, 0.5, threshold_z, False)
    observed = [
        token_id_score(secret_key, first_word, position, token_id)
        for position, token_id in enumerate(token_ids, start=1)
    ]
    n = len(observed)
    mean_score = sum(observed) / n
    z_score = (sum(observed) - 0.5 * n) / math.sqrt(n / 12.0)
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return DetectionResult(n, mean_score, z_score, p_value, threshold_z, z_score >= threshold_z)


def _optional_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError(
            "Real-LLM support is optional. Install it with: pip install -e '.[real-llm]'"
        ) from exc
    return torch, AutoTokenizer, AutoModelForMultimodalLM


def _load_tokenizer(auto_tokenizer: Any, model_name: str) -> Any:
    """Load only text tokenizer assets, never multimodal processors."""
    tokenizer = auto_tokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("The selected model tokenizer has no EOS token")
    return tokenizer


def _choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QwenSeedMark:
    """Load one Qwen3.5 model and generate marked/unmarked matched samples."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        torch, AutoTokenizer, AutoModelForMultimodalLM = _optional_stack()
        self.torch = torch
        self.model_name = model_name
        self.device = _choose_device(torch, device)
        self.tokenizer = _load_tokenizer(AutoTokenizer, model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        if self.device == "cpu":
            self.model.float()
        self.model.eval()

    def _encode(self, text: str) -> tuple[Any, Any]:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def generate(
        self,
        *,
        prompt: str = "Research is",
        max_new_tokens: int = 64,
        secret_key: str = "seedmark-demo-key",
        strength: float = 1.5,
        top_k: int = 20,
        temperature: float = 1.0,
        threshold_z: float = 3.0,
        rng_seed: int = 20260817,
        watermarked: bool = True,
    ) -> HFGenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if top_k < 2:
            raise ValueError("top_k must be >= 2")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if strength < 0:
            raise ValueError("strength must be >= 0")

        torch = self.torch
        seed_word = normalize_word(prompt)
        input_ids, attention_mask = self._encode(prompt)
        prompt_token_ids = tuple(int(x) for x in input_ids[0].detach().cpu().tolist())
        generated_ids: list[int] = []
        trace: list[HFStepTrace] = []
        scores_seen: list[float] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng_seed)

        with torch.inference_mode():
            for position in range(1, max_new_tokens + 1):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :].float() / temperature
                k = min(top_k, int(logits.shape[-1]))
                values, ids = torch.topk(logits, k=k)
                base = torch.softmax(values, dim=-1).detach().cpu()
                candidate_ids = [int(x) for x in ids.detach().cpu().tolist()]
                keyed_scores = [
                    token_id_score(secret_key, seed_word, position, token_id)
                    for token_id in candidate_ids
                ]
                score_tensor = torch.tensor(keyed_scores, dtype=torch.float32)
                tilted = base * torch.exp(strength * (2.0 * score_tensor - 1.0))
                marked = tilted / tilted.sum()
                generation = marked if watermarked else base
                chosen_index = int(torch.multinomial(generation, 1, generator=generator).item())
                chosen_id = candidate_ids[chosen_index]
                chosen_score = keyed_scores[chosen_index]
                scores_seen.append(chosen_score)
                cumulative_n = len(scores_seen)
                cumulative_z = (
                    sum(scores_seen) - 0.5 * cumulative_n
                ) / math.sqrt(cumulative_n / 12.0)

                candidate_trace = []
                for idx, token_id in enumerate(candidate_ids):
                    candidate_trace.append(HFCandidateTrace(
                        token_id=token_id,
                        token_text=self.tokenizer.decode(
                            [token_id], clean_up_tokenization_spaces=False
                        ),
                        base_probability=float(base[idx]),
                        generation_probability=float(generation[idx]),
                        watermark_score=keyed_scores[idx],
                        chosen=idx == chosen_index,
                    ))
                trace.append(HFStepTrace(
                    position=position,
                    chosen_token_id=chosen_id,
                    chosen_token_text=self.tokenizer.decode(
                        [chosen_id], clean_up_tokenization_spaces=False
                    ),
                    chosen_base_probability=float(base[chosen_index]),
                    chosen_generation_probability=float(generation[chosen_index]),
                    chosen_watermark_score=chosen_score,
                    cumulative_z=cumulative_z,
                    candidates=tuple(candidate_trace),
                ))
                generated_ids.append(chosen_id)
                next_id = torch.tensor([[chosen_id]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_id)), dim=1
                )
                eos_id = self.tokenizer.eos_token_id
                if chosen_id == int(eos_id):
                    break

        continuation = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        full_text = prompt + continuation
        detection = detect_token_ids(
            generated_ids,
            secret_key=secret_key,
            first_word=seed_word,
            threshold_z=threshold_z,
        )
        return HFGenerationResult(
            model_name=self.model_name,
            prompt=prompt,
            first_word=seed_word,
            text=full_text,
            continuation=continuation,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=tuple(generated_ids),
            watermarked=watermarked,
            temperature=temperature,
            top_k=top_k,
            strength=strength,
            rng_seed=rng_seed,
            detection=detection,
            trace=tuple(trace),
        )


def detect_text_with_tokenizer(
    *,
    model_name: str,
    text: str,
    prompt: str,
    secret_key: str,
    threshold_z: float = 3.0,
) -> DetectionResult:
    """Retokenize text using only the public tokenizer, then run the detector."""
    _, AutoTokenizer, _ = _optional_stack()
    tokenizer = _load_tokenizer(AutoTokenizer, model_name)
    full_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("text does not tokenize with the supplied prompt as an exact prefix")
    generated_ids = full_ids[len(prompt_ids):]
    return detect_token_ids(
        generated_ids,
        secret_key=secret_key,
        first_word=normalize_word(prompt),
        threshold_z=threshold_z,
    )
