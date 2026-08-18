"""Semantic self-keyed chat generation for SeedMark's Hugging Face backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .chat_llm import (
    DEFAULT_CHAT_QUESTION,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    ChatQwenSeedMark,
    _display_prompt,
    chat_messages,
    render_chat_prompt,
)
from .core import DetectionResult
from .hf_llm import DEFAULT_MODEL, _load_tokenizer, _optional_stack
from .semantic import (
    DEFAULT_SEMANTIC_MODEL,
    HFMeanPoolingSemanticEncoder,
    SemanticBucketizer,
    SemanticContextTracker,
    SemanticEncoder,
    detect_semantic_token_ids,
    semantic_token_score,
)


@dataclass(frozen=True, slots=True)
class SemanticCandidateTrace:
    token_id: int
    token_text: str
    base_probability: float
    generation_probability: float
    watermark_score: float
    chosen: bool


@dataclass(frozen=True, slots=True)
class SemanticStepTrace:
    position: int
    sentence_index: int
    token_offset: int
    semantic_bucket: int
    semantic_margin: float
    semantic_context: str
    chosen_token_id: int
    chosen_token_text: str
    chosen_base_probability: float
    chosen_generation_probability: float
    chosen_watermark_score: float
    cumulative_z: float
    candidates: tuple[SemanticCandidateTrace, ...]


@dataclass(frozen=True, slots=True)
class SemanticGenerationResult:
    model_name: str
    semantic_model_name: str
    prompt: str
    question: str
    text: str
    continuation: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    watermarked: bool
    temperature: float
    top_k: int
    strength: float
    bucket_count: int
    context_sentences: int
    rng_seed: int
    detection: DetectionResult
    trace: tuple[SemanticStepTrace, ...]


class SemanticChatQwenSeedMark(ChatQwenSeedMark):
    """Generate assistant answers keyed by the evolving semantics of the answer.

    The user question bootstraps sentence 0.  Once an answer sentence completes,
    its semantic bucket (or the configured recent-sentence window) keys the token
    sampler for the next sentence.  Sentence-local token offsets reset at each
    boundary, giving the detector a natural re-synchronisation point after edits.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        *,
        semantic_model: str = DEFAULT_SEMANTIC_MODEL,
        semantic_device: str = "cpu",
        semantic_encoder: SemanticEncoder | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.semantic_encoder = semantic_encoder or HFMeanPoolingSemanticEncoder(
            semantic_model,
            device=semantic_device,
        )
        self.semantic_model_name = getattr(self.semantic_encoder, "model_name", semantic_model)

    def generate(
        self,
        *,
        question: str = DEFAULT_CHAT_QUESTION,
        system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        secret_key: str = "seedmark-demo-key",
        strength: float = 1.5,
        top_k: int = 20,
        temperature: float = 1.0,
        threshold_z: float = 3.0,
        rng_seed: int = 20260817,
        watermarked: bool = True,
        bucket_count: int = 32,
        context_sentences: int = 1,
    ) -> SemanticGenerationResult:
        """Generate a semantically self-keyed assistant answer."""

        question = question.strip()
        system_prompt = system_prompt.strip()
        chat_messages(question, system_prompt)  # validation
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if top_k < 2:
            raise ValueError("top_k must be >= 2")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if strength < 0:
            raise ValueError("strength must be >= 0")
        if threshold_z <= 0:
            raise ValueError("threshold_z must be > 0")
        if bucket_count < 4:
            raise ValueError("bucket_count must be >= 4")
        if context_sentences < 1:
            raise ValueError("context_sentences must be >= 1")
        if not secret_key:
            raise ValueError("secret_key must not be empty")

        torch = self.torch
        input_ids, attention_mask = self._encode_chat(question, system_prompt)
        prompt_token_ids = tuple(int(x) for x in input_ids[0].detach().cpu().tolist())
        generated_ids: list[int] = []
        trace: list[SemanticStepTrace] = []
        scores_seen: list[float] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng_seed)
        eos_id = int(self.tokenizer.eos_token_id)

        bucketizer = SemanticBucketizer(secret_key=secret_key, bucket_count=bucket_count)
        tracker = SemanticContextTracker(
            encoder=self.semantic_encoder,
            bucketizer=bucketizer,
            bootstrap_text=question,
            context_sentences=context_sentences,
        )

        with torch.inference_mode():
            for position in range(1, max_new_tokens + 1):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :].float() / temperature
                k = min(top_k, int(logits.shape[-1]))
                values, ids = torch.topk(logits, k=k)
                base = torch.softmax(values, dim=-1).detach().cpu()
                candidate_ids = [int(value) for value in ids.detach().cpu().tolist()]

                semantic_key = tracker.current_key
                token_offset = tracker.next_token_offset
                keyed_scores = [
                    semantic_token_score(
                        secret_key,
                        semantic_key.bucket,
                        token_offset,
                        token_id,
                    )
                    for token_id in candidate_ids
                ]
                score_tensor = torch.tensor(keyed_scores, dtype=torch.float32)
                tilted = base * torch.exp(strength * (2.0 * score_tensor - 1.0))
                marked = tilted / tilted.sum()
                generation = marked if watermarked else base
                chosen_index = int(torch.multinomial(generation, 1, generator=generator).item())
                chosen_id = candidate_ids[chosen_index]

                if chosen_id == eos_id:
                    break

                chosen_piece = self.tokenizer.decode(
                    [chosen_id], clean_up_tokenization_spaces=False
                )
                chosen_score = keyed_scores[chosen_index]
                scores_seen.append(chosen_score)
                n = len(scores_seen)
                cumulative_z = (sum(scores_seen) - 0.5 * n) / math.sqrt(n / 12.0)

                candidates = tuple(
                    SemanticCandidateTrace(
                        token_id=token_id,
                        token_text=self.tokenizer.decode(
                            [token_id], clean_up_tokenization_spaces=False
                        ),
                        base_probability=float(base[index]),
                        generation_probability=float(generation[index]),
                        watermark_score=keyed_scores[index],
                        chosen=index == chosen_index,
                    )
                    for index, token_id in enumerate(candidate_ids)
                )
                trace.append(
                    SemanticStepTrace(
                        position=position,
                        sentence_index=tracker.sentence_index,
                        token_offset=token_offset,
                        semantic_bucket=semantic_key.bucket,
                        semantic_margin=semantic_key.margin,
                        semantic_context=tracker.semantic_context,
                        chosen_token_id=chosen_id,
                        chosen_token_text=chosen_piece,
                        chosen_base_probability=float(base[chosen_index]),
                        chosen_generation_probability=float(generation[chosen_index]),
                        chosen_watermark_score=chosen_score,
                        cumulative_z=cumulative_z,
                        candidates=candidates,
                    )
                )

                generated_ids.append(chosen_id)
                next_id = torch.tensor([[chosen_id]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_id)), dim=1
                )
                tracker.observe_token_piece(chosen_piece)

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        detection = detect_semantic_token_ids(
            generated_ids,
            tokenizer=self.tokenizer,
            encoder=self.semantic_encoder,
            secret_key=secret_key,
            bootstrap_text=question,
            threshold_z=threshold_z,
            bucket_count=bucket_count,
            context_sentences=context_sentences,
        )
        return SemanticGenerationResult(
            model_name=self.model_name,
            semantic_model_name=self.semantic_model_name,
            prompt=_display_prompt(question),
            question=question,
            text=answer,
            continuation=answer,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=tuple(generated_ids),
            watermarked=watermarked,
            temperature=temperature,
            top_k=top_k,
            strength=strength,
            bucket_count=bucket_count,
            context_sentences=context_sentences,
            rng_seed=rng_seed,
            detection=detection,
            trace=tuple(trace),
        )


def detect_semantic_chat_text_with_tokenizer(
    *,
    model_name: str,
    text: str,
    question: str = DEFAULT_CHAT_QUESTION,
    system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
    secret_key: str,
    semantic_model: str = DEFAULT_SEMANTIC_MODEL,
    semantic_device: str = "cpu",
    threshold_z: float = 3.0,
    bucket_count: int = 32,
    context_sentences: int = 1,
) -> DetectionResult:
    """Retokenize an assistant answer and run semantic self-key detection.

    Only the generator tokenizer and the lightweight semantic encoder are loaded;
    generator model weights and generation logits are not required.
    """

    _, AutoTokenizer, _ = _optional_stack()
    tokenizer = _load_tokenizer(AutoTokenizer, model_name)
    encoder = HFMeanPoolingSemanticEncoder(semantic_model, device=semantic_device)
    rendered = render_chat_prompt(tokenizer, question, system_prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(rendered + text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant text does not tokenize with the reconstructed chat prefix")
    generated_ids = full_ids[len(prompt_ids) :]
    return detect_semantic_token_ids(
        generated_ids,
        tokenizer=tokenizer,
        encoder=encoder,
        secret_key=secret_key,
        bootstrap_text=question,
        threshold_z=threshold_z,
        bucket_count=bucket_count,
        context_sentences=context_sentences,
    )
