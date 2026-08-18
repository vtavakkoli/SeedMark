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
    DEFAULT_SEMANTIC_SCOPE,
    SEMANTIC_SCOPES,
    HFMeanPoolingSemanticEncoder,
    SemanticBucketizer,
    SemanticContextTracker,
    SemanticEncoder,
    SemanticKey,
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
    semantic_scope: str
    segment_index: int
    token_occurrence: int
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
class _SemanticPass:
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    text: str
    trace: tuple[SemanticStepTrace, ...]


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
    semantic_scope: str
    context_paragraphs: int
    rng_seed: int
    detection: DetectionResult
    semantic_bucket: int | None
    semantic_margin: float | None
    semantic_key_text: str | None
    draft_text: str | None
    answer_key_attempts: int
    answer_key_stable: bool
    trace: tuple[SemanticStepTrace, ...]


class SemanticChatQwenSeedMark(ChatQwenSeedMark):
    """Generate assistant answers keyed by paragraph or complete-answer semantics.

    ``answer`` scope is the default. It first generates an ordinary semantic draft,
    derives a coarse secret bucket from the complete draft, then generates the
    watermarked answer. The marked answer is accepted only when its own complete
    semantics map to that same bucket, allowing detection to reconstruct the key
    from the final answer alone.

    ``paragraph`` scope is the streaming alternative. The question bootstraps the
    first paragraph; each completed paragraph keys the next paragraph. Sentence
    punctuation never re-keys the watermark.
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

    def _generate_pass(
        self,
        *,
        question: str,
        system_prompt: str,
        max_new_tokens: int,
        secret_key: str,
        strength: float,
        top_k: int,
        temperature: float,
        rng_seed: int,
        watermarked: bool,
        bucketizer: SemanticBucketizer,
        semantic_scope: str,
        context_paragraphs: int,
        fixed_key: SemanticKey | None = None,
        fixed_context: str = "",
    ) -> _SemanticPass:
        torch = self.torch
        input_ids, attention_mask = self._encode_chat(question, system_prompt)
        prompt_token_ids = tuple(int(x) for x in input_ids[0].detach().cpu().tolist())
        generated_ids: list[int] = []
        trace: list[SemanticStepTrace] = []
        scores_seen: list[float] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng_seed)
        eos_id = int(self.tokenizer.eos_token_id)

        tracker = None
        answer_occurrences: dict[int, int] = {}
        if semantic_scope == "paragraph":
            tracker = SemanticContextTracker(
                encoder=self.semantic_encoder,
                bucketizer=bucketizer,
                bootstrap_text=question,
                context_paragraphs=context_paragraphs,
            )
        elif fixed_key is None:
            raise ValueError("answer scope requires a fixed semantic key for a generation pass")

        with torch.inference_mode():
            for position in range(1, max_new_tokens + 1):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :].float() / temperature
                k = min(top_k, int(logits.shape[-1]))
                values, ids = torch.topk(logits, k=k)
                base = torch.softmax(values, dim=-1).detach().cpu()
                candidate_ids = [int(value) for value in ids.detach().cpu().tolist()]

                if semantic_scope == "answer":
                    assert fixed_key is not None
                    semantic_key = fixed_key
                    segment_index = 0
                    semantic_context = fixed_context
                    candidate_occurrences = [
                        answer_occurrences.get(token_id, 0) + 1
                        for token_id in candidate_ids
                    ]
                else:
                    assert tracker is not None
                    semantic_key = tracker.current_key
                    segment_index = tracker.paragraph_index
                    semantic_context = tracker.semantic_context
                    candidate_occurrences = [
                        tracker.next_token_occurrence(token_id)
                        for token_id in candidate_ids
                    ]

                keyed_scores = [
                    semantic_token_score(
                        secret_key,
                        semantic_key.bucket,
                        candidate_occurrences[index],
                        token_id,
                    )
                    for index, token_id in enumerate(candidate_ids)
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
                chosen_occurrence = candidate_occurrences[chosen_index]
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
                        semantic_scope=semantic_scope,
                        segment_index=segment_index,
                        token_occurrence=chosen_occurrence,
                        semantic_bucket=semantic_key.bucket,
                        semantic_margin=semantic_key.margin,
                        semantic_context=semantic_context,
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
                if semantic_scope == "answer":
                    answer_occurrences[chosen_id] = chosen_occurrence
                else:
                    assert tracker is not None
                    tracker.observe_token(chosen_id, chosen_piece)

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return _SemanticPass(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=tuple(generated_ids),
            text=answer,
            trace=tuple(trace),
        )

    def _rescore_answer_trace(
        self,
        trace: tuple[SemanticStepTrace, ...],
        *,
        semantic_key: SemanticKey,
        secret_key: str,
        semantic_context: str,
    ) -> tuple[SemanticStepTrace, ...]:
        """Attach the final complete-answer key to an already generated trace."""

        occurrences: dict[int, int] = {}
        scores_seen: list[float] = []
        rescored: list[SemanticStepTrace] = []
        for step in trace:
            candidate_occurrences = [
                occurrences.get(candidate.token_id, 0) + 1
                for candidate in step.candidates
            ]
            candidate_scores = [
                semantic_token_score(
                    secret_key,
                    semantic_key.bucket,
                    candidate_occurrences[index],
                    candidate.token_id,
                )
                for index, candidate in enumerate(step.candidates)
            ]
            candidates = tuple(
                SemanticCandidateTrace(
                    token_id=candidate.token_id,
                    token_text=candidate.token_text,
                    base_probability=candidate.base_probability,
                    generation_probability=candidate.generation_probability,
                    watermark_score=candidate_scores[index],
                    chosen=candidate.chosen,
                )
                for index, candidate in enumerate(step.candidates)
            )
            chosen_occurrence = occurrences.get(step.chosen_token_id, 0) + 1
            chosen_score = semantic_token_score(
                secret_key,
                semantic_key.bucket,
                chosen_occurrence,
                step.chosen_token_id,
            )
            scores_seen.append(chosen_score)
            n = len(scores_seen)
            cumulative_z = (sum(scores_seen) - 0.5 * n) / math.sqrt(n / 12.0)
            rescored.append(
                SemanticStepTrace(
                    position=step.position,
                    semantic_scope="answer",
                    segment_index=0,
                    token_occurrence=chosen_occurrence,
                    semantic_bucket=semantic_key.bucket,
                    semantic_margin=semantic_key.margin,
                    semantic_context=semantic_context,
                    chosen_token_id=step.chosen_token_id,
                    chosen_token_text=step.chosen_token_text,
                    chosen_base_probability=step.chosen_base_probability,
                    chosen_generation_probability=step.chosen_generation_probability,
                    chosen_watermark_score=chosen_score,
                    cumulative_z=cumulative_z,
                    candidates=candidates,
                )
            )
            occurrences[step.chosen_token_id] = chosen_occurrence
        return tuple(rescored)

    def _build_result(
        self,
        *,
        generated: _SemanticPass,
        question: str,
        max_new_tokens: int,
        secret_key: str,
        strength: float,
        top_k: int,
        temperature: float,
        threshold_z: float,
        rng_seed: int,
        watermarked: bool,
        bucket_count: int,
        semantic_scope: str,
        context_paragraphs: int,
        semantic_key: SemanticKey | None,
        semantic_key_text: str | None,
        draft_text: str | None,
        answer_key_attempts: int,
        answer_key_stable: bool,
        trace: tuple[SemanticStepTrace, ...] | None = None,
    ) -> SemanticGenerationResult:
        del max_new_tokens
        detection = detect_semantic_token_ids(
            generated.generated_token_ids,
            tokenizer=self.tokenizer,
            encoder=self.semantic_encoder,
            secret_key=secret_key,
            bootstrap_text=question,
            threshold_z=threshold_z,
            bucket_count=bucket_count,
            semantic_scope=semantic_scope,
            context_paragraphs=context_paragraphs,
        )
        return SemanticGenerationResult(
            model_name=self.model_name,
            semantic_model_name=self.semantic_model_name,
            prompt=_display_prompt(question),
            question=question,
            text=generated.text,
            continuation=generated.text,
            prompt_token_ids=generated.prompt_token_ids,
            generated_token_ids=generated.generated_token_ids,
            watermarked=watermarked,
            temperature=temperature,
            top_k=top_k,
            strength=strength,
            bucket_count=bucket_count,
            semantic_scope=semantic_scope,
            context_paragraphs=context_paragraphs,
            rng_seed=rng_seed,
            detection=detection,
            semantic_bucket=semantic_key.bucket if semantic_key is not None else None,
            semantic_margin=semantic_key.margin if semantic_key is not None else None,
            semantic_key_text=semantic_key_text,
            draft_text=draft_text,
            answer_key_attempts=answer_key_attempts,
            answer_key_stable=answer_key_stable,
            trace=trace if trace is not None else generated.trace,
        )

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
        semantic_scope: str = DEFAULT_SEMANTIC_SCOPE,
        context_paragraphs: int = 1,
        max_answer_passes: int = 4,
    ) -> SemanticGenerationResult:
        """Generate a semantically keyed assistant answer."""

        question = question.strip()
        system_prompt = system_prompt.strip()
        chat_messages(question, system_prompt)
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
        if semantic_scope not in SEMANTIC_SCOPES:
            raise ValueError(f"semantic_scope must be one of {SEMANTIC_SCOPES}")
        if context_paragraphs < 1:
            raise ValueError("context_paragraphs must be >= 1")
        if max_answer_passes < 1:
            raise ValueError("max_answer_passes must be >= 1")
        if not secret_key:
            raise ValueError("secret_key must not be empty")

        bucketizer = SemanticBucketizer(secret_key=secret_key, bucket_count=bucket_count)

        if semantic_scope == "paragraph":
            generated = self._generate_pass(
                question=question,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                secret_key=secret_key,
                strength=strength,
                top_k=top_k,
                temperature=temperature,
                rng_seed=rng_seed,
                watermarked=watermarked,
                bucketizer=bucketizer,
                semantic_scope="paragraph",
                context_paragraphs=context_paragraphs,
            )
            return self._build_result(
                generated=generated,
                question=question,
                max_new_tokens=max_new_tokens,
                secret_key=secret_key,
                strength=strength,
                top_k=top_k,
                temperature=temperature,
                threshold_z=threshold_z,
                rng_seed=rng_seed,
                watermarked=watermarked,
                bucket_count=bucket_count,
                semantic_scope="paragraph",
                context_paragraphs=context_paragraphs,
                semantic_key=None,
                semantic_key_text=None,
                draft_text=None,
                answer_key_attempts=0,
                answer_key_stable=True,
            )

        bootstrap_key = bucketizer.key_for_text(question, self.semantic_encoder)

        if not watermarked:
            generated = self._generate_pass(
                question=question,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                secret_key=secret_key,
                strength=strength,
                top_k=top_k,
                temperature=temperature,
                rng_seed=rng_seed,
                watermarked=False,
                bucketizer=bucketizer,
                semantic_scope="answer",
                context_paragraphs=context_paragraphs,
                fixed_key=bootstrap_key,
                fixed_context=question,
            )
            final_key = bucketizer.key_for_text(generated.text, self.semantic_encoder)
            trace = self._rescore_answer_trace(
                generated.trace,
                semantic_key=final_key,
                secret_key=secret_key,
                semantic_context=generated.text,
            )
            return self._build_result(
                generated=generated,
                question=question,
                max_new_tokens=max_new_tokens,
                secret_key=secret_key,
                strength=strength,
                top_k=top_k,
                temperature=temperature,
                threshold_z=threshold_z,
                rng_seed=rng_seed,
                watermarked=False,
                bucket_count=bucket_count,
                semantic_scope="answer",
                context_paragraphs=context_paragraphs,
                semantic_key=final_key,
                semantic_key_text=generated.text,
                draft_text=None,
                answer_key_attempts=0,
                answer_key_stable=True,
                trace=trace,
            )

        draft = self._generate_pass(
            question=question,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            secret_key=secret_key,
            strength=strength,
            top_k=top_k,
            temperature=temperature,
            rng_seed=rng_seed,
            watermarked=False,
            bucketizer=bucketizer,
            semantic_scope="answer",
            context_paragraphs=context_paragraphs,
            fixed_key=bootstrap_key,
            fixed_context=question,
        )
        target_key = bucketizer.key_for_text(draft.text, self.semantic_encoder)

        for attempt in range(1, max_answer_passes + 1):
            generated = self._generate_pass(
                question=question,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                secret_key=secret_key,
                strength=strength,
                top_k=top_k,
                temperature=temperature,
                rng_seed=rng_seed + attempt - 1,
                watermarked=True,
                bucketizer=bucketizer,
                semantic_scope="answer",
                context_paragraphs=context_paragraphs,
                fixed_key=target_key,
                fixed_context=draft.text,
            )
            final_key = bucketizer.key_for_text(generated.text, self.semantic_encoder)
            if final_key.bucket != target_key.bucket:
                continue
            trace = self._rescore_answer_trace(
                generated.trace,
                semantic_key=final_key,
                secret_key=secret_key,
                semantic_context=generated.text,
            )
            return self._build_result(
                generated=generated,
                question=question,
                max_new_tokens=max_new_tokens,
                secret_key=secret_key,
                strength=strength,
                top_k=top_k,
                temperature=temperature,
                threshold_z=threshold_z,
                rng_seed=rng_seed + attempt - 1,
                watermarked=True,
                bucket_count=bucket_count,
                semantic_scope="answer",
                context_paragraphs=context_paragraphs,
                semantic_key=final_key,
                semantic_key_text=generated.text,
                draft_text=draft.text,
                answer_key_attempts=attempt,
                answer_key_stable=True,
                trace=trace,
            )

        raise RuntimeError(
            "complete-answer semantic key did not stabilize after "
            f"{max_answer_passes} marked passes; try a smaller --bucket-count or "
            "increase --max-answer-passes"
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
    semantic_scope: str = DEFAULT_SEMANTIC_SCOPE,
    context_paragraphs: int = 1,
) -> DetectionResult:
    """Retokenize an assistant answer and run semantic-key detection.

    Only the generator tokenizer and lightweight semantic encoder are loaded;
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
        semantic_scope=semantic_scope,
        context_paragraphs=context_paragraphs,
    )
