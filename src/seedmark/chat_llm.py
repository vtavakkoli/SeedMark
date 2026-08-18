"""Chat-oriented real-Qwen adapter for SeedMark.

The public demo should behave like a normal assistant conversation rather than a
raw text-completion benchmark. The user asks a question, Qwen's own chat template
builds the model input, and SeedMark modifies only the assistant's next-token
sampling distribution. Detection still sees only the generated assistant token
IDs, the first word of the user question, and the secret key.
"""

from __future__ import annotations

import math
from typing import Any

from .core import normalize_word
from .hf_llm import (
    DEFAULT_MODEL,
    HFCandidateTrace,
    HFGenerationResult,
    HFStepTrace,
    QwenSeedMark,
    _load_tokenizer,
    _optional_stack,
    detect_token_ids,
    token_id_score,
)

DEFAULT_CHAT_QUESTION = "What is AI?"
DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly as a short "
    "plain-language article. Explain what AI is, where it is used, its main benefits, "
    "its main risks, and end with a brief conclusion. Do not expose chain-of-thought, "
    "analysis tags, or internal reasoning."
)


def chat_messages(question: str, system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Build the two-message conversation used by the real-Qwen demo."""
    question = question.strip()
    system_prompt = system_prompt.strip()
    if not question:
        raise ValueError("question must not be empty")
    if not system_prompt:
        raise ValueError("system_prompt must not be empty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def render_chat_prompt(
    tokenizer: Any,
    question: str,
    system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> str:
    """Render Qwen's native chat template with an assistant-generation marker."""
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("the selected tokenizer does not provide a chat template")
    rendered = tokenizer.apply_chat_template(
        chat_messages(question, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("chat template returned an empty prompt")
    return rendered


def _display_prompt(question: str) -> str:
    """Human-readable transcript prefix used in reports and animations."""
    return f"User: {question.strip()}\nAssistant:"


class ChatQwenSeedMark(QwenSeedMark):
    """Generate a marked or control assistant answer from a real Qwen chat prompt."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
        super().__init__(model_name=model_name, device=device)

    def _encode_chat(self, question: str, system_prompt: str) -> tuple[Any, Any]:
        rendered = render_chat_prompt(self.tokenizer, question, system_prompt)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = self.torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        return input_ids, attention_mask

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
    ) -> HFGenerationResult:
        """Generate only the assistant answer, using Qwen's native chat template."""
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

        torch = self.torch
        seed_word = normalize_word(question)
        input_ids, attention_mask = self._encode_chat(question, system_prompt)
        prompt_token_ids = tuple(int(x) for x in input_ids[0].detach().cpu().tolist())
        generated_ids: list[int] = []
        trace: list[HFStepTrace] = []
        scores_seen: list[float] = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng_seed)
        eos_id = int(self.tokenizer.eos_token_id)

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

                # EOS is a chat-control token, not visible assistant text. Do not add it
                # to the watermark trace so saved text can be retokenized consistently.
                if chosen_id == eos_id:
                    break

                chosen_score = keyed_scores[chosen_index]
                scores_seen.append(chosen_score)
                n = len(scores_seen)
                cumulative_z = (sum(scores_seen) - 0.5 * n) / math.sqrt(n / 12.0)

                candidate_trace = tuple(
                    HFCandidateTrace(
                        token_id=token_id,
                        token_text=self.tokenizer.decode(
                            [token_id], clean_up_tokenization_spaces=False
                        ),
                        base_probability=float(base[idx]),
                        generation_probability=float(generation[idx]),
                        watermark_score=keyed_scores[idx],
                        chosen=idx == chosen_index,
                    )
                    for idx, token_id in enumerate(candidate_ids)
                )
                trace.append(
                    HFStepTrace(
                        position=position,
                        chosen_token_id=chosen_id,
                        chosen_token_text=self.tokenizer.decode(
                            [chosen_id], clean_up_tokenization_spaces=False
                        ),
                        chosen_base_probability=float(base[chosen_index]),
                        chosen_generation_probability=float(generation[chosen_index]),
                        chosen_watermark_score=chosen_score,
                        cumulative_z=cumulative_z,
                        candidates=candidate_trace,
                    )
                )
                generated_ids.append(chosen_id)
                next_id = torch.tensor([[chosen_id]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_id)), dim=1
                )

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        detection = detect_token_ids(
            generated_ids,
            secret_key=secret_key,
            first_word=seed_word,
            threshold_z=threshold_z,
        )
        return HFGenerationResult(
            model_name=self.model_name,
            prompt=_display_prompt(question),
            first_word=seed_word,
            text=answer,
            continuation=answer,
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


def detect_chat_text_with_tokenizer(
    *,
    model_name: str,
    text: str,
    question: str = DEFAULT_CHAT_QUESTION,
    system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
    secret_key: str,
    threshold_z: float = 3.0,
):
    """Retokenize a saved assistant answer using the same Qwen chat prefix.

    Saved token IDs in the JSON trace remain authoritative. This convenience path
    intentionally loads only the tokenizer and does not load model weights.
    """
    _, AutoTokenizer, _ = _optional_stack()
    tokenizer = _load_tokenizer(AutoTokenizer, model_name)
    rendered = render_chat_prompt(tokenizer, question, system_prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(rendered + text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant text does not tokenize with the reconstructed chat prefix")
    generated_ids = full_ids[len(prompt_ids):]
    return detect_token_ids(
        generated_ids,
        secret_key=secret_key,
        first_word=normalize_word(question),
        threshold_z=threshold_z,
    )
