# Real-LLM chat experiment

SeedMark's real-model experiment works with Hugging Face LLMs while preserving the same scientific question: **can the detector recover keyed token-selection correlation without receiving the model's probability distribution?**

## Model-agnostic API

The preferred Python API is now model-agnostic:

```python
from seedmark import LLMSeedMark, ChatLLMSeedMark, SemanticChatLLMSeedMark
```

Use `LLMSeedMark` for raw text completion, `ChatLLMSeedMark` for tokenizers that provide a native chat template, and `SemanticChatLLMSeedMark` for the semantic self-keyed chat experiment.

Standard text checkpoints are loaded through Hugging Face `AutoModelForCausalLM`. If a checkpoint is not registered for that auto class, SeedMark falls back to `AutoModelForMultimodalLM` when available. This preserves support for the current Qwen3.5 default while allowing ordinary causal LLMs to use the same code path.

The historical names `QwenSeedMark`, `ChatQwenSeedMark`, and `SemanticChatQwenSeedMark` remain available for backward compatibility, but new code should use the `LLM` names.

## Models

Default: `Qwen/Qwen3.5-0.8B`.

Optional higher-quality comparison: `Qwen/Qwen3.5-2B`.

The API is not limited to Qwen. Standard Hugging Face causal checkpoints can be selected with `model_name`, for example:

```python
lab = ChatLLMSeedMark(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="auto",
)
```

For chat generation, the selected tokenizer must provide `apply_chat_template(...)`. Raw completion through `LLMSeedMark` does not require a chat template.

## Real chat state

The public demo is intentionally **chat-based**, not a raw completion from text such as `Research is`.

The default conversation is:

```text
system: You are a helpful assistant. Answer the user's question directly as a
        short plain-language article. Explain what AI is, where it is used,
        its main benefits, its main risks, and end with a brief conclusion.

user:   What is AI?

assistant: <generated article>
```

SeedMark calls the tokenizer's native `apply_chat_template(...)` with an assistant-generation prompt. If the model's template explicitly advertises an `enable_thinking` option, SeedMark sets it to `False` for the public demonstration. Other model templates are left unchanged.

The system message and user message are normal context. **Only generated assistant tokens are watermarked and scored.**

## Matched experiment

The same system message, user question, model settings, and RNG seed are used for two runs:

```text
watermarked assistant answer  → keyed probability nudge enabled
control assistant answer      → original model sampling
```

The report makes the intended detector contrast explicit:

```text
watermarked output          → Detected
control / without watermark → Not detected
```

These are not hard-coded outcomes. Badges reflect the actual detector results, and the report displays **Review this run** if the expected contrast is not achieved.

## Generation

At each assistant-token position, SeedMark asks the selected Hugging Face model for next-token logits, applies temperature, keeps the model's top-k candidates, and converts them to base probabilities `p(v)`.

The first normalized word of the **user question** is the public seed word. With the default question `What is AI?`, the seed word is `what`.

For each candidate token ID:

```text
u(t,v) = HMAC-SHA256(
    key,
    SHA256(first_word_of_user_question) || t || token_id
) -> [0,1)
```

Watermarked generation samples from:

```text
q(v) ∝ p(v) exp(strength * (2u(t,v)-1)).
```

The matched control samples from `p(v)` without the keyed reweighting.

The trace preserves both `p(v)` and the actual generation probability so the watermark nudge remains inspectable token by token.

## Detection

The detector receives the observed generated **assistant token IDs**, first-word seed, and secret key. It does **not** receive model logits, probabilities, hidden states, or model weights.

Exact token IDs saved in `watermarked-trace.json` and `control-trace.json` are the authoritative representation. The tokenizer-only detection helper reconstructs the same chat prefix, retokenizes the saved assistant answer, and then applies the detector without loading generator weights.

The detector reports cumulative z-score, one-sided p-value / `1-p`, and prioritized-token share. `detection.gif` overlays the marked and control z-curves with the decision threshold.

## Output semantics

`generated_watermarked.txt` and `generated_control.txt` contain the assistant answers only. They do not contain the serialized system/user chat template.

The human-readable report displays the conversation separately:

```text
User: What is AI?
Assistant: <article>
```

This makes the demonstration resemble a normal local-AI interaction while keeping the scientific trace precise.

## Docker Compose and CLI compatibility

The current Compose workflow and command names retain their Qwen-oriented names for backward compatibility with existing scripts. They still default to the Qwen checkpoint, while the Python library API is now generic.

The real chat demo is the default Compose workflow:

```bash
docker compose up --build
```

It uses a persistent Hugging Face cache outside the repository, runs the marked/control chat experiment, and serves:

```text
http://localhost:8081/report.html
```

The toy bigram baseline is opt-in:

```bash
docker compose --profile toy up --build experiment report
```

## Scientific boundary

This is a deliberately simple teaching and ablation baseline. It is not a reproduction of SynthID-Text or any vendor watermark. Results should be reported with model/revision, system prompt, user question, generation settings, text length, edit conditions, quality observations, and false-positive calibration.

`1-p` is confidence against this detector's null model, **not** a posterior probability that a passage was generated by AI.
