# Real-LLM experiment

SeedMark's second experiment replaces the transparent bigram model with a real Qwen model while preserving the same scientific question: **can the detector recover a keyed token-selection correlation without receiving the model's probability distribution?**

## Models

Default: `Qwen/Qwen3.5-0.8B` (0.8B parameters, Apache-2.0 model weights).

Higher-quality optional comparison: `Qwen/Qwen3.5-2B` (2B parameters, Apache-2.0). Qwen's published model-card benchmarks report materially higher scores for the 2B model on most listed language/reasoning evaluations; it therefore serves as a useful quality-vs-compute comparison.

There is no official `Qwen3.6-0.8B` model in the Qwen collection at the time this adapter was added. Qwen3.6 is offered at much larger sizes; the intended small model here is **Qwen3.5-0.8B**.

## Generation

At each position, SeedMark asks the actual Qwen model for next-token logits, applies temperature, keeps the model's top-k candidates, converts them to base probabilities `p(v)`, computes a secret-keyed pseudorandom value for each token ID,

```text
u(t,v) = HMAC-SHA256(key, SHA256(first_word) || t || token_id) -> [0,1)
```

and samples from

```text
q(v) ∝ p(v) exp(strength * (2u(t,v)-1)).
```

The report preserves both `p(v)` and `q(v)` so the effect is inspectable token by token.

## Detection

The detector receives the observed generated token IDs, first word, and key. It does **not** receive Qwen logits, probabilities, hidden states, or model weights. The text-only convenience detector downloads only the public tokenizer and retokenizes the text.

Exact token IDs saved in `watermarked-trace.json` are the authoritative representation because decode-then-retokenize round trips can change token boundaries after edits or normalization.

## Scientific boundary

This is a deliberately distortionary baseline designed for teaching and ablation. It is not a reproduction of SynthID-Text or any vendor watermark. Results must be reported together with text quality, generation settings, model name/revision, text length, edit conditions, and false-positive calibration.
