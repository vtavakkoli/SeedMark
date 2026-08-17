# SeedMark 🌱🔐

**An interactive, scientific teaching lab for first-word-seeded keyed pseudorandom text watermarking.**

SeedMark demonstrates a precise idea: **a detector can recover statistical correlation with a secret pseudorandom token-selection rule without having the language model's original next-token probability distribution.**

> [!IMPORTANT]
> SeedMark is a deliberately simplified educational algorithm. It is **not** Anthropic's production watermark, **not** Google SynthID-Text, **not** C2PA, and does not claim that any vendor uses a first-word seed.

## Two experiment levels

SeedMark now has both an inspectable toy baseline and a **real-LLM experiment**:

| Backend | Purpose | Model probabilities used in generation? | Needed for detection? |
|---|---|---:|---:|
| `ToyBigramLM` | transparent mathematical baseline | yes | no |
| `Qwen/Qwen3.5-0.8B` | real Transformer logits | yes | **no** |
| `Qwen/Qwen3.5-2B` | stronger optional real-model comparison | yes | **no** |

The small default is **Qwen3.5-0.8B**. There is currently no official Qwen3.6 0.8B model; the official Qwen3.6 family is much larger. Both Qwen3.5 models above use Apache-2.0 model weights, so SeedMark can run them locally without an inference API fee; local compute/storage still have a cost.

## What it does

The first word becomes a public SHA-256 seed. For every later candidate, SeedMark derives a deterministic HMAC-SHA256 score from:

```text
first word → SHA-256 seed
                + secret key
                + token position
                + candidate token / token ID
                      ↓
              pseudorandom u ∈ [0,1)
```

During **watermarked generation**, high-scoring candidates receive a small probability boost. During **detection**, SeedMark reconstructs the score of each chosen token and tests whether the average is unusually high.

For the real Qwen path, the generator uses genuine model logits and actual tokenizer token IDs. The detector receives the token IDs (or retokenized text), first-word seed, and secret key—but **not Qwen logits, hidden states, or probability distributions**.

## Real Qwen3.5 experiment

Install the optional real-model stack:

```bash
python -m pip install -e ".[real-llm]"
```

Run the small 0.8B model:

```bash
seedmark qwen-demo \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Research is" \
  --max-new-tokens 64 \
  --top-k 20 \
  --output-dir results/qwen
```

For better language/reasoning quality at higher memory/compute cost:

```bash
seedmark qwen-demo \
  --model Qwen/Qwen3.5-2B \
  --prompt "Research is" \
  --max-new-tokens 64 \
  --output-dir results/qwen-2b
```

The Qwen model is downloaded directly from Hugging Face on the first run. `qwen-demo` loads the model **once** and produces a matched watermarked/control pair using the same RNG seed.

The exact detector can operate on saved token IDs. The convenience text detector loads only the public tokenizer, not model weights:

```bash
seedmark qwen-detect \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Research is" \
  --text-file results/qwen/generated_watermarked.txt
```

See [`docs/real-llm.md`](docs/real-llm.md) for the method and scientific caveats.

## Docker: real LLM

The real-model containers are behind a Compose profile so the lightweight toy demo stays fast:

```bash
docker compose --profile real-llm up --build qwen qwen-report
```

Then open:

```text
http://localhost:8081/report.html
```

The Hugging Face cache is stored in a named Docker volume, so the model does not need to be downloaded on every run.

## Interactive report

Every experiment produces a standalone `report.html`. The Qwen report shows:

- ▶️ animated **real token-by-token** playback;
- decoded candidate token and tokenizer ID;
- actual base probability from Qwen's logits;
- watermark-adjusted sampling probability;
- keyed pseudorandom score for every top-k candidate;
- chosen token and cumulative detector z-score;
- matched watermarked and unwatermarked outputs;
- JSON traces containing every model/watermark decision.

The original toy experiment additionally provides Monte Carlo TPR/FPR, 95% Wilson intervals and empirical ROC/AUC.

## One-command toy Docker demo

```bash
docker compose up --build
```

The experiment service generates `results/run/`; after it completes, the report service serves:

```text
http://localhost:8080/report.html
```

## Local Python 3.13 toy baseline

The base SeedMark package has **zero runtime dependencies**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e .
seedmark experiment --output-dir results/run --trials 300 --length 80
```

## The teaching algorithm

For first word `w0`:

```text
s = SHA256(normalize(w0))
u(t,v) = HMAC_SHA256(key, s || position || token) → [0,1)
```

The real-model version substitutes the tokenizer's integer `token_id` for `token`.

For the base top-k distribution `p_t(v)`, marked sampling uses:

```text
q_t(v) ∝ p_t(v) · exp(strength · (2u(t,v) - 1))
```

The detector reconstructs only the chosen-token values `u_t`. Under an unwatermarked null, those keyed values should average approximately `0.5`, so SeedMark computes:

```text
z = Σ(u_t - 0.5) / sqrt(n / 12)
```

This is the central experiment: **generation needs a language-model distribution; detection does not.**

## Repository layout

```text
src/seedmark/          core algorithm, toy LM, Qwen adapter, detector, reports, CLI
examples/              minimal no-distribution detector example
tests/                 deterministic dependency-free unit tests
docs/                  method, real-LLM design, limitations
results/                generated reports and reference notes
.github/workflows/      Python 3.13 CI + experiment artifact
Dockerfile              lightweight toy experiment
Dockerfile.qwen         real Hugging Face/Qwen experiment
docker-compose.yml      toy + opt-in real-LLM profile
```

## Reproducibility

Toy defaults:

| Parameter | Value |
|---|---:|
| Python | 3.13 |
| first word | `research` |
| top-k | 8 |
| watermark strength | 1.5 |
| z threshold | 3.0 |
| generated words | 80 |
| Monte Carlo trials | 300 |
| experiment RNG seed | 20260817 |

Qwen defaults use `Qwen/Qwen3.5-0.8B`, prompt `Research is`, top-k 20, temperature 1.0, strength 1.5 and 64 new tokens. The report records the selected model, prompt, token IDs, real base probabilities, keyed scores and adjusted probabilities.

## Tests

```bash
python -m unittest discover -s tests -v
```

The normal CI does not download Qwen weights. It tests the real-model watermark mathematics independently of Transformers, while the optional Docker path performs the actual model-backed experiment locally.

## Scientific use

SeedMark is useful for lectures, research discussions, and as a minimal baseline before experimenting with production-grade tokenizer-level watermark algorithms. The Qwen path makes it possible to measure watermark strength against real-model text quality, temperature, top-k, generation length and model size.

Please read [`docs/limitations.md`](docs/limitations.md) and [`docs/real-llm.md`](docs/real-llm.md) before interpreting detector scores.

## License

SeedMark code is MIT. Qwen3.5 model weights are distributed separately under their model license (Apache-2.0 at the time this adapter was added).
