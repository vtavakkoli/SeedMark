# SeedMark 🌱🔐

**An interactive, scientific teaching lab for first-word-seeded keyed pseudorandom text watermarking.**

SeedMark demonstrates a precise idea: **a detector can recover statistical correlation with a secret pseudorandom token-selection rule without having the language model's original next-token probability distribution.**

> [!IMPORTANT]
> SeedMark is a deliberately simplified educational algorithm. It is **not** Anthropic's production watermark, **not** Google SynthID-Text, **not** C2PA, and does not claim that any vendor uses a first-word seed.

## Two experiment levels

| Backend | Purpose | Model probabilities used in generation? | Needed for detection? |
|---|---|---:|---:|
| `ToyBigramLM` | transparent mathematical baseline | yes | no |
| `Qwen/Qwen3.5-0.8B` | real Transformer logits | yes | **no** |
| `Qwen/Qwen3.5-2B` | stronger optional real-model comparison | yes | **no** |

The small default is **Qwen3.5-0.8B**. The real-model workflow is local and uses Hugging Face model weights rather than a paid inference API.

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

For the real Qwen path, generation uses genuine model logits and tokenizer token IDs. Detection receives the token IDs (or retokenized text), first-word seed, and secret key—but **not Qwen logits, hidden states, or probability distributions**.

## Real Qwen3.5 experiment

Install locally:

```bash
python -m pip install -e ".[real-llm]"
```

Run:

```bash
seedmark qwen-demo \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Research is" \
  --max-new-tokens 64 \
  --top-k 20 \
  --output-dir results/qwen
```

The exact detector can operate on saved token IDs. The convenience text detector loads only the public tokenizer, not model weights:

```bash
seedmark qwen-detect \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Research is" \
  --text-file results/qwen/generated_watermarked.txt
```

See [`docs/real-llm.md`](docs/real-llm.md) for the method and scientific caveats.

## Docker: fast rebuilds + persistent model cache

The Qwen image is structured for Docker layer reuse:

```text
requirements/real-llm.txt
        ↓
install torch / transformers / huggingface_hub   ← expensive cached layer
        ↓
copy src/                                        ← cheap source layer
        ↓
python -m seedmark.cli                           ← no package reinstall
```

This means editing SeedMark Python source does **not** reinstall PyTorch/Transformers on every Docker build.

The model cache is a normal host directory **outside the repository**. By default Compose uses:

```text
../seedmark-model-cache
```

The first run downloads Qwen into that directory. Later runs reuse Hugging Face's cached snapshot instead of downloading the same weights again. To use a different location, copy `.env.example` to `.env` and set:

```text
SEEDMARK_MODEL_CACHE=D:/model-cache/seedmark
```

You can explicitly prefetch/inspect the model cache:

```bash
docker compose --profile real-llm run --rm qwen-cache
```

or locally:

```bash
seedmark qwen-cache --model Qwen/Qwen3.5-0.8B
```

Run the full experiment:

```bash
docker compose --profile real-llm up --build qwen qwen-report
```

Then open:

```text
http://localhost:8081/report.html
```

On the second run, the dependency layer and model files are reused unless their dependency/model revision changed.

## Versioning

SeedMark follows **Semantic Versioning**. The package version has one source of truth and is available from Python and the CLI:

```bash
seedmark --version
```

```python
import seedmark
print(seedmark.__version__)
```

See [`CHANGELOG.md`](CHANGELOG.md) and [`docs/versioning.md`](docs/versioning.md). Package version and Qwen model revision are tracked separately so experiments can be reproduced precisely.

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

The toy experiment additionally provides Monte Carlo TPR/FPR, 95% Wilson intervals and empirical ROC/AUC.

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

## Teaching algorithm

For first word `w0`:

```text
s = SHA256(normalize(w0))
u(t,v) = HMAC_SHA256(key, s || position || token) → [0,1)
```

The real-model version substitutes the tokenizer's integer `token_id` for `token`.

For base top-k distribution `p_t(v)`, marked sampling uses:

```text
q_t(v) ∝ p_t(v) · exp(strength · (2u(t,v) - 1))
```

The detector reconstructs only the chosen-token values `u_t`. Under the teaching null, those keyed values should average approximately `0.5`:

```text
z = Σ(u_t - 0.5) / sqrt(n / 12)
```

This is the central experiment: **generation needs a language-model distribution; detection does not.**

## Repository layout

```text
src/seedmark/              algorithms, Qwen adapter, cache helper, reports, CLI
requirements/              Docker dependency manifests
tests/                     deterministic unit + packaging contract tests
docs/                      method, real-LLM design, limitations, version policy
results/                   generated reports and reference notes
.github/workflows/          Python 3.13 CI + package build + report artifact
Dockerfile                  lightweight toy experiment
Dockerfile.qwen             optimized real-Qwen image
docker-compose.yml          toy + opt-in real-LLM/cache services
CHANGELOG.md                release history
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

Qwen defaults use `Qwen/Qwen3.5-0.8B`, prompt `Research is`, top-k 20, temperature 1.0, strength 1.5 and 64 new tokens. For archival experiments, pin the Hugging Face model revision as well as the SeedMark version.

## Tests

```bash
python -m unittest discover -s tests -v
```

CI also builds the Python distribution to validate package metadata and the single-source version contract. Normal CI does not download Qwen weights.

## Scientific use

SeedMark is useful for lectures, research discussions, and as a minimal baseline before experimenting with production-grade tokenizer-level watermark algorithms. Please read [`docs/limitations.md`](docs/limitations.md) and [`docs/real-llm.md`](docs/real-llm.md) before interpreting detector scores.

## License

SeedMark code is MIT. Qwen3.5 model weights are distributed separately under their model license.
