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

## Real Qwen3.5 article demo

The default Qwen demo now uses an article-style prompt that is easier to read and compare than a short sentence continuation:

> **Write a short plain-language article answering: What is AI? Explain what AI is, where it is used, benefits, risks, and conclude briefly.**

Install locally:

```bash
python -m pip install -e ".[real-llm]"
```

Run the default matched experiment:

```bash
seedmark qwen-demo \
  --model Qwen/Qwen3.5-0.8B \
  --output-dir results/qwen
```

The default allows up to **128 new tokens** per marked/control generation so the short article has room to cover the requested points. You can override the prompt or length normally:

```bash
seedmark qwen-demo \
  --prompt "Explain edge AI in plain language." \
  --max-new-tokens 96
```

The exact detector can operate on saved token IDs. The convenience text detector loads only the public tokenizer, not model weights:

```bash
seedmark qwen-detect \
  --text-file results/qwen/generated_watermarked.txt
```

See [`docs/real-llm.md`](docs/real-llm.md) for the method and scientific caveats.

## What the Qwen report makes explicit

The top of `report.html` is a matched comparison, not a single detector score:

```text
watermarked output          → Detected
control / without watermark → Not detected
```

Badges always reflect the actual run. If a run does not achieve the expected contrast, the report says **Review this run** rather than silently presenting it as a success.

The report also contains:

- two clear result cards with z-score, `1-p`, prioritized-token share and threshold;
- `generation.gif` embedded directly in the report;
- `detection.gif` embedded directly in the report;
- watermarked and control text side by side;
- a sliding recent-context sentence view that wraps cleanly and highlights **only the current token**;
- a marked-vs-control z-score chart with the decision threshold;
- one-sided confidence (`1-p`) and prioritized-token-share views.

The displayed `1-p` is confidence against the detector's null model. It is **not** a posterior probability that a passage was written by AI.

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

## Report outputs

Every real-Qwen experiment produces a standalone `report.html` plus the two GIFs, static previews, marked/control text files and JSON traces. The generation animation exposes base-vs-marked candidate probabilities and the current selected token; the detection animation focuses on marked/control statistical evidence.

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

Qwen defaults use `Qwen/Qwen3.5-0.8B`, the **“What is AI?” article prompt** above, top-k 20, temperature 1.0, strength 1.5 and up to 128 new tokens. For archival experiments, pin the Hugging Face model revision as well as the SeedMark version.

## Tests

```bash
python -m unittest discover -s tests -v
```

CI also builds the Python distribution to validate package metadata and the single-source version contract. Normal CI does not download Qwen weights.

## Scientific use

SeedMark is useful for lectures, research discussions, and as a minimal baseline before experimenting with production-grade tokenizer-level watermark algorithms. Please read [`docs/limitations.md`](docs/limitations.md) and [`docs/real-llm.md`](docs/real-llm.md) before interpreting detector scores.

## License

SeedMark code is MIT. Qwen3.5 model weights are distributed separately under their model license.
