# SeedMark 🌱🔐

<p align="center">
  <strong>A chat-first, inspectable, model-agnostic teaching lab for keyed pseudorandom text watermarking.</strong>
</p>

<p align="center">
  <img alt="Python 3.13+" src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-2ea44f">
  <img alt="Purpose Research and Education" src="https://img.shields.io/badge/Purpose-Research%20%26%20Education-6f42c1">
</p>

SeedMark demonstrates one focused idea:

> **A detector can recover statistical correlation with a secret token-selection rule without receiving the language model's original next-token probability distribution.**

The **SeedMark method is LLM-agnostic**: it operates at the token-sampling layer and can be integrated with autoregressive language models that expose token IDs and next-token scores/probabilities. The repository currently includes **Qwen3.5 as the bundled reference implementation and default demo backend**.

The default experiment is deliberately easy to inspect. SeedMark asks the reference LLM **“What is AI?”**, generates the answer once **with** watermarking and once **without** watermarking, and then runs the same detector on both outputs.

| Watermarked answer | Control / no watermark |
|---|---|
| expected **Detected** | expected **Not detected** |
| token sampling is keyed and biased | ordinary model sampling |
| detector reconstructs keyed token scores | same detector, same threshold |

> [!IMPORTANT]
> SeedMark is a deliberately simplified **educational algorithm**. It is **not** Anthropic's production watermark, **not** Google SynthID-Text, **not** C2PA, and does not claim that any vendor uses this first-word-seeded construction.

---

## 🎬 See SeedMark work

### 1. Watermarked generation

The generation animation shows the answer being produced token by token. The display keeps a sliding recent-context window so the text stays readable and highlights **only the token currently being generated**.

<p align="center">
  <img src="generation.gif" alt="SeedMark watermarked LLM text generation animation" width="900">
</p>

**What to watch:**

- the LLM produces its normal next-token distribution;
- SeedMark assigns deterministic keyed pseudorandom values to candidate token IDs;
- the candidate probabilities are reweighted before sampling;
- only the **current token** is highlighted;
- previously generated text remains visible in a moving context window;
- the visible answer still reads like a normal assistant response.

The bundled animation was produced with the repository's **Qwen3.5 reference backend**, but the watermarking logic itself is not tied to Qwen.

[Open `generation.gif` directly](generation.gif)

### 2. Statistical detection

The detection animation shows what happens after generation. It compares the accumulated detector statistic for the **watermarked output** with the matched **control output**.

<p align="center">
  <img src="detection.gif" alt="SeedMark marked versus control detection animation" width="900">
</p>

**What to watch:**

- the **marked z-curve** accumulates evidence from the selected tokens;
- the **control z-curve** shows the same detector on text generated without watermarking;
- the horizontal line is the configured detection threshold;
- the report displays one-sided confidence as `1-p`;
- the intended demonstration is **watermarked → Detected** and **control → Not detected**.

[Open `detection.gif` directly](detection.gif)

> [!NOTE]
> `1-p` is confidence against this detector's null hypothesis. It is **not** a posterior probability that a text was written by AI.

---

## 🧩 LLM and backend support

SeedMark is designed around a small model-facing contract rather than a particular model family.

A compatible LLM backend needs to provide:

1. a tokenizer with stable token IDs;
2. access to next-token logits or probabilities during generation;
3. a way to append the selected token and continue autoregressive generation;
4. for chat models, an appropriate chat template or prompt formatter.

Conceptually, the flow is:

```text
any compatible autoregressive LLM
             │
             ▼
      next-token scores
             │
             ▼
     SeedMark reweighting
             │
       ┌─────┴─────┐
       │           │
   watermarked   control
    sampling     sampling
       │           │
       └─────┬─────┘
             ▼
       generated token IDs
             │
             ▼
      SeedMark detector
```

### Current reference backend

This repository currently ships a working Hugging Face reference implementation using:

```text
Qwen/Qwen3.5-0.8B
```

Qwen is used here because it provides a practical public model for the demonstration. It should be read as **an example backend, not as a requirement of the SeedMark algorithm**.

Other autoregressive LLMs can be integrated by providing the same token-level generation information. The current CLI and Docker convenience commands are still named around Qwen because that is the backend implemented in this repository today.

---

## 🚀 Quick start with Docker Compose

The bundled real-LLM reference workflow uses **Qwen3.5** by default:

```bash
docker compose up --build
```

SeedMark will:

1. build the reference LLM image;
2. download `Qwen/Qwen3.5-0.8B` only when it is not already in the persistent cache;
3. ask **“What is AI?”** through the model's chat template;
4. generate one watermarked assistant article;
5. generate one matched control article without the watermark;
6. detect both outputs;
7. create the GIFs, traces, text outputs, summary data, and `report.html`;
8. serve the report at:

```text
http://localhost:8081/report.html
```

Stop the services with:

```bash
docker compose down
```

### Persistent model cache

By default, Hugging Face model files are stored outside the repository:

```text
../seedmark-model-cache
```

That prevents the reference model from being downloaded again on every run.

To choose another location, set for example:

```text
SEEDMARK_MODEL_CACHE=D:/model-cache/seedmark
```

in `.env`.

The reference-model Dockerfile installs heavy dependencies **before** application source is copied, so normal source edits do not force PyTorch and Transformers to be reinstalled.

---

## 💬 Default experience: a real LLM chat

SeedMark behaves like an assistant conversation rather than a raw completion benchmark.

```text
System:
You are a helpful assistant. Answer the user's question directly as a short
plain-language article. Explain what AI is, where it is used, its main benefits,
its main risks, and end with a brief conclusion.

User:
What is AI?

Assistant:
<article generated by the selected LLM>
```

For the bundled demo, the selected LLM is **Qwen3.5** and SeedMark uses Qwen's native chat template before sampling. A different chat-capable backend should use that model's own compatible chat template or prompt formatter.

Thinking/reasoning output is disabled for the current Qwen demo so the visible output is the requested article rather than internal reasoning text.

The identical chat request is evaluated through two generation paths:

```text
                      What is AI?
                           │
                  same chat context
                           │
               ┌───────────┴───────────┐
               │                       │
       SeedMark enabled         SeedMark disabled
               │                       │
     watermarked answer          control answer
               │                       │
               └───────────┬───────────┘
                           │
                    same detector
                           │
              ┌────────────┴────────────┐
              │                         │
          Detected              expected Not detected
```

The detector uses:

- the generated assistant token IDs;
- the first word of the user question (`what` for `What is AI?`);
- the secret key.

It does **not** need the LLM's logits, hidden states, model weights, or original next-token probability distribution during detection.

---

## 🧠 How the watermark works

Only the **assistant response tokens** are watermarked. The system message and user question remain ordinary chat context.

At every assistant-token position, the selected LLM provides its normal top-k next-token distribution `p(v)`.

SeedMark assigns each candidate token ID a deterministic keyed value:

```text
u(t,v) = HMAC-SHA256(
    secret_key,
    SHA256(first_word_of_user_question) || position || token_id
) → [0,1)
```

Watermarked generation samples from a reweighted distribution:

```text
q(v) ∝ p(v) · exp(strength · (2u(t,v) - 1))
```

The matched control samples directly from the original LLM distribution:

```text
p(v)
```

So the experiment changes the sampling rule, **not** the detector after the fact.

### Detection

For the selected output tokens, the detector reconstructs their keyed `u` values and computes:

```text
z = Σ(u_t - 0.5) / sqrt(n / 12)
```

The central separation is therefore:

> **Generation needs the LLM distribution. Detection does not.**

That is the core property SeedMark is designed to make visible and easy to study.

---

## 🧪 Why the control matters

A watermark demo is much more informative when it shows both sides of the experiment.

SeedMark therefore reports the pair explicitly:

```text
Watermarked output  → detector → expected DETECTED
Control output      → detector → expected NOT DETECTED
```

The control is generated from the same LLM and chat setup but without SeedMark's probability reweighting. This makes it easier to see whether the detector is responding to the keyed sampling signal rather than simply producing a high score for any model-generated text.

The report does **not** hard-code success. If a stochastic run does not produce the intended marked/control separation, the result is labeled **Review this run**.

---

## 📊 Generated report

A default run with the bundled Qwen reference backend produces:

```text
results/qwen/
├── report.html
├── generation.gif
├── detection.gif
├── generation-preview.png
├── detection-preview.png
├── generated_watermarked.txt
├── generated_control.txt
├── watermarked-trace.json
├── control-trace.json
└── summary.json
```

`generated_watermarked.txt` and `generated_control.txt` contain the **assistant answers only**, so they can be compared directly as normal chat responses.

The HTML report includes:

- the system/user chat context;
- the LLM/model identifier used for the run;
- a clear **Watermarked → Detected** result card;
- a clear **Control / without watermark → Not detected** result card when that contrast is achieved;
- side-by-side assistant articles;
- the animated generation walkthrough;
- the animated detection walkthrough;
- marked and control z-curves;
- the decision threshold;
- one-sided confidence (`1-p`);
- prioritized-token share;
- raw token traces for inspection.

---

## 🖥️ Run locally

Install the real-LLM dependencies:

```bash
python -m pip install -e ".[real-llm]"
```

Run the bundled Qwen reference experiment:

```bash
seedmark qwen-demo --output-dir results/qwen
```

The `qwen-demo` command name identifies the **currently implemented reference adapter**; it does not mean the SeedMark watermarking method itself requires Qwen.

### Default reference-demo parameters

| Parameter | Default |
|---|---|
| Reference model | `Qwen/Qwen3.5-0.8B` |
| User question | `What is AI?` |
| Output style | short plain-language article |
| Max assistant tokens | `128` |
| Top-k | `20` |
| Temperature | `1.0` |
| Watermark strength | `1.5` |
| Detector threshold | `z = 3.0` |
| RNG seed | `20260817` |

Ask another question with the reference backend:

```bash
seedmark qwen-demo \
  --question "What is edge AI?" \
  --max-new-tokens 128 \
  --output-dir results/edge-ai
```

`--prompt` remains available as a backward-compatible alias for `--question`.

Override the system instruction:

```bash
seedmark qwen-demo \
  --question "What is AI?" \
  --system-prompt "Answer in simple language for a general audience."
```

---

## 🔎 Detect a saved assistant answer

The authoritative detector input is the saved token-ID trace.

A detector does not need the original LLM inference graph or generation logits. It needs the output token IDs interpreted with the tokenizer corresponding to the model that generated them, plus the SeedMark key/seed inputs.

For the bundled Qwen reference backend, a convenience text detector is available and loads only the tokenizer, not the model weights:

```bash
seedmark qwen-detect \
  --question "What is AI?" \
  --text-file results/qwen/generated_watermarked.txt
```

The same chat template and question must be supplied because they define the generation context and the first-word seed used by this educational construction.

---

## 🧸 Toy baseline

The transparent `ToyBigramLM` experiment remains available for teaching, debugging, and Monte Carlo calibration, but it is no longer the default Docker experience.

Run it locally:

```bash
seedmark experiment --output-dir results/run --trials 300 --length 80
```

Or with Compose:

```bash
docker compose --profile toy up --build experiment report
```

Then open:

```text
http://localhost:8080/report.html
```

---

## 📦 Reference-model cache helper

Prefetch the currently bundled Qwen model explicitly with:

```bash
docker compose run --rm qwen-cache
```

or locally:

```bash
seedmark qwen-cache --model Qwen/Qwen3.5-0.8B
```

These command names are specific to the current reference backend, not to the SeedMark algorithm.

---

## 🗂️ Repository layout

```text
src/seedmark/chat_llm.py    chat-oriented reference LLM workflow (currently Qwen)
src/seedmark/hf_llm.py      Hugging Face token-watermark primitives / reference adapter
src/seedmark/animation.py   generation and detection GIFs
src/seedmark/reporting.py   standalone HTML comparison report
src/seedmark/model_cache.py persistent model-cache helpers
src/seedmark/cli.py         command-line interface
requirements/               Docker dependency manifests
tests/                      deterministic tests and GIF smoke tests
docs/                       method, limitations, real-LLM notes
Dockerfile.qwen             current Qwen reference-backend image
docker-compose.yml          default reference demo + opt-in toy profile
```

---

## ✅ Tests

```bash
python -m unittest discover -s tests -v
```

CI validates unit tests, package build, reproducible toy reporting, multi-frame GIF rendering, and the Docker build. Normal CI does not download the reference LLM weights.

---

## 🏷️ Versioning

SeedMark follows Semantic Versioning and keeps one package-version source of truth.

```bash
seedmark --version
```

```python
import seedmark
print(seedmark.__version__)
```

See [`CHANGELOG.md`](CHANGELOG.md) and [`docs/versioning.md`](docs/versioning.md).

---

## 🔬 Scientific scope and limitations

SeedMark is intended for:

- lectures and demonstrations;
- reproducible research discussions;
- inspection of token-level watermark mechanics;
- experiments comparing marked and unmarked LLM generation;
- experimentation across compatible language-model backends;
- a minimal baseline before studying production-grade text watermarking systems.

SeedMark should **not** be treated as a production provenance mechanism, universal AI-text detector, or claim about any commercial watermark implementation.

Model-agnostic here means that the **algorithm is not mathematically tied to Qwen**. It does not mean every model or API can be used without integration work. Hosted APIs that do not expose token-level next-token scores generally cannot apply this generation-time watermark directly.

Before interpreting detector scores, read:

- [`docs/limitations.md`](docs/limitations.md)
- [`docs/real-llm.md`](docs/real-llm.md)

---

## 🤝 Contributing

Contributions, new LLM adapters, experiments, bug reports, and reproducibility improvements are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

If you use SeedMark in research work, repository citation metadata is available in [`CITATION.cff`](CITATION.cff).

## 📄 License

SeedMark code is released under the [`MIT License`](LICENSE).

The bundled Qwen3.5 reference model weights are distributed separately under their own model license. Other model backends remain subject to their respective licenses.
