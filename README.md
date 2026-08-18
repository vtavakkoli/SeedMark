# SeedMark 🌱🔐

<p align="center">
  <strong>An inspectable research and teaching lab for token-level and semantic-keyed LLM watermarking.</strong>
</p>

<p align="center">
  <img alt="Python 3.13+" src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white">
  <img alt="Version 0.7.0" src="https://img.shields.io/badge/version-0.7.0-0A7EA4">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-2ea44f">
  <img alt="Purpose Research and Education" src="https://img.shields.io/badge/Purpose-Research%20%26%20Education-6f42c1">
</p>

SeedMark demonstrates how an autoregressive language model can embed a
statistically detectable signal **during token sampling**, while a detector later
reconstructs that keyed signal without receiving the generator's original
next-token probability distribution.

Version **0.7.0** contains two families of experiments:

| Mode | Key source | Token synchronization | Purpose |
|---|---|---|---|
| **Baseline token watermark** | first word of user question | absolute token position | minimal transparent teaching baseline |
| **Semantic watermark — answer** **(default semantic mode)** | complete answer semantics | per-token occurrence count | robust research mode for normal assistant answers |
| **Semantic watermark — paragraph** | previous paragraph semantics | per-token occurrence count reset per paragraph | streaming mode for long structured answers |

The generator-side algorithm is model-agnostic: a compatible autoregressive LLM
needs stable token IDs and access to next-token logits or probabilities. The
repository currently ships **Qwen3.5** as the reference Hugging Face backend.

> [!IMPORTANT]
> SeedMark is a research/education prototype. It is **not** Anthropic's production
> watermark, **not** Google SynthID-Text, **not** C2PA, and does not claim that its
> watermark is impossible to remove or sufficient as standalone proof of
> authorship.

---

## 🧠 Semantic watermarking from the complete answer

The default semantic mode uses the **meaning of the complete answer as the key
context**.

This is intentionally broader than sentence-level keying. Sentence boundaries are
not stable semantic units: a harmless edit can split one sentence into two or
combine two sentences without materially changing the answer. SeedMark therefore
does **not** re-key on periods, question marks, or other sentence punctuation.

### Complete-answer draft/commit flow

A complete answer is not available before generation, so answer-wide keying uses a
two-pass semantic commitment:

```text
User question
     │
     ▼
ordinary draft answer
     │
     ▼
semantic encoder
     │
     ▼
secret coarse semantic bucket B
     │
     ▼
watermarked final generation using B
     │
     ▼
embed final answer
     │
     ├── bucket(final) == B  → accept
     │
     └── bucket(final) != B  → retry marked generation
```

The detector never needs the draft. It embeds only the **final observed answer**.
A marked answer is accepted only when its final semantics map to the same bucket
that keyed generation.

This makes the complete answer a practical semantic key while keeping detection
self-contained.

### Secret semantic buckets

SeedMark does not hash the exact floating-point embedding. That would be brittle:
a tiny vector perturbation could completely change the hash.

Instead:

```text
complete answer
     │
     ▼
normalized embedding
     │
     ▼
nearest secret deterministic anchor
     │
     ▼
coarse semantic bucket
```

The trace records the selected bucket plus the top-1/top-2 similarity **margin**.
A small margin means the answer lies close to a semantic bucket boundary and is
more likely to change key after editing.

---

## 📄 Paragraph semantic mode

For long structured responses, SeedMark also supports paragraph-level streaming
keying:

```text
question semantics
      │
      ▼
watermark paragraph 0
      │
      ▼
paragraph 0 semantics
      │
      ▼
watermark paragraph 1
      │
      ▼
paragraph 1 semantics
      │
      └── repeat
```

A paragraph boundary is a blank line. Sentence punctuation does not change the
key.

The first paragraph is bootstrapped from the user question because its own
semantics are not yet available causally. Later paragraphs use one or more recent
completed paragraphs.

---

## 🔁 Why semantic mode does not use absolute token position

The original SeedMark baseline uses absolute token position. That is easy to
inspect, but inserting one word shifts every later PRF input.

Semantic mode instead uses the occurrence number of each **specific candidate
token ID** inside the semantic scope:

```text
u = HMAC-SHA256(
    secret_key,
    semantic_bucket ||
    occurrence_of_this_token_id ||
    candidate_token_id
) → [0,1)
```

This means inserting an unrelated token does **not** shift the keyed state for all
later tokens. Only later occurrences of the edited token ID are directly
re-indexed.

Watermarked generation still applies the same transparent probability tilt:

```text
q(v) ∝ p(v) · exp(strength · (2u(v) - 1))
```

The matched control samples directly from the original model distribution `p(v)`.

---

## 🚀 Semantic quick start

Install the real-LLM dependencies:

```bash
python -m pip install -e ".[real-llm]"
```

### Complete-answer mode — default

```bash
seedmark semantic-qwen-demo \
  --question "What is AI?" \
  --semantic-scope answer \
  --output-dir results/semantic-answer
```

Because `answer` is the default, this is equivalent:

```bash
seedmark semantic-qwen-demo \
  --question "What is AI?" \
  --output-dir results/semantic-answer
```

Useful options:

```bash
seedmark semantic-qwen-demo \
  --question "What is edge AI?" \
  --bucket-count 32 \
  --max-answer-passes 4 \
  --semantic-device cpu \
  --max-new-tokens 128 \
  --output-dir results/edge-ai-semantic
```

### Paragraph mode

```bash
seedmark semantic-qwen-demo \
  --question "Explain industrial AI in several paragraphs." \
  --semantic-scope paragraph \
  --context-paragraphs 1 \
  --output-dir results/semantic-paragraph
```

### Detect saved text

```bash
seedmark semantic-qwen-detect \
  --question "What is AI?" \
  --semantic-scope answer \
  --text-file results/semantic-answer/generated_watermarked.txt
```

The detector loads the generator tokenizer and the semantic encoder, but **not the
generator model weights**.

The default semantic encoder is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is loaded with Hugging Face `AutoTokenizer` + `AutoModel` and masked mean
pooling; the separate `sentence-transformers` Python package is not required. By
default, the semantic encoder runs on CPU so it does not consume the generator GPU
memory budget.

---

## 📦 Semantic experiment outputs

A semantic run writes:

```text
results/semantic-answer/
├── generated_watermarked.txt
├── generated_control.txt
├── watermarked-trace.json
├── control-trace.json
└── summary.json
```

The summary records, among other fields:

- semantic scope;
- generator model;
- semantic encoder model;
- bucket count;
- selected answer bucket and margin;
- answer-key stabilization attempts;
- marked detector result;
- control detector result.

The full trace records every candidate probability and keyed score.

---

## 🔬 Semantic traceability

Semantic generation traces include:

- semantic scope;
- semantic segment index;
- token occurrence number;
- semantic bucket;
- semantic bucket margin;
- semantic context;
- candidate token IDs and decoded text;
- base probability;
- generation probability;
- keyed PRF score;
- chosen token;
- cumulative detector z-score.

Answer-wide results additionally retain the semantic draft and stabilization
metadata so robustness experiments can inspect whether the final answer stayed in
the intended semantic region.

---

## 💬 Python API

```python
from seedmark.semantic_chat import SemanticChatQwenSeedMark

lab = SemanticChatQwenSeedMark(
    model_name="Qwen/Qwen3.5-0.8B",
    semantic_device="cpu",
)

marked = lab.generate(
    question="What is AI?",
    watermarked=True,
    semantic_scope="answer",
    bucket_count=32,
    max_answer_passes=4,
)

control = lab.generate(
    question="What is AI?",
    watermarked=False,
    semantic_scope="answer",
    bucket_count=32,
)

print(marked.text)
print(marked.semantic_bucket)
print(marked.semantic_margin)
print(marked.detection)
print(control.detection)
```

Use `semantic_scope="paragraph"` and `context_paragraphs=1` for paragraph-level
streaming keying.

---

## 🧪 Why the control matters

Every useful watermark experiment should include an unwatermarked comparison.
SeedMark therefore reports:

```text
Watermarked output  → detector → expected DETECTED
Control output      → detector → expected NOT DETECTED
```

The control is generated from the same model and chat setup without probability
reweighting. In answer semantic mode, the detector still derives a semantic bucket
from the control answer itself; because ordinary generation was not correlated
with that secret keyed scoring rule, it should not systematically exceed the
threshold.

---

## 🌱 Original SeedMark baseline

The original first-word-seeded experiment remains unchanged for reproducibility.
It asks the reference LLM **“What is AI?”**, generates the answer once with the
watermark and once without it, and runs the same baseline detector on both.

Run the baseline with Docker Compose:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8081/report.html
```

Or locally:

```bash
seedmark qwen-demo --output-dir results/qwen
```

### Baseline generation animation

<p align="center">
  <img src="generation.gif" alt="SeedMark watermarked LLM text generation animation" width="900">
</p>

### Baseline detection animation

<p align="center">
  <img src="detection.gif" alt="SeedMark marked versus control detection animation" width="900">
</p>

The baseline uses:

```text
u(t,v) = HMAC-SHA256(
    secret_key,
    SHA256(first_word_of_user_question) || position || token_id
) → [0,1)
```

The detector reconstructs selected `u` values and computes:

```text
z = Σ(u_t - 0.5) / sqrt(n / 12)
```

Its central teaching property remains:

> **Generation needs the LLM distribution. Detection does not.**

---

## 🏗️ Architecture

```text
                       compatible autoregressive LLM
                                  │
                                  ▼
                          next-token scores
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
               ▼                                     ▼
      baseline SeedMark                    semantic SeedMark
 first word + absolute pos.          semantic bucket + token occurrence
               │                                     │
               └──────────────────┬──────────────────┘
                                  ▼
                         reweighted sampling
                                  │
                         generated token IDs
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
        baseline detector                    semantic detector
                                                  + encoder
               └──────────────────┬──────────────────┘
                                  ▼
                           statistical score
```

A compatible generator backend needs:

1. stable token IDs;
2. next-token logits or probabilities;
3. autoregressive token appending;
4. a compatible prompt/chat-template formatter.

Semantic mode additionally needs a deterministic semantic encoder shared by
generation and detection.

---

## 🤗 Reference models and cache

The bundled reference generator is:

```text
Qwen/Qwen3.5-0.8B
```

Qwen is an example backend, not a mathematical requirement of SeedMark.

The default host-side Hugging Face cache is:

```text
../seedmark-model-cache
```

Override it with, for example:

```text
SEEDMARK_MODEL_CACHE=D:/model-cache/seedmark
```

Prefetch the reference generator model with:

```bash
docker compose run --rm qwen-cache
```

or:

```bash
seedmark qwen-cache --model Qwen/Qwen3.5-0.8B
```

---

## 🧸 Toy baseline

The dependency-free `ToyBigramLM` remains useful for teaching, debugging, and
Monte Carlo calibration:

```bash
seedmark experiment --output-dir results/run --trials 300 --length 80
```

Or:

```bash
docker compose --profile toy up --build experiment report
```

---

## 🗂️ Repository layout

```text
src/seedmark/core.py           original keyed token watermark primitives
src/seedmark/chat_llm.py       chat-oriented reference generator adapter
src/seedmark/hf_llm.py         Hugging Face token watermark primitives
src/seedmark/semantic.py       semantic encoder, buckets, scopes, PRF, detector
src/seedmark/semantic_chat.py  answer/paragraph semantic chat generation
src/seedmark/animation.py      baseline generation/detection GIFs
src/seedmark/reporting.py      standalone baseline HTML comparison report
src/seedmark/cli.py            CLI including semantic-qwen-* commands
examples/semantic_chat.py      minimal semantic marked/control example
tests/                         deterministic unit and smoke tests
docs/semantic-watermark.md     semantic design, threat model, benchmark plan
docs/limitations.md            scientific boundaries and attack limitations
```

---

## ✅ Tests

```bash
python -m unittest discover -s tests -v
```

Normal CI does not download generator or semantic model weights. Semantic-core and
CLI contract tests use deterministic lightweight test doubles.

---

## ⚠️ Scientific scope and limitations

Semantic keying makes the prototype less brittle, but it does **not** make the
watermark unbreakable.

Important remaining failure modes include:

- aggressive whole-answer paraphrasing that crosses a semantic bucket boundary;
- translation that changes tokenizer and embedding geometry;
- deletion/replacement of enough carrier tokens to destroy statistical evidence;
- answer-wide multi-pass inference cost;
- failure of a very fine bucket configuration to stabilize within the allowed
  number of marked attempts;
- adaptive attacks with repeated detector access;
- insufficient statistical power in very short text.

A statistically detected watermark is evidence of correlation with the configured
keyed generation rule under the tested assumptions—not proof of authorship, truth,
or model identity.

Read:

- [`docs/limitations.md`](docs/limitations.md)
- [`docs/semantic-watermark.md`](docs/semantic-watermark.md)
- [`docs/real-llm.md`](docs/real-llm.md)

### Related semantic-watermark research

The semantic mode is original SeedMark project code; it is not copied code from
these systems. Relevant research includes:

- Ren et al. (2024), **SemaMark: A Robust Semantics-based Watermark for Large
  Language Model against Paraphrasing**, Findings of NAACL 2024.
  https://aclanthology.org/2024.findings-naacl.40/
- Hou et al. (2024), **SemStamp: A Semantic Watermark with Paraphrastic Robustness
  for Text Generation**, NAACL 2024.
  https://aclanthology.org/2024.naacl-long.226/
- Ye et al. (2026), **SWAN: Semantic Watermarking with Abstract Meaning
  Representation**, ACL 2026.
  https://aclanthology.org/2026.acl-long.1681/

---

## 🏷️ Versioning

SeedMark follows Semantic Versioning and uses one package-version source of truth:

```bash
seedmark --version
```

```python
import seedmark
print(seedmark.__version__)
```

See [`CHANGELOG.md`](CHANGELOG.md) and [`docs/versioning.md`](docs/versioning.md).

---

## 🤝 Contributing

Contributions are welcome for new generator adapters, semantic encoders,
robustness attacks, calibration experiments, report visualizations, and
reproducibility improvements. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

Repository citation metadata is available in [`CITATION.cff`](CITATION.cff).

## 📄 License

SeedMark code is released under the [`MIT License`](LICENSE).

The bundled/reference model weights are distributed separately under their own
model licenses.
