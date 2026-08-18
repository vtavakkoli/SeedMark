# Semantic self-key watermarking

SeedMark 0.7 includes an experimental **semantic self-key** mode for chat answers.
The key difference from the original first-word baseline is that semantic mode can
derive its watermark context from the **meaning of the answer itself** rather than
from a fragile exact lexical prefix.

> **Research status:** experimental. The design aims to improve robustness to
> ordinary lexical edits and paraphrases. It is not a claim of paraphrase-proof,
> translation-proof, or adversarially secure provenance.

## Semantic scopes

SeedMark supports two semantic scopes:

| Scope | Key source | Generation style | Best use |
|---|---|---|---|
| `answer` **(default)** | complete answer semantics | two-pass draft/commit | normal assistant answers, including one-paragraph replies |
| `paragraph` | previous completed paragraph(s) | streaming one-pass | long structured answers with explicit paragraph breaks |

Sentence punctuation does **not** re-key semantic mode.

---

## Complete-answer mode

Complete-answer mode is the default because many real assistant replies contain
only one paragraph. If the algorithm waited for a paragraph boundary, short
answers would never become self-keyed from their own semantics.

Generation therefore uses a semantic **draft/commit** workflow:

```text
user question
    │
    ▼
ordinary draft answer
    │
    ▼
semantic encoder
    │
    ▼
secret coarse bucket B
    │
    ▼
watermarked final generation using B
    │
    ▼
embed final answer
    │
    ├── final bucket == B  → accept
    │
    └── final bucket != B  → retry marked generation
```

The important property is that the detector does **not** need the draft. It embeds
the final observed answer. A marked answer is accepted only when its final
semantics map to the same bucket that was used during generation.

The number of marked attempts is bounded by `max_answer_passes` (CLI:
`--max-answer-passes`, default `4`). If the semantic bucket does not stabilize,
SeedMark raises an explicit error instead of returning a watermark that the
answer-only detector cannot reconstruct.

### Why not hash the exact embedding?

A cryptographic hash of floating-point embedding values would be highly brittle:
a tiny semantic-vector perturbation could completely change the downstream PRF
seed. SeedMark instead maps the normalized embedding to the nearest of a set of
secret deterministic random anchors.

```text
complete answer
     │
     ▼
normalized embedding
     │
     ▼
nearest secret anchor
     │
     ▼
coarse semantic bucket
```

The trace records the top-1/top-2 similarity margin. A small margin indicates that
the answer is close to a bucket boundary and may be less stable under editing.

---

## Paragraph mode

Paragraph mode is a streaming alternative:

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

A paragraph boundary is a blank line (`\n\n` or equivalent CRLF form). Periods,
question marks, and other sentence-ending punctuation do not change the semantic
key.

The first paragraph is bootstrapped from the user question because its own
semantics are not yet available causally. Subsequent paragraphs use the most
recent completed paragraph(s), configured by `context_paragraphs`.

---

## Token occurrence keying

Semantic mode intentionally avoids absolute token positions. For each candidate
token, SeedMark uses the occurrence number of that **specific token ID** inside
the current semantic scope.

```text
u = HMAC-SHA256(
      secret_key,
      "seedmark-semantic-v2" ||
      semantic_bucket ||
      occurrence_of_this_token_id ||
      candidate_token_id
    ) -> [0, 1)
```

This matters for edit robustness. If an unrelated token is inserted or deleted,
it does not shift the PRF input of every token that follows. Only later
occurrences of the edited token ID are directly re-indexed.

The normal model probability is tilted with the same transparent rule used by the
baseline:

```text
q(v) ∝ p(v) · exp(strength · (2u(v) - 1))
```

The control path samples directly from `p(v)`.

---

## Detection

### Answer scope

The detector:

1. retokenizes the final assistant answer;
2. embeds the complete answer;
3. reconstructs its secret semantic bucket;
4. reconstructs per-token occurrence numbers;
5. evaluates the keyed scores and cumulative detector statistic.

No generator logits, generator hidden states, next-token probability history, or
generator model weights are required.

### Paragraph scope

The detector reconstructs paragraph boundaries from the visible answer, derives
the same previous-paragraph semantic buckets, and resets token-occurrence counters
at each paragraph boundary.

### Required configuration

Semantic detection needs:

- the generated token IDs, or text that can be retokenized consistently;
- the generator tokenizer;
- the secret key;
- the same semantic encoder model;
- the same bucket count;
- the same semantic scope;
- for paragraph mode, the same `context_paragraphs` value;
- the user question for chat-prefix reconstruction and first-paragraph bootstrap.

The default semantic encoder is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is loaded through Hugging Face `AutoTokenizer` + `AutoModel` with masked mean
pooling. The generator and semantic encoder remain separate components.

---

## CLI

Complete-answer mode is the default:

```bash
seedmark semantic-qwen-demo \
  --question "What is AI?" \
  --semantic-scope answer \
  --output-dir results/semantic-answer
```

Use paragraph mode for long structured output:

```bash
seedmark semantic-qwen-demo \
  --question "Explain industrial AI." \
  --semantic-scope paragraph \
  --context-paragraphs 1 \
  --output-dir results/semantic-paragraph
```

Detect a saved answer:

```bash
seedmark semantic-qwen-detect \
  --question "What is AI?" \
  --semantic-scope answer \
  --text-file results/semantic-answer/generated_watermarked.txt
```

---

## Python API

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

print(marked.text)
print(marked.semantic_bucket)
print(marked.semantic_margin)
print(marked.detection)
```

Paragraph mode is selected with `semantic_scope="paragraph"` and optionally
`context_paragraphs=2` or another small context window.

---

## Trace fields

Each generation trace records:

- semantic scope;
- semantic segment index (`0` for answer-wide mode, paragraph index otherwise);
- chosen token occurrence count;
- semantic bucket;
- bucket stability margin;
- semantic context used for the generation pass;
- candidate token IDs and text;
- base and watermarked probabilities;
- keyed score;
- chosen token;
- cumulative detector z-score.

Answer-wide results additionally record:

- the unwatermarked semantic draft;
- the final answer-derived semantic key text;
- semantic bucket and margin;
- number of marked attempts;
- whether the answer-key commitment stabilized.

---

## What this improves

Compared with the original first-word + absolute-position baseline, semantic mode
is designed to be less brittle to:

- changing the first answer word;
- synonym replacement that preserves overall meaning;
- insertion/deletion of unrelated tokens;
- sentence splitting/merging in complete-answer mode;
- ordinary sentence-boundary changes, because sentences are no longer key
  boundaries;
- local paragraph edits when later paragraphs preserve their semantic bucket.

## What it does not solve

Semantic keying does **not** make a token-selection watermark impossible to
remove. Important limitations remain:

- sufficiently strong paraphrasing can move the answer or paragraph into another
  semantic bucket;
- translation can alter both tokenizer behavior and embedding geometry;
- deleting or replacing many carrier tokens reduces statistical evidence even if
  the semantic bucket remains stable;
- answer-wide generation adds computational cost because it requires a draft plus
  at least one marked generation;
- very fine bucket counts can make answer-key commitment less stable;
- repeated detector access may enable adaptive optimization against the signal;
- short answers may still contain too few tokens for strong statistical power;
- the z-test is an educational approximation and requires empirical calibration
  for real deployment conditions.

---

## Suggested robustness benchmark

A serious evaluation should compare both semantic scopes across:

1. unmodified marked vs. control text;
2. synonym substitution at several edit rates;
3. random insertion/deletion;
4. sentence split/merge attacks;
5. sentence-level paraphrasing;
6. paragraph-level paraphrasing;
7. whole-answer paraphrasing;
8. back-translation;
9. bucket counts such as `8`, `16`, `32`, and `64`;
10. answer-key stabilization rate and number of generation attempts;
11. text-quality metrics alongside ROC/AUC and false-positive rate.

The central scientific question is how detection power, semantic-bucket stability,
and output quality degrade as edits become stronger.

## Related work

This SeedMark implementation is original project code, not copied from the systems
below. Relevant semantic-watermark research includes:

- Ren et al. (2024), **A Robust Semantics-based Watermark for Large Language
  Model against Paraphrasing (SemaMark)**, Findings of NAACL 2024.
  https://aclanthology.org/2024.findings-naacl.40/
- Hou et al. (2024), **SemStamp: A Semantic Watermark with Paraphrastic Robustness
  for Text Generation**, NAACL 2024.
  https://aclanthology.org/2024.naacl-long.226/
- Ye et al. (2026), **SWAN: Semantic Watermarking with Abstract Meaning
  Representation**, ACL 2026.
  https://aclanthology.org/2026.acl-long.1681/

See [`limitations.md`](limitations.md) before interpreting detector results as
provenance evidence.
