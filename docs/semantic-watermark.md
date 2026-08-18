# Semantic self-key watermarking

SeedMark 0.7 adds an experimental **semantic self-key** mode for chat answers.
Instead of deriving every watermark decision from one visible seed word and a
single global token position, the semantic mode periodically derives the next
watermark context from the meaning of the answer itself.

> **Research status:** experimental. This mode is designed to make common lexical
> edits less catastrophic and to provide sentence-level re-synchronisation. It is
> not a claim of paraphrase-proof, translation-proof, or adversarially secure
> provenance.

## Motivation

Token watermarks that hash exact preceding tokens can be fragile: synonym
replacement, paraphrasing, insertion, or deletion can change the hash state and
shift later token positions. Semantic watermarking research instead uses semantic
representations because paraphrases often preserve meaning even when surface
forms change substantially.

SeedMark's semantic mode follows that high-level direction while keeping the
implementation intentionally small and inspectable.

## Design

### 1. Bootstrap

The first assistant sentence has no previous answer sentence. SeedMark therefore
bootstraps the initial semantic context from the user question.

```text
question
   │
semantic encoder
   │
secret semantic bucket
   │
watermark sentence 0
```

### 2. Self-key from the answer

After a sentence completes, SeedMark embeds the most recent completed answer
sentence (or the configured recent-sentence window). The normalized embedding is
assigned to the nearest of a set of secret deterministic random anchors.

```text
completed answer sentence(s)
          │
          ▼
   semantic embedding
          │
          ▼
 nearest secret anchor
          │
          ▼
 semantic bucket id
```

The bucket ID is deliberately coarse. Hashing every embedding bit would be
brittle: a very small embedding perturbation could change the entire downstream
PRF seed. A nearest-anchor bucket has a region of stability around each anchor.
The trace records both the selected bucket and the top-1/top-2 similarity margin
so experiments can inspect how close a context was to a bucket boundary.

### 3. Sentence-local watermark score

For each candidate token in the next sentence, SeedMark computes

```text
u = HMAC-SHA256(
      secret_key,
      "seedmark-semantic-v1" ||
      semantic_bucket ||
      sentence_local_token_offset ||
      candidate_token_id
    ) -> [0, 1)
```

and applies the same transparent exponential tilt used by the original SeedMark
prototype:

```text
q(v) ∝ p(v) · exp(strength · (2u(v) - 1))
```

The control path samples from `p(v)` unchanged.

### 4. Re-synchronisation

The token offset resets at each detected sentence boundary. That matters for edit
robustness. If a token is inserted or deleted in one sentence, the position shift
is local to that sentence instead of permanently changing every later score.

When the edited sentence retains similar meaning and therefore remains in the
same semantic bucket, the following sentence can re-enter the same watermark
state.

```text
sentence N edited
      │
      ├─ local token evidence may be damaged
      │
      ▼
semantic encoder
      │
      ├─ same bucket (when semantics stay sufficiently close)
      ▼
sentence N+1 re-synchronises
```

## Detector inputs

Semantic detection requires:

- the generated token IDs, or text that can be retokenized consistently;
- the generator tokenizer;
- the secret key;
- the user question for first-sentence bootstrap;
- the same semantic encoder model;
- the same bucket count and semantic context-window configuration.

It **does not require** the generator model weights, generation logits, or the
original next-token probability distributions.

The default semantic encoder is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It is loaded through Hugging Face `AutoTokenizer` + `AutoModel` and masked mean
pooling, so no additional `sentence-transformers` Python dependency is required.
The generator and semantic encoder are separate components and can be replaced
independently.

## Python API

```python
from seedmark.semantic_chat import SemanticChatQwenSeedMark

model = SemanticChatQwenSeedMark(
    model_name="Qwen/Qwen3.5-0.8B",
    semantic_device="cpu",
)

marked = model.generate(
    question="What is AI?",
    watermarked=True,
    bucket_count=32,
    context_sentences=1,
)

control = model.generate(
    question="What is AI?",
    watermarked=False,
    bucket_count=32,
    context_sentences=1,
)

print(marked.text)
print(marked.detection)
print(control.detection)
```

A tokenizer-only generator path plus the semantic encoder is also available for
saved text:

```python
from seedmark.semantic_chat import detect_semantic_chat_text_with_tokenizer

result = detect_semantic_chat_text_with_tokenizer(
    model_name="Qwen/Qwen3.5-0.8B",
    question="What is AI?",
    text=marked.text,
    secret_key="seedmark-demo-key",
)
```

## Trace fields

Every semantic generation step records:

- sentence index;
- sentence-local token offset;
- semantic bucket;
- bucket stability margin;
- semantic context text used for the bucket;
- candidate token IDs/text;
- base and watermarked probabilities;
- keyed watermark score;
- selected token;
- cumulative detector z-score.

These fields are intended to support ablation studies and robustness experiments.

## What this improves

Compared with the original first-word + absolute-position demonstration, semantic
mode is designed to improve resilience to:

- synonym replacement that largely preserves sentence meaning;
- local insertions/deletions whose position shift should end at the next sentence;
- partial sentence rewriting where later semantic buckets remain stable;
- changes to the first answer word, because later answer sentences self-key from
  answer semantics rather than from that word.

## What it does not solve

A semantic key alone does **not** make a token-selection watermark impossible to
remove. Important failure modes remain:

- a sufficiently strong paraphrase can move a sentence into another semantic
  bucket;
- rewriting the current sentence changes the actual token IDs carrying that
  sentence's evidence;
- translation can change both tokenization and embedding geometry;
- sentence splitting/merging can alter re-synchronisation boundaries;
- an attacker with repeated detector access may be able to estimate or optimise
  against the signal;
- short answers may contain too few scored tokens for reliable statistical
  separation;
- the simple z-test remains an educational approximation and should be calibrated
  empirically for a real deployment, model, language, attack set, and decoding
  configuration.

For stronger paraphrase invariance, research systems can encode the watermark at
the sentence semantic-structure level rather than only using semantics to key a
token-level signal. SeedMark keeps this implementation intentionally compact so
the mechanism is inspectable.

## Suggested robustness benchmark

A useful evaluation should compare at least:

1. unmodified marked vs. control text;
2. synonym substitution at several edit rates;
3. random insertion/deletion at several edit rates;
4. sentence-level paraphrasing;
5. paragraph-level paraphrasing;
6. sentence split/merge attacks;
7. back-translation;
8. different semantic bucket counts (`8`, `16`, `32`, `64`);
9. one- vs. two-sentence semantic context windows;
10. quality metrics alongside detector ROC/AUC and false-positive rate.

The main scientific question is not whether one example remains detected. It is
how detection power and text quality degrade as semantic and lexical distance
increase.

## Related work

This SeedMark mode is an original compact teaching implementation, not copied
code from the following systems. The design is informed by the broader semantic
watermarking literature:

- Ren et al. (2024), **A Robust Semantics-based Watermark for Large Language
  Model against Paraphrasing (SemaMark)**, Findings of NAACL 2024.
  https://aclanthology.org/2024.findings-naacl.40/
- Hou et al. (2024), **SemStamp: A Semantic Watermark with Paraphrastic Robustness
  for Text Generation**, NAACL 2024.
  https://aclanthology.org/2024.naacl-long.226/
- Ye et al. (2026), **SWAN: Semantic Watermarking with Abstract Meaning
  Representation**, ACL 2026. SWAN illustrates a stronger semantic-structure
  direction in which the signature is encoded at the meaning-representation
  level rather than only using semantics as a contextual key.
  https://aclanthology.org/2026.acl-long.1681/

See also [`limitations.md`](limitations.md) before interpreting detector results
as provenance evidence.
