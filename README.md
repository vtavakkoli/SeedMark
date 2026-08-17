# SeedMark 🌱🔐

**An interactive, scientific teaching lab for first-word-seeded keyed pseudorandom text watermarking.**

SeedMark demonstrates a precise idea: **a detector can recover statistical correlation with a secret pseudorandom token-selection rule without having the language model's original next-token probability distribution.**

> [!IMPORTANT]
> SeedMark is a deliberately simplified educational algorithm. It is **not** Anthropic's production watermark, **not** Google SynthID-Text, **not** C2PA, and does not claim that any vendor uses a first-word seed.

## What it does

The first word becomes a public SHA-256 seed. For every later token candidate, SeedMark derives a deterministic HMAC-SHA256 score from:

```text
first word → SHA-256 seed
                + secret key
                + token position
                + candidate token
                      ↓
              pseudorandom u ∈ [0,1)
```

During **watermarked generation**, high-scoring candidates receive a small probability boost. During **detection**, SeedMark sees only the final text + the secret key. It reconstructs the score of each chosen token and tests whether the average is unusually high.

The included toy bigram language model is intentionally transparent: its only job is to provide inspectable base probabilities.

## Interactive report

Every experiment produces a standalone `report.html` with:

- ▶️ animated token-by-token playback;
- base probability vs watermark-adjusted probability bars;
- the pseudorandom score for every candidate;
- the selected token at each step;
- cumulative detector z-score;
- watermarked vs unwatermarked generated text;
- Monte Carlo true-positive / false-positive rates with 95% Wilson intervals;
- empirical ROC/AUC;
- JSON and CSV traces;
- dependency-free SVG plots.

## One-command Docker demo

```bash
docker compose up --build
```

The experiment service generates `results/run/`; after it completes, the report service serves the interactive report at:

```text
http://localhost:8080/report.html
```

To regenerate only the experiment:

```bash
docker compose run --rm experiment
```

## Local Python 3.13

SeedMark has **zero runtime dependencies**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e .
seedmark experiment --output-dir results/run --trials 300 --length 80
```

Generate one sequence:

```bash
seedmark generate --first-word research --length 80 --secret-key my-demo-key
```

Detect an existing SeedMark-formatted sequence **without supplying any model probabilities**:

```bash
seedmark detect --text-file results/run/generated_watermarked.txt --secret-key seedmark-demo-key
```

## The teaching algorithm

For first word `w0`:

```text
s = SHA256(normalize(w0))
u(t,v) = HMAC_SHA256(key, s || position || token) → [0,1)
```

For the base top-k distribution `p_t(v)`, marked sampling uses:

```text
q_t(v) ∝ p_t(v) · exp(strength · (2u(t,v) - 1))
```

The detector reconstructs only the chosen-token values `u_t`. Under an unwatermarked null, those keyed values should average approximately `0.5`, so SeedMark computes:

```text
z = Σ(u_t - 0.5) / sqrt(n / 12)
```

This is the central experiment: **the detector needs the key and observed token sequence, not the original language-model distribution.** See [`docs/method.md`](docs/method.md) for the derivation.

## Repository layout

```text
src/seedmark/          core algorithm, toy LM, detector, reports, CLI
examples/              minimal no-distribution detector example
tests/                 deterministic unit tests
docs/                  method and limitations
results/                generated reports and reference notes
.github/workflows/      Python 3.13 CI + experiment artifact
Dockerfile              reproducible Python 3.13 image
docker-compose.yml      experiment + local report server
```

## Reproducibility

Default settings:

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

The report records the public first-word seed, a one-way fingerprint of the secret key, every selected probability/score, and every trial z-score.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite checks seed reproducibility, key sensitivity, probability normalization, trace completeness, marked-vs-control separation, and—most importantly—that the detector API operates without a language model or probability distribution.

## Scientific use

SeedMark is useful for lectures, research discussions, and as a minimal baseline before experimenting with real tokenizer-level watermark algorithms. Please read [`docs/limitations.md`](docs/limitations.md) before interpreting detector scores.

Contributions that add calibration, edit robustness, alternate keyed score constructions, or real-model adapters are welcome.

## License

MIT. See [`LICENSE`](LICENSE).
