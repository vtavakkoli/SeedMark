# Method

SeedMark is a **pedagogical keyed pseudorandom text-watermark experiment** designed to make one statistical idea easy to inspect:

> A detector can test correlation with a secret pseudorandom sequence without reconstructing the language model's original next-token distribution.

It is intentionally simpler than production watermark schemes.

## First word as the public seed

Let the first generated word be `w0`. SeedMark normalizes it and computes `s = SHA256(w0)`. Changing the first word changes the seed and therefore every subsequent keyed score. The first word is not secret; a separate secret key `K` prevents outsiders from predicting favored tokens.

## Keyed pseudorandom token score

At position `t >= 1`, candidate token `v` receives:

```text
u(t,v) = U(HMAC_K(s || t || v)),  u(t,v) in [0,1)
```

`U` interprets 64 hash bits as a uniform-looking floating-point value. The same first word, key, position, and token always produce the same score.

## Generation

The toy language model supplies a top-k distribution `p_t(v)`. Watermarked generation uses the transparent exponential tilt:

```text
q_t(v) = p_t(v) * exp(lambda * (2u(t,v)-1)) / Z_t
```

A high keyed score does not force a token; it only increases its probability among tokens that the base model already considers plausible.

## Detection without the model distribution

Given only the final token sequence `x0,...,xn`, the visible first word `x0`, and key `K`, the detector reconstructs `u_t = u(t, x_t)`. Under the teaching null hypothesis, unwatermarked choices are independent of the secret PRF values, so approximately `E[u_t]=1/2` and `Var[u_t]=1/12`.

SeedMark reports:

```text
z = sum(u_t - 1/2) / sqrt(n / 12)
```

A large positive z-score means the observed sequence selected unusually many high-scoring tokens relative to chance. **The detector never receives `p_t(v)` or `q_t(v)`.**

## Experimental evaluation

`seedmark experiment` creates matched watermarked and unwatermarked sequences and reports mean/std z-score, empirical true- and false-positive rates, 95% Wilson confidence intervals, ROC/AUC, per-trial CSV data, and a token-level interactive report showing base probability, adjusted probability, PRF score, selected token, and cumulative z-score.

## Why this is not SynthID-Text

SeedMark uses a simple distortionary exponential tilt and a first-word seed because those choices are easy to teach and verify. This repository makes **no claim** that Anthropic, Google DeepMind, or another vendor uses this algorithm, seed construction, detector, or parameters.
