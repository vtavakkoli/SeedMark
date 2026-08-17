# Animated generation output

Real-Qwen experiments create `results/qwen/generation.gif` by default.

The animation is generated **after** model inference from the saved SeedMark trace; it does not re-run Qwen. Every frame represents one generated token.

## What is highlighted?

The sentence panel contains the prompt plus all tokens that have already been generated. The **pink highlighted token is exactly the token selected at the current generation step**.

For example, if the trace is:

```text
prompt: Research is
step 1: " useful"
step 2: " because"
step 3: " evidence"
```

then frame 2 conceptually renders:

```text
Research is useful [because]
                    ^ current token
```

The highlighted token is intentionally excluded from the normal prefix until its own frame, so the visualization cannot accidentally highlight a previously generated token.

## What each frame shows

- the sentence being constructed, with the current token highlighted;
- current step and total generated-token count;
- the top candidate tokens from the real Qwen logits;
- base probability and watermarked sampling probability bars;
- keyed pseudorandom score `u` for each displayed candidate;
- chosen token ID and probabilities;
- cumulative detector z-score and the detection threshold.

The GIF therefore connects the three important layers of the experiment in one view:

```text
real Qwen distribution
        ↓
SeedMark probability tilt
        ↓
selected token → highlighted in the sentence
        ↓
cumulative statistical signal
```

## CLI

Default:

```bash
seedmark qwen-demo --output-dir results/qwen
```

This writes:

```text
results/qwen/
├── report.html
├── generation.gif
├── generated_watermarked.txt
├── generated_control.txt
├── watermarked-trace.json
├── control-trace.json
└── summary.json
```

Change playback speed:

```bash
seedmark qwen-demo --gif-frame-ms 450
```

Disable GIF creation for benchmark-only runs:

```bash
seedmark qwen-demo --no-gif
```

When using Docker Compose, the GIF is available from the same report server:

```text
http://localhost:8081/generation.gif
```

The animation dependency (`Pillow`) belongs only to the optional `real-llm` extra; the toy/core SeedMark package remains dependency-free.
