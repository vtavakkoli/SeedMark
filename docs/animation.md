# Animated generation and detection report

Real-Qwen experiments now create **two complementary animations** plus static poster previews:

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

All visuals are rendered **after inference from the recorded SeedMark trace**. Qwen is not run a second time.

## 1. `generation.gif` — how the watermark changes token choice

Each frame is one generation step and contains three linked views:

1. **Next-token distribution.** The gray dashed curve is the real Qwen base distribution. The overlaid curve is the SeedMark-adjusted distribution.
2. **Live sentence construction.** The prompt and already generated text are normal. The **token selected at the current step is the only token highlighted in pink**.
3. **Signal history.** The cumulative z-score and the share of selected tokens in the keyed preferred half show how tiny local changes accumulate.

Candidate markers use:

- **teal:** keyed score `u ≥ 0.5`, so the multiplicative watermark factor is at least one;
- **coral:** keyed score `u < 0.5`, so the multiplicative factor is below one;
- **pink outline:** the token actually selected at that step.

For a trace

```text
prompt: Research is
step 1: " useful"
step 2: " because"
step 3: " evidence"
```

frame 2 renders conceptually as:

```text
Research is useful [because]
                    ^ current token
```

The current token is intentionally excluded from the normal prefix until its own frame. This prevents an older occurrence of the same word from being highlighted accidentally.

## 2. `detection.gif` — how detection works without Qwen probabilities

The detection animation replays the observed token sequence from the detector's point of view.

It shows:

- the observed sentence, with selected tokens colored by their keyed score;
- the current token outlined in pink;
- cumulative keyed-correlation z-score;
- the configured decision threshold;
- matched unwatermarked-control z-score as a dashed reference when available;
- one-sided detector confidence `1 - p`;
- the cumulative share of selected tokens with `u ≥ 0.5`.

The detector uses only:

```text
generated token IDs
+ first-word seed
+ secret key
────────────────────
keyed score sequence
→ cumulative z-score
→ threshold decision
```

It does **not** receive Qwen logits, hidden states, top-k probabilities, or model weights.

### Important statistical wording

`1 - p` in the visualization is confidence against SeedMark's null hypothesis. It is **not** a posterior probability that the passage is AI-generated.

With the default `z = 3` decision threshold, the equivalent one-sided normal-test confidence is about 99.865%, not 95%.

## GIF reliability

SeedMark writes every frame using a common adaptive palette and saves GIFs with:

- `save_all=True`;
- all remaining frames passed through `append_images`;
- `loop=0`;
- `optimize=False`;
- `disposal=2`;
- an extended final-frame duration.

CI opens the generated files with Pillow and verifies that both GIFs contain multiple frames. Static PNG previews are generated as an additional inspection/fallback artifact.

## CLI

Default:

```bash
seedmark qwen-demo --output-dir results/qwen
```

Adjust animation speed and dimensions:

```bash
seedmark qwen-demo \
  --gif-frame-ms 450 \
  --gif-width 1200 \
  --gif-height 900
```

Disable visual assets for benchmark-only runs:

```bash
seedmark qwen-demo --no-gif
```

With Docker Compose, all assets are served beside the HTML report:

```text
http://localhost:8081/report.html
http://localhost:8081/generation.gif
http://localhost:8081/detection.gif
```

`Pillow` belongs only to the optional `real-llm` dependency group; the toy/core SeedMark package remains dependency-free.
