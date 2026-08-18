# Animated generation and detection report

Real-Qwen runs create two complementary GIFs plus static PNG previews:

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

The visualizations are rendered from the recorded traces after inference; Qwen is not run again to build the GIFs.

## `generation.gif`

Each frame links the model distribution to the chosen token and the accumulating detector signal:

- dashed curve: base Qwen top-k probabilities;
- solid curve: SeedMark-adjusted probabilities;
- teal/coral candidate markers: keyed preference direction;
- pink ring: candidate selected on this step;
- sentence panel: a **sliding recent-context window**;
- pink sentence highlight: **only the current appended token**;
- lower charts: cumulative z-score and prioritized-token share.

The recent-context window is intentionally left-clipped as a generation grows. Historical whitespace is compacted, long fragments are split to the available width, and the current token is kept separate from history. This prevents long article output from overlapping the panel or hiding the current token.

## `detection.gif`

The detection animation shows the detector's view of the same experiment. It uses observed token IDs, the normalized first-word seed and the secret key; it does not use Qwen logits or next-token probabilities.

The main evidence panel contains:

- **solid marked z-curve**;
- **dashed control / without-watermark z-curve**;
- **red decision-threshold line**.

The lower panels show:

- one-sided confidence `1-p`;
- prioritized-token share, with the null reference at `0.5`.

The sentence view also uses a sliding recent-context window. Historical tokens are deliberately not highlighted; only the current observed token is outlined so the animation cannot visually confuse an old occurrence with the current step.

> `1-p` is confidence against SeedMark's detector null model. It is not a posterior probability that a passage was written by AI.

## Default article demonstration

The real-Qwen demo defaults to:

> Write a short plain-language article answering: What is AI? Explain what AI is, where it is used, benefits, risks, and conclude briefly.

This makes the marked/control comparison easier to read than the earlier `Research is` continuation. The default generation budget is 128 new tokens per run.

## CLI

```bash
seedmark qwen-demo --output-dir results/qwen
```

Adjust animation dimensions or speed:

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

GIF encoding uses all recorded frames, a shared adaptive palette, looping, `optimize=False`, and `disposal=2`. CI opens the generated GIFs and verifies that they remain multi-frame files.
