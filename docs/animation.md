# Animated generation and detection report

Real-Qwen chat runs create two complementary GIFs plus static PNG previews:

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

## Chat context

The default run uses Qwen's native chat template:

```text
User: What is AI?
Assistant: <article generated token by token>
```

The system instruction asks for a short plain-language article and disables visible thinking output. Only assistant-response tokens are watermarked. The human-readable transcript prefix is kept in the sentence view so the animation looks like a normal chat interaction rather than an arbitrary completion.

## `generation.gif`

Each frame links the model distribution to the chosen assistant token and the accumulating detector signal:

- dashed curve: base Qwen top-k probabilities;
- solid curve: SeedMark-adjusted probabilities;
- teal/coral candidate markers: keyed preference direction;
- pink ring: candidate selected on this step;
- sentence panel: a **sliding recent-context window**;
- pink sentence highlight: **only the current appended assistant token**;
- lower charts: cumulative z-score and prioritized-token share.

The recent-context window is intentionally left-clipped as an answer grows. Historical whitespace is compacted, long fragments are split to the available width, and the current token is kept separate from history. This prevents article output from overlapping the panel or hiding the current token.

## `detection.gif`

The detection animation shows the detector's view of the same assistant answer. It uses observed assistant token IDs, the normalized first word of the user question, and the secret key; it does not use Qwen logits or next-token probabilities.

The main evidence panel contains:

- **solid marked z-curve**;
- **dashed control / without-watermark z-curve**;
- **red decision-threshold line**.

The lower panels show:

- one-sided confidence `1-p`;
- prioritized-token share, with the null reference at `0.5`.

The sentence view also uses a sliding recent-context window. Historical tokens are deliberately not highlighted; only the current observed token is outlined so the animation cannot visually confuse an old occurrence with the current step.

> `1-p` is confidence against SeedMark's detector null model. It is not a posterior probability that a passage was written by AI.

## Default demonstration

The user asks:

> **What is AI?**

The system message asks Qwen to answer as a short plain-language article covering what AI is, where it is used, benefits, risks, and a brief conclusion. The default generation budget is 128 assistant tokens per marked/control run.

## CLI

```bash
seedmark qwen-demo --output-dir results/qwen
```

Ask another question:

```bash
seedmark qwen-demo --question "What is edge AI?"
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
