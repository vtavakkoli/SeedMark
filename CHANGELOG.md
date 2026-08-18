# Changelog

SeedMark follows [Semantic Versioning](https://semver.org/) for the public Python package and CLI.

## [Unreleased]

### Changed
- The default Qwen demonstration now asks for a short plain-language article answering “What is AI?” and uses a 128-token generation budget.
- The real-LLM HTML report now leads with two explicit result cards and a marked-vs-control contrast: watermarked output vs control / without watermark.
- `generation.gif` and `detection.gif` use a sliding recent-context window for long text and highlight only the current token.
- `detection.gif` now presents the marked z-curve, dashed control z-curve, decision threshold, one-sided confidence (`1-p`), and prioritized-token share in non-overlapping chart regions.

### Fixed
- Long generated text no longer overlaps or hides the current highlighted token in GIF sentence panels.
- Report status badges reflect the actual detector result and explicitly flag runs where the expected marked/control contrast is not achieved.

## [0.5.0] - 2026-08-17

### Added
- `detection.gif`, which replays the detector's keyed-correlation signal token by token without using Qwen logits.
- `generation-preview.png` and `detection-preview.png` as static poster-quality fallbacks/inspection assets.
- A redesigned standalone `report.html` that embeds both animations, explains generation and detection, compares marked/control output, includes an interactive token microscope, live charts, and a raw trace table.
- `--gif-width` and `--gif-height` controls in addition to `--gif-frame-ms`.
- Dedicated `seedmark.reporting` module and report tests.

### Fixed
- GIF encoding now uses a consistent adaptive palette with `save_all=True`, `append_images`, `optimize=False`, `disposal=2`, and explicit looping so multi-frame animations remain animated across common viewers.
- The current generated token is isolated and highlighted only on its own frame.
- Detection visualization uses statistically precise wording: `1-p` is confidence against the null, not a posterior probability of AI authorship.

### Changed
- Both visualizations are rendered from the already-recorded generation trace, so creating the report never runs Qwen a second time.
- CI now opens and verifies both generated GIFs as true multi-frame files.
- Real-LLM report generation is separated from the model adapter for a cleaner scientific code structure.

## [0.4.0] - 2026-08-17

### Added
- `generation.gif` for real-Qwen runs: a polished token-by-token animation that keeps the generated sentence visible and highlights exactly the token being appended at each step.
- Candidate probability bars, keyed PRF score, selected token metadata and cumulative detector z-score in every animation frame.
- `--gif-frame-ms` to control animation speed and `--no-gif` for benchmark-only runs.
- `docs/animation.md` documenting what is highlighted and how the animation maps model probabilities to the statistical watermark signal.

### Changed
- Added Pillow only to the optional `real-llm` dependency set for GIF rendering; the dependency-free core package remains unchanged.

## [0.3.0] - 2026-08-17

### Added
- Persistent host-side Hugging Face model cache via `SEEDMARK_MODEL_CACHE`.
- `seedmark qwen-cache` for explicit model prefetching and cache inspection.
- Single-source package version in `src/seedmark/_version.py`.
- `seedmark --version` CLI support.
- Packaging tests that keep Docker real-LLM requirements synchronized with `pyproject.toml`.

### Changed
- Reordered the Qwen Docker build so heavy Python dependencies are installed before application source is copied.
- SeedMark source now runs directly from `PYTHONPATH` in the real-LLM container, avoiding an editable package reinstall on source-only changes.
- Added BuildKit pip-download caching for faster dependency-layer rebuilds.

## [0.2.2] - 2026-08-17

### Fixed
- Qwen3.5 text-only loading uses `AutoTokenizer` and no longer constructs the multimodal video processor.

## [0.2.0] - 2026-08-17

### Added
- Real Qwen3.5 generation path with real next-token logits and token-ID watermark detection.

## [0.1.0] - 2026-08-17

### Added
- Initial toy language-model watermarking laboratory, detector, reports, Docker workflow, and CI.
