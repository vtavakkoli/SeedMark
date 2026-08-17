# Changelog

SeedMark follows [Semantic Versioning](https://semver.org/) for the public Python package and CLI.

## [0.3.0] - 2026-08-17

### Added
- Persistent host-side Hugging Face model cache via `SEEDMARK_MODEL_CACHE`.
- `seedmark qwen-cache` for explicit model prefetching and cache inspection.
- Single-source package version in `src/seedmark/_version.py`.
- `seedmark --version` CLI support.
- Packaging tests that keep Docker real-LLM requirements synchronized with `pyproject.toml`.
- `generation.gif` for real-Qwen runs: a polished token-by-token animation that keeps the generated sentence visible and highlights exactly the token being appended at each step.
- Candidate probability bars, keyed PRF score, selected token metadata and cumulative detector z-score in each GIF frame.

### Changed
- Reordered the Qwen Docker build so heavy Python dependencies are installed before application source is copied.
- SeedMark source now runs directly from `PYTHONPATH` in the real-LLM container, avoiding an editable package reinstall on source-only changes.
- Added BuildKit pip-download caching for faster dependency-layer rebuilds.
- Added Pillow only to the optional `real-llm` dependency set for deterministic local GIF rendering; the base SeedMark package remains dependency-free.

## [0.2.2] - 2026-08-17

### Fixed
- Qwen3.5 text-only loading uses `AutoTokenizer` and no longer constructs the multimodal video processor.

## [0.2.0] - 2026-08-17

### Added
- Real Qwen3.5 generation path with real next-token logits and token-ID watermark detection.

## [0.1.0] - 2026-08-17

### Added
- Initial toy language-model watermarking laboratory, detector, reports, Docker workflow, and CI.
