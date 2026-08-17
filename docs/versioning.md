# Versioning and release policy

SeedMark uses Semantic Versioning (`MAJOR.MINOR.PATCH`).

- **MAJOR**: incompatible public API, CLI, report schema, or detector-contract changes.
- **MINOR**: backward-compatible research features, new model adapters, new report fields, or new CLI commands.
- **PATCH**: backward-compatible bug fixes, packaging fixes, documentation corrections, and reproducibility improvements.

The single source of truth is `src/seedmark/_version.py`. `pyproject.toml` reads that value dynamically, and the CLI exposes it with:

```bash
seedmark --version
```

Every release should update `CHANGELOG.md` and `CITATION.cff`. Generated scientific reports should record the SeedMark version whenever practical so results can be traced back to the exact software contract.

## Model versions

Package versioning and model versioning are independent. Experiments should record the full Hugging Face model identifier and, for archival studies, pin a model revision/commit. `seedmark qwen-cache --revision ...` supports prefetching a specific model revision into the persistent cache.
