"""Persistent Hugging Face model-cache helpers used by the real-LLM workflow."""

from __future__ import annotations

import os
from pathlib import Path


def cache_home() -> Path:
    """Return the effective Hugging Face cache root used by SeedMark."""
    configured = os.environ.get("HF_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").resolve()


def prefetch_model(model_name: str, *, revision: str | None = None) -> Path:
    """Download/cache a model snapshot and return the immutable snapshot path.

    Hugging Face's cache is content-addressed. Repeating this call reuses files
    already present in HF_HOME/HF_HUB_CACHE instead of downloading the weights
    again unless the requested upstream revision has changed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "Model caching requires the real-LLM dependencies: pip install -e '.[real-llm]'"
        ) from exc

    snapshot = snapshot_download(repo_id=model_name, revision=revision, repo_type="model")
    return Path(snapshot).resolve()
