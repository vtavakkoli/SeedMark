"""Semantic self-keying primitives for SeedMark.

Semantic mode uses coarse secret semantic buckets rather than exact lexical
prefixes. Two scopes are supported:

* ``answer`` (default): one semantic bucket is derived from the complete answer.
  Generation uses a draft/commit pass so the final answer lands in the same
  bucket that keyed token sampling; detection derives the bucket from the final
  answer itself.
* ``paragraph``: the user question bootstraps the first paragraph and each
  completed paragraph keys the following paragraph. Paragraph boundaries provide
  streaming re-synchronisation points without reacting to sentence punctuation.

Within either scope, token scores use a token-specific occurrence counter instead
of an absolute text position. Inserting an unrelated token therefore does not
shift the keyed score stream for every later token.

This is an experimental research mode inspired by semantics-based watermarking
work such as SemaMark and SemStamp. It is not a reimplementation of either system
and does not claim paraphrase-proof detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import random
from typing import Any, Protocol, Sequence

from .core import DetectionResult

_TWO64 = float(1 << 64)
_SEMANTIC_NAMESPACE = b"seedmark-semantic-v2\x00"

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_SCOPE = "answer"
SEMANTIC_SCOPES = ("answer", "paragraph")


class SemanticEncoder(Protocol):
    """Minimal semantic-encoder contract used by generation and detection."""

    model_name: str

    def encode(self, text: str) -> Sequence[float]:
        """Return one dense semantic vector for *text*."""


@dataclass(frozen=True, slots=True)
class SemanticKey:
    """Stable semantic bucket selected for one context window."""

    bucket: int
    margin: float
    similarity: float


@dataclass(frozen=True, slots=True)
class SemanticWatermarkConfig:
    """Configuration for semantic self-keyed token sampling."""

    secret_key: str = "seedmark-demo-key"
    strength: float = 1.5
    top_k: int = 20
    threshold_z: float = 3.0
    bucket_count: int = 32
    semantic_scope: str = DEFAULT_SEMANTIC_SCOPE
    context_paragraphs: int = 1
    semantic_model: str = DEFAULT_SEMANTIC_MODEL
    max_answer_passes: int = 4

    def __post_init__(self) -> None:
        if not self.secret_key:
            raise ValueError("secret_key must not be empty")
        if self.strength < 0:
            raise ValueError("strength must be >= 0")
        if self.top_k < 2:
            raise ValueError("top_k must be >= 2")
        if self.threshold_z <= 0:
            raise ValueError("threshold_z must be > 0")
        if self.bucket_count < 4:
            raise ValueError("bucket_count must be >= 4")
        if self.semantic_scope not in SEMANTIC_SCOPES:
            raise ValueError(f"semantic_scope must be one of {SEMANTIC_SCOPES}")
        if self.context_paragraphs < 1:
            raise ValueError("context_paragraphs must be >= 1")
        if not self.semantic_model:
            raise ValueError("semantic_model must not be empty")
        if self.max_answer_passes < 1:
            raise ValueError("max_answer_passes must be >= 1")


class HFMeanPoolingSemanticEncoder:
    """Small Hugging Face encoder with masked mean pooling.

    The implementation intentionally depends only on the existing ``real-llm``
    optional dependency set (PyTorch + Transformers). It does not require the
    sentence-transformers Python package.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Semantic watermarking requires the real-LLM dependencies: "
                "pip install -e '.[real-llm]'"
            ) from exc
        if max_length < 8:
            raise ValueError("max_length must be >= 8")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def encode(self, text: str) -> tuple[float, ...]:
        text = text.strip()
        if not text:
            raise ValueError("semantic text must not be empty")
        torch = self.torch
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            norm = pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            pooled = pooled / norm
        return tuple(float(value) for value in pooled[0].detach().cpu().tolist())


class SemanticBucketizer:
    """Map dense semantic vectors to coarse secret random-projection buckets."""

    def __init__(self, *, secret_key: str, bucket_count: int = 32) -> None:
        if not secret_key:
            raise ValueError("secret_key must not be empty")
        if bucket_count < 4:
            raise ValueError("bucket_count must be >= 4")
        self.secret_key = secret_key
        self.bucket_count = bucket_count
        self._anchors_by_dimension: dict[int, tuple[tuple[float, ...], ...]] = {}

    def _anchors(self, dimension: int) -> tuple[tuple[float, ...], ...]:
        if dimension < 2:
            raise ValueError("semantic embedding must have at least two dimensions")
        cached = self._anchors_by_dimension.get(dimension)
        if cached is not None:
            return cached

        anchors: list[tuple[float, ...]] = []
        key = self.secret_key.encode("utf-8")
        for bucket in range(self.bucket_count):
            seed_material = hmac.new(
                key,
                _SEMANTIC_NAMESPACE + b"anchor\x00" + bucket.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
            rng = random.Random(int.from_bytes(seed_material, "big"))
            values = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            anchors.append(tuple(value / norm for value in values))
        result = tuple(anchors)
        self._anchors_by_dimension[dimension] = result
        return result

    def key_for_embedding(self, embedding: Sequence[float]) -> SemanticKey:
        vector = tuple(float(value) for value in embedding)
        if len(vector) < 2:
            raise ValueError("semantic embedding must have at least two dimensions")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            raise ValueError("semantic embedding must not be the zero vector")
        unit = tuple(value / norm for value in vector)
        similarities = [
            sum(value * anchor_value for value, anchor_value in zip(unit, anchor, strict=True))
            for anchor in self._anchors(len(unit))
        ]
        order = sorted(range(len(similarities)), key=similarities.__getitem__, reverse=True)
        best = order[0]
        second = order[1]
        return SemanticKey(
            bucket=best,
            margin=similarities[best] - similarities[second],
            similarity=similarities[best],
        )

    def key_for_text(self, text: str, encoder: SemanticEncoder) -> SemanticKey:
        return self.key_for_embedding(encoder.encode(text))


def semantic_token_score(
    secret_key: str,
    semantic_bucket: int,
    token_occurrence: int,
    token_id: int,
) -> float:
    """Map (semantic bucket, token occurrence, token id) to U[0,1).

    ``token_occurrence`` is the occurrence number of this *specific candidate
    token ID* within the current semantic scope, not an absolute text position.
    This keeps unrelated insertions/deletions from shifting all later PRF inputs.
    """

    if not secret_key:
        raise ValueError("secret_key must not be empty")
    if semantic_bucket < 0:
        raise ValueError("semantic_bucket must be >= 0")
    if token_occurrence < 1:
        raise ValueError("token_occurrence must be >= 1")
    if token_id < 0:
        raise ValueError("token_id must be >= 0")
    message = (
        _SEMANTIC_NAMESPACE
        + b"token\x00"
        + semantic_bucket.to_bytes(4, "big")
        + token_occurrence.to_bytes(4, "big")
        + token_id.to_bytes(8, "big")
    )
    digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / _TWO64


def is_paragraph_boundary(text: str) -> bool:
    """Return whether accumulated visible text ends at a blank-line boundary."""

    if not text:
        return False
    normalized = text.replace("\r\n", "\n")
    return normalized.endswith("\n\n")


class SemanticContextTracker:
    """Maintain paragraph semantic state for streaming generation/detection."""

    def __init__(
        self,
        *,
        encoder: SemanticEncoder,
        bucketizer: SemanticBucketizer,
        bootstrap_text: str,
        context_paragraphs: int = 1,
    ) -> None:
        if context_paragraphs < 1:
            raise ValueError("context_paragraphs must be >= 1")
        bootstrap_text = bootstrap_text.strip()
        if not bootstrap_text:
            raise ValueError("bootstrap_text must not be empty")
        self.encoder = encoder
        self.bucketizer = bucketizer
        self.context_paragraphs = context_paragraphs
        self.completed_paragraphs: list[str] = []
        self.current_paragraph_text = ""
        self.paragraph_index = 0
        self.current_key = bucketizer.key_for_text(bootstrap_text, encoder)
        self.bootstrap_text = bootstrap_text
        self._token_occurrences: dict[int, int] = {}

    def next_token_occurrence(self, token_id: int) -> int:
        return self._token_occurrences.get(int(token_id), 0) + 1

    @property
    def semantic_context(self) -> str:
        if not self.completed_paragraphs:
            return self.bootstrap_text
        return "\n\n".join(self.completed_paragraphs[-self.context_paragraphs :])

    def observe_token(self, token_id: int, piece: str) -> bool:
        """Observe one chosen token and re-key only at paragraph boundaries."""

        token_id = int(token_id)
        self._token_occurrences[token_id] = self.next_token_occurrence(token_id)
        self.current_paragraph_text += piece
        if not is_paragraph_boundary(self.current_paragraph_text):
            return False

        paragraph = self.current_paragraph_text.strip()
        if paragraph:
            self.completed_paragraphs.append(paragraph)
            context = "\n\n".join(
                self.completed_paragraphs[-self.context_paragraphs :]
            )
            self.current_key = self.bucketizer.key_for_text(context, self.encoder)
        self.current_paragraph_text = ""
        self._token_occurrences.clear()
        self.paragraph_index += 1
        return True


def decode_generated_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    """Decode visible generated IDs with a small tokenizer-compatibility fallback."""

    ids = [int(token_id) for token_id in token_ids]
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def _detection_from_scores(
    observed: Sequence[float], *, threshold_z: float
) -> DetectionResult:
    if not observed:
        return DetectionResult(0, 0.5, 0.0, 0.5, threshold_z, False)
    n = len(observed)
    mean_score = sum(observed) / n
    z_score = (sum(observed) - 0.5 * n) / math.sqrt(n / 12.0)
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return DetectionResult(
        n_scored_tokens=n,
        mean_score=mean_score,
        z_score=z_score,
        p_value_one_sided=p_value,
        threshold_z=threshold_z,
        detected=z_score >= threshold_z,
    )


def detect_semantic_token_ids(
    token_ids: Sequence[int],
    *,
    tokenizer: Any,
    encoder: SemanticEncoder,
    secret_key: str,
    bootstrap_text: str,
    threshold_z: float = 3.0,
    bucket_count: int = 32,
    semantic_scope: str = DEFAULT_SEMANTIC_SCOPE,
    context_paragraphs: int = 1,
) -> DetectionResult:
    """Detect semantic-keyed correlation from visible generated token IDs.

    ``answer`` scope embeds the complete observed answer and uses one bucket for
    all token occurrences. ``paragraph`` scope reconstructs the previous-paragraph
    bucket while resetting occurrence counters at blank-line boundaries.
    """

    if threshold_z <= 0:
        raise ValueError("threshold_z must be > 0")
    if semantic_scope not in SEMANTIC_SCOPES:
        raise ValueError(f"semantic_scope must be one of {SEMANTIC_SCOPES}")
    if context_paragraphs < 1:
        raise ValueError("context_paragraphs must be >= 1")
    if not token_ids:
        return DetectionResult(0, 0.5, 0.0, 0.5, threshold_z, False)

    bucketizer = SemanticBucketizer(secret_key=secret_key, bucket_count=bucket_count)
    observed: list[float] = []

    if semantic_scope == "answer":
        answer_text = decode_generated_text(tokenizer, token_ids).strip()
        if not answer_text:
            return DetectionResult(0, 0.5, 0.0, 0.5, threshold_z, False)
        semantic_key = bucketizer.key_for_text(answer_text, encoder)
        occurrences: dict[int, int] = {}
        for token_id in token_ids:
            token_id = int(token_id)
            occurrence = occurrences.get(token_id, 0) + 1
            occurrences[token_id] = occurrence
            observed.append(
                semantic_token_score(
                    secret_key,
                    semantic_key.bucket,
                    occurrence,
                    token_id,
                )
            )
        return _detection_from_scores(observed, threshold_z=threshold_z)

    tracker = SemanticContextTracker(
        encoder=encoder,
        bucketizer=bucketizer,
        bootstrap_text=bootstrap_text,
        context_paragraphs=context_paragraphs,
    )
    for token_id in token_ids:
        token_id = int(token_id)
        occurrence = tracker.next_token_occurrence(token_id)
        observed.append(
            semantic_token_score(
                secret_key,
                tracker.current_key.bucket,
                occurrence,
                token_id,
            )
        )
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        tracker.observe_token(token_id, piece)

    return _detection_from_scores(observed, threshold_z=threshold_z)
