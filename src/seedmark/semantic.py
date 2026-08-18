"""Semantic self-keying primitives for SeedMark.

Semantic mode replaces the fragile first-word/absolute-position seed with a
sentence-synchronised semantic context key.  The first answer sentence is
bootstrapped from the user question.  After a sentence is completed, a compact
semantic embedding of the recent answer is assigned to a secret random semantic
bucket; that bucket becomes the key context for watermarking the next sentence.

This is an experimental research mode inspired by semantics-based watermarking
work such as SemaMark and SemStamp.  It is not a reimplementation of either
system and does not claim paraphrase-proof detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import random
import re
from typing import Any, Protocol, Sequence

from .core import DetectionResult

_TWO64 = float(1 << 64)
_SEMANTIC_NAMESPACE = b"seedmark-semantic-v1\x00"
_SENTENCE_END_RE = re.compile(r"[.!?][\"'\u201d\u2019)\]]*\s*$")

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
    context_sentences: int = 1
    semantic_model: str = DEFAULT_SEMANTIC_MODEL

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
        if self.context_sentences < 1:
            raise ValueError("context_sentences must be >= 1")
        if not self.semantic_model:
            raise ValueError("semantic_model must not be empty")


class HFMeanPoolingSemanticEncoder:
    """Small Hugging Face encoder with masked mean pooling.

    The implementation intentionally depends only on the existing ``real-llm``
    optional dependency set (PyTorch + Transformers).  It does not require the
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
    """Map dense semantic vectors to coarse secret random-projection buckets.

    Coarse nearest-anchor buckets are deliberately used instead of hashing every
    embedding bit.  Small semantic changes can then remain in the same bucket,
    while the secret key still hides the bucket geometry from an attacker.
    """

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
    token_offset: int,
    token_id: int,
) -> float:
    """Map (semantic bucket, sentence-local offset, token id) to U[0,1)."""

    if not secret_key:
        raise ValueError("secret_key must not be empty")
    if semantic_bucket < 0:
        raise ValueError("semantic_bucket must be >= 0")
    if token_offset < 1:
        raise ValueError("token_offset must be >= 1")
    if token_id < 0:
        raise ValueError("token_id must be >= 0")
    message = (
        _SEMANTIC_NAMESPACE
        + b"token\x00"
        + semantic_bucket.to_bytes(4, "big")
        + token_offset.to_bytes(4, "big")
        + token_id.to_bytes(8, "big")
    )
    digest = hmac.new(secret_key.encode("utf-8"), message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / _TWO64


def is_sentence_boundary(text: str) -> bool:
    """Return whether accumulated visible text ends a semantic sentence segment."""

    if not text:
        return False
    return bool(_SENTENCE_END_RE.search(text)) or text.endswith("\n\n")


class SemanticContextTracker:
    """Maintain the semantic key and sentence-local token offset during streaming."""

    def __init__(
        self,
        *,
        encoder: SemanticEncoder,
        bucketizer: SemanticBucketizer,
        bootstrap_text: str,
        context_sentences: int = 1,
    ) -> None:
        if context_sentences < 1:
            raise ValueError("context_sentences must be >= 1")
        bootstrap_text = bootstrap_text.strip()
        if not bootstrap_text:
            raise ValueError("bootstrap_text must not be empty")
        self.encoder = encoder
        self.bucketizer = bucketizer
        self.context_sentences = context_sentences
        self.completed_sentences: list[str] = []
        self.current_sentence_text = ""
        self.current_sentence_tokens = 0
        self.sentence_index = 0
        self.current_key = bucketizer.key_for_text(bootstrap_text, encoder)
        self.bootstrap_text = bootstrap_text

    @property
    def next_token_offset(self) -> int:
        return self.current_sentence_tokens + 1

    @property
    def semantic_context(self) -> str:
        if not self.completed_sentences:
            return self.bootstrap_text
        return " ".join(self.completed_sentences[-self.context_sentences :])

    def observe_token_piece(self, piece: str) -> bool:
        """Observe one decoded token piece and re-key at sentence boundaries."""

        self.current_sentence_tokens += 1
        self.current_sentence_text += piece
        if not is_sentence_boundary(self.current_sentence_text):
            return False

        sentence = self.current_sentence_text.strip()
        if sentence:
            self.completed_sentences.append(sentence)
            context = " ".join(self.completed_sentences[-self.context_sentences :])
            self.current_key = self.bucketizer.key_for_text(context, self.encoder)
        self.current_sentence_text = ""
        self.current_sentence_tokens = 0
        self.sentence_index += 1
        return True


def detect_semantic_token_ids(
    token_ids: Sequence[int],
    *,
    tokenizer: Any,
    encoder: SemanticEncoder,
    secret_key: str,
    bootstrap_text: str,
    threshold_z: float = 3.0,
    bucket_count: int = 32,
    context_sentences: int = 1,
) -> DetectionResult:
    """Detect semantic self-keyed correlation from visible generated token IDs.

    The detector reconstructs semantic buckets from the observed text itself.
    Sentence boundaries reset the local token offset, so insertions/deletions in
    one sentence do not permanently shift every later token position.
    """

    if threshold_z <= 0:
        raise ValueError("threshold_z must be > 0")
    if not token_ids:
        return DetectionResult(0, 0.5, 0.0, 0.5, threshold_z, False)

    bucketizer = SemanticBucketizer(secret_key=secret_key, bucket_count=bucket_count)
    tracker = SemanticContextTracker(
        encoder=encoder,
        bucketizer=bucketizer,
        bootstrap_text=bootstrap_text,
        context_sentences=context_sentences,
    )
    observed: list[float] = []
    for token_id in token_ids:
        token_id = int(token_id)
        observed.append(
            semantic_token_score(
                secret_key,
                tracker.current_key.bucket,
                tracker.next_token_offset,
                token_id,
            )
        )
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        tracker.observe_token_piece(piece)

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
