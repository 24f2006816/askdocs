"""Text embedders. All return L2-normalised float32 matrices, so dot product = cosine."""

from __future__ import annotations

import logging
import math
import re
import zlib
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Dependency-free lexical embedder (feature hashing of words and word bigrams).

    It needs no model download, so tests and CI run anywhere. Semantic quality is
    lower than a neural model; use it as a fallback, not as the default.
    """

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> dict[int, float]:
        tokens = _TOKEN_RE.findall(text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        counts: dict[int, float] = {}
        for gram in grams:
            bucket = zlib.crc32(gram.encode()) % self.dim
            counts[bucket] = counts.get(bucket, 0.0) + 1.0
        return {k: 1.0 + math.log(v) for k, v in counts.items()}

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for col, weight in self._features(text).items():
                matrix[row, col] = weight
        return _normalise(matrix)


class SentenceTransformerEmbedder:
    """Neural embedder (default: all-MiniLM-L6-v2, 384 dimensions, runs on CPU)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashingEmbedder()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if kind == "auto":
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # missing package or model download blocked
            log.warning("Falling back to the hashing embedder (%s). Install requirements-embeddings.txt "
                        "for semantic search.", exc)
            return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDER: {kind}")
