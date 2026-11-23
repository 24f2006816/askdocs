"""Optional second-stage reranking with a cross-encoder.

Retrieval (bi-encoder + BM25) is fast but scores the query and each passage
separately. A cross-encoder reads the query and passage together, which ranks
much more precisely, so we run it on the top ~20 candidates only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Protocol

from .index import SearchHit

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def scores(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small enough for CPU)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=os.getenv("MODEL_DEVICE") or None)
        self.name = model_name

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, p) for p in passages], show_progress_bar=False)]


def rerank(reranker: Reranker, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
    if not hits:
        return []
    scores = reranker.scores(query, [h.chunk.index_text for h in hits])
    ranked = sorted(zip(scores, hits, strict=True), key=lambda pair: -pair[0])
    return [replace(hit, score=score) for score, hit in ranked[:k]]


def build_reranker(model_name: str) -> Reranker | None:
    if not model_name:
        return None
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:  # missing package or model download blocked
        log.warning("Reranker disabled (%s).", exc)
        return None
