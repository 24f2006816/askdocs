"""Wiring: build (or load) the index and the chat engine from settings."""

from __future__ import annotations

import logging

from .chat import ChatEngine
from .config import Settings
from .embeddings import Embedder, build_embedder
from .index import KnowledgeBase, docs_fingerprint
from .ingest import chunk_documents, load_directory
from .llm import build_llm
from .rerank import Reranker, build_reranker

log = logging.getLogger(__name__)


def build_knowledge_base(settings: Settings, embedder: Embedder, force: bool = False) -> KnowledgeBase:
    """Load the saved index if it matches the current documents; otherwise rebuild it."""
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = docs_fingerprint(settings.docs_dir)
    if not force:
        kb = KnowledgeBase.load(settings.index_dir, embedder, fingerprint=fingerprint)
        if kb is not None:
            log.info("Loaded index: %d chunks", len(kb.chunks))
            return kb
    docs = load_directory(settings.docs_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    kb = KnowledgeBase.build(embedder, chunks)
    kb.save(settings.index_dir, fingerprint)
    log.info("Built index: %d documents/pages -> %d chunks (%s)", len(docs), len(chunks), embedder.name)
    return kb


def build_engine(settings: Settings, embedder: Embedder | None = None,
                 reranker: Reranker | None = None) -> ChatEngine:
    embedder = embedder or build_embedder(settings.embedder, settings.embedding_model)
    kb = build_knowledge_base(settings, embedder)
    return ChatEngine(
        kb,
        llm=build_llm(settings),
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        min_keyword_coverage=settings.min_keyword_coverage,
        max_history_turns=settings.max_history_turns,
        reranker=reranker if reranker is not None else build_reranker(settings.reranker_model),
        rerank_candidates=settings.rerank_candidates,
    )
