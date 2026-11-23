"""Hybrid retrieval: dense vectors (FAISS) + BM25 keywords, merged with reciprocal rank fusion.

Dense search catches paraphrases ("money back" -> "refund"); BM25 catches exact terms
(batch codes, numbers, names) that embeddings blur. Fusing ranks avoids having to
calibrate two very different score scales against each other.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder
from .ingest import SUPPORTED_EXTENSIONS, Chunk

try:  # FAISS is optional; numpy is exact and fast enough for small corpora.
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

_WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "so",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stop words removed and a light plural stemmer."""
    tokens = []
    for word in _WORD_RE.findall(text.lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif word.endswith("sses") or (len(word) > 4 and word.endswith(("ches", "shes", "xes", "zes"))):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        tokens.append(word)
    return tokens


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) with a non-negative IDF."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(doc) for doc in corpus]
        self.doc_lens = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avg_len = float(self.doc_lens.mean()) if len(corpus) else 0.0
        n_docs = len(corpus)
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.term_freqs), dtype=np.float32)
        if not self.term_freqs or self.avg_len == 0:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.doc_lens / self.avg_len)
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self.term_freqs], dtype=np.float32)
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


class VectorIndex:
    """Exact inner-product search; uses FAISS when installed."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self._faiss = None
        if faiss is not None and len(vectors):
            self._faiss = faiss.IndexFlatIP(vectors.shape[1])
            self._faiss.add(vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.vectors):
            return []
        k = min(k, len(self.vectors))
        if self._faiss is not None:
            scores, ids = self._faiss.search(query.reshape(1, -1), k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]
        sims = self.vectors @ query
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # final ranking score
    dense_score: float  # cosine similarity to the query
    keyword_score: float  # BM25 score (0 = no shared keywords)


def docs_fingerprint(directory: Path) -> str:
    """Content hash of every supported file, used to detect when the index is stale."""
    digest = hashlib.sha256()
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                digest.update(path.name.encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class KnowledgeBase:
    def __init__(self, embedder: Embedder, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.embedder = embedder
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self._dense = VectorIndex(self.vectors)
        self._bm25 = BM25([tokenize(c.index_text) for c in chunks])

    # ---------------------------------------------------------------- construction

    @classmethod
    def build(cls, embedder: Embedder, chunks: list[Chunk]) -> KnowledgeBase:
        if chunks:
            vectors = embedder.embed([c.index_text for c in chunks])
        else:
            vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(embedder, chunks, vectors)

    def save(self, directory: Path, fingerprint: str = "") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder.name, "dim": self.embedder.dim,
                "chunks": len(self.chunks), "fingerprint": fingerprint}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, fingerprint: str | None = None) -> KnowledgeBase | None:
        """Load a saved index, or return None if it is missing or was built differently."""
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != embedder.name or meta.get("dim") != embedder.dim:
            return None
        if fingerprint is not None and meta.get("fingerprint") != fingerprint:
            return None
        with (directory / "chunks.jsonl").open(encoding="utf-8") as fh:
            chunks = [Chunk.from_dict(json.loads(line)) for line in fh if line.strip()]
        vectors = np.load(directory / "vectors.npy")
        return cls(embedder, chunks, vectors)

    # ---------------------------------------------------------------- search

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[SearchHit]:
        if not self.chunks:
            return []
        n_candidates = min(len(self.chunks), max(k * 5, 20))
        query_vec = self.embedder.embed([query])[0]
        keyword_scores = self._bm25.scores(tokenize(query))

        dense_ranking = [i for i, _ in self._dense.search(query_vec, n_candidates)]
        keyword_ranking = [int(i) for i in np.argsort(-keyword_scores)[:n_candidates] if keyword_scores[i] > 0]

        if mode == "dense":
            order = [(i, 0.0) for i in dense_ranking]
        elif mode == "bm25":
            order = [(i, float(keyword_scores[i])) for i in keyword_ranking]
        elif mode == "hybrid":
            fused = reciprocal_rank_fusion([dense_ranking, keyword_ranking])
            order = sorted(fused.items(), key=lambda item: -item[1])
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        hits = []
        for idx, score in order[:k]:
            dense = float(self.vectors[idx] @ query_vec)
            hits.append(SearchHit(self.chunks[idx], score if mode != "dense" else dense, dense,
                                  float(keyword_scores[idx])))
        return hits

    def sources(self) -> list[dict]:
        counts: Counter[str] = Counter(c.source for c in self.chunks)
        return [{"source": name, "chunks": n} for name, n in sorted(counts.items())]
