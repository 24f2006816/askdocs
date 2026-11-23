"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines; variables already in the environment win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    # Storage
    docs_dir: Path
    index_dir: Path
    feedback_path: Path
    # Retrieval
    embedder: str  # "auto" | "sentence-transformers" | "hash"
    embedding_model: str
    reranker_model: str  # empty string = no reranking
    rerank_candidates: int
    chunk_size: int  # characters per chunk
    chunk_overlap: int  # characters repeated between consecutive chunks
    top_k: int
    min_similarity: float  # refuse if the best dense score is below this...
    min_keyword_coverage: float  # ...and fewer than this share of query words appear in any hit
    # Generation
    llm_provider: str  # "auto" | "gemini" | "openai" | "extractive"
    gemini_api_key: str | None
    gemini_model: str
    openai_api_key: str | None
    openai_model: str
    max_history_turns: int
    # Public deployment
    admin_token: str | None  # when set, uploads and deletes need this token
    rate_limit_per_minute: int  # per client, for chat and feedback; 0 disables
    max_upload_mb: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            docs_dir=_path("DOCS_DIR", PROJECT_ROOT / "data" / "docs"),
            index_dir=_path("INDEX_DIR", PROJECT_ROOT / "data" / "index"),
            feedback_path=_path("FEEDBACK_PATH", PROJECT_ROOT / "data" / "feedback" / "feedback.jsonl"),
            embedder=os.getenv("EMBEDDER", "auto").lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            reranker_model=os.getenv("RERANKER_MODEL", ""),
            rerank_candidates=_int("RERANK_CANDIDATES", 20),
            chunk_size=_int("CHUNK_SIZE", 700),
            chunk_overlap=_int("CHUNK_OVERLAP", 120),
            top_k=_int("TOP_K", 4),
            min_similarity=_float("MIN_SIMILARITY", 0.30),
            min_keyword_coverage=_float("MIN_KEYWORD_COVERAGE", 0.30),
            llm_provider=os.getenv("LLM_PROVIDER", "auto").lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
            max_history_turns=_int("MAX_HISTORY_TURNS", 6),
            admin_token=os.getenv("ADMIN_TOKEN") or None,
            rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 30),
            max_upload_mb=_int("MAX_UPLOAD_MB", 10),
        )
