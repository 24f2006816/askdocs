import shutil
from pathlib import Path

import pytest

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.service import build_knowledge_base

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    shutil.copytree(ROOT / "data" / "docs", docs)
    return Settings(
        docs_dir=docs, index_dir=tmp_path / "index", feedback_path=tmp_path / "feedback" / "feedback.jsonl",
        embedder="hash", embedding_model="unused", reranker_model="", rerank_candidates=20,
        chunk_size=700, chunk_overlap=120, top_k=4, min_similarity=0.30, min_keyword_coverage=0.30,
        llm_provider="extractive", gemini_api_key=None, gemini_model="unused",
        openai_api_key=None, openai_model="unused", max_history_turns=6,
        allow_uploads=True, admin_token=None, rate_limit_per_minute=0, max_upload_mb=1,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def kb(settings, embedder):
    return build_knowledge_base(settings, embedder)


class FakeLLM:
    """Scripted LLM. Each queued reply is a string (streamed in small pieces) or an exception.

    A list of strings streams exactly those pieces; an exception inside the list is raised
    mid-stream, after the earlier pieces were sent.
    """

    name = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def _next(self, system, prompt):
        self.calls.append((system, prompt))
        return self.replies.pop(0)

    def generate(self, system, prompt):
        reply = self._next(system, prompt)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream(self, system, prompt):
        reply = self._next(system, prompt)
        if isinstance(reply, Exception):
            raise reply
        pieces = reply if isinstance(reply, list) else [reply[i:i + 5] for i in range(0, len(reply), 5)]
        for piece in pieces:
            if isinstance(piece, Exception):
                raise piece
            yield piece


@pytest.fixture
def fake_llm():
    return FakeLLM
