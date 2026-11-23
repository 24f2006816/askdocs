"""FastAPI app: REST and streaming endpoints, plus the single-page chat UI."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__
from .chat import ChatEngine
from .config import Settings
from .embeddings import Embedder, build_embedder
from .ingest import SUPPORTED_EXTENSIONS
from .rerank import Reranker
from .security import RateLimiter, check_admin, client_ip
from .service import build_engine, build_knowledge_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("askdocs")

STATIC_DIR = Path(__file__).parent / "static"
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class SourceOut(BaseModel):
    n: int
    source: str
    page: int | None
    section: str | None
    location: str
    snippet: str
    similarity: float
    cited: bool


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    grounded: bool
    mode: str
    standalone_question: str
    latency_ms: float
    sources: list[SourceOut]


class FeedbackRequest(BaseModel):
    session_id: str = Field(max_length=64)
    question: str = Field(max_length=2000)
    answer: str = Field(max_length=8000)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


def safe_filename(name: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", Path(name).name).strip(" .")
    return cleaned[:120]


def create_app(settings: Settings | None = None, embedder: Embedder | None = None,
               reranker: Reranker | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    state: dict = {"started": time.time()}
    rebuild_lock = threading.Lock()
    feedback_lock = threading.Lock()
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["embedder"] = embedder or build_embedder(settings.embedder, settings.embedding_model)
        state["engine"] = build_engine(settings, state["embedder"], reranker)
        eng = state["engine"]
        log.info("AskDocs ready: %d chunks | embedder=%s | reranker=%s | answers=%s | uploads=%s",
                 len(eng.kb.chunks), state["embedder"].name, eng.reranker.name if eng.reranker else "off",
                 eng.mode, "locked" if settings.admin_token else "open")
        yield

    app = FastAPI(title="AskDocs", version=__version__, lifespan=lifespan,
                  description="Retrieval-augmented chatbot that answers from your documents with citations.")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            log.info("%s %s %d %.0fms id=%s", request.method, request.url.path, response.status_code,
                     elapsed, request_id)
        return response

    def engine() -> ChatEngine:
        return state["engine"]

    def rebuild_index() -> None:
        with rebuild_lock:
            engine().kb = build_knowledge_base(settings, state["embedder"], force=True)

    # ----------------------------------------------------------------- status

    def require_write_access(request: Request) -> None:
        if not settings.allow_uploads:
            raise HTTPException(403, "Uploads are turned off on this demo. Run AskDocs yourself to use your own files.")
        check_admin(request, settings.admin_token)

    @app.get("/api/health")
    def health() -> dict:
        eng = engine()
        return {
            "status": "ok", "version": __version__, "uptime_seconds": int(time.time() - state["started"]),
            "documents": len(eng.kb.sources()), "chunks": len(eng.kb.chunks),
            "embedder": eng.kb.embedder.name, "reranker": eng.reranker.name if eng.reranker else None,
            "answer_mode": eng.mode, "uploads_enabled": settings.allow_uploads,
            "uploads_locked": bool(settings.admin_token), "stats": eng.stats.snapshot(),
        }

    # ----------------------------------------------------------------- documents

    @app.get("/api/documents")
    def list_documents() -> list[dict]:
        return engine().kb.sources()

    @app.post("/api/documents")
    async def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> dict:  # noqa: B008
        require_write_access(request)
        limit = settings.max_upload_mb * 1024 * 1024
        saved: list[Path] = []
        for upload in files:
            name = safe_filename(upload.filename or "")
            if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(400, f"Unsupported file '{upload.filename}'. Use PDF, Word, Markdown or TXT.")
            data = await upload.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"'{name}' is larger than {settings.max_upload_mb} MB.")
            target = settings.docs_dir / name
            target.write_bytes(data)
            saved.append(target)
        try:
            await run_in_threadpool(rebuild_index)
        except Exception as exc:  # e.g. a corrupt PDF: remove it and keep the old index
            for path in saved:
                path.unlink(missing_ok=True)
            await run_in_threadpool(rebuild_index)
            raise HTTPException(400, f"Could not read the uploaded file: {exc}") from exc
        return {"added": [p.name for p in saved], "chunks": len(engine().kb.chunks)}

    @app.delete("/api/documents/{name}")
    async def delete_document(name: str, request: Request) -> dict:
        require_write_access(request)
        target = settings.docs_dir / safe_filename(name)
        if not target.is_file():
            raise HTTPException(404, "No such document.")
        target.unlink()
        await run_in_threadpool(rebuild_index)
        return {"deleted": target.name, "chunks": len(engine().kb.chunks)}

    # ----------------------------------------------------------------- chat

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        limiter.check(client_ip(request))
        session_id = body.session_id or uuid.uuid4().hex
        result = engine().ask(session_id, body.message)
        return ChatResponse(
            session_id=session_id, answer=result.answer, grounded=result.grounded, mode=result.mode,
            standalone_question=result.standalone_question, latency_ms=result.latency_ms,
            sources=[SourceOut(**{k: getattr(s, k) for k in SourceOut.model_fields}) for s in result.sources],
        )

    @app.post("/api/chat/stream")
    def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
        """Server-sent events: `meta` (sources), many `token`s, then `done` (citations)."""
        limiter.check(client_ip(request))
        session_id = body.session_id or uuid.uuid4().hex

        def events() -> Iterator[str]:
            try:
                for event in engine().ask_stream(session_id, body.message):
                    if event["type"] == "meta":
                        event["session_id"] = session_id
                    yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception:  # never leave the client hanging without a final event
                log.exception("Streaming failed")
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'detail': 'Something went wrong.'})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.delete("/api/chat/{session_id}")
    def reset_chat(session_id: str) -> dict:
        engine().memory.reset(session_id)
        return {"reset": session_id}

    @app.post("/api/feedback")
    def feedback(body: FeedbackRequest, request: Request) -> dict:
        limiter.check(client_ip(request))
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **body.model_dump()}
        settings.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with feedback_lock, settings.feedback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"saved": True}

    # ----------------------------------------------------------------- UI

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
