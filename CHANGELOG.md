# Changelog

## 2.0.0

- Streaming answers over server-sent events (`POST /api/chat/stream`) for Gemini and OpenAI, with a hold-back that keeps the `NOT_FOUND` marker from ever reaching users.
- Optional cross-encoder reranking (`RERANKER_MODEL`), enabled by default in the Docker image.
- Word (`.docx`) documents, including headings, lists and tables.
- Public-demo safety: `ADMIN_TOKEN` for uploads and deletes, per-client rate limiting, request IDs, security headers.
- Answer feedback (`POST /api/feedback`), logged to JSONL; usage stats in `/api/health`.
- Graceful degradation: extractive fallback if the LLM fails, partial answer kept if it fails mid-stream.
- New UI: streaming, dark mode, mobile layout, clickable citations, copy and feedback buttons.
- Automatic Hugging Face Spaces deployment from CI; ruff lint in CI; 43 tests.

## 1.0.0

- Hybrid retrieval (FAISS + BM25 with reciprocal rank fusion), cited answers, refusal gate, follow-up rewriting, extractive mode, evaluation script, FastAPI backend, web UI, Docker and CI.
