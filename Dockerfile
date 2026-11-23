FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
COPY requirements.txt requirements-embeddings.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt -r requirements-embeddings.txt

COPY app ./app
COPY scripts ./scripts
COPY data/docs ./data/docs

# Download the embedding and reranker models and build the index now, so the container starts fast.
RUN python -m scripts.build_index \
 && python -c "import os; from app.rerank import build_reranker; build_reranker(os.environ['RERANKER_MODEL'])"

# Hugging Face Spaces runs containers as user 1000; the first user created here gets that id.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/api/health')"

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
