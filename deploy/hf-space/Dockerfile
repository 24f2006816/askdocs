# Hugging Face Space build for AskDocs.
# The Space holds only this file and a README: the app itself is pulled from the public GitHub repo,
# so no Hugging Face token has to be stored on GitHub. Use "Factory rebuild" in the Space settings
# to pick up new commits (or add HF_TOKEN to the GitHub repo secrets for automatic deploys).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    ALLOW_UPLOADS=false \
    PORT=7860

ARG REPO=24f2006816/askdocs
ARG BRANCH=main
WORKDIR /app

# Docker re-downloads this archive on every build and only reuses the cache if it is unchanged,
# so a rebuild always picks up the latest commit.
ADD https://codeload.github.com/${REPO}/tar.gz/refs/heads/${BRANCH} /tmp/src.tar.gz
RUN tar -xzf /tmp/src.tar.gz -C /app --strip-components=1 && rm /tmp/src.tar.gz

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt -r requirements-embeddings.txt

# Download the models and build the index at build time, so the Space starts fast.
RUN python -m scripts.build_index \
 && python -c "import os; from app.rerank import build_reranker; build_reranker(os.environ['RERANKER_MODEL'])"

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
