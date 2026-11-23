"""Hugging Face Space entry point for AskDocs.

The Space holds only this file, requirements.txt and README.md. At startup it downloads the latest
AskDocs code from GitHub, then serves the FastAPI app on port 7860. Restarting the Space picks up
new commits, and no Hugging Face token has to be stored anywhere.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = os.getenv("ASKDOCS_REPO", "24f2006816/askdocs")
BRANCH = os.getenv("ASKDOCS_BRANCH", "main")
SOURCE_DIR = Path(__file__).resolve().parent / "askdocs-src"


def download_source() -> None:
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/{BRANCH}"
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = []
        for member in tar.getmembers():
            _, _, relative = member.name.partition("/")  # drop the "askdocs-main/" prefix
            if relative:
                member.name = relative
                members.append(member)
        try:
            tar.extractall(SOURCE_DIR, members=members, filter="data")
        except TypeError:  # Python without tarfile extraction filters
            tar.extractall(SOURCE_DIR, members=members)


def main() -> None:
    download_source()
    # Public demo defaults; Space variables and secrets (e.g. GEMINI_API_KEY) still take precedence.
    os.environ.setdefault("ALLOW_UPLOADS", "false")
    os.environ.setdefault("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Make the downloaded "app" package win over this file, which is also called app.py.
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in ("", here)]
    sys.path.insert(0, str(SOURCE_DIR))
    os.chdir(SOURCE_DIR)

    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")),
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
