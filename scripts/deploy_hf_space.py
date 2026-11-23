"""Deploy this repository to a Hugging Face Docker Space (creates the Space if needed).

Environment variables:
  HF_TOKEN        required: a Hugging Face access token with write permission
  HF_SPACE        optional: "username/space-name" (default: "<your username>/askdocs")
  GEMINI_API_KEY, OPENAI_API_KEY, ADMIN_TOKEN
                  optional: copied into the Space as secrets

Usage: python scripts/deploy_hf_space.py
The GitHub Actions workflow runs this on every push to main once HF_TOKEN is set.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "ADMIN_TOKEN")
IGNORE = shutil.ignore_patterns(".git", ".github", ".venv", "venv", "__pycache__", ".pytest_cache",
                                ".ruff_cache", "tests", ".env", "index", "feedback")

# Hugging Face reads the Space configuration from YAML at the top of README.md.
FRONT_MATTER = """---
title: AskDocs
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Chat with your documents, with page-level citations
---

"""


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set.")
    api = HfApi(token=token)
    space = os.getenv("HF_SPACE") or f"{api.whoami()['name']}/askdocs"

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    for key in SECRETS:
        if os.getenv(key):
            api.add_space_secret(space, key, os.environ[key])
            print(f"Set secret {key}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "space"
        shutil.copytree(ROOT, staging, ignore=IGNORE)
        readme = staging / "README.md"
        readme.write_text(FRONT_MATTER + readme.read_text(encoding="utf-8"), encoding="utf-8")
        sha = os.getenv("GITHUB_SHA", "")[:7]
        api.upload_folder(folder_path=str(staging), repo_id=space, repo_type="space",
                          commit_message=f"Deploy from GitHub {sha}".strip())

    print(f"Deployed: https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
