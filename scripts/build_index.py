"""Build (or rebuild) the search index from the documents folder.

Usage: python -m scripts.build_index [--force]
"""

import argparse
import logging

from app.config import Settings
from app.embeddings import build_embedder
from app.service import build_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if the index is up to date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings.from_env()
    embedder = build_embedder(settings.embedder, settings.embedding_model)
    kb = build_knowledge_base(settings, embedder, force=args.force)
    for doc in kb.sources():
        print(f"  {doc['source']}: {doc['chunks']} chunks")
    print(f"Index ready in {settings.index_dir} ({len(kb.chunks)} chunks, embedder={embedder.name})")


if __name__ == "__main__":
    main()
