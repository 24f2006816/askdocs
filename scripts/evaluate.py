"""Evaluate retrieval quality and refusal behaviour on a labelled question set.

For each in-scope question we know which document (and page, section or phrase) holds
the answer. We report hit@k (is a correct chunk in the top k?) and MRR (mean reciprocal
rank of the first correct chunk) for dense-only, BM25-only and hybrid retrieval.
Out-of-scope questions check that the bot refuses instead of making something up.

Usage: python -m scripts.evaluate [--questions data/eval/questions.jsonl] [--json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.chat import ChatEngine
from app.config import PROJECT_ROOT, Settings
from app.embeddings import build_embedder
from app.index import KnowledgeBase
from app.ingest import Chunk, chunk_documents, load_directory
from app.rerank import Reranker, build_reranker, rerank

MODES = ("dense", "bm25", "hybrid")
KS = (1, 3, 5)


def is_relevant(chunk: Chunk, item: dict) -> bool:
    if chunk.source != item["source"]:
        return False
    if "page" in item and chunk.page != item["page"]:
        return False
    if "section" in item and chunk.section != item["section"]:
        return False
    return "contains" not in item or item["contains"].lower() in chunk.text.lower()


def retrieval_metrics(kb: KnowledgeBase, items: list[dict], mode: str,
                      reranker: Reranker | None = None, candidates: int = 20) -> dict:
    hits = {k: 0 for k in KS}
    reciprocal_ranks = []
    for item in items:
        if mode == "hybrid+rerank":
            results = rerank(reranker, item["question"], kb.search(item["question"], k=candidates), max(KS))
        else:
            results = kb.search(item["question"], k=max(KS), mode=mode)
        rank = next((i for i, r in enumerate(results, start=1) if is_relevant(r.chunk, item)), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        for k in KS:
            hits[k] += int(rank is not None and rank <= k)
    n = len(items)
    return {**{f"hit@{k}": hits[k] / n for k in KS}, "mrr": sum(reciprocal_ranks) / n}


def refusal_metrics(engine: ChatEngine, in_scope: list[dict], out_of_scope: list[dict]) -> dict:
    def grounded(question: str) -> bool:
        return engine.is_grounded(question, engine.kb.search(question, k=engine.top_k))

    correct_refusals = sum(not grounded(i["question"]) for i in out_of_scope)
    false_refusals = sum(not grounded(i["question"]) for i in in_scope)
    return {
        "out_of_scope_refused": correct_refusals / len(out_of_scope) if out_of_scope else None,
        "in_scope_wrongly_refused": false_refusals / len(in_scope) if in_scope else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data" / "eval" / "questions.jsonl")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    parser.add_argument("--sweep", action="store_true", help="also print refusal trade-offs across thresholds")
    args = parser.parse_args()

    settings = Settings.from_env()
    items = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    in_scope = [i for i in items if not i.get("out_of_scope")]
    out_of_scope = [i for i in items if i.get("out_of_scope")]

    embedder = build_embedder(settings.embedder, settings.embedding_model)
    chunks = chunk_documents(load_directory(settings.docs_dir), settings.chunk_size, settings.chunk_overlap)
    start = time.perf_counter()
    kb = KnowledgeBase.build(embedder, chunks)
    build_seconds = time.perf_counter() - start

    reranker = build_reranker(settings.reranker_model)
    modes = MODES + (("hybrid+rerank",) if reranker else ())
    start = time.perf_counter()
    results = {mode: retrieval_metrics(kb, in_scope, mode) for mode in MODES}
    per_query_ms = (time.perf_counter() - start) * 1000 / (len(in_scope) * len(MODES))
    if reranker:
        results["hybrid+rerank"] = retrieval_metrics(kb, in_scope, "hybrid+rerank", reranker,
                                                     settings.rerank_candidates)
    engine = ChatEngine(kb, llm=None, top_k=settings.top_k, min_similarity=settings.min_similarity,
                        min_keyword_coverage=settings.min_keyword_coverage)
    refusal = refusal_metrics(engine, in_scope, out_of_scope)

    report = {"embedder": embedder.name, "reranker": reranker.name if reranker else None,
              "chunks": len(chunks), "questions": len(in_scope), "out_of_scope_questions": len(out_of_scope),
              "index_build_seconds": round(build_seconds, 3), "avg_search_ms": round(per_query_ms, 2),
              "retrieval": results, "refusal": refusal}
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Embedder: {embedder.name} | reranker: {reranker.name if reranker else 'off'} | "
          f"{len(chunks)} chunks | {len(in_scope)} questions | {len(out_of_scope)} out-of-scope\n")
    print("| Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR |")
    print("| --- | --- | --- | --- | --- |")
    for mode in modes:
        r = results[mode]
        print(f"| {mode} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} | {r['mrr']:.3f} |")
    print(f"\nOut-of-scope questions refused: {refusal['out_of_scope_refused']:.0%}")
    print(f"In-scope questions wrongly refused: {refusal['in_scope_wrongly_refused']:.0%}")
    print(f"Average search latency: {per_query_ms:.1f} ms")

    if args.sweep and out_of_scope:
        print("\nRefusal trade-off (choose MIN_SIMILARITY / MIN_KEYWORD_COVERAGE for your embedder):\n")
        print("| MIN_SIMILARITY | MIN_KEYWORD_COVERAGE | Out-of-scope refused | In-scope wrongly refused |")
        print("| --- | --- | --- | --- |")
        for sim in (0.2, 0.3, 0.4, 0.5):
            for cov in (0.2, 0.3, 0.4, 0.5):
                engine.min_similarity, engine.min_keyword_coverage = sim, cov
                r = refusal_metrics(engine, in_scope, out_of_scope)
                refused, wrong = r["out_of_scope_refused"], r["in_scope_wrongly_refused"]
                print(f"| {sim:.1f} | {cov:.1f} | {refused:.0%} | {wrong:.0%} |")


if __name__ == "__main__":
    main()
