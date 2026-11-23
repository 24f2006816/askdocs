from app.index import BM25, KnowledgeBase, docs_fingerprint, reciprocal_rank_fusion, tokenize


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("What are the refunds for classes?") == ["refund", "class"]


def test_bm25_prefers_documents_with_rare_query_terms():
    corpus = [tokenize(t) for t in ["fees and payment", "refund policy and refund timeline", "mock test schedule"]]
    scores = BM25(corpus).scores(tokenize("refund"))
    assert scores.argmax() == 1
    assert scores[2] == 0


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert max(fused, key=fused.get) in {1, 2}
    assert fused[2] > fused[3] and fused[1] > fused[4]


def test_hybrid_search_finds_the_refund_page(kb):
    hits = kb.search("Can I get my money back if I quit after three weeks?", k=3)
    assert any(h.chunk.source == "fee-and-refund-policy.pdf" and h.chunk.page == 3 for h in hits)


def test_keyword_search_finds_batch_code(kb):
    top = kb.search("CR-27", k=1, mode="bm25")[0]
    assert top.chunk.section == "Batches and timings"


def test_index_round_trip_and_staleness(settings, kb, embedder):
    fingerprint = docs_fingerprint(settings.docs_dir)
    loaded = KnowledgeBase.load(settings.index_dir, embedder, fingerprint=fingerprint)
    assert loaded is not None and len(loaded.chunks) == len(kb.chunks)

    (settings.docs_dir / "new.txt").write_text("A brand new document.", encoding="utf-8")
    assert KnowledgeBase.load(settings.index_dir, embedder, fingerprint=docs_fingerprint(settings.docs_dir)) is None
