from app.chat import CUT_OFF_NOTE, NOT_FOUND_ANSWER, ChatEngine, ConversationStore, Turn, heuristic_rewrite
from app.llm import LLMError


def test_extractive_answer_quotes_and_cites(kb):
    result = ChatEngine(kb).ask("s", "When are the doubt-clearing sessions?")
    assert result.grounded and result.mode == "extractive"
    assert "Tuesday and Thursday" in result.answer and "[1]" in result.answer
    assert result.sources[0].cited and result.sources[0].section == "Doubt-clearing sessions"


def test_out_of_scope_question_is_refused(kb):
    result = ChatEngine(kb).ask("s", "Who won the 2023 cricket world cup?")
    assert not result.grounded
    assert result.answer == NOT_FOUND_ANSWER


def test_heuristic_rewrite_handles_follow_ups():
    turns = [Turn("How much does the crash course cost?", "How much does the crash course cost?", "...")]
    assert heuristic_rewrite("And is it refundable?", turns).startswith("How much does the crash course cost?")
    assert heuristic_rewrite("When are the doubt-clearing sessions held?", turns) == \
        "When are the doubt-clearing sessions held?"


def test_llm_answer_marks_cited_sources(kb, fake_llm):
    llm = fake_llm("You get a 50% refund if you leave between day 8 and day 30 [1].")
    result = ChatEngine(kb, llm=llm).ask("s", "Refund if I leave after three weeks?")
    assert result.grounded and result.mode == "fake"
    assert result.answer.startswith("You get a 50% refund")
    assert result.sources[0].cited
    system, prompt = llm.calls[0]
    assert "NOT_FOUND" in system
    assert "[1] (" in prompt  # numbered context passages are sent


def test_stream_emits_meta_tokens_done_in_order(kb, fake_llm):
    llm = fake_llm("The full course costs **Rs. 38,000** [1].")
    events = list(ChatEngine(kb, llm=llm).ask_stream("s", "How much does the full GATE DA course cost?"))
    assert events[0]["type"] == "meta" and events[0]["sources"]
    assert events[-1]["type"] == "done" and events[-1]["cited"] == [1]
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed == events[-1]["answer"]
    assert sum(e["type"] == "token" for e in events) > 1  # really streamed in pieces


def test_streamed_not_found_is_never_shown(kb, fake_llm):
    llm = fake_llm(["NOT_", "FOUND"])
    events = list(ChatEngine(kb, llm=llm).ask_stream("s", "What is the hostel fee at IIT Madras?"))
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert "NOT_" not in tokens and tokens == NOT_FOUND_ANSWER
    assert events[-1]["grounded"] is False


def test_llm_failure_before_any_token_falls_back_to_extractive(kb, fake_llm):
    result = ChatEngine(kb, llm=fake_llm(LLMError("timeout"))).ask("s", "Can I pay the fee in instalments?")
    assert result.grounded
    assert result.mode == "extractive (LLM unavailable)"
    assert "three instalments" in result.answer
    assert ChatEngine(kb).stats.snapshot()["llm_errors"] == 0


def test_llm_failure_mid_stream_keeps_partial_answer(kb, fake_llm):
    llm = fake_llm(["You can pay in three ", "instalments [1]", LLMError("connection reset")])
    engine = ChatEngine(kb, llm=llm)
    result = engine.ask("s", "Can I pay the fee in instalments?")
    assert result.answer.startswith("You can pay in three instalments [1]")
    assert result.answer.endswith(CUT_OFF_NOTE)
    assert engine.stats.snapshot()["llm_errors"] == 1


def test_llm_rewrites_follow_up_before_searching(kb, fake_llm):
    llm = fake_llm(
        "The crash course costs Rs. 12,500 [1].",
        "Is the crash course fee refundable?",  # rewrite of the follow-up (generate)
        "It is non-refundable once the first class has started [1].",
    )
    engine = ChatEngine(kb, llm=llm)
    engine.ask("s", "How much is the crash course?")
    result = engine.ask("s", "Is it refundable?")
    assert result.standalone_question == "Is the crash course fee refundable?"
    assert any(s.page == 3 for s in result.sources)
    assert "Earlier in this conversation" in llm.calls[-1][1]  # history reaches the answer prompt


def test_reranker_reorders_candidates(kb):
    class PreferFaq:
        name = "fake-reranker"

        def scores(self, query, passages):
            return [1.0 if "laptop" in p.lower() else 0.0 for p in passages]

    engine = ChatEngine(kb, reranker=PreferFaq(), rerank_candidates=15)
    hits = engine.retrieve("Do I need a computer for mock tests?")
    assert len(hits) == engine.top_k
    assert "laptop" in hits[0].chunk.text.lower() and hits[0].score == 1.0


def test_usage_stats_track_refusals(kb):
    engine = ChatEngine(kb)
    engine.ask("s", "When are the doubt-clearing sessions?")
    engine.ask("s", "What is the capital of France?")
    stats = engine.stats.snapshot()
    assert stats["questions"] == 2 and stats["refusal_rate"] == 0.5


def test_memory_is_capped_per_session():
    store = ConversationStore(max_turns=2)
    for i in range(5):
        store.append("a", Turn(f"q{i}", f"q{i}", f"a{i}"))
    assert [t.question for t in store.turns("a")] == ["q3", "q4"]
    store.reset("a")
    assert store.turns("a") == []
