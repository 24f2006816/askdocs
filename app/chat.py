"""The chat engine: memory -> query rewriting -> retrieval -> (rerank) -> grounded answer.

`ask_stream` yields events as the answer is produced, so the API can stream tokens:

    {"type": "meta",  "standalone_question": ..., "sources": [...]}   once, before the answer
    {"type": "token", "text": ...}                                     many
    {"type": "done",  "answer": ..., "grounded": ..., "cited": [...], "sources": [...], ...}

`ask` consumes the same stream and returns a ChatResult, so both paths share one pipeline.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Generator, Iterator
from dataclasses import asdict, dataclass, field

from .index import KnowledgeBase, SearchHit, tokenize
from .ingest import split_units
from .llm import LLM, LLMError
from .rerank import Reranker, rerank

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are AskDocs, an assistant that answers questions using only the numbered \
context passages you are given.

Rules:
- Use only facts stated in the passages. Never use outside knowledge or guess.
- Cite every fact with its passage number in square brackets, for example [1] or [2][3].
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Be concise: one to four sentences, or a short bulleted list when listing several items.
- Use **bold** for key numbers, dates and deadlines."""

REWRITE_PROMPT = """Rewrite the user's latest message as a standalone question that can be \
understood without the conversation. Keep every name, number and detail. Reply with the \
question only. If it is already standalone, return it unchanged."""

NOT_FOUND_ANSWER = ("I couldn't find that in the documents. Try rephrasing, or ask about "
                    "something the documents cover.")
CUT_OFF_NOTE = "\n\n_(The answer was cut off because the language model stopped responding.)_"

_CITATION_RE = re.compile(r"\[(\d+)\]")
_FOLLOW_UP_WORDS = {"it", "its", "that", "this", "those", "these", "they", "them", "their", "there", "same"}
_FOLLOW_UP_STARTS = ("and ", "what about", "how about", "also ", "but ", "then ")
_NOT_FOUND = "NOT_FOUND"

Event = dict


@dataclass
class Turn:
    question: str
    standalone: str
    answer: str


@dataclass
class SourceRef:
    n: int
    chunk_id: str
    source: str
    page: int | None
    section: str | None
    location: str
    snippet: str
    similarity: float
    cited: bool


@dataclass
class ChatResult:
    answer: str
    standalone_question: str
    grounded: bool
    mode: str
    sources: list[SourceRef] = field(default_factory=list)
    latency_ms: float = 0.0


class ConversationStore:
    """In-memory, thread-safe chat history with a cap on turns and on sessions."""

    def __init__(self, max_turns: int = 6, max_sessions: int = 1000) -> None:
        self.max_turns, self.max_sessions = max_turns, max_sessions
        self._sessions: OrderedDict[str, deque[Turn]] = OrderedDict()
        self._lock = threading.Lock()

    def turns(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def append(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, deque(maxlen=self.max_turns))
            history.append(turn)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class UsageStats:
    """Counters for the health endpoint: volume, refusal rate, latency, LLM failures."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.questions = self.refused = self.llm_errors = 0
        self.total_ms = 0.0

    def record(self, grounded: bool, latency_ms: float, llm_error: bool) -> None:
        with self._lock:
            self.questions += 1
            self.refused += int(not grounded)
            self.llm_errors += int(llm_error)
            self.total_ms += latency_ms

    def snapshot(self) -> dict:
        with self._lock:
            n = self.questions
            return {"questions": n, "refusal_rate": round(self.refused / n, 3) if n else 0.0,
                    "avg_latency_ms": round(self.total_ms / n, 1) if n else 0.0, "llm_errors": self.llm_errors}


def heuristic_rewrite(question: str, turns: list[Turn]) -> str:
    """Without an LLM: treat short or pronoun-heavy messages as follow-ups to the last question."""
    if not turns:
        return question
    words = re.findall(r"[a-z']+", question.lower())
    follow_up = (
        len(words) <= 5
        or any(w in _FOLLOW_UP_WORDS for w in words)
        or question.lower().startswith(_FOLLOW_UP_STARTS)
    )
    return f"{turns[-1].standalone} {question}" if follow_up else question


def extractive_answer(question: str, hits: list[SearchHit], max_sentences: int = 3,
                      focus: str | None = None) -> tuple[str, set[int]]:
    """No-LLM mode: quote the sentences from the top passages that best match the question.

    ``focus`` is the user's latest message; its words count double, so a follow-up such as
    "and for the crash course?" pulls out the crash-course sentence.
    """
    query = set(tokenize(question))
    focus_terms = set(tokenize(focus or question))
    scored = []
    for n, hit in enumerate(hits[:3], start=1):
        for position, sentence in enumerate(split_units(hit.chunk.text)):
            if len(sentence) < 60 and sentence[-1] not in ".!?)":
                continue  # a title line, not an answer
            words = set(tokenize(sentence))
            overlap = len(query & words) + len(focus_terms & words)
            if overlap:
                scored.append((overlap, n, position, sentence))
    # Keep sentences that cover at least half as much of the query as the best one,
    # preferring higher-ranked passages.
    best_overlap = max((s[0] for s in scored), default=0)
    candidates = [(o + (3 - n) * 0.25, n, p, s) for o, n, p, s in scored if o * 2 >= best_overlap]
    if not candidates:
        first = split_units(hits[0].chunk.text)[0]
        candidates = [(0.0, 1, 0, first)]
    best = sorted(candidates, key=lambda c: -c[0])[:max_sentences]
    best.sort(key=lambda c: (c[1], c[2]))
    bullet_chars = "-*• "
    lines = [f"- {sentence.lstrip(bullet_chars).strip()} [{n}]" for _, n, _, sentence in best]
    return "Here's what the documents say:\n" + "\n".join(lines), {n for _, n, _, _ in best}


class ChatEngine:
    def __init__(self, kb: KnowledgeBase, llm: LLM | None = None, top_k: int = 4,
                 min_similarity: float = 0.30, min_keyword_coverage: float = 0.30,
                 max_history_turns: int = 6, reranker: Reranker | None = None,
                 rerank_candidates: int = 20) -> None:
        self.kb = kb
        self.llm = llm
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.min_keyword_coverage = min_keyword_coverage
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.memory = ConversationStore(max_turns=max_history_turns)
        self.stats = UsageStats()

    @property
    def mode(self) -> str:
        return self.llm.name if self.llm else "extractive"

    # ------------------------------------------------------------------ public API

    def ask(self, session_id: str, question: str) -> ChatResult:
        done: Event = {}
        for event in self.ask_stream(session_id, question):
            if event["type"] == "done":
                done = event
        return ChatResult(answer=done["answer"], standalone_question=done["standalone_question"],
                          grounded=done["grounded"], mode=done["mode"],
                          sources=[SourceRef(**s) for s in done["sources"]], latency_ms=done["latency_ms"])

    def ask_stream(self, session_id: str, question: str) -> Iterator[Event]:
        started = time.perf_counter()
        question = question.strip()
        turns = self.memory.turns(session_id)
        standalone = self.rewrite(question, turns)
        hits = self.retrieve(standalone)
        grounded = self.is_grounded(standalone, hits)
        yield {"type": "meta", "standalone_question": standalone, "sources": self._sources(hits, set())}

        llm_error = False
        if not grounded:
            answer, cited, mode = NOT_FOUND_ANSWER, set(), self.mode
            yield {"type": "token", "text": answer}
        elif self.llm is None:
            answer, cited = extractive_answer(standalone, hits, focus=question)
            mode = "extractive"
            yield from _word_chunks(answer)
        else:
            answer, cited, grounded, mode, llm_error = yield from self._stream_llm(question, standalone, hits, turns)

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self.memory.append(session_id, Turn(question, standalone, answer))
        self.stats.record(grounded, latency_ms, llm_error)
        yield {"type": "done", "answer": answer, "standalone_question": standalone, "grounded": grounded,
               "mode": mode, "cited": sorted(cited), "sources": self._sources(hits, cited),
               "latency_ms": latency_ms}

    # ------------------------------------------------------------------ pipeline steps

    def rewrite(self, question: str, turns: list[Turn]) -> str:
        if not turns:
            return question
        if self.llm is None:
            return heuristic_rewrite(question, turns)
        prompt = f"Conversation so far:\n{_transcript(turns)}\n\nLatest message: {question}"
        try:
            rewritten = self.llm.generate(REWRITE_PROMPT, prompt)
        except LLMError as exc:
            log.warning("Query rewrite failed, using heuristic: %s", exc)
            return heuristic_rewrite(question, turns)
        rewritten = rewritten.strip().strip('"').splitlines()[0] if rewritten.strip() else ""
        return rewritten or question

    def retrieve(self, query: str) -> list[SearchHit]:
        if self.reranker is None:
            return self.kb.search(query, k=self.top_k)
        candidates = self.kb.search(query, k=max(self.rerank_candidates, self.top_k))
        return rerank(self.reranker, query, candidates, self.top_k)

    def is_grounded(self, question: str, hits: list[SearchHit]) -> bool:
        """Refuse when nothing retrieved is semantically close AND few query words appear."""
        if not hits:
            return False
        if max(h.dense_score for h in hits) >= self.min_similarity:
            return True
        query = set(tokenize(question))
        if not query:
            return False
        coverage = max(len(query & set(tokenize(h.chunk.index_text))) / len(query) for h in hits)
        return coverage >= self.min_keyword_coverage

    def _stream_llm(self, question: str, standalone: str, hits: list[SearchHit],
                    turns: list[Turn]) -> Generator[Event, None, tuple[str, set[int], bool, str, bool]]:
        """Stream the LLM answer, holding back the first few characters to catch NOT_FOUND."""
        context = "\n\n".join(f"[{n}] ({h.chunk.location})\n{h.chunk.text}" for n, h in enumerate(hits, start=1))
        history = f"Earlier in this conversation:\n{_transcript(turns)}\n\n" if turns else ""
        prompt = f"{history}Context passages:\n{context}\n\nQuestion: {standalone}"

        parts: list[str] = []
        released = False  # have we started sending tokens to the client?
        try:
            for delta in self.llm.stream(SYSTEM_PROMPT, prompt):  # type: ignore[union-attr]
                parts.append(delta)
                if released:
                    yield {"type": "token", "text": delta}
                    continue
                held = "".join(parts).lstrip()
                if len(held) < len(_NOT_FOUND) or held.upper().startswith(_NOT_FOUND):
                    continue  # not sure yet, or it is a refusal: keep holding
                released = True
                yield {"type": "token", "text": held}
        except LLMError as exc:
            log.warning("LLM failed: %s", exc)
            if released:
                yield {"type": "token", "text": CUT_OFF_NOTE}
                answer = "".join(parts).strip() + CUT_OFF_NOTE
                return answer, _citations(answer, len(hits)), True, self.mode, True
            answer, cited = extractive_answer(standalone, hits, focus=question)
            yield from _word_chunks(answer)
            return answer, cited, True, "extractive (LLM unavailable)", True

        answer = "".join(parts).strip()
        if not answer or answer.upper().startswith(_NOT_FOUND):
            yield {"type": "token", "text": NOT_FOUND_ANSWER}
            return NOT_FOUND_ANSWER, set(), False, self.mode, False
        if not released:  # a very short answer never passed the hold-back threshold
            yield {"type": "token", "text": answer}
        return answer, _citations(answer, len(hits)), True, self.mode, False

    @staticmethod
    def _sources(hits: list[SearchHit], cited: set[int]) -> list[dict]:
        refs = []
        for n, hit in enumerate(hits, start=1):
            text = hit.chunk.text
            refs.append(asdict(SourceRef(
                n=n, chunk_id=hit.chunk.id, source=hit.chunk.source, page=hit.chunk.page,
                section=hit.chunk.section, location=hit.chunk.location,
                snippet=text if len(text) <= 320 else text[:317].rsplit(" ", 1)[0] + "...",
                similarity=round(hit.dense_score, 3), cited=n in cited,
            )))
        return refs


def _citations(answer: str, n_hits: int) -> set[int]:
    return {int(n) for n in _CITATION_RE.findall(answer) if 1 <= int(n) <= n_hits}


def _word_chunks(text: str, size: int = 4) -> Iterator[Event]:
    """Send a pre-built answer a few words at a time so every mode streams the same way."""
    words = re.split(r"(\s+)", text)
    for i in range(0, len(words), size * 2):
        yield {"type": "token", "text": "".join(words[i:i + size * 2])}


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def _transcript(turns: list[Turn], last: int = 3) -> str:
    return "\n".join(f"User: {t.question}\nAssistant: {_strip_citations(t.answer)}" for t in turns[-last:])
