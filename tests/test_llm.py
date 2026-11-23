import contextlib
import dataclasses

import httpx
import pytest

from app import llm as llm_module
from app.llm import GeminiLLM, LLMError, OpenAILLM, build_llm


def fake_post(status: int, body: dict, captured: dict):
    def _post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))
    return _post


def test_gemini_request_and_parsing(monkeypatch):
    captured: dict = {}
    body = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True},
                                                  {"text": "Refunds take 14 days [1]."}]}}]}
    monkeypatch.setattr(llm_module.httpx, "post", fake_post(200, body, captured))
    answer = GeminiLLM("key-123", "gemini-test").generate("system rules", "the prompt")
    assert answer == "Refunds take 14 days [1]."
    assert captured["url"].endswith("/models/gemini-test:generateContent")
    assert captured["headers"] == {"x-goog-api-key": "key-123"}
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == "system rules"
    assert "generationConfig" not in captured["json"]


def test_openai_request_and_parsing(monkeypatch):
    captured: dict = {}
    body = {"choices": [{"message": {"content": " An answer [2]. "}}]}
    monkeypatch.setattr(llm_module.httpx, "post", fake_post(200, body, captured))
    assert OpenAILLM("sk-test", "model-x").generate("sys", "prompt") == "An answer [2]."
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def fake_stream(status: int, sse_text: str, captured: dict):
    @contextlib.contextmanager
    def _stream(method, url, params=None, headers=None, json=None, timeout=None):
        captured.update(method=method, url=url, params=params, json=json)
        yield httpx.Response(status, content=sse_text.encode(), request=httpx.Request(method, url))
    return _stream


def test_gemini_streaming(monkeypatch):
    captured: dict = {}
    sse = "".join(f'data: {{"candidates": [{{"content": {{"parts": [{{"text": "{t}"}}]}}}}]}}\n\n'
                  for t in ["Refunds ", "take 14 days ", "[1]."])
    monkeypatch.setattr(llm_module.httpx, "stream", fake_stream(200, sse, captured))
    pieces = list(GeminiLLM("k", "gemini-test").stream("s", "p"))
    assert pieces == ["Refunds ", "take 14 days ", "[1]."]
    assert captured["url"].endswith(":streamGenerateContent") and captured["params"] == {"alt": "sse"}


def test_openai_streaming(monkeypatch):
    captured: dict = {}
    sse = ('data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
           'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
           'data: {"choices": [{"delta": {"content": " world [1]"}}]}\n\n'
           "data: [DONE]\n\n")
    monkeypatch.setattr(llm_module.httpx, "stream", fake_stream(200, sse, captured))
    assert list(OpenAILLM("sk", "m").stream("s", "p")) == ["Hello", " world [1]"]
    assert captured["json"]["stream"] is True


def test_stream_errors_become_llm_errors(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "stream", fake_stream(500, "", {}))
    with pytest.raises(LLMError):
        list(OpenAILLM("sk", "m").stream("s", "p"))


def test_http_errors_become_llm_errors(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "post", fake_post(429, {"error": "rate limited"}, {}))
    with pytest.raises(LLMError):
        GeminiLLM("k", "m").generate("s", "p")


def test_build_llm_picks_provider(settings):
    assert build_llm(settings) is None  # extractive
    auto = dataclasses.replace(settings, llm_provider="auto", gemini_api_key=None, openai_api_key="sk")
    assert isinstance(build_llm(auto), OpenAILLM)
    with pytest.raises(ValueError):
        build_llm(dataclasses.replace(settings, llm_provider="gemini", gemini_api_key=None))
