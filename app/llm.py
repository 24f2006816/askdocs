"""LLM providers, called over plain HTTPS (no vendor SDKs to pin or break).

Each request is one system prompt plus one user message. The chat engine folds the
conversation history into that message, which keeps providers interchangeable and
avoids provider-specific rules about replaying model turns. Sampling parameters are
not sent: newer models ignore or reject them.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Protocol

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> Iterator[str]: ...


def _sse_data(response: httpx.Response) -> Iterator[dict]:
    """Yield the JSON payload of every `data:` line in a server-sent event stream."""
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        yield json.loads(data)


class GeminiLLM:
    BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}"

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.name = f"gemini:{model}"

    def _payload(self, system: str, prompt: str) -> dict:
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        }

    @staticmethod
    def _text(body: dict) -> str:
        parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        # Skip internal "thought" parts if the model returns any.
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = httpx.post(f"{self.BASE.format(model=self.model)}:generateContent",
                                  headers={"x-goog-api-key": self.api_key},
                                  json=self._payload(system, prompt), timeout=self.timeout)
            response.raise_for_status()
            return self._text(response.json()).strip()
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        try:
            with httpx.stream("POST", f"{self.BASE.format(model=self.model)}:streamGenerateContent",
                              params={"alt": "sse"}, headers={"x-goog-api-key": self.api_key},
                              json=self._payload(system, prompt), timeout=self.timeout) as response:
                response.raise_for_status()
                for body in _sse_data(response):
                    text = self._text(body)
                    if text:
                        yield text
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"Gemini stream failed: {exc}") from exc


class OpenAILLM:
    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.name = f"openai:{model}"

    def _payload(self, system: str, prompt: str, stream: bool) -> dict:
        return {"model": self.model, "stream": stream,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = httpx.post(self.URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                  json=self._payload(system, prompt, stream=False), timeout=self.timeout)
            response.raise_for_status()
            return (response.json()["choices"][0]["message"]["content"] or "").strip()
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        try:
            with httpx.stream("POST", self.URL, headers={"Authorization": f"Bearer {self.api_key}"},
                              json=self._payload(system, prompt, stream=True), timeout=self.timeout) as response:
                response.raise_for_status()
                for body in _sse_data(response):
                    choices = body.get("choices") or [{}]
                    text = (choices[0].get("delta") or {}).get("content")
                    if text:
                        yield text
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"OpenAI stream failed: {exc}") from exc


def build_llm(settings: Settings) -> LLM | None:
    """Return a configured LLM, or None to run in extractive (no-LLM) mode."""
    provider = settings.llm_provider
    if provider == "extractive":
        return None
    if provider in {"gemini", "auto"} and settings.gemini_api_key:
        return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    if provider in {"openai", "auto"} and settings.openai_api_key:
        return OpenAILLM(settings.openai_api_key, settings.openai_model)
    if provider in {"gemini", "openai"}:
        raise ValueError(f"LLM_PROVIDER={provider} but its API key is not set")
    return None
