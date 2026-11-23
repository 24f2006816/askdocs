"""Small guards for running AskDocs as a public demo: an admin token and a rate limiter."""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque

from fastapi import HTTPException, Request


def client_ip(request: Request) -> str:
    """First address in X-Forwarded-For (set by proxies such as Hugging Face), else the socket peer."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_admin(request: Request, admin_token: str | None) -> None:
    """When ADMIN_TOKEN is set, require it as `Authorization: Bearer <token>` or `X-Admin-Token`."""
    if not admin_token:
        return
    supplied = request.headers.get("x-admin-token", "")
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, admin_token):
        raise HTTPException(401, "This action needs the admin token.", headers={"WWW-Authenticate": "Bearer"})


class RateLimiter:
    """Sliding-window limit of N requests per 60 seconds per key. limit=0 disables it."""

    def __init__(self, limit_per_minute: int, window: float = 60.0, max_keys: int = 10_000) -> None:
        self.limit, self.window, self.max_keys = limit_per_minute, window, max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > self.max_keys:  # drop idle clients
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window - (now - hits[0])) + 1)
                raise HTTPException(429, "Too many requests. Please wait a moment.",
                                    headers={"Retry-After": str(retry_after)})
            hits.append(now)
