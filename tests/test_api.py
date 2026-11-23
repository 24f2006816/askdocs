import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from app.api import create_app


@pytest.fixture
def client(settings, embedder):
    with TestClient(create_app(settings, embedder=embedder)) as test_client:
        yield test_client


def make_client(settings, embedder, **overrides):
    return TestClient(create_app(dataclasses.replace(settings, **overrides), embedder=embedder))


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        data = next(line[6:] for line in block.splitlines() if line.startswith("data: "))
        events.append(json.loads(data))
    return events


def test_health_and_documents(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["chunks"] > 0 and health["answer_mode"] == "extractive"
    assert health["uploads_locked"] is False and health["reranker"] is None
    names = {d["source"] for d in client.get("/api/documents").json()}
    assert {"faq.txt", "fee-and-refund-policy.pdf", "student-handbook.md"} <= names


def test_chat_returns_cited_answer_and_session(client):
    response = client.post("/api/chat", json={"message": "Can I pay the fee in instalments?"})
    data = response.json()
    assert data["grounded"] and data["session_id"] and data["latency_ms"] >= 0
    assert any(s["cited"] and s["page"] == 2 for s in data["sources"])
    assert response.headers["x-request-id"]

    follow_up = client.post("/api/chat", json={"message": "And the late fee?", "session_id": data["session_id"]})
    assert follow_up.json()["standalone_question"].startswith("Can I pay the fee in instalments?")
    assert client.delete(f"/api/chat/{data['session_id']}").status_code == 200
    assert client.get("/api/health").json()["stats"]["questions"] == 2


def test_streaming_endpoint(client):
    response = client.post("/api/chat/stream", json={"message": "When are the doubt-clearing sessions?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[0]["type"] == "meta" and events[0]["session_id"]
    assert events[-1]["type"] == "done" and events[-1]["grounded"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == events[-1]["answer"]


def test_chat_validates_input(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 2001}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "session_id": "../../etc"}).status_code == 422


def test_upload_then_ask_then_delete(client):
    doc = b"# Library\n\nThe Northstar library is open on Sundays from 10 AM to 4 PM for enrolled students."
    response = client.post("/api/documents", files={"files": ("library.md", doc, "text/markdown")})
    assert response.status_code == 200 and response.json()["added"] == ["library.md"]

    data = client.post("/api/chat", json={"message": "When is the library open?"}).json()
    assert data["grounded"] and data["sources"][0]["source"] == "library.md"

    assert client.delete("/api/documents/library.md").status_code == 200
    assert "library.md" not in {d["source"] for d in client.get("/api/documents").json()}


def test_upload_rejects_bad_files(client):
    bad_type = client.post("/api/documents", files={"files": ("run.exe", b"MZ", "application/octet-stream")})
    assert bad_type.status_code == 400
    too_big = client.post("/api/documents", files={"files": ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain")})
    assert too_big.status_code == 413
    corrupt = client.post("/api/documents", files={"files": ("broken.pdf", b"%PDF-1.4 nope", "application/pdf")})
    assert corrupt.status_code == 400
    traversal = client.post("/api/documents", files={"files": ("../../evil.md", b"# hi\n\ntext", "text/markdown")})
    assert traversal.status_code == 200 and traversal.json()["added"] == ["evil.md"]


def test_admin_token_locks_uploads_and_deletes(settings, embedder):
    with make_client(settings, embedder, admin_token="s3cret") as client:
        assert client.get("/api/health").json()["uploads_locked"] is True
        files = {"files": ("note.txt", b"Office hours are on Mondays.", "text/plain")}
        assert client.post("/api/documents", files=files).status_code == 401
        assert client.post("/api/documents", files=files, headers={"X-Admin-Token": "wrong"}).status_code == 401
        ok = client.post("/api/documents", files=files, headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        assert client.delete("/api/documents/note.txt").status_code == 401
        assert client.delete("/api/documents/note.txt", headers={"X-Admin-Token": "s3cret"}).status_code == 200
        assert client.post("/api/chat", json={"message": "When are the doubt-clearing sessions?"}).status_code == 200


def test_rate_limit(settings, embedder):
    with make_client(settings, embedder, rate_limit_per_minute=2) as client:
        body = {"message": "When are the doubt-clearing sessions?"}
        assert client.post("/api/chat", json=body).status_code == 200
        assert client.post("/api/chat/stream", json=body).status_code == 200
        limited = client.post("/api/chat", json=body)
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
        # A different client (by X-Forwarded-For) has its own budget.
        assert client.post("/api/chat", json=body, headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 200


def test_feedback_is_logged(client, settings):
    body = {"session_id": "abc", "question": "q?", "answer": "a.", "rating": "down", "comment": "wrong page"}
    assert client.post("/api/feedback", json=body).json() == {"saved": True}
    assert client.post("/api/feedback", json={**body, "rating": "meh"}).status_code == 422
    record = json.loads(settings.feedback_path.read_text().strip())
    assert record["rating"] == "down" and record["comment"] == "wrong page" and "time" in record


def test_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "AskDocs" in page.text
    assert page.headers["x-content-type-options"] == "nosniff"
