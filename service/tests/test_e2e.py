"""End-to-end tests for the public RAG service boundary."""

import importlib
import os
import pathlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_service.app import (
    MAX_DOCUMENT_CHARS,
    MAX_QUESTION_CHARS,
    MAX_REQUEST_BYTES,
    MAX_TITLE_CHARS,
    _add_cors_middleware,
    _cors_origins,
    app,
)
from rag_service.generation import GenerationTimeoutError

SAMPLE_DOCS = pathlib.Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def client():
    c = TestClient(app)
    previous = os.environ.get("RAG_UPLOADS_ENABLED")
    os.environ["RAG_UPLOADS_ENABLED"] = "1"
    try:
        for file in sorted(SAMPLE_DOCS.glob("*.md")):
            response = c.post(
                "/documents",
                json={
                    "title": file.stem.replace("_", " "),
                    "text": file.read_text(),
                },
            )
            assert response.status_code == 200, response.text
    finally:
        if previous is None:
            os.environ.pop("RAG_UPLOADS_ENABLED", None)
        else:
            os.environ["RAG_UPLOADS_ENABLED"] = previous
    return c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_stats_after_seed(client):
    stats = client.get("/stats").json()
    assert stats["documents"] == 3
    assert stats["chunks"] >= 3
    assert stats["metric"] == "cosine"
    assert stats["min_score"] == pytest.approx(0.09)
    assert stats["dim"] == 384
    assert stats["m"] == 16
    assert stats["ef_construction"] == 200
    assert stats["embedder"] == "HashedEmbedder"
    assert stats["generation"] == "mock"
    assert stats["uploads_enabled"] is False
    assert stats["limits"] == {
        "title_chars": MAX_TITLE_CHARS,
        "document_chars": MAX_DOCUMENT_CHARS,
        "question_chars": MAX_QUESTION_CHARS,
        "request_bytes": MAX_REQUEST_BYTES,
    }


def test_documents_listed(client):
    titles = {document["title"] for document in client.get("/documents").json()}
    assert titles == {"hnsw", "brute force", "embeddings"}


def test_uploads_disabled_do_not_modify_the_index(client):
    before = client.get("/stats").json()
    response = client.post(
        "/documents",
        json={"title": "blocked", "text": "this must not be indexed"},
    )
    after = client.get("/stats").json()

    assert response.status_code == 403
    assert response.json()["detail"] == "document uploads are disabled"
    assert after["documents"] == before["documents"]
    assert after["chunks"] == before["chunks"]


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "x" * (MAX_TITLE_CHARS + 1), "text": "valid"},
        {"title": "valid", "text": "x" * (MAX_DOCUMENT_CHARS + 1)},
        {"title": "valid", "text": "   "},
        {"title": "valid", "text": "valid", "max_words": 0},
        {"title": "valid", "text": "valid", "max_words": 1001},
        {"title": "valid", "text": "valid", "overlap": -1},
        {"title": "valid", "text": "valid", "overlap": 1000},
        {"title": "valid", "text": "valid", "max_words": 40, "overlap": 40},
    ],
)
def test_invalid_document_inputs_return_422_without_mutation(client, monkeypatch, payload):
    monkeypatch.setenv("RAG_UPLOADS_ENABLED", "1")
    before = client.get("/stats").json()
    response = client.post("/documents", json=payload)
    after = client.get("/stats").json()

    assert response.status_code == 422
    assert after["documents"] == before["documents"]
    assert after["chunks"] == before["chunks"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
        {"question": "valid", "ef_search": 0},
        {"question": "valid", "ef_search": 2001},
    ],
)
def test_invalid_query_inputs_return_422(client, payload):
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_oversized_http_body_returns_413_before_json_parsing(client):
    oversized = b'{"question":"' + (b"x" * MAX_REQUEST_BYTES) + b'"}'
    response = client.post(
        "/query",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_claude_timeout_is_exposed_as_504(client, monkeypatch):
    app_module = importlib.import_module("rag_service.app")

    def time_out(_question, _chunks):
        raise GenerationTimeoutError(30)

    monkeypatch.setattr(app_module, "generate_answer", time_out)
    response = client.post(
        "/query",
        json={"question": "How does HNSW search its graph?", "k": 1},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == "Claude generation timed out after 30 seconds"


def test_closed_cors_emits_no_headers(client):
    response = client.options(
        "/query",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_configured_cors_allows_only_the_exact_origin():
    cors_app = FastAPI()

    @cors_app.get("/healthz")
    def cors_healthz():
        return {"status": "ok"}

    _add_cors_middleware(cors_app, ["https://allowed.example"])
    cors_client = TestClient(cors_app)
    preflight_headers = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }

    allowed = cors_client.options(
        "/healthz",
        headers={"Origin": "https://allowed.example", **preflight_headers},
    )
    blocked = cors_client.options(
        "/healthz",
        headers={"Origin": "https://blocked.example", **preflight_headers},
    )

    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"
    assert "access-control-allow-origin" not in blocked.headers


def test_cors_origin_parser_supports_exact_lists_and_explicit_wildcard(monkeypatch):
    monkeypatch.setenv(
        "RAG_CORS_ORIGINS",
        "https://one.example, https://two.example,https://one.example",
    )
    assert _cors_origins() == ["https://one.example", "https://two.example"]

    monkeypatch.setenv("RAG_CORS_ORIGINS", "*")
    assert _cors_origins() == ["*"]

    monkeypatch.setenv("RAG_CORS_ORIGINS", "*,https://one.example")
    with pytest.raises(RuntimeError):
        _cors_origins()


@pytest.mark.parametrize(
    "question,expected_doc",
    [
        ("What does HNSW stand for and how does its layered graph work?", "hnsw"),
        ("Why is brute-force search exact but slow?", "brute force"),
        ("What is cosine distance and why is it scale invariant?", "embeddings"),
    ],
)
def test_query_retrieves_relevant_document(client, question, expected_doc):
    response = client.post("/query", json={"question": question, "k": 3})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["sources"], "expected at least one retrieved source"
    retrieved_docs = {source["doc_title"] for source in body["sources"]}
    assert expected_doc in retrieved_docs
    scores = [source["score"] for source in body["sources"]]
    assert scores == sorted(scores, reverse=True)
    assert body["model"] in ("mock", "claude-sonnet-5")
    assert body["answer"]


def test_unrelated_question_returns_no_sources(client):
    response = client.post(
        "/query", json={"question": "how do I prune roses", "k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["model"] == "mock"
    assert body["answer"] == (
        "I don't have any indexed documents that address that question."
    )


def test_uploads_enabled_continue_indexing(client, monkeypatch):
    monkeypatch.setenv("RAG_UPLOADS_ENABLED", "1")
    before = client.get("/stats").json()
    response = client.post(
        "/documents",
        json={"title": "hardening regression", "text": "bounded public input"},
    )
    after = client.get("/stats").json()

    assert response.status_code == 200, response.text
    assert after["documents"] == before["documents"] + 1
    assert after["chunks"] == before["chunks"] + 1


def test_store_filters_weak_matches_and_keeps_relevant_ones():
    from rag_service.store import DocumentStore

    class OrthogonalEmbedder:
        dim = 2

        def embed(self, texts):
            return [
                [1.0, 0.0] if text in {"known", "known query"} else [0.0, 1.0]
                for text in texts
            ]

    store = DocumentStore(embedder=OrthogonalEmbedder(), min_score=0.15)
    store.add_document("known document", "known", max_words=10, overlap=0)

    assert store.retrieve("unrelated", k=1) == []
    relevant = store.retrieve("known query", k=1)
    assert len(relevant) == 1
    assert relevant[0].doc_title == "known document"
    assert relevant[0].score == pytest.approx(1.0)
