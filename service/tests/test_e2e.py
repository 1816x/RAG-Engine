"""End-to-end test of the RAG service via FastAPI's in-process TestClient.

Uploads the sample corpus, runs canned questions, and asserts the retrieval
layer returns the right sources. Runs keyless (mock generation mode), so it
works in CI without an API key or network.
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

from rag_service.app import app

SAMPLE_DOCS = pathlib.Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.fixture(scope="module")
def client():
    c = TestClient(app)
    # Seed the shared in-process store once for the module.
    for f in sorted(SAMPLE_DOCS.glob("*.md")):
        resp = c.post(
            "/documents", json={"title": f.stem.replace("_", " "), "text": f.read_text()}
        )
        assert resp.status_code == 200, resp.text
    return c


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_stats_after_seed(client):
    stats = client.get("/stats").json()
    assert stats["documents"] == 3
    assert stats["chunks"] >= 3
    assert stats["metric"] == "cosine"


def test_documents_listed(client):
    titles = {d["title"] for d in client.get("/documents").json()}
    assert titles == {"hnsw", "brute force", "embeddings"}


@pytest.mark.parametrize(
    "question,expected_doc",
    [
        ("What does HNSW stand for and how does its layered graph work?", "hnsw"),
        ("Why is brute-force search exact but slow?", "brute force"),
        ("What is cosine distance and why is it scale invariant?", "embeddings"),
    ],
)
def test_query_retrieves_relevant_document(client, question, expected_doc):
    resp = client.post("/query", json={"question": question, "k": 3})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["sources"], "expected at least one retrieved source"
    # The expected document should be among the top-k retrieved sources. We
    # don't assert exact rank 1: CI runs the deterministic hashed fallback
    # embedder, which ranks by raw term overlap; a real embedding model would
    # discriminate more sharply.
    retrieved_docs = {s["doc_title"] for s in body["sources"]}
    assert expected_doc in retrieved_docs
    # Scores are cosine similarities, descending.
    scores = [s["score"] for s in body["sources"]]
    assert scores == sorted(scores, reverse=True)
    # Keyless CI runs in mock mode.
    assert body["model"] in ("mock", "claude-sonnet-5")
    assert body["answer"]


def test_unrelated_question_still_responds_cleanly(client):
    # An off-topic query returns whatever is nearest; the contract is only that
    # the endpoint responds cleanly with a well-formed body.
    resp = client.post("/query", json={"question": "how do I prune roses", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and "sources" in body
