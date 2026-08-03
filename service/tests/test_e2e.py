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
    assert stats["min_score"] == pytest.approx(0.09)


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


def test_unrelated_question_returns_no_sources(client):
    resp = client.post("/query", json={"question": "how do I prune roses", "k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
    assert body["model"] == "mock"
    assert body["answer"] == (
        "I don't have any indexed documents that address that question."
    )


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
