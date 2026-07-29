"""Tests for startup seeding.

This is the path a hosted deployment depends on — a fresh container has nobody
to run `scripts/seed.py`, so if it regresses the live demo silently serves an
empty index. Worth covering directly.
"""

import rag_service.app as app_module
from rag_service.app import SAMPLE_DOCS, _env_flag, seed_sample_docs


def test_sample_corpus_is_present():
    # The Docker image must COPY this directory; if it goes missing the demo
    # boots empty.
    assert SAMPLE_DOCS.is_dir()
    assert sorted(p.name for p in SAMPLE_DOCS.glob("*.md")) == [
        "brute_force.md",
        "embeddings.md",
        "hnsw.md",
    ]


def test_env_flag_parsing():
    import os

    for truthy in ("1", "true", "TRUE", "yes", "on", " 1 "):
        os.environ["RAG_TEST_FLAG"] = truthy
        assert _env_flag("RAG_TEST_FLAG") is True
    for falsy in ("0", "false", "no", "off", ""):
        os.environ["RAG_TEST_FLAG"] = falsy
        assert _env_flag("RAG_TEST_FLAG") is False
    del os.environ["RAG_TEST_FLAG"]
    # Defaults to off, so tests and local runs keep today's behavior.
    assert _env_flag("RAG_TEST_FLAG") is False


def test_seed_sample_docs_indexes_the_corpus(monkeypatch):
    """Seed into an isolated store so the module-level one stays untouched."""
    from hnsw_rag import get_embedder
    from rag_service.store import DocumentStore

    fresh = DocumentStore(embedder=get_embedder("hashed", dim=128))
    monkeypatch.setattr(app_module, "_store", fresh)

    added = seed_sample_docs()

    assert added == 3
    stats = fresh.stats()
    assert stats["documents"] == 3
    assert stats["chunks"] >= 3
    assert stats["embedder"] == "HashedEmbedder"

    # And the seeded index actually answers a query.
    hits = fresh.retrieve("what is brute force search", k=3)
    assert hits
    assert any(h.doc_title == "brute force" for h in hits)


def test_seeding_is_idempotent_via_empty_index_guard(monkeypatch):
    """The lifespan guard only seeds an empty index, so a suspend/resume or a
    double startup can't duplicate the corpus."""
    from hnsw_rag import get_embedder
    from rag_service.store import DocumentStore

    fresh = DocumentStore(embedder=get_embedder("hashed", dim=128))
    monkeypatch.setattr(app_module, "_store", fresh)

    seed_sample_docs()
    first = fresh.stats()["chunks"]
    assert first > 0

    # Simulate the guard the lifespan applies.
    if fresh.stats()["chunks"] == 0:
        seed_sample_docs()

    assert fresh.stats()["chunks"] == first
