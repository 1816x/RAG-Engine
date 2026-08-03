"""Tests for the hnsw_rag helper package (chunking + embeddings) and a small
end-to-end retrieval check that wires helpers to the Rust index."""

import math

import pytest

from hnsw_engine import Hnsw
from hnsw_rag import HashedEmbedder, chunk_text, get_embedder


def test_chunk_short_text_single_chunk():
    chunks = chunk_text("one two three", max_words=180, source="doc")
    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert chunks[0].index == 0
    assert chunks[0].source == "doc"


def test_chunk_respects_word_budget_and_overlaps():
    words = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(words, max_words=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 100
    # Consecutive chunks share the overlap tail.
    first_tail = chunks[0].text.split()[-20:]
    second_head = chunks[1].text.split()[:20]
    assert first_tail == second_head
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_splits_on_paragraphs():
    text = "alpha beta\n\ngamma delta\n\nepsilon"
    chunks = chunk_text(text, max_words=4, overlap=0)
    # Each paragraph is under budget but together they exceed it, so they land
    # in separate chunks rather than being merged past the limit.
    assert len(chunks) >= 2


def test_chunk_overlap_validation():
    with pytest.raises(ValueError):
        chunk_text("x y z", max_words=10, overlap=10)
    with pytest.raises(ValueError, match="greater than zero"):
        chunk_text("x y z", max_words=0, overlap=0)
    with pytest.raises(ValueError, match="non-negative"):
        chunk_text("x y z", max_words=10, overlap=-1)


def test_paragraph_overlap_never_exceeds_word_budget():
    first = " ".join(f"first{i}" for i in range(100))
    second = " ".join(f"second{i}" for i in range(100))
    chunks = chunk_text(f"{first}\n\n{second}", max_words=100, overlap=20)
    assert [len(chunk.text.split()) for chunk in chunks] == [100, 100]


def test_long_paragraph_has_no_redundant_final_window():
    words = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(words, max_words=100, overlap=20)
    assert len(chunks) == 6
    assert chunks[-1].text.split()[0] == "w400"
    assert chunks[-1].text.split()[-1] == "w499"


def test_hashed_embedder_is_deterministic_and_normalized():
    emb = HashedEmbedder(dim=64)
    a = emb.embed(["the quick brown fox"])[0]
    b = emb.embed(["the quick brown fox"])[0]
    assert a == b
    assert len(a) == 64
    assert math.sqrt(sum(x * x for x in a)) == pytest.approx(1.0, abs=1e-6)


def test_hashed_embedder_reflects_term_overlap():
    emb = HashedEmbedder(dim=256)

    def cos(u, v):
        return sum(x * y for x, y in zip(u, v))

    shared, related, unrelated = emb.embed(
        [
            "vector search with hnsw graphs",
            "hnsw graphs power vector search engines",
            "the weather in spring is mild",
        ]
    )
    assert cos(shared, related) > cos(shared, unrelated)


def test_get_embedder_hashed_backend():
    emb = get_embedder("hashed", dim=128)
    assert isinstance(emb, HashedEmbedder)
    assert emb.dim == 128


def test_get_embedder_rejects_unknown():
    with pytest.raises(ValueError):
        get_embedder("nonsense")


def test_end_to_end_retrieval_with_helpers():
    """Chunk a tiny corpus, embed with the deterministic backend, index it,
    and confirm the most relevant chunk is retrieved for a query."""
    doc = (
        "HNSW builds a layered proximity graph for approximate nearest "
        "neighbor search.\n\n"
        "Brute force search compares the query against every vector and is "
        "exact but slow.\n\n"
        "Cosine distance measures the angle between two vectors regardless "
        "of their magnitude."
    )
    chunks = chunk_text(doc, max_words=30, overlap=5, source="notes")
    emb = get_embedder("hashed", dim=256)
    vectors = emb.embed([c.text for c in chunks])

    index = Hnsw(dim=emb.dim, metric="cosine", seed=1)
    index.insert_batch(vectors)

    qvec = emb.embed(["what is brute force search"])[0]
    hits = index.search(qvec, k=1)
    assert chunks[hits[0][0]].source == "notes"
    assert "brute force" in chunks[hits[0][0]].text.lower()
