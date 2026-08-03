"""Tests for the hnsw_engine Python bindings.

Mirrors the Rust test suite through the FFI boundary: API surface, error
mapping, and recall against a pure-Python brute-force baseline (seeded, so
assertions are exact and reproducible).
"""

import math
import random

import pytest

from hnsw_engine import Hnsw


def brute_force(vectors, query, k):
    """Exact cosine k-NN, no dependencies."""

    def cos_dist(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - dot / (na * nb)

    hits = sorted(
        ((cos_dist(query, v), i) for i, v in enumerate(vectors)),
        key=lambda t: t[0],
    )
    return [i for _, i in hits[:k]]


def clustered_vectors(rng, n, dim, n_clusters):
    centers = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(n_clusters)]
    return [
        [c + rng.gauss(0, 0.05) for c in centers[rng.randrange(n_clusters)]]
        for _ in range(n)
    ]


def test_constructor_validation():
    with pytest.raises(ValueError, match="metric"):
        Hnsw(dim=4, metric="manhattan")
    with pytest.raises(ValueError):
        Hnsw(dim=0)
    with pytest.raises(ValueError):
        Hnsw(dim=4, m=1)
    with pytest.raises(ValueError):
        Hnsw(dim=4, ef_construction=0)


def test_dimension_mismatch_raises():
    index = Hnsw(dim=4)
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.insert([1.0, 2.0])
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.search([1.0, 2.0], k=1)


def test_non_finite_values_are_rejected():
    index = Hnsw(dim=2)
    with pytest.raises(ValueError, match="non-finite"):
        index.insert([float("nan"), 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        index.search([0.0, float("inf")], k=1)


def test_insert_batch_is_atomic_on_dimension_error():
    index = Hnsw(dim=2)
    with pytest.raises(ValueError, match="batch vector 1.*dimension mismatch"):
        index.insert_batch([[1.0, 0.0], [1.0]])
    assert len(index) == 0


def test_insert_batch_is_atomic_on_non_finite_value():
    index = Hnsw(dim=2)
    with pytest.raises(ValueError, match="batch vector 1.*non-finite"):
        index.insert_batch([[1.0, 0.0], [float("inf"), 1.0]])
    assert len(index) == 0


def test_basic_roundtrip():
    index = Hnsw(dim=3, metric="euclidean")
    ids = index.insert_batch([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    assert ids == [0, 1, 2]
    assert len(index) == 3
    assert index.dim == 3
    assert index.metric == "euclidean"

    hits = index.search([0.9, 0.0, 0.0], k=2)
    assert hits[0][0] == 1
    assert hits[0][1] == pytest.approx(0.1, abs=1e-5)
    assert hits[1][0] == 0

    assert index.vector(1) == pytest.approx([1.0, 0.0, 0.0])
    with pytest.raises(IndexError):
        index.vector(99)


def test_empty_index_search():
    index = Hnsw(dim=4)
    assert index.search([0.0, 0.0, 0.0, 0.0], k=5) == []


def test_cosine_normalizes_stored_vectors():
    index = Hnsw(dim=2, metric="cosine")
    index.insert([3.0, 4.0])
    norm = math.sqrt(sum(x * x for x in index.vector(0)))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_repr():
    index = Hnsw(dim=8)
    assert repr(index) == 'Hnsw(dim=8, metric="cosine", len=0)'


def test_recall_against_brute_force():
    rng = random.Random(2024)
    dim, k = 32, 10
    vectors = clustered_vectors(rng, 2000, dim, 50)
    queries = clustered_vectors(rng, 30, dim, 50)

    index = Hnsw(dim=dim, metric="cosine", seed=7)
    index.insert_batch(vectors)

    total = 0.0
    for q in queries:
        truth = set(brute_force(vectors, q, k))
        got = {i for i, _ in index.search(q, k=k, ef_search=100)}
        total += len(truth & got) / k
    recall = total / len(queries)
    assert recall >= 0.95, f"recall@{k} too low: {recall:.4f}"
