import pytest

from hnsw_rag.embeddings import HashedEmbedder
from rag_service.store import DocumentNotFoundError, DocumentStore


class CountingEmbedder:
    def __init__(self, dim=32):
        self.inner = HashedEmbedder(dim=dim)
        self.dim = dim
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return self.inner.embed(texts)


def assert_dense(store):
    assert len(store._index) == len(store._chunks)
    assert set(store._chunks) == set(range(len(store._index)))
    assert all(chunk.doc_id in store._documents for chunk in store._chunks.values())
    for document in store.documents():
        assert document.n_chunks == sum(
            chunk.doc_id == document.id for chunk in store._chunks.values()
        )


def test_delete_rebuilds_dense_ids_without_reembedding_and_keeps_doc_ids_monotonic():
    embedder = CountingEmbedder()
    store = DocumentStore(embedder, min_score=None)
    alpha = store.add_document("A", "alpha-specific content")
    beta = store.add_document("B", "beta-specific content")
    embedder.calls.clear()

    assert store.delete_document(alpha.id) == alpha
    assert embedder.calls == []
    assert store.documents() == [beta]
    assert all(hit.doc_id == beta.id for hit in store.retrieve("alpha", k=5))
    assert store.retrieve("beta-specific", k=1)[0].doc_id == beta.id
    assert_dense(store)
    assert store.add_document("C", "gamma-specific content").id == 2


def test_replace_preserves_document_identity_and_only_embeds_new_chunks():
    embedder = CountingEmbedder()
    store = DocumentStore(embedder, min_score=None)
    survivor = store.add_document("survivor", "Neptune blue planet")
    original = store.add_document("Saturn", "Saturn has distinctive rings")
    embedder.calls.clear()

    replaced = store.replace_document(
        original.id, "Jupiter", "Jupiter has a giant red storm", max_words=3, overlap=0
    )

    assert replaced.id == original.id
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == ["Jupiter has a", "giant red storm"]
    chunks = sorted(
        (chunk for chunk in store._chunks.values() if chunk.doc_id == original.id),
        key=lambda chunk: chunk.ordinal,
    )
    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert all("Saturn" not in chunk.text for chunk in store._chunks.values())
    assert store.retrieve("Jupiter storm", k=1)[0].doc_id == original.id
    assert any(chunk.doc_id == survivor.id for chunk in store._chunks.values())
    assert_dense(store)


def test_unknown_lifecycle_targets_are_clear_errors():
    store = DocumentStore(HashedEmbedder(dim=8))
    with pytest.raises(DocumentNotFoundError, match="document 9 was not found"):
        store.delete_document(9)
    with pytest.raises(DocumentNotFoundError, match="document 9 was not found"):
        store.replace_document(9, "missing", "text")


def test_delete_last_document_persists_empty_corpus_and_allows_next_insert(tmp_path):
    path = tmp_path / "state"
    store = DocumentStore(HashedEmbedder(dim=16), state_path=path, min_score=None)
    first = store.add_document("only", "only content")
    store.delete_document(first.id)
    assert store.documents() == []
    assert store.stats()["chunks"] == 0
    assert store.retrieve("only") == []

    reloaded = DocumentStore(HashedEmbedder(dim=16), state_path=path, min_score=None)
    assert reloaded.documents() == []
    assert reloaded.add_document("next", "new content").id == 1


def test_delete_and_replace_survive_restart(tmp_path):
    path = tmp_path / "state"
    store = DocumentStore(HashedEmbedder(dim=16), state_path=path, min_score=None)
    removed = store.add_document("A", "alpha unique")
    kept = store.add_document("B", "beta unique")
    store.delete_document(removed.id)
    store.replace_document(kept.id, "B2", "jupiter unique")

    reloaded = DocumentStore(HashedEmbedder(dim=16), state_path=path, min_score=None)
    assert [(doc.id, doc.title) for doc in reloaded.documents()] == [(kept.id, "B2")]
    assert reloaded.retrieve("jupiter", k=1)[0].doc_id == kept.id
    assert all("alpha" not in chunk.text for chunk in reloaded._chunks.values())
    assert_dense(reloaded)


@pytest.mark.parametrize("operation", ["delete", "replace"])
def test_failed_lifecycle_persistence_leaves_live_and_disk_state_unchanged(
    tmp_path, monkeypatch, operation
):
    path = tmp_path / "state"
    store = DocumentStore(HashedEmbedder(dim=16), state_path=path, min_score=None)
    original = store.add_document("original", "saturn original content")
    store.add_document("survivor", "neptune survivor content")
    before_bytes = path.read_bytes()
    before_documents = store.documents()
    before_chunks = dict(store._chunks)
    monkeypatch.setattr(
        store, "_persist", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )

    with pytest.raises(OSError, match="disk full"):
        if operation == "delete":
            store.delete_document(original.id)
        else:
            store.replace_document(original.id, "replacement", "jupiter replacement")

    assert store.documents() == before_documents
    assert store._chunks == before_chunks
    assert store.retrieve("saturn", k=1)[0].doc_id == original.id
    assert path.read_bytes() == before_bytes
    reloaded = DocumentStore(HashedEmbedder(dim=16), state_path=path, min_score=None)
    assert reloaded.documents() == before_documents
