import pytest

from hnsw_rag.embeddings import HashedEmbedder
from rag_service.store import DocumentStore, STATE_VERSION


def test_complete_store_survives_restart(tmp_path):
    path = tmp_path / "corpus.state"
    first = DocumentStore(HashedEmbedder(dim=32), state_path=path, min_score=None)
    doc = first.add_document("durable title", "persistent graph metadata provenance")
    before = first.retrieve("persistent metadata", k=1)

    second = DocumentStore(HashedEmbedder(dim=32), state_path=path, min_score=None)
    after = second.retrieve("persistent metadata", k=1)
    assert second.documents() == [doc]
    assert [(x.id, x.text, x.doc_id, x.doc_title, x.ordinal) for x in after] == [
        (x.id, x.text, x.doc_id, x.doc_title, x.ordinal) for x in before
    ]
    assert second.stats()["persistence"] == {
        "enabled": True,
        "loaded": True,
        "format_version": STATE_VERSION,
    }
    assert second.add_document("next", "another durable chunk").id == 1


def test_corrupt_and_incompatible_state_rejected(tmp_path):
    path = tmp_path / "corpus.state"
    DocumentStore(HashedEmbedder(dim=16), state_path=path).add_document(
        "one", "content"
    )
    with pytest.raises(RuntimeError, match="incompatible"):
        DocumentStore(HashedEmbedder(dim=8), state_path=path)
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="checksum"):
        DocumentStore(HashedEmbedder(dim=16), state_path=path)


def test_failed_commit_does_not_publish_staged_mutation(tmp_path, monkeypatch):
    store = DocumentStore(HashedEmbedder(dim=8), state_path=tmp_path / "state")
    store.add_document("first", "one")
    before = store.stats()
    monkeypatch.setattr(
        store, "_persist", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        store.add_document("second", "two")
    assert store.stats()["documents"] == before["documents"]
    assert store.stats()["chunks"] == before["chunks"]
