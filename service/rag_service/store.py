"""In-memory document + chunk store backed by the HNSW index.

Keeps three things in lockstep:
- the HNSW index (chunk vectors, keyed by dense id = insertion order),
- `chunks[id]` -> the chunk's text + provenance,
- `documents` -> per-document metadata.

The index is created lazily on the first insert, because its dimension is
whatever the configured embedder produces. Everything lives in memory; this
is a demo/portfolio service, not a durable database, and it says so.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hnsw_engine import Hnsw
from hnsw_rag import Chunk, chunk_text
from hnsw_rag.embeddings import Embedder


@dataclass
class StoredChunk:
    id: int
    text: str
    doc_id: int
    doc_title: str
    ordinal: int  # position within the source document


@dataclass
class Document:
    id: int
    title: str
    n_chunks: int


@dataclass
class RetrievedChunk:
    id: int
    text: str
    doc_id: int
    doc_title: str
    ordinal: int
    score: float  # cosine similarity in [~-1, 1]; higher = more relevant


@dataclass
class DocumentStore:
    embedder: Embedder
    metric: str = "cosine"
    min_score: Optional[float] = None
    m: int = 16
    ef_construction: int = 200
    seed: int = 0x5EED

    _index: Optional[Hnsw] = None
    _chunks: Dict[int, StoredChunk] = field(default_factory=dict)
    _documents: Dict[int, Document] = field(default_factory=dict)
    _next_doc_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def dim(self) -> int:
        return self.embedder.dim

    def add_document(
        self, title: str, text: str, *, max_words: int = 180, overlap: int = 40
    ) -> Document:
        """Chunk, embed, and index a document. Returns its metadata."""
        chunks: List[Chunk] = chunk_text(
            text, max_words=max_words, overlap=overlap, source=title
        )
        if not chunks:
            with self._lock:
                doc_id = self._next_doc_id
                self._next_doc_id += 1
                doc = Document(id=doc_id, title=title, n_chunks=0)
                self._documents[doc_id] = doc
                return doc

        vectors = self.embedder.embed([c.text for c in chunks])

        with self._lock:
            if self._index is None:
                self._index = Hnsw(
                    dim=self.dim,
                    metric=self.metric,
                    m=self.m,
                    ef_construction=self.ef_construction,
                    seed=self.seed,
                )
            doc_id = self._next_doc_id
            self._next_doc_id += 1
            ids = self._index.insert_batch(vectors)
            for chunk, cid in zip(chunks, ids):
                self._chunks[cid] = StoredChunk(
                    id=cid,
                    text=chunk.text,
                    doc_id=doc_id,
                    doc_title=title,
                    ordinal=chunk.index,
                )
            doc = Document(id=doc_id, title=title, n_chunks=len(chunks))
            self._documents[doc_id] = doc
            return doc

    def retrieve(self, query: str, k: int = 5, ef_search: int = 100) -> List[RetrievedChunk]:
        """Embed the query and return the k most relevant chunks."""
        with self._lock:
            if self._index is None or len(self._index) == 0:
                return []
            index = self._index

        qvec = self.embedder.embed([query])[0]
        hits = index.search(qvec, k=k, ef_search=ef_search)
        out: List[RetrievedChunk] = []
        for cid, distance in hits:
            sc = self._chunks[cid]
            # Cosine distance is 1 - similarity; report similarity so higher
            # is more relevant, which is what a reader expects from a score.
            score = 1.0 - distance if self.metric == "cosine" else -distance
            if self.min_score is not None and score < self.min_score:
                # Hits are closest-first, so their relevance scores only
                # decrease. Weak nearest neighbors are not useful grounding.
                break
            out.append(
                RetrievedChunk(
                    id=sc.id,
                    text=sc.text,
                    doc_id=sc.doc_id,
                    doc_title=sc.doc_title,
                    ordinal=sc.ordinal,
                    score=score,
                )
            )
        return out

    def stats(self) -> dict:
        with self._lock:
            return {
                "documents": len(self._documents),
                "chunks": len(self._chunks),
                "dim": self.dim,
                "metric": self.metric,
                "min_score": self.min_score,
                "m": self.m,
                "ef_construction": self.ef_construction,
                # Which embedding backend is actually live. Worth surfacing:
                # HashedEmbedder matches on term overlap, not meaning, so a
                # reader should not mistake it for semantic search.
                "embedder": type(self.embedder).__name__,
            }

    def documents(self) -> List[Document]:
        with self._lock:
            return list(self._documents.values())
