"""Thread-safe document store with optional durable, atomic snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from hnsw_engine import Hnsw
from hnsw_rag import Chunk, chunk_text
from hnsw_rag.embeddings import Embedder

STATE_MAGIC = b"RAGSTATE"
STATE_VERSION = 1
HNSW_FORMAT_VERSION = 1
_HEADER = struct.Struct("<8sIQQ")
_CHECKSUM_BYTES = 32
_MAX_METADATA_BYTES = 256 * 1024 * 1024
_MAX_INDEX_BYTES = 16 * 1024 * 1024 * 1024


@dataclass
class StoredChunk:
    id: int
    text: str
    doc_id: int
    doc_title: str
    ordinal: int


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
    score: float


def _embedder_identity(embedder: Embedder) -> dict:
    identity = {"type": type(embedder).__name__, "dim": embedder.dim}
    model_name = getattr(embedder, "model_name", None)
    if model_name is not None:
        identity["model_name"] = model_name
    return identity


@dataclass
class DocumentStore:
    embedder: Embedder
    metric: str = "cosine"
    min_score: Optional[float] = None
    m: int = 16
    ef_construction: int = 200
    seed: int = 0x5EED
    state_path: Optional[pathlib.Path | str] = None

    _index: Optional[Hnsw] = None
    _chunks: Dict[int, StoredChunk] = field(default_factory=dict)
    _documents: Dict[int, Document] = field(default_factory=dict)
    _next_doc_id: int = 0
    _loaded: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.state_path is not None:
            raw = str(self.state_path).strip()
            self.state_path = pathlib.Path(raw) if raw else None
        if self.state_path is not None and self.state_path.exists():
            self._load_state(self.state_path)
            self._loaded = True

    @property
    def dim(self) -> int:
        return self.embedder.dim

    def _new_index(self) -> Hnsw:
        return Hnsw(
            dim=self.dim,
            metric=self.metric,
            m=self.m,
            ef_construction=self.ef_construction,
            seed=self.seed,
        )

    def _metadata(
        self,
        chunks: Dict[int, StoredChunk],
        documents: Dict[int, Document],
        next_doc_id: int,
    ) -> dict:
        return {
            "state_version": STATE_VERSION,
            "hnsw_format_version": HNSW_FORMAT_VERSION,
            "index": {
                "dim": self.dim,
                "metric": self.metric,
                "m": self.m,
                "ef_construction": self.ef_construction,
                "seed": self.seed,
            },
            "embedder": _embedder_identity(self.embedder),
            "next_doc_id": next_doc_id,
            "chunks": [asdict(chunks[key]) for key in sorted(chunks)],
            "documents": [asdict(documents[key]) for key in sorted(documents)],
        }

    def _persist(
        self,
        index: Optional[Hnsw],
        chunks: Dict[int, StoredChunk],
        documents: Dict[int, Document],
        next_doc_id: int,
    ) -> None:
        assert self.state_path is not None
        actual_index = index if index is not None else self._new_index()
        metadata = json.dumps(
            self._metadata(chunks, documents, next_doc_id),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        index_data = bytes(actual_index.to_bytes())
        payload = (
            _HEADER.pack(STATE_MAGIC, STATE_VERSION, len(metadata), len(index_data))
            + metadata
            + index_data
        )
        data = payload + hashlib.blake2b(payload, digest_size=_CHECKSUM_BYTES).digest()
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temporary.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _load_state(self, path: pathlib.Path) -> None:
        data = path.read_bytes()
        if len(data) < _HEADER.size + _CHECKSUM_BYTES:
            raise RuntimeError("persisted RAG state is truncated")
        payload, digest = data[:-_CHECKSUM_BYTES], data[-_CHECKSUM_BYTES:]
        if hashlib.blake2b(payload, digest_size=_CHECKSUM_BYTES).digest() != digest:
            raise RuntimeError("persisted RAG state checksum mismatch")
        magic, version, metadata_len, index_len = _HEADER.unpack_from(payload)
        if magic != STATE_MAGIC:
            raise RuntimeError("persisted RAG state has an invalid magic header")
        if version != STATE_VERSION:
            raise RuntimeError(f"unsupported RAG state format version {version}")
        if metadata_len > _MAX_METADATA_BYTES or index_len > _MAX_INDEX_BYTES:
            raise RuntimeError("persisted RAG state exceeds supported size limits")
        expected = _HEADER.size + metadata_len + index_len
        if expected != len(payload):
            raise RuntimeError("persisted RAG state lengths are invalid")
        metadata_end = _HEADER.size + metadata_len
        try:
            metadata = json.loads(payload[_HEADER.size : metadata_end])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("persisted RAG metadata is invalid") from exc
        expected_index = {
            "dim": self.dim,
            "metric": self.metric,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "seed": self.seed,
        }
        if metadata.get("index") != expected_index:
            raise RuntimeError(
                f"persisted index configuration is incompatible with runtime configuration (stored={metadata.get('index')!r}, runtime={expected_index!r})"
            )
        identity = _embedder_identity(self.embedder)
        if metadata.get("embedder") != identity:
            raise RuntimeError(
                f"persisted embedder is incompatible with runtime embedder (stored={metadata.get('embedder')!r}, runtime={identity!r})"
            )
        try:
            index = Hnsw.from_bytes(payload[metadata_end:])
            chunks = {item["id"]: StoredChunk(**item) for item in metadata["chunks"]}
            documents = {item["id"]: Document(**item) for item in metadata["documents"]}
            next_doc_id = int(metadata["next_doc_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("persisted RAG metadata has an invalid schema") from exc
        if (
            metadata.get("state_version") != STATE_VERSION
            or metadata.get("hnsw_format_version") != HNSW_FORMAT_VERSION
        ):
            raise RuntimeError("persisted RAG metadata version is incompatible")
        if index.dim != self.dim or index.metric != self.metric:
            raise RuntimeError(
                "persisted HNSW configuration does not match its metadata"
            )
        if len(index) != len(chunks) or set(chunks) != set(range(len(index))):
            raise RuntimeError("persisted HNSW ids and chunk metadata are not aligned")
        if next_doc_id < 0 or any(doc_id >= next_doc_id for doc_id in documents):
            raise RuntimeError("persisted document insertion state is invalid")
        if any(chunk.doc_id not in documents for chunk in chunks.values()):
            raise RuntimeError("persisted chunk references an unknown document")
        chunk_counts = {doc_id: 0 for doc_id in documents}
        for chunk in chunks.values():
            chunk_counts[chunk.doc_id] += 1
            if chunk.doc_title != documents[chunk.doc_id].title:
                raise RuntimeError("persisted chunk and document titles disagree")
        if any(
            document.n_chunks != chunk_counts[doc_id]
            for doc_id, document in documents.items()
        ):
            raise RuntimeError("persisted document chunk counts are invalid")
        self._index, self._chunks, self._documents, self._next_doc_id = (
            index,
            chunks,
            documents,
            next_doc_id,
        )

    def add_document(
        self, title: str, text: str, *, max_words: int = 180, overlap: int = 40
    ) -> Document:
        chunks: List[Chunk] = chunk_text(
            text, max_words=max_words, overlap=overlap, source=title
        )
        vectors = self.embedder.embed([c.text for c in chunks]) if chunks else []
        if len(vectors) != len(chunks):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        with self._lock:
            # With persistence enabled, mutate a snapshot clone, commit it, then
            # publish it in memory. A failed fsync/replace leaves both old states active.
            staging = self.state_path is not None
            index = (
                Hnsw.from_bytes(bytes(self._index.to_bytes()))
                if staging and self._index is not None
                else (
                    self._new_index() if chunks and self._index is None else self._index
                )
            )
            chunk_map = dict(self._chunks) if staging else self._chunks
            document_map = dict(self._documents) if staging else self._documents
            doc_id = self._next_doc_id
            next_doc_id = doc_id + 1
            ids = index.insert_batch(vectors) if chunks and index is not None else []
            if len(ids) != len(chunks):
                raise RuntimeError(
                    f"index returned {len(ids)} ids for {len(chunks)} chunks"
                )
            for chunk, cid in zip(chunks, ids):
                chunk_map[cid] = StoredChunk(
                    cid, chunk.text, doc_id, title, chunk.index
                )
            doc = Document(doc_id, title, len(chunks))
            document_map[doc_id] = doc
            if staging:
                self._persist(index, chunk_map, document_map, next_doc_id)
            self._index, self._chunks, self._documents = index, chunk_map, document_map
            self._next_doc_id = next_doc_id
            return doc

    def retrieve(
        self, query: str, k: int = 5, ef_search: int = 100
    ) -> List[RetrievedChunk]:
        qvec = self.embedder.embed([query])[0]
        with self._lock:
            if self._index is None or len(self._index) == 0:
                return []
            hits = self._index.search(qvec, k=k, ef_search=ef_search)
            out = []
            for cid, distance in hits:
                sc = self._chunks[cid]
                score = 1.0 - distance if self.metric == "cosine" else -distance
                if self.min_score is not None and score < self.min_score:
                    break
                out.append(
                    RetrievedChunk(
                        sc.id, sc.text, sc.doc_id, sc.doc_title, sc.ordinal, score
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
                "embedder": type(self.embedder).__name__,
                "persistence": {
                    "enabled": self.state_path is not None,
                    "loaded": self._loaded,
                    "format_version": STATE_VERSION,
                },
            }

    def documents(self) -> List[Document]:
        with self._lock:
            return list(self._documents.values())
