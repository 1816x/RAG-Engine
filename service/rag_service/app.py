"""FastAPI RAG service.

Owns the index, retrieval, and generation — the Next.js app is a pure UI that
talks to this. Endpoints:

    POST /documents   upload text → chunk → embed → insert into HNSW
    POST /query       embed question → HNSW search → Claude answer + sources
    GET  /stats       index size / config
    GET  /documents   list indexed documents
    GET  /healthz     liveness

Run with:  uvicorn rag_service.app:app --reload
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .generation import generate_answer
from .store import DocumentStore
from hnsw_rag import get_embedder

log = logging.getLogger(__name__)

# One in-memory store for the process. The embedder backend is chosen at
# startup: real model if available, deterministic hashed fallback otherwise.
_embedder = get_embedder(os.environ.get("RAG_EMBEDDER", "auto"))
_store = DocumentStore(embedder=_embedder)

SAMPLE_DOCS = pathlib.Path(__file__).resolve().parent.parent / "sample_docs"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def seed_sample_docs() -> int:
    """Index the bundled sample corpus in-process. Returns documents added.

    The index is in-memory, so a fresh container starts empty. `scripts/seed.py`
    solves that over HTTP for a running instance, but a hosted deployment has
    nobody to run it — hence this in-process path for startup.
    """
    if not SAMPLE_DOCS.is_dir():
        log.warning("sample corpus not found at %s; starting with an empty index", SAMPLE_DOCS)
        return 0
    added = 0
    for path in sorted(SAMPLE_DOCS.glob("*.md")):
        _store.add_document(path.stem.replace("_", " "), path.read_text())
        added += 1
    return added


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Optionally seed the sample corpus on boot.

    Off by default so local runs and the test suite keep their existing
    behavior (seed explicitly via `scripts/seed.py` or a fixture). Deployments
    turn it on — see `RAG_SEED_ON_STARTUP` in fly.toml. Also guarded on an
    empty index so it can never double-seed.
    """
    if _env_flag("RAG_SEED_ON_STARTUP") and _store.stats()["chunks"] == 0:
        added = seed_sample_docs()
        log.info("seeded %d sample document(s) on startup", added)
    yield


app = FastAPI(title="RAG Engine service", version="0.1.0", lifespan=lifespan)

# The Next.js dev server calls this cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DocumentIn(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    max_words: int = 180
    overlap: int = 40


class DocumentOut(BaseModel):
    id: int
    title: str
    n_chunks: int


class QueryIn(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    ef_search: int = Field(100, ge=1)


class SourceOut(BaseModel):
    chunk_id: int
    doc_id: int
    doc_title: str
    ordinal: int
    score: float
    text: str
    cited: bool


class QueryOut(BaseModel):
    question: str
    answer: str
    model: str
    sources: List[SourceOut]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    return _store.stats()


@app.get("/documents", response_model=List[DocumentOut])
def list_documents() -> List[DocumentOut]:
    return [
        DocumentOut(id=d.id, title=d.title, n_chunks=d.n_chunks)
        for d in _store.documents()
    ]


@app.post("/documents", response_model=DocumentOut)
def add_document(doc: DocumentIn) -> DocumentOut:
    result = _store.add_document(
        doc.title, doc.text, max_words=doc.max_words, overlap=doc.overlap
    )
    return DocumentOut(id=result.id, title=result.title, n_chunks=result.n_chunks)


@app.post("/query", response_model=QueryOut)
def query(q: QueryIn) -> QueryOut:
    chunks = _store.retrieve(q.question, k=q.k, ef_search=q.ef_search)
    answer = generate_answer(q.question, chunks)
    cited = set(answer.cited_chunk_ids)
    sources = [
        SourceOut(
            chunk_id=c.id,
            doc_id=c.doc_id,
            doc_title=c.doc_title,
            ordinal=c.ordinal,
            score=round(c.score, 4),
            text=c.text,
            cited=c.id in cited,
        )
        for c in chunks
    ]
    return QueryOut(
        question=q.question, answer=answer.text, model=answer.model, sources=sources
    )
