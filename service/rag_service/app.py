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

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .generation import generate_answer
from .store import DocumentStore
from hnsw_rag import get_embedder

app = FastAPI(title="RAG Engine service", version="0.1.0")

# The Next.js dev server calls this cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One in-memory store for the process. The embedder backend is chosen at
# startup: real model if available, deterministic hashed fallback otherwise.
_embedder = get_embedder(os.environ.get("RAG_EMBEDDER", "auto"))
_store = DocumentStore(embedder=_embedder)


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
