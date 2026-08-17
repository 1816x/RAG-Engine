"""FastAPI RAG service.

Owns the index, retrieval, and generation — the Next.js app is a pure UI that
talks to this. Endpoints:

    POST /documents   upload text → chunk → embed → insert into HNSW
    PUT/DELETE /documents/{id} replace or delete through HNSW compaction
    POST /query       embed question → HNSW search → Claude answer + sources
    GET  /stats       index size / config
    GET  /documents   list indexed documents
    GET  /healthz     liveness

Run with:  uvicorn rag_service.app:app --reload
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import pathlib
from typing import AsyncIterator, List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .generation import GenerationTimeoutError, generate_answer
from .store import DocumentNotFoundError, DocumentStore
from hnsw_rag import get_embedder

log = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _cors_origins() -> List[str]:
    raw = os.environ.get("RAG_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]

    origins = list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))
    if "*" in origins:
        raise RuntimeError("RAG_CORS_ORIGINS must be '*' or a list of exact origins")
    return origins


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _finite_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite")
    return value


def _add_cors_middleware(target: FastAPI, origins: List[str]) -> None:
    if not origins:
        return
    target.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )


MAX_TITLE_CHARS = _positive_env_int("RAG_MAX_TITLE_CHARS", 200)
MAX_DOCUMENT_CHARS = _positive_env_int("RAG_MAX_DOCUMENT_CHARS", 1_000_000)
MAX_QUESTION_CHARS = _positive_env_int("RAG_MAX_QUESTION_CHARS", 2_000)
MAX_REQUEST_BYTES = _positive_env_int("RAG_MAX_REQUEST_BYTES", 4 * 1024 * 1024)


class BodySizeLimitMiddleware:
    """Reject oversized request bodies before JSON parsing."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
                if content_length < 0:
                    raise ValueError
                if content_length > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse(
                    {"detail": "invalid Content-Length header"}, status_code=400
                )
                await response(scope, receive, send)
                return

        messages: List[Message] = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
                more_body = message.get("more_body", False)

        position = 0

        async def replay() -> Message:
            nonlocal position
            if position < len(messages):
                message = messages[position]
                position += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": f"request body exceeds {self.max_bytes} bytes"},
            status_code=413,
        )
        await response(scope, receive, send)


# One process-local store; durability is opt-in through RAG_STATE_PATH. The embedder backend is chosen at
# startup: real model if available, deterministic hashed fallback otherwise.
_embedder = get_embedder(os.environ.get("RAG_EMBEDDER", "auto"))
_store = DocumentStore(
    embedder=_embedder,
    min_score=_finite_env_float("RAG_MIN_SCORE", 0.09),
    state_path=os.environ.get("RAG_STATE_PATH"),
)

SAMPLE_DOCS = pathlib.Path(__file__).resolve().parent.parent / "sample_docs"


def seed_sample_docs() -> int:
    """Index the bundled sample corpus in-process. Returns documents added.

    The hosted demo has persistence disabled, so startup seeding remains useful.
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
    """Optionally seed the sample corpus on boot."""
    if _env_flag("RAG_SEED_ON_STARTUP") and _store.stats()["chunks"] == 0:
        added = seed_sample_docs()
        log.info("seeded %d sample document(s) on startup", added)
    yield


app = FastAPI(title="RAG Engine service", version="0.1.0", lifespan=lifespan)

_add_cors_middleware(app, _cors_origins())
app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)


def _require_uploads_enabled() -> None:
    if not _env_flag("RAG_UPLOADS_ENABLED"):
        raise HTTPException(status_code=403, detail="document uploads are disabled")


class DocumentIn(BaseModel):
    title: str = Field(..., min_length=1, max_length=MAX_TITLE_CHARS)
    text: str = Field(..., min_length=1, max_length=MAX_DOCUMENT_CHARS)
    max_words: int = Field(180, ge=1, le=1000)
    overlap: int = Field(40, ge=0, le=999)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @model_validator(mode="after")
    def validate_chunk_window(self):
        if self.overlap >= self.max_words:
            raise ValueError("overlap must be smaller than max_words")
        return self


class DocumentOut(BaseModel):
    id: int
    title: str
    n_chunks: int


class QueryIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int = Field(5, ge=1, le=50)
    ef_search: int = Field(100, ge=1, le=2000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


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
    data = _store.stats()
    data["generation"] = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "mock"
    data["uploads_enabled"] = _env_flag("RAG_UPLOADS_ENABLED")
    data["limits"] = {
        "title_chars": MAX_TITLE_CHARS,
        "document_chars": MAX_DOCUMENT_CHARS,
        "question_chars": MAX_QUESTION_CHARS,
        "request_bytes": MAX_REQUEST_BYTES,
    }
    return data


@app.get("/documents", response_model=List[DocumentOut])
def list_documents() -> List[DocumentOut]:
    return [
        DocumentOut(id=d.id, title=d.title, n_chunks=d.n_chunks)
        for d in _store.documents()
    ]


@app.post(
    "/documents",
    response_model=DocumentOut,
    dependencies=[Depends(_require_uploads_enabled)],
)
def add_document(doc: DocumentIn) -> DocumentOut:
    result = _store.add_document(
        doc.title, doc.text, max_words=doc.max_words, overlap=doc.overlap
    )
    return DocumentOut(id=result.id, title=result.title, n_chunks=result.n_chunks)


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentOut,
    dependencies=[Depends(_require_uploads_enabled)],
)
def delete_document(document_id: int) -> DocumentOut:
    try:
        result = _store.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DocumentOut(id=result.id, title=result.title, n_chunks=result.n_chunks)


@app.put(
    "/documents/{document_id}",
    response_model=DocumentOut,
    dependencies=[Depends(_require_uploads_enabled)],
)
def replace_document(document_id: int, doc: DocumentIn) -> DocumentOut:
    try:
        result = _store.replace_document(
            document_id,
            doc.title,
            doc.text,
            max_words=doc.max_words,
            overlap=doc.overlap,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DocumentOut(id=result.id, title=result.title, n_chunks=result.n_chunks)


@app.post("/query", response_model=QueryOut)
def query(q: QueryIn) -> QueryOut:
    chunks = _store.retrieve(q.question, k=q.k, ef_search=q.ef_search)
    try:
        answer = generate_answer(q.question, chunks)
    except GenerationTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
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
