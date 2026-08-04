"""Answer generation with cited sources.

Given a question and the chunks retrieved for it, produce an answer that cites
which chunks it used. When ANTHROPIC_API_KEY is set, this calls Claude; when it
isn't, a deterministic "mock mode" produces an extractive answer so the whole
RAG app is demoable — and testable in CI — without a key or network.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from .store import RetrievedChunk

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_CLAUDE_TIMEOUT_SECONDS = 30.0

SYSTEM_PROMPT = """You answer questions using only the provided source chunks.

Rules:
- Use only information present in the sources. If they don't contain the answer, \
say so plainly — do not invent facts.
- Cite the chunks you used with inline markers like [1], [2] that refer to the \
numbered sources.
- Be concise. Lead with the answer, then support it."""


class GenerationTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Claude generation timed out after {timeout_seconds:g} seconds"
        )


@dataclass
class Answer:
    text: str
    cited_chunk_ids: List[int]
    model: str  # "claude-…" or "mock"


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _format_sources(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {chunk.doc_title})\n{chunk.text}")
    return "\n\n".join(blocks)


def parse_citations(text: str, chunks: List[RetrievedChunk]) -> List[int]:
    """Map answer citation markers back to chunk ids."""
    cited: List[int] = []
    for marker in re.findall(r"\[(\d+)\]", text):
        idx = int(marker) - 1
        if 0 <= idx < len(chunks):
            chunk_id = chunks[idx].id
            if chunk_id not in cited:
                cited.append(chunk_id)
    return cited


def _mock_answer(question: str, chunks: List[RetrievedChunk]) -> Answer:
    """Return the most relevant chunk as a deterministic, keyless answer."""
    if not chunks:
        return Answer(
            text="I don't have any indexed documents that address that question.",
            cited_chunk_ids=[],
            model="mock",
        )
    top = chunks[0]
    text = (
        f"(mock answer — set ANTHROPIC_API_KEY for a generated one)\n\n"
        f"Based on the most relevant source [1]:\n\n{top.text}"
    )
    return Answer(text=text, cited_chunk_ids=[top.id], model="mock")


def generate_answer(
    question: str,
    chunks: List[RetrievedChunk],
    *,
    model: Optional[str] = None,
) -> Answer:
    """Generate an answer grounded in chunks, or use keyless mock mode."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _mock_answer(question, chunks)

    try:
        import anthropic
    except ImportError:
        return _mock_answer(question, chunks)

    if not chunks:
        return Answer(
            text="I don't have any indexed documents that address that question.",
            cited_chunk_ids=[],
            model="mock",
        )

    timeout_seconds = _positive_env_float(
        "RAG_CLAUDE_TIMEOUT_SECONDS", DEFAULT_CLAUDE_TIMEOUT_SECONDS
    )
    model = model or os.environ.get("RAG_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic(timeout=timeout_seconds, max_retries=0)
    user_content = (
        f"Sources:\n\n{_format_sources(chunks)}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the sources above, citing them with [n] markers."
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APITimeoutError as exc:
        raise GenerationTimeoutError(timeout_seconds) from exc

    if response.stop_reason == "refusal":
        return Answer(
            text="The model declined to answer this question.",
            cited_chunk_ids=[],
            model=model,
        )

    text = "".join(block.text for block in response.content if block.type == "text")
    return Answer(
        text=text,
        cited_chunk_ids=parse_citations(text, chunks),
        model=model,
    )
