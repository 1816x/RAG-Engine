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

# Latest Sonnet is a good default for grounded RAG answers: fast, cheap, and
# strong at instruction-following. Override with RAG_MODEL if you want Opus.
DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You answer questions using only the provided source chunks.

Rules:
- Use only information present in the sources. If they don't contain the answer, \
say so plainly — do not invent facts.
- Cite the chunks you used with inline markers like [1], [2] that refer to the \
numbered sources.
- Be concise. Lead with the answer, then support it."""


@dataclass
class Answer:
    text: str
    cited_chunk_ids: List[int]
    model: str  # "claude-…" or "mock"


def _format_sources(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source: {c.doc_title})\n{c.text}")
    return "\n\n".join(blocks)


def parse_citations(text: str, chunks: List[RetrievedChunk]) -> List[int]:
    """Map the `[n]` markers in an answer back to the chunk ids they refer to.

    The sources are numbered 1..len(chunks) in the prompt, so `[2]` means
    `chunks[1]`. Out-of-range markers (the model inventing `[9]` for three
    sources) are ignored rather than trusted. Returns ids in first-mention
    order, deduplicated — so a "cited" flag actually means the answer used
    that chunk, instead of just "we retrieved it".
    """
    cited: List[int] = []
    for marker in re.findall(r"\[(\d+)\]", text):
        idx = int(marker) - 1
        if 0 <= idx < len(chunks):
            cid = chunks[idx].id
            if cid not in cited:
                cited.append(cid)
    return cited


def _mock_answer(question: str, chunks: List[RetrievedChunk]) -> Answer:
    """Deterministic, keyless fallback: return the single most relevant chunk
    as the answer, cited. Good enough to prove the pipeline end to end."""
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
    """Generate an answer to `question` grounded in `chunks`.

    Falls back to mock mode when ANTHROPIC_API_KEY is unset or the `anthropic`
    package isn't installed.
    """
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

    model = model or os.environ.get("RAG_MODEL", DEFAULT_MODEL)
    client = anthropic.Anthropic()
    user_content = (
        f"Sources:\n\n{_format_sources(chunks)}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the sources above, citing them with [n] markers."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

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
