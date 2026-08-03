"""Paragraph-aware text chunking.

Splitting strategy: break on blank lines (paragraphs), then greedily pack
paragraphs into chunks up to a word budget. When a single paragraph is larger
than the budget it is split on word boundaries. Consecutive chunks share an
overlap of trailing words so a fact that straddles a boundary is still fully
present in at least one chunk — the usual reason naive fixed-size chunking
loses answers.

Word counts stand in for tokens: it keeps the helper dependency-free, and the
approximation is close enough for chunk sizing (real token counts run ~1.3x
higher for English prose).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text and where it came from."""

    text: str
    index: int  # position of this chunk within its source document
    source: str | None = None


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_long_paragraph(
    words: list[str], max_words: int, overlap: int
) -> list[list[str]]:
    pieces: list[list[str]] = []
    start = 0
    step = max_words - overlap
    while start < len(words):
        end = min(start + max_words, len(words))
        pieces.append(words[start:end])
        if end == len(words):
            break
        start += step
    return pieces


def chunk_text(
    text: str,
    *,
    max_words: int = 180,
    overlap: int = 40,
    source: str | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping chunks of at most ~`max_words` words.

    Args:
        text: the document text.
        max_words: soft upper bound on words per chunk.
        overlap: words of trailing context repeated at the start of the next
            chunk. Must be < max_words.
        source: optional label carried onto every chunk (e.g. a filename).

    Returns:
        Chunks in document order, each tagged with its running index.
    """
    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:
            flush()
            for piece in _split_long_paragraph(words, max_words, overlap):
                chunks.append(" ".join(piece))
            continue
        if current and len(current) + len(words) > max_words:
            # Carry as much overlap as fits beside the next paragraph. A
            # full-size paragraph leaves no room for an overlap tail.
            tail_size = min(overlap, max_words - len(words))
            tail = current[-tail_size:] if tail_size else []
            flush()
            current.extend(tail)
        current.extend(words)
    flush()

    return [Chunk(text=c, index=i, source=source) for i, c in enumerate(chunks)]
