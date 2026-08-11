"""Pluggable embedding backends.

`get_embedder()` returns a callable that maps a list of strings to a list of
equal-length float vectors. The default prefers a real model
(`sentence-transformers`, `all-MiniLM-L6-v2`, 384-d) and falls back to a
deterministic hashed embedder when that isn't installed — so the test suite
and the demo run anywhere, with no network and no heavyweight dependency,
while production code gets real semantics simply by installing the model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol, Sequence


class Embedder(Protocol):
    """Anything that turns texts into fixed-width, L2-normalizable vectors."""

    dim: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashedEmbedder:
    """Deterministic hashing-trick embedder — no model, no network, no deps.

    Each lowercased token is hashed into a bucket of a `dim`-wide vector with
    a signed contribution; the vector is then L2-normalized. This is a real
    (if crude) bag-of-words embedding: it captures exact term overlap between
    a query and a chunk, which is enough to demonstrate end-to-end retrieval
    deterministically. It does *not* capture synonymy or paraphrase — install
    sentence-transformers for that. Same text always yields the same vector,
    which keeps tests exact.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    def _tokens(self, text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                code = int.from_bytes(h, "little")
                bucket = code % self.dim
                sign = 1.0 if (code >> 63) & 1 else -1.0
                vec[bucket] += sign
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            out.append(vec)
        return out


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Imported lazily so the dependency
    is optional."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=False,
        )
        return [list(map(float, v)) for v in vectors]


def get_embedder(
    prefer: str = "auto",
    *,
    dim: int = 384,
    model_name: str = "all-MiniLM-L6-v2",
) -> Embedder:
    """Return an embedder.

    Args:
        prefer: "auto" (real model if available, else hashed), "model" (force
            sentence-transformers, raising if absent), or "hashed" (force the
            dependency-free fallback).
        dim: vector width for the hashed fallback.
        model_name: sentence-transformers model id.
    """
    if prefer not in {"auto", "model", "hashed"}:
        raise ValueError(f"unknown prefer={prefer!r}")

    if prefer == "hashed":
        return HashedEmbedder(dim=dim)

    if prefer in {"auto", "model"}:
        try:
            return SentenceTransformerEmbedder(model_name=model_name)
        except Exception:
            if prefer == "model":
                raise
            return HashedEmbedder(dim=dim)

    return HashedEmbedder(dim=dim)  # unreachable, keeps type checkers happy
