"""hnsw_rag: thin Python helpers around the Rust HNSW engine.

The engine (`hnsw_engine.Hnsw`) only knows about vectors. This package adds
the two pieces that turn documents into vectors and back:

- `chunking`  — split text into overlapping, retrievable chunks.
- `embeddings` — turn chunks into vectors via a pluggable backend.
"""

from .chunking import Chunk, chunk_text
from .embeddings import Embedder, HashedEmbedder, get_embedder

__all__ = [
    "Chunk",
    "chunk_text",
    "Embedder",
    "HashedEmbedder",
    "get_embedder",
]
