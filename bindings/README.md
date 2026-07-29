# bindings/ — Python bindings (PyO3)

The Rust HNSW engine (`engine/`) exposed as a Python package, so it plugs into the Python embeddings ecosystem.

```python
from hnsw_engine import Hnsw

index = Hnsw(dim=384, metric="cosine", m=16, ef_construction=200)
index.insert([0.1, 0.2, ...])                 # -> 0 (dense ids, insert order)
index.insert_batch(vectors)                   # -> [1, 2, ...]
index.search(query, k=10, ef_search=100)      # -> [(id, distance), ...] closest first
index.vector(0)                               # stored vector (normalized if cosine)
len(index), index.dim, index.metric
```

`metric` is `"cosine"` (default) or `"euclidean"`. Distances are `1 - cos` and true L2 respectively. `ef_search` defaults to `max(4*k, 50)`.

Vectors cross the boundary as plain sequences of floats — no NumPy dependency by design; call `list(arr)` on arrays.

## The `hnsw_rag` helper package

`python/hnsw_rag/` is a small pure-Python layer that turns raw text into something the index can hold:

- `chunking.chunk_text(text, ...)` — paragraph-aware chunker with token-ish overlap.
- `embeddings.get_embedder(...)` — pluggable embedding backend. Defaults to `sentence-transformers` (`all-MiniLM-L6-v2`, 384-d) when installed, and falls back to a deterministic zero-dependency hashed embedder so tests and demos run anywhere.

## Build & test

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install maturin pytest
maturin develop --release   # builds the Rust extension into the venv
pytest tests/
```
