# bindings/ — Python bindings (PyO3)

Lands in **Phase 3**. This package will expose the Rust HNSW engine (`engine/`) as a Python library via PyO3 + maturin, so it can plug into the Python embeddings ecosystem (sentence-transformers or an embeddings API).

Planned surface:

```python
from hnsw_engine import Hnsw

index = Hnsw(dim=384, metric="cosine", m=16, ef_construction=200)
index.insert(vector)          # -> id
index.search(query, k=10)     # -> [(id, distance), ...]
```
