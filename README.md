# RAG Engine — vector search from scratch

A from-scratch implementation of an **HNSW (Hierarchical Navigable Small World) vector index in Rust** — no LangChain, no heavyweight RAG frameworks — with Python bindings (PyO3) and a small TypeScript RAG app on top to prove real end-to-end usage. Includes honest, reproducible benchmarks against brute-force search.

The point of this project is not to beat qdrant. The point is that the wheel *is* the skill being demonstrated: understanding how an AI retrieval system works on the inside, not just how to call one.

## Architecture

```
documents → [Python: chunking + embeddings]
                        ↓
        [Rust: HNSW index — insert / search]  ←── benchmarks vs. brute force
                        ↓ (via PyO3 bindings)
              [Python: retrieval layer]
                        ↓
        [Claude API: answer generation with cited sources]
                        ↓
                [Next.js: chat UI + sources]
```

## Repository layout

| Path        | Language   | What it is |
|-------------|------------|------------|
| `engine/`   | Rust       | The HNSW index itself: insert, layered greedy search, neighbor-selection heuristic, brute-force baseline. Zero runtime dependencies. |
| `bindings/` | Rust + Python | PyO3 bindings exposing the engine as a Python package (Phase 3). |
| `app/`      | TypeScript | Next.js RAG app: upload documents, ask questions, see retrieved sources + Claude-generated answers (Phase 4). |

## Roadmap

- [x] **Phase 0 — Scaffolding**: workspace layout, README with architecture.
- [x] **Phase 1 — Vertical slice**: minimal HNSW supporting insert + search over random vectors, correctness-tested against brute force.
- [ ] **Phase 2 — Iteration**: benchmarks vs. brute-force search (real numbers in this README, not invented ones), recall tests, parameter tuning (`M`, `ef_construction`).
- [ ] **Phase 3 — Bindings + real embeddings**: PyO3 bindings, real embedding model, document chunking.
- [ ] **Phase 4 — Full RAG**: Claude API integration with cited sources, Next.js app.
- [ ] **Phase 5 — Release**: `v0.1.0`, benchmark table (latency/recall vs. brute force).

## Benchmarks

All numbers measured with `cargo run --release --example bench` on this project's dev container (single-threaded, Linux x86-64; expect different absolute numbers on your hardware — relative behavior holds). Dataset: 50,000 random vectors, dim 128, drawn from 1,000 Gaussian-ish clusters to mimic how real embedding vectors distribute (see the honesty note below). Metric: Euclidean; k = 10; 200 queries; fully seeded and reproducible.

**HNSW (M=16, ef_construction=200) vs. exact brute force:**

| method      | ef_search | avg latency/query | recall@10 | speedup |
|-------------|-----------|-------------------|-----------|---------|
| brute force | —         | 5472 µs           | 1.000     | 1.0x    |
| HNSW        | 10        | 49 µs             | 0.980     | 112x    |
| HNSW        | 25        | 64 µs             | 0.995     | 86x     |
| HNSW        | 50        | 106 µs            | 1.000     | 52x     |
| HNSW        | 100       | 204 µs            | 1.000     | 27x     |
| HNSW        | 200       | 379 µs            | 1.000     | 14x     |

Build time: 29.7 s (1,686 inserts/s). Sweeping M: M=8 builds in 19.3 s but drops to 0.91 recall at ef=10; M=32 builds in 47.5 s and only costs latency at this corpus size. M=16 is the default for a reason.

**The honesty note (uniform random data).** On i.i.d. *uniform* random vectors at dim 128 — a distribution real embeddings never follow — recall degrades badly (0.53 at ef=100, same corpus size). This is the curse of dimensionality: uniform high-dim points have near-identical pairwise distances, so *no* proximity structure can exploit the geometry (the same effect hits production HNSW implementations). Clustered data restores the structure HNSW navigates. Run `BENCH_CLUSTERS=0 cargo run --release --example bench` to reproduce the pathology yourself; benchmarks that only show the flattering case aren't benchmarks.

Knobs: `BENCH_N`, `BENCH_DIM`, `BENCH_CLUSTERS` (0 = uniform), `BENCH_M`, `BENCH_EFC`.

## Design decisions

This section grows as the project does; each phase documents the trade-offs it makes.

- **Why HNSW instead of a flat index**: brute-force search is exact but O(n) per query; HNSW trades a small amount of recall for approximately logarithmic search through a layered proximity graph. The benchmarks in this README exist to show that trade-off with real numbers rather than assert it: at 50k×128 clustered, you give up 2% recall for a 112x speedup, or 0% for 52x.
- **Why Rust for the index**: the index is the one component where performance genuinely matters, and where manual control over memory layout pays off.
- **Why Python bindings**: the embeddings ecosystem lives in Python; the engine should be usable from it rather than compete with it.
- **Zero dependencies in the engine**: even the RNG is a 10-line SplitMix64. Not dogma — the crate exists to show the algorithm, and every dependency would blur what's actually implemented here. It also makes builds deterministic: same seed, same graph, byte-for-byte.
- **Squared L2 internally, true L2 at the API**: distance comparisons don't need the sqrt (monotonic transform), so the hot loop skips it; user-facing results convert back.
- **Contiguous vector storage**: vectors live in one flat `Vec<f32>` (row stride = dim), not `Vec<Vec<f32>>`. Distance evaluation is the hot loop; removing a pointer indirection per evaluation roughly halved both build and query time in measurement.
- **Bitset visited-set**: the per-search visited set is a reusable bitset (n/8 bytes, cleared with one memset) instead of a `HashSet` — cheaper to clear, cheaper to probe, and keeps `search(&self)` shareable across threads by using a per-call instance.
- **Diversity heuristic over "closest M"** (Algorithm 4 of the paper): neighbors are kept only if they're closer to the query than to already-kept neighbors, spreading links across directions. This is what keeps the graph navigable through sparse regions; without it, recall on clustered data collapses at cluster boundaries.
