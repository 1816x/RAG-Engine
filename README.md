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

Pending — will be populated with real measured numbers once Phase 2 lands. Run them yourself with:

```sh
cargo run --release --example bench
```

## Design decisions

This section grows as the project does; each phase documents the trade-offs it makes.

- **Why HNSW instead of a flat index**: brute-force search is exact but O(n) per query; HNSW trades a small amount of recall for approximately logarithmic search through a layered proximity graph. The benchmarks in this README exist to show that trade-off with real numbers rather than assert it.
- **Why Rust for the index**: the index is the one component where performance genuinely matters, and where manual control over memory layout pays off.
- **Why Python bindings**: the embeddings ecosystem lives in Python; the engine should be usable from it rather than compete with it.
