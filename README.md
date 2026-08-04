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
| `bindings/` | Rust + Python | PyO3 bindings exposing the engine as a Python package, plus `hnsw_rag` helpers (chunking + pluggable embeddings). |
| `service/`  | Python | FastAPI RAG service: chunk → embed → HNSW retrieve → Claude answer with cited sources. Runs keyless in mock mode. |
| `app/`      | TypeScript | Next.js chat UI: ask questions, see the answer and retrieved sources (cited chunks highlighted). |

## Deploying it

The repo ships everything needed to host the demo: the service runs in a container, the UI deploys to Vercel.

**Backend (Fly.io):**

```sh
fly launch --no-deploy   # claims an app name, keeps the committed fly.toml
fly deploy
```

The `Dockerfile` is multi-stage — stage one compiles the PyO3 wheel with the Rust toolchain, stage two installs only that wheel (it's `abi3`, so it's portable across CPython ≥ 3.9). Final image is ~270 MB and runs as a non-root user. `fly.toml` scales to zero when idle and suspends rather than stops, so the in-memory index survives a resume.

**Frontend (Vercel):** deploy `app/` as its own project. Point it at the backend with `RAG_SERVICE_URL`, or set `DEFAULT_RAG_SERVICE_URL` in `app/app/lib/config.ts` (it's a public URL, not a secret).

**What the hosted demo actually runs — worth being clear about:**

- **No `ANTHROPIC_API_KEY`.** It runs in mock mode: answers are extractive, tagged with a visible `mock` badge. Nothing calls Claude, so there's no key on a public endpoint and no spend to burn. Retrieval — the HNSW index, which is the point of the project — is fully real.
- **The hashed fallback embedder**, not `sentence-transformers` (which would drag `torch` into the image). That means retrieval matches on *term overlap, not meaning*. Don't mistake the demo for semantic search; install `sentence-transformers` and set `RAG_EMBEDDER=model` for that. `GET /stats` reports which backend is live so you never have to guess.
- **Cold starts.** Scaled to zero, the first request after an idle period waits a second or two for the machine to wake.
- **A visible, locked document workspace.** The UI lists every indexed document and shows the active corpus, embedding, HNSW, relevance, generation, and upload configuration. Public uploads are disabled by default; enabling them is an explicit deployment choice. Documents are held in memory and disappear when the service starts fresh.

## Quick start

Each layer runs on its own; you only need the ones you care about.

**1. The Rust engine (no Python, no Node):**

```sh
cargo test                              # 24 tests, incl. seeded recall vs. brute force
cargo run --release --example bench     # the benchmark table below
```

**2. Python bindings + helpers:**

```sh
cd bindings
python3 -m venv .venv && . .venv/bin/activate
pip install maturin pytest
maturin develop --release               # builds the Rust extension into the venv
pytest tests/                           # 21 tests through the FFI boundary
```

```python
from hnsw_engine import Hnsw
idx = Hnsw(dim=384, metric="cosine")
idx.insert_batch(vectors)               # -> [0, 1, 2, ...]
idx.search(query, k=10)                 # -> [(id, distance), ...] closest first
```

**3. Full RAG (service + app):** the service runs **without an API key** in mock mode, so you can see the whole pipeline before adding one.

```sh
# terminal 1 — the Python service (reuses the bindings venv)
cd service && pip install -r requirements.txt
RAG_UPLOADS_ENABLED=1 uvicorn rag_service.app:app --port 8000
python scripts/seed.py                  # uploads require the flag above

# terminal 2 — the Next.js UI
cd app && npm install
RAG_SERVICE_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

Set `ANTHROPIC_API_KEY` before starting the service to get real Claude-generated answers instead of the extractive mock. The retrieval layer (HNSW search + cited sources) is identical either way.

Retrieval drops chunks whose relevance score is below `RAG_MIN_SCORE`
(default: `0.09`). If no chunk clears the threshold, the service returns no
sources and says the indexed documents do not address the question. Tune the
threshold for a different embedding model or set it to `-1` to preserve every
cosine-search hit.

The web UI makes the runtime modes explicit instead of collapsing them into a
single model badge. **Retrieval** is labeled lexical/hashed or semantic based on
the active embedder; **answer generation** is labeled extractive mock or Claude.
The HNSW index remains real in every mode. Its live document count, chunk count,
vector dimension, metric, `M`, `ef_construction`, and relevance floor are
visible alongside the document library.

The upload panel remains visible when uploads are disabled, but its controls
are locked and explain how to opt in. To add material locally, start both Fly.io
and Vercel-compatible environments with `RAG_UPLOADS_ENABLED=1`, then select a
`.md`, `.markdown`, or `.txt` file (or paste text), give it a title, and
choose **Add to index**. The browser reads the file as text; no file is stored
on disk. `scripts/seed.py` uses the same protected upload endpoint and
therefore also requires uploads to be enabled. Startup seeding is unaffected
because it inserts directly into the in-process store.

### Public API hardening

| Variable | Default | Behavior |
|----------|---------|----------|
| `RAG_UPLOADS_ENABLED` | `0` | Enables `POST /documents` only when explicitly set to `1`, `true`, `yes`, or `on`. |
| `RAG_MAX_TITLE_CHARS` | `200` | Maximum document title length. |
| `RAG_MAX_DOCUMENT_CHARS` | `1000000` | Maximum document text length. |
| `RAG_MAX_QUESTION_CHARS` | `2000` | Maximum query length. |
| `RAG_MAX_REQUEST_BYTES` | `4194304` | Maximum HTTP request body size (4 MiB), enforced before JSON parsing. |
| `RAG_CLAUDE_TIMEOUT_SECONDS` | `30` | Claude request timeout; automatic SDK retries are disabled. |
| `RAG_CORS_ORIGINS` | empty | Exact comma-separated allowed origins. Empty installs no CORS middleware; `*` restores wildcard access explicitly. |

Chunking accepts `max_words` from 1 through 1000 and `overlap` from 0
through 999, with overlap strictly smaller than the chunk size. Query `k`
remains 1 through 50 and `ef_search` is limited to 1 through 2000. Oversized
bodies return `413`, invalid fields return `422`, disabled uploads return
`403`, and Claude timeouts return `504`. The Next.js same-origin proxy
preserves these status codes and service messages.

If deployment-specific limits are changed, set the same `RAG_MAX_*` values in
both the service environment (Fly.io) and the Next.js environment (Vercel) so
the UI, proxy, and backend enforce one contract.

## Roadmap

- [x] **Phase 0 — Scaffolding**: workspace layout, README with architecture.
- [x] **Phase 1 — Vertical slice**: minimal HNSW supporting insert + search over random vectors, correctness-tested against brute force.
- [x] **Phase 2 — Iteration**: benchmarks vs. brute-force search (real numbers in this README, not invented ones), recall tests, parameter tuning (`M`, `ef_construction`).
- [x] **Phase 3 — Bindings + real embeddings**: PyO3 bindings, real embedding model, document chunking.
- [x] **Phase 4 — Full RAG**: Claude API integration with cited sources, Next.js app.
- [x] **Phase 5 — Release**: `v0.1.0`, benchmark table (latency/recall vs. brute force).

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
- **Keyless mock mode in the RAG service**: the demo answers questions with an extractive fallback when no `ANTHROPIC_API_KEY` is set, and the embeddings layer falls back to a deterministic hashed embedder when `sentence-transformers` isn't installed. Both keep the end-to-end app — and its tests — runnable anywhere, with real Claude answers and real embeddings one env var / one `pip install` away.

## Tests

| Layer | Command | Count |
|-------|---------|-------|
| Rust engine | `cargo test` | 24 (unit + seeded recall vs. brute force + doctest) |
| Python bindings + helpers | `pytest` in `bindings/` | 21 (FFI surface, atomic batches, validation, chunking, embeddings, E2E retrieval) |
| RAG service | `PYTHONPATH=. pytest` in `service/` | 36 (keyless E2E, hardening limits, CORS, Claude timeout, relevance filtering, citation parsing, startup seeding) |
| Next.js app | `npm run build` | type-checked production build |

CI (`.github/workflows/ci.yml`) runs all four on every push, in parallel jobs: `cargo fmt --check` + `cargo clippy -- -D warnings` + `cargo test`, the bindings suite, the service suite, and the app build.

The bindings job installs with `pip install ./bindings` rather than `maturin develop`. That is deliberate: `maturin develop` installs *editable*, which would mask whether the built wheel actually ships the pure-Python `hnsw_rag` package alongside the compiled module. Installing the real wheel is what catches that.

## License

MIT.
