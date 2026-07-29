//! `hnsw-engine`: an HNSW (Hierarchical Navigable Small World) vector index
//! written from scratch, plus a brute-force baseline used for correctness
//! tests and benchmarks.
//!
//! Zero runtime dependencies by design — the index is the point of the
//! project, not the glue around it.
//!
//! Phase 1 lands the index itself (insert + layered search).
