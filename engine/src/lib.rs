//! `hnsw-engine`: an HNSW (Hierarchical Navigable Small World) vector index
//! written from scratch, plus a brute-force baseline used for correctness
//! tests and benchmarks.
//!
//! Zero runtime dependencies by design — the index is the point of the
//! project, not the glue around it.
//!
//! ```
//! use hnsw_engine::{Hnsw, HnswParams, Metric};
//!
//! let mut index = Hnsw::new(3, Metric::Cosine, HnswParams::default());
//! index.insert(vec![1.0, 0.0, 0.0]).unwrap();
//! index.insert(vec![0.0, 1.0, 0.0]).unwrap();
//! let hits = index.search(&[0.9, 0.1, 0.0], 1, 10).unwrap();
//! assert_eq!(hits[0].id, 0);
//! ```

pub mod brute;
mod distance;
mod hnsw;
pub mod rng;

pub use distance::Metric;
pub use hnsw::{Error, Hnsw, HnswParams, Neighbor};
