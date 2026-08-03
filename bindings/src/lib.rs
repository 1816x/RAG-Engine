//! Python bindings for the `hnsw-engine` crate.
//!
//! Exposes a single `Hnsw` class mirroring the Rust API. Vectors cross the
//! boundary as plain Python sequences of floats — no NumPy dependency, which
//! keeps the module tiny; convert with `list(array)` if you have arrays.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use hnsw_engine as engine;

/// When `ef_search` is not given, use `max(4 * k, 50)` — comfortably above
/// the knee of the recall curve for typical corpus sizes (see the README
/// benchmark table) while staying far cheaper than brute force.
fn default_ef(k: usize) -> usize {
    (4 * k).max(50)
}

fn parse_metric(metric: &str) -> PyResult<engine::Metric> {
    match metric {
        "cosine" => Ok(engine::Metric::Cosine),
        "euclidean" | "l2" => Ok(engine::Metric::Euclidean),
        other => Err(PyValueError::new_err(format!(
            "unknown metric {other:?}; expected \"cosine\" or \"euclidean\""
        ))),
    }
}

/// An HNSW approximate-nearest-neighbor index over `dim`-dimensional vectors.
#[pyclass]
struct Hnsw {
    inner: engine::Hnsw,
}

#[pymethods]
impl Hnsw {
    #[new]
    #[pyo3(signature = (dim, metric="cosine", m=16, ef_construction=200, seed=0x5EED))]
    fn new(
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be > 0"));
        }
        if m < 2 {
            return Err(PyValueError::new_err("m must be >= 2"));
        }
        if ef_construction == 0 {
            return Err(PyValueError::new_err("ef_construction must be >= 1"));
        }
        Ok(Self {
            inner: engine::Hnsw::new(
                dim,
                metric,
                engine::HnswParams {
                    m,
                    ef_construction,
                    seed,
                },
            ),
        })
    }

    /// Insert one vector; returns its id (dense, insertion order: 0, 1, 2…).
    fn insert(&mut self, vector: Vec<f32>) -> PyResult<u32> {
        self.inner
            .insert(vector)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Insert many vectors; returns their ids.
    fn insert_batch(&mut self, vectors: Vec<Vec<f32>>) -> PyResult<Vec<u32>> {
        // Validate the complete batch before mutating the index. Otherwise a
        // bad vector halfway through would leave earlier vectors inserted even
        // though Python receives an exception for the overall operation.
        for (batch_index, vector) in vectors.iter().enumerate() {
            if vector.len() != self.inner.dim() {
                return Err(PyValueError::new_err(format!(
                    "batch vector {batch_index}: dimension mismatch: index holds {}-d vectors, got {}-d",
                    self.inner.dim(),
                    vector.len()
                )));
            }
            if let Some(position) = vector.iter().position(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(format!(
                    "batch vector {batch_index} contains a non-finite value at position {position}"
                )));
            }
        }
        vectors.into_iter().map(|v| self.insert(v)).collect()
    }

    /// Return up to `k` nearest neighbors as `(id, distance)` tuples, closest
    /// first. `ef_search` (beam width) defaults to `max(4 * k, 50)`; raise it
    /// for better recall, lower it for speed.
    #[pyo3(signature = (query, k=10, ef_search=None))]
    fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<Vec<(u32, f32)>> {
        let ef = ef_search.unwrap_or_else(|| default_ef(k));
        self.inner
            .search(&query, k, ef)
            .map(|hits| hits.into_iter().map(|n| (n.id, n.distance)).collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The stored vector for `id` (L2-normalized if the metric is cosine).
    fn vector(&self, id: u32) -> PyResult<Vec<f32>> {
        self.inner
            .vector(id)
            .map(|s| s.to_vec())
            .ok_or_else(|| PyIndexError::new_err(format!("no vector with id {id}")))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn metric(&self) -> &'static str {
        match self.inner.metric() {
            engine::Metric::Cosine => "cosine",
            engine::Metric::Euclidean => "euclidean",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Hnsw(dim={}, metric=\"{}\", len={})",
            self.inner.dim(),
            self.metric(),
            self.inner.len()
        )
    }
}

/// Native module. Re-exported by the `hnsw_engine` Python package so users
/// write `from hnsw_engine import Hnsw`, not the private name.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Hnsw>()?;
    Ok(())
}
