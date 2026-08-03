//! HNSW (Hierarchical Navigable Small World) index, implemented from the
//! Malkov & Yashunin paper (arXiv:1603.09320).
//!
//! The structure is a stack of proximity graphs. Every vector lives in layer
//! 0; each node is additionally promoted to higher layers with exponentially
//! decaying probability. A query greedily descends from the sparse top layers
//! (long hops across the space) into layer 0, where a beam search of width
//! `ef` collects the nearest neighbors.
//!
//! Performance notes (measured, see README benchmarks):
//! - Vectors live in one contiguous `Vec<f32>` (row stride = dim) rather than
//!   `Vec<Vec<f32>>` — distance evaluation is the hot loop and pointer
//!   chasing there dominates everything else.
//! - The visited set is a reusable bitset, not a `HashSet` — clearing it is
//!   one small memset instead of a rehash, and lookups are branch + mask.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::distance::{self, Metric};
use crate::rng::SplitMix64;

/// Construction parameters.
#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    /// Max links per node on layers > 0 (layer 0 allows `2 * m`). Higher `m`
    /// means better recall and more memory. Typical range: 8–48.
    pub m: usize,
    /// Beam width while building. Higher means better graph quality and
    /// slower inserts. Typical range: 100–500.
    pub ef_construction: usize,
    /// Seed for level assignment; a fixed seed makes builds reproducible.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            seed: 0x5EED,
        }
    }
}

/// A single search hit. `distance` is in the index's metric: true L2 for
/// [`Metric::Euclidean`], `1 - cos` for [`Metric::Cosine`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    DimensionMismatch { expected: usize, got: usize },
    NonFiniteValue { position: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "dimension mismatch: index holds {expected}-d vectors, got {got}-d"
                )
            }
            Error::NonFiniteValue { position } => {
                write!(
                    f,
                    "vector contains a non-finite value at position {position}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// f32 wrapper with total order so distances can live in heaps.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Contiguous vector storage: row `id` lives at `data[id * dim .. (id+1) * dim]`.
struct VectorStore {
    dim: usize,
    metric: Metric,
    data: Vec<f32>,
}

impl VectorStore {
    fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    fn get(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
    }

    fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(v);
    }

    /// Comparison distance: squared L2 for Euclidean (monotonic in true L2),
    /// `1 - dot` for Cosine (vectors are pre-normalized).
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::Euclidean => distance::l2_sq(a, b),
            Metric::Cosine => 1.0 - distance::dot(a, b),
        }
    }

    fn dist_to(&self, q: &[f32], id: u32) -> f32 {
        self.dist(q, self.get(id))
    }
}

/// Reusable visited-bitset. `reset` is one memset of n/8 bytes; the insert
/// path keeps one instance alive across calls, `search` uses a per-call one
/// so it can stay `&self` (and thus be called concurrently).
struct Visited {
    bits: Vec<u64>,
}

impl Visited {
    fn new() -> Self {
        Self { bits: Vec::new() }
    }

    fn reset(&mut self, n: usize) {
        let words = n.div_ceil(64);
        self.bits.clear();
        self.bits.resize(words, 0);
    }

    /// Mark `id` visited; returns true if it was not visited before.
    fn insert(&mut self, id: u32) -> bool {
        let word = (id / 64) as usize;
        let mask = 1u64 << (id % 64);
        let seen = self.bits[word] & mask != 0;
        self.bits[word] |= mask;
        !seen
    }
}

pub struct Hnsw {
    params: HnswParams,
    store: VectorStore,
    /// `links[id][level]` = neighbor ids of `id` at that level. A node's top
    /// level is `links[id].len() - 1`.
    links: Vec<Vec<Vec<u32>>>,
    /// Entry point: the node with the highest level.
    entry: Option<u32>,
    top_level: usize,
    /// Max links on layer 0 (`2 * m`, per the paper).
    m_max0: usize,
    /// `1 / ln(m)` — normalization for the level distribution.
    level_mult: f64,
    rng: SplitMix64,
    /// Scratch for the `&mut self` insert path.
    visited: Visited,
}

impl Hnsw {
    /// Create an empty index for `dim`-dimensional vectors.
    ///
    /// # Panics
    /// If `dim == 0`, `m < 2`, or `ef_construction == 0`.
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(params.m >= 2, "m must be >= 2");
        assert!(params.ef_construction > 0, "ef_construction must be > 0");
        Self {
            store: VectorStore {
                dim,
                metric,
                data: Vec::new(),
            },
            links: Vec::new(),
            entry: None,
            top_level: 0,
            m_max0: params.m * 2,
            level_mult: 1.0 / (params.m as f64).ln(),
            rng: SplitMix64::new(params.seed),
            visited: Visited::new(),
            params,
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.data.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.store.dim
    }

    pub fn metric(&self) -> Metric {
        self.store.metric
    }

    /// The stored vector for `id` (normalized if the metric is Cosine).
    pub fn vector(&self, id: u32) -> Option<&[f32]> {
        if (id as usize) < self.store.len() {
            Some(self.store.get(id))
        } else {
            None
        }
    }

    /// Insert a vector, returning its id. Ids are dense: 0, 1, 2, …
    pub fn insert(&mut self, mut vector: Vec<f32>) -> Result<u32, Error> {
        if vector.len() != self.store.dim {
            return Err(Error::DimensionMismatch {
                expected: self.store.dim,
                got: vector.len(),
            });
        }
        if let Some(position) = vector.iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteValue { position });
        }
        if self.store.metric == Metric::Cosine {
            distance::normalize(&mut vector);
        }

        let id = self.store.len() as u32;
        let level = self.random_level();
        self.store.push(&vector);
        self.links.push(vec![Vec::new(); level + 1]);
        let q = vector; // owned copy doubles as the query — no re-clone

        let Some(entry) = self.entry else {
            self.entry = Some(id);
            self.top_level = level;
            return Ok(id);
        };

        let mut ep = vec![(self.store.dist_to(&q, entry), entry)];

        // Layers above the new node's level: pure greedy descent (beam of 1).
        for lc in (level + 1..=self.top_level).rev() {
            ep = search_layer(&self.store, &self.links, &mut self.visited, &q, ep, 1, lc);
            ep.truncate(1);
        }

        // Layers the new node belongs to: beam-search candidates, pick a
        // diverse subset, and wire links both ways.
        for lc in (0..=level.min(self.top_level)).rev() {
            let found = search_layer(
                &self.store,
                &self.links,
                &mut self.visited,
                &q,
                ep,
                self.params.ef_construction,
                lc,
            );
            let selected = select_neighbors(&self.store, &found, self.params.m);
            let max_links = if lc == 0 { self.m_max0 } else { self.params.m };
            for &(_, e) in &selected {
                self.links[id as usize][lc].push(e);
                let e_links = &mut self.links[e as usize][lc];
                e_links.push(id);
                if e_links.len() > max_links {
                    prune_links(&self.store, e_links, e, max_links);
                }
            }
            ep = found;
        }

        if level > self.top_level {
            self.top_level = level;
            self.entry = Some(id);
        }
        Ok(id)
    }

    /// Return the `k` approximate nearest neighbors of `query`, closest
    /// first. `ef_search` is the layer-0 beam width; it is clamped to at
    /// least `k`. Larger values improve recall at the cost of latency.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<Neighbor>, Error> {
        if query.len() != self.store.dim {
            return Err(Error::DimensionMismatch {
                expected: self.store.dim,
                got: query.len(),
            });
        }
        if let Some(position) = query.iter().position(|value| !value.is_finite()) {
            return Err(Error::NonFiniteValue { position });
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        let normalized;
        let q: &[f32] = if self.store.metric == Metric::Cosine {
            let mut v = query.to_vec();
            distance::normalize(&mut v);
            normalized = v;
            &normalized
        } else {
            query
        };

        let ef = ef_search.max(k);
        let mut visited = Visited::new();
        let mut ep = vec![(self.store.dist_to(q, entry), entry)];
        for lc in (1..=self.top_level).rev() {
            ep = search_layer(&self.store, &self.links, &mut visited, q, ep, 1, lc);
            ep.truncate(1);
        }
        let found = search_layer(&self.store, &self.links, &mut visited, q, ep, ef, 0);
        Ok(found
            .into_iter()
            .take(k)
            .map(|(d, id)| Neighbor {
                id,
                distance: self.report(d),
            })
            .collect())
    }

    /// Convert a comparison distance to the user-facing one.
    fn report(&self, d: f32) -> f32 {
        match self.store.metric {
            Metric::Euclidean => d.max(0.0).sqrt(),
            Metric::Cosine => d,
        }
    }

    /// Sample a level: floor(-ln(U) / ln(m)), the paper's exponential decay.
    fn random_level(&mut self) -> usize {
        let r = self.rng.next_f64().max(f64::MIN_POSITIVE);
        (-r.ln() * self.level_mult) as usize
    }
}

/// Algorithm 2 from the paper: beam search within one layer. Takes entry
/// points as `(comparison_distance, id)` pairs, returns up to `ef` closest
/// nodes, sorted ascending by distance.
///
/// Free function over explicit fields (rather than `&self`) so the insert
/// path can hold disjoint borrows: the store is read while links are being
/// rewired.
fn search_layer(
    store: &VectorStore,
    links: &[Vec<Vec<u32>>],
    visited: &mut Visited,
    q: &[f32],
    entry_points: Vec<(f32, u32)>,
    ef: usize,
    level: usize,
) -> Vec<(f32, u32)> {
    visited.reset(store.len());
    for &(_, id) in &entry_points {
        visited.insert(id);
    }
    // Min-heap of nodes still to expand.
    let mut candidates: BinaryHeap<Reverse<(OrdF32, u32)>> = entry_points
        .iter()
        .map(|&(d, id)| Reverse((OrdF32(d), id)))
        .collect();
    // Max-heap of current best: worst kept result sits on top.
    let mut results: BinaryHeap<(OrdF32, u32)> = entry_points
        .into_iter()
        .map(|(d, id)| (OrdF32(d), id))
        .collect();
    while results.len() > ef {
        results.pop();
    }

    while let Some(Reverse((OrdF32(c_dist), c))) = candidates.pop() {
        let worst = results
            .peek()
            .map(|&(OrdF32(d), _)| d)
            .unwrap_or(f32::INFINITY);
        if c_dist > worst && results.len() >= ef {
            break; // the closest unexpanded node can't improve results
        }
        for &nb in &links[c as usize][level] {
            if !visited.insert(nb) {
                continue;
            }
            let d = store.dist_to(q, nb);
            let worst = results
                .peek()
                .map(|&(OrdF32(w), _)| w)
                .unwrap_or(f32::INFINITY);
            if results.len() < ef || d < worst {
                candidates.push(Reverse((OrdF32(d), nb)));
                results.push((OrdF32(d), nb));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    let mut out: Vec<(f32, u32)> = results.into_iter().map(|(OrdF32(d), id)| (d, id)).collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

/// Algorithm 4 from the paper (the diversity heuristic): walk candidates
/// closest-first and keep one only if it is closer to the query than to
/// every already-kept neighbor. This spreads links across directions
/// instead of clustering them, which is what keeps the graph navigable.
/// Pruned candidates backfill remaining slots (keepPrunedConnections).
fn select_neighbors(store: &VectorStore, candidates: &[(f32, u32)], m: usize) -> Vec<(f32, u32)> {
    if candidates.len() <= m {
        return candidates.to_vec();
    }
    let mut selected: Vec<(f32, u32)> = Vec::with_capacity(m);
    let mut pruned: Vec<(f32, u32)> = Vec::new();
    for &(d, e) in candidates {
        if selected.len() >= m {
            break;
        }
        let ev = store.get(e);
        let diverse = selected
            .iter()
            .all(|&(_, s)| store.dist(ev, store.get(s)) >= d);
        if diverse {
            selected.push((d, e));
        } else {
            pruned.push((d, e));
        }
    }
    for &(d, e) in &pruned {
        if selected.len() >= m {
            break;
        }
        selected.push((d, e));
    }
    selected
}

/// Re-select `node`'s link list down to `max_links`, using the same
/// diversity heuristic as insertion.
fn prune_links(store: &VectorStore, list: &mut Vec<u32>, node: u32, max_links: usize) {
    let nv = store.get(node);
    let mut with_dist: Vec<(f32, u32)> = list
        .iter()
        .map(|&n| (store.dist(nv, store.get(n)), n))
        .collect();
    with_dist.sort_by(|a, b| a.0.total_cmp(&b.0));
    let selected = select_neighbors(store, &with_dist, max_links);
    list.clear();
    list.extend(selected.into_iter().map(|(_, id)| id));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn index_with(vectors: &[Vec<f32>], metric: Metric) -> Hnsw {
        let mut idx = Hnsw::new(vectors[0].len(), metric, HnswParams::default());
        for v in vectors {
            idx.insert(v.clone()).unwrap();
        }
        idx
    }

    #[test]
    fn empty_index_returns_nothing() {
        let idx = Hnsw::new(4, Metric::Euclidean, HnswParams::default());
        assert!(idx.search(&[0.0; 4], 5, 10).unwrap().is_empty());
        assert!(idx.is_empty());
    }

    #[test]
    fn insert_rejects_wrong_dimension() {
        let mut idx = Hnsw::new(4, Metric::Euclidean, HnswParams::default());
        assert_eq!(
            idx.insert(vec![1.0, 2.0]),
            Err(Error::DimensionMismatch {
                expected: 4,
                got: 2
            })
        );
        assert!(idx.search(&[0.0; 3], 1, 10).is_err());
    }

    #[test]
    fn finds_exact_match_first() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
            vec![-3.0, 2.0],
        ];
        let idx = index_with(&vectors, Metric::Euclidean);
        for (i, v) in vectors.iter().enumerate() {
            let hits = idx.search(v, 1, 10).unwrap();
            assert_eq!(hits[0].id, i as u32);
            assert_eq!(hits[0].distance, 0.0);
        }
    }

    #[test]
    fn k_larger_than_index_returns_all() {
        let vectors = vec![vec![0.0f32, 0.0], vec![1.0, 1.0]];
        let idx = index_with(&vectors, Metric::Euclidean);
        let hits = idx.search(&[0.5, 0.5], 10, 10).unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn k_zero_returns_nothing() {
        let idx = index_with(&[vec![0.0f32, 0.0]], Metric::Euclidean);
        assert!(idx.search(&[0.0, 0.0], 0, 10).unwrap().is_empty());
    }

    #[test]
    fn euclidean_distance_is_true_l2() {
        let idx = index_with(&[vec![0.0f32, 0.0]], Metric::Euclidean);
        let hits = idx.search(&[3.0, 4.0], 1, 10).unwrap();
        assert!((hits[0].distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_is_scale_invariant() {
        let vectors = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];
        let idx = index_with(&vectors, Metric::Cosine);
        // Query along [1, 0, 0] at any magnitude must rank id 0 first with
        // distance ~0, then id 2 (45°), then id 1 (90°).
        for scale in [0.001f32, 1.0, 1000.0] {
            let hits = idx.search(&[scale, 0.0, 0.0], 3, 10).unwrap();
            assert_eq!(hits[0].id, 0);
            assert!(hits[0].distance.abs() < 1e-5);
            assert_eq!(hits[1].id, 2);
            assert_eq!(hits[2].id, 1);
        }
    }

    #[test]
    fn duplicate_vectors_are_all_retrievable() {
        let vectors = vec![vec![1.0f32, 1.0]; 5];
        let idx = index_with(&vectors, Metric::Euclidean);
        let hits = idx.search(&[1.0, 1.0], 5, 10).unwrap();
        assert_eq!(hits.len(), 5);
        let ids: HashSet<u32> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn stored_vectors_are_retrievable_by_id() {
        let vectors = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let idx = index_with(&vectors, Metric::Euclidean);
        assert_eq!(idx.vector(0), Some(&[1.0f32, 2.0][..]));
        assert_eq!(idx.vector(1), Some(&[3.0f32, 4.0][..]));
        assert_eq!(idx.vector(2), None);
    }

    #[test]
    fn link_counts_respect_caps() {
        let params = HnswParams {
            m: 4,
            ef_construction: 32,
            seed: 1,
        };
        let mut idx = Hnsw::new(8, Metric::Euclidean, params);
        let mut rng = crate::rng::SplitMix64::new(9);
        for _ in 0..500 {
            let v: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
            idx.insert(v).unwrap();
        }
        for node in &idx.links {
            for (level, nbrs) in node.iter().enumerate() {
                let cap = if level == 0 { idx.m_max0 } else { idx.params.m };
                assert!(nbrs.len() <= cap, "level {level}: {} > {cap}", nbrs.len());
            }
        }
    }
}
