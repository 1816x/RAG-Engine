//! HNSW (Hierarchical Navigable Small World) index, implemented from the
//! Malkov & Yashunin paper (arXiv:1603.09320).
//!
//! The structure is a stack of proximity graphs. Every vector lives in layer
//! 0; each node is additionally promoted to higher layers with exponentially
//! decaying probability. A query greedily descends from the sparse top layers
//! (long hops across the space) into layer 0, where a beam search of width
//! `ef` collects the nearest neighbors.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

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
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: index holds {expected}-d vectors, got {got}-d")
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

pub struct Hnsw {
    params: HnswParams,
    metric: Metric,
    dim: usize,
    /// Vector data, indexed by id (ids are dense, assigned in insert order).
    vectors: Vec<Vec<f32>>,
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
            metric,
            dim,
            vectors: Vec::new(),
            links: Vec::new(),
            entry: None,
            top_level: 0,
            m_max0: params.m * 2,
            level_mult: 1.0 / (params.m as f64).ln(),
            rng: SplitMix64::new(params.seed),
            params,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The stored vector for `id` (normalized if the metric is Cosine).
    pub fn vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(id as usize).map(|v| v.as_slice())
    }

    /// Insert a vector, returning its id. Ids are dense: 0, 1, 2, …
    pub fn insert(&mut self, mut vector: Vec<f32>) -> Result<u32, Error> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.metric == Metric::Cosine {
            distance::normalize(&mut vector);
        }

        let id = self.vectors.len() as u32;
        let level = self.random_level();
        let q = vector.clone();
        self.vectors.push(vector);
        self.links.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            self.entry = Some(id);
            self.top_level = level;
            return Ok(id);
        };

        let mut ep = vec![(self.dist_to(&q, entry), entry)];

        // Layers above the new node's level: pure greedy descent (beam of 1).
        for lc in (level + 1..=self.top_level).rev() {
            ep = self.search_layer(&q, ep, 1, lc);
            ep.truncate(1);
        }

        // Layers the new node belongs to: beam-search candidates, pick a
        // diverse subset, and wire links both ways.
        for lc in (0..=level.min(self.top_level)).rev() {
            let found = self.search_layer(&q, ep, self.params.ef_construction, lc);
            let selected = self.select_neighbors(&found, self.params.m);
            let max_links = if lc == 0 { self.m_max0 } else { self.params.m };
            for &(_, e) in &selected {
                self.links[id as usize][lc].push(e);
                self.links[e as usize][lc].push(id);
                if self.links[e as usize][lc].len() > max_links {
                    self.prune_links(e, lc, max_links);
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
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<Neighbor>, Error> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };
        if k == 0 {
            return Ok(Vec::new());
        }

        let q: Vec<f32>;
        let q = if self.metric == Metric::Cosine {
            let mut v = query.to_vec();
            distance::normalize(&mut v);
            q = v;
            &q[..]
        } else {
            query
        };

        let ef = ef_search.max(k);
        let mut ep = vec![(self.dist_to(q, entry), entry)];
        for lc in (1..=self.top_level).rev() {
            ep = self.search_layer(q, ep, 1, lc);
            ep.truncate(1);
        }
        let found = self.search_layer(q, ep, ef, 0);
        Ok(found
            .into_iter()
            .take(k)
            .map(|(d, id)| Neighbor {
                id,
                distance: self.report(d),
            })
            .collect())
    }

    /// Comparison distance (squared L2 / `1 - dot`) — see `distance` module.
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            Metric::Euclidean => distance::l2_sq(a, b),
            Metric::Cosine => 1.0 - distance::dot(a, b),
        }
    }

    fn dist_to(&self, q: &[f32], id: u32) -> f32 {
        self.dist(q, &self.vectors[id as usize])
    }

    /// Convert a comparison distance to the user-facing one.
    fn report(&self, d: f32) -> f32 {
        match self.metric {
            Metric::Euclidean => d.max(0.0).sqrt(),
            Metric::Cosine => d,
        }
    }

    /// Sample a level: floor(-ln(U) / ln(m)), the paper's exponential decay.
    fn random_level(&mut self) -> usize {
        let r = self.rng.next_f64().max(f64::MIN_POSITIVE);
        (-r.ln() * self.level_mult) as usize
    }

    /// Algorithm 2 from the paper: beam search within one layer. Takes entry
    /// points as `(comparison_distance, id)` pairs, returns up to `ef`
    /// closest nodes, sorted ascending by distance.
    fn search_layer(
        &self,
        q: &[f32],
        entry_points: Vec<(f32, u32)>,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, u32)> {
        let mut visited: HashSet<u32> = entry_points.iter().map(|&(_, id)| id).collect();
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
            let worst = results.peek().map(|&(OrdF32(d), _)| d).unwrap_or(f32::INFINITY);
            if c_dist > worst && results.len() >= ef {
                break; // the closest unexpanded node can't improve results
            }
            for &nb in &self.links[c as usize][level] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = self.dist_to(q, nb);
                let worst = results.peek().map(|&(OrdF32(w), _)| w).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(Reverse((OrdF32(d), nb)));
                    results.push((OrdF32(d), nb));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(f32, u32)> = results
            .into_iter()
            .map(|(OrdF32(d), id)| (d, id))
            .collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// Algorithm 4 from the paper (the diversity heuristic): walk candidates
    /// closest-first and keep one only if it is closer to the query than to
    /// every already-kept neighbor. This spreads links across directions
    /// instead of clustering them, which is what keeps the graph navigable.
    /// Pruned candidates backfill remaining slots (keepPrunedConnections).
    fn select_neighbors(&self, candidates: &[(f32, u32)], m: usize) -> Vec<(f32, u32)> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }
        let mut selected: Vec<(f32, u32)> = Vec::with_capacity(m);
        let mut pruned: Vec<(f32, u32)> = Vec::new();
        for &(d, e) in candidates {
            if selected.len() >= m {
                break;
            }
            let ev = &self.vectors[e as usize];
            let diverse = selected
                .iter()
                .all(|&(_, s)| self.dist(ev, &self.vectors[s as usize]) >= d);
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

    /// Re-select `node`'s links at `level` down to `max_links`, using the
    /// same diversity heuristic as insertion.
    fn prune_links(&mut self, node: u32, level: usize, max_links: usize) {
        let nv = self.vectors[node as usize].clone();
        let mut with_dist: Vec<(f32, u32)> = self.links[node as usize][level]
            .iter()
            .map(|&n| (self.dist(&nv, &self.vectors[n as usize]), n))
            .collect();
        with_dist.sort_by(|a, b| a.0.total_cmp(&b.0));
        let selected = self.select_neighbors(&with_dist, max_links);
        self.links[node as usize][level] = selected.into_iter().map(|(_, id)| id).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Err(Error::DimensionMismatch { expected: 4, got: 2 })
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
    fn link_counts_respect_caps() {
        let params = HnswParams { m: 4, ef_construction: 32, seed: 1 };
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
