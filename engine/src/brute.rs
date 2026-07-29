//! Exact brute-force k-NN — the baseline HNSW is measured against.
//!
//! O(n) per query and exact by construction, which makes it both the recall
//! ground truth in tests and the latency baseline in benchmarks.

use crate::distance;
use crate::{Metric, Neighbor};

/// Exact k nearest neighbors of `query` over `vectors`, closest first.
/// Distances match [`crate::Hnsw::search`]: true L2 for Euclidean,
/// `1 - cos` for Cosine (computed from raw vectors, no pre-normalization
/// required).
pub fn search(vectors: &[Vec<f32>], query: &[f32], k: usize, metric: Metric) -> Vec<Neighbor> {
    let mut hits: Vec<Neighbor> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| Neighbor {
            id: i as u32,
            distance: exact_distance(query, v, metric),
        })
        .collect();
    hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    hits.truncate(k);
    hits
}

fn exact_distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Euclidean => distance::l2_sq(a, b).max(0.0).sqrt(),
        Metric::Cosine => {
            let na = distance::dot(a, a).sqrt();
            let nb = distance::dot(b, b).sqrt();
            if na == 0.0 || nb == 0.0 {
                return 1.0;
            }
            1.0 - distance::dot(a, b) / (na * nb)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ordering() {
        let vectors = vec![vec![10.0f32], vec![1.0], vec![5.0]];
        let hits = search(&vectors, &[0.0], 3, Metric::Euclidean);
        let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids, vec![1, 2, 0]);
        assert_eq!(hits[0].distance, 1.0);
    }

    #[test]
    fn truncates_to_k() {
        let vectors = vec![vec![0.0f32]; 10];
        assert_eq!(search(&vectors, &[0.0], 3, Metric::Euclidean).len(), 3);
    }
}
