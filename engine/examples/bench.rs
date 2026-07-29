//! Benchmark: HNSW vs. brute-force search on random vectors.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example bench
//! ```
//!
//! Reports build time, per-query latency, and recall@10 for several values
//! of `ef_search`, next to the exact brute-force baseline. Fully seeded, so
//! numbers vary only with hardware.

use std::time::Instant;

use hnsw_engine::rng::SplitMix64;
use hnsw_engine::{brute, Hnsw, HnswParams, Metric};

const N_QUERIES: usize = 200;
const K: usize = 10;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn uniform_vectors(rng: &mut SplitMix64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect()
}

/// Mixture of Gaussian-ish clusters — much closer to how real embedding
/// vectors are distributed than i.i.d. uniform noise (embeddings live on a
/// low-dimensional manifold; uniform high-dim noise has near-identical
/// pairwise distances, which no proximity structure can exploit).
fn clustered_vectors(
    rng: &mut SplitMix64,
    n: usize,
    dim: usize,
    n_clusters: usize,
) -> Vec<Vec<f32>> {
    let centers = uniform_vectors(rng, n_clusters, dim);
    (0..n)
        .map(|_| {
            let c = &centers[(rng.next_u64() as usize) % n_clusters];
            // sum of 4 uniforms ~ Gaussian-ish noise around the center
            (0..dim)
                .map(|j| {
                    let noise: f32 = (0..4).map(|_| rng.next_f32() - 0.5).sum::<f32>() * 0.1;
                    c[j] + noise
                })
                .collect()
        })
        .collect()
}

fn main() {
    // Defaults chosen to look like a real RAG corpus: 50k chunks with
    // 128-d embeddings, grouped into topical clusters.
    let n = env_usize("BENCH_N", 50_000);
    let dim = env_usize("BENCH_DIM", 128);
    let clusters = env_usize("BENCH_CLUSTERS", 1_000); // 0 = uniform random

    let mut rng = SplitMix64::new(0xBE7C);
    let dist_kind = if clusters == 0 {
        "uniform".into()
    } else {
        format!("{clusters} clusters")
    };
    println!("generating {n} random vectors, dim={dim} ({dist_kind}) ...");
    let (vectors, queries);
    if clusters == 0 {
        vectors = uniform_vectors(&mut rng, n, dim);
        queries = uniform_vectors(&mut rng, N_QUERIES, dim);
    } else {
        let mut all = clustered_vectors(&mut rng, n + N_QUERIES, dim, clusters);
        queries = all.split_off(n);
        vectors = all;
    }

    let params = HnswParams {
        m: env_usize("BENCH_M", HnswParams::default().m),
        ef_construction: env_usize("BENCH_EFC", HnswParams::default().ef_construction),
        ..HnswParams::default()
    };
    println!(
        "building HNSW (m={}, ef_construction={}) ...",
        params.m, params.ef_construction
    );
    let t = Instant::now();
    let mut index = Hnsw::new(dim, Metric::Euclidean, params);
    for v in &vectors {
        index.insert(v.clone()).unwrap();
    }
    let build = t.elapsed();
    println!(
        "build: {:.2}s ({:.0} inserts/s)\n",
        build.as_secs_f64(),
        n as f64 / build.as_secs_f64()
    );

    // Exact ground truth + brute-force latency baseline.
    let t = Instant::now();
    let truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            brute::search(&vectors, q, K, Metric::Euclidean)
                .iter()
                .map(|n| n.id)
                .collect()
        })
        .collect();
    let brute_total = t.elapsed();
    let brute_per_query_us = brute_total.as_micros() as f64 / N_QUERIES as f64;

    println!("| method       | ef  | avg latency/query | recall@{K} | speedup vs brute |");
    println!("|--------------|-----|-------------------|-----------|------------------|");
    println!(
        "| brute force  |  —  | {:>14.1} µs |    1.0000 |             1.0x |",
        brute_per_query_us
    );

    for ef in [10, 25, 50, 100, 200] {
        let t = Instant::now();
        let results: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| {
                index
                    .search(q, K, ef)
                    .unwrap()
                    .iter()
                    .map(|n| n.id)
                    .collect()
            })
            .collect();
        let elapsed = t.elapsed();
        let per_query_us = elapsed.as_micros() as f64 / N_QUERIES as f64;

        let recall: f64 = results
            .iter()
            .zip(&truth)
            .map(|(got, tr)| got.iter().filter(|id| tr.contains(id)).count() as f64 / K as f64)
            .sum::<f64>()
            / N_QUERIES as f64;

        println!(
            "| hnsw         | {ef:>3} | {:>14.1} µs | {:>9.4} | {:>15.1}x |",
            per_query_us,
            recall,
            brute_per_query_us / per_query_us
        );
    }
}
