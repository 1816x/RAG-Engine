//! Correctness of the index as a whole: recall@k measured against exact
//! brute-force ground truth on random vectors. Everything is seeded, so
//! these are exact, reproducible assertions — not flaky statistical ones.

use hnsw_engine::rng::SplitMix64;
use hnsw_engine::{brute, Hnsw, HnswParams, Metric};

fn random_vectors(rng: &mut SplitMix64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect()
}

/// Mean recall@k of the index over `queries`, against brute-force truth.
fn measure_recall(
    index: &Hnsw,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
    metric: Metric,
) -> f64 {
    let mut total = 0.0;
    for q in queries {
        let truth: Vec<u32> = brute::search(vectors, q, k, metric)
            .iter()
            .map(|n| n.id)
            .collect();
        let got: Vec<u32> = index
            .search(q, k, ef)
            .unwrap()
            .iter()
            .map(|n| n.id)
            .collect();
        let hits = got.iter().filter(|id| truth.contains(id)).count();
        total += hits as f64 / k as f64;
    }
    total / queries.len() as f64
}

#[test]
fn recall_euclidean_random_vectors() {
    let mut rng = SplitMix64::new(2024);
    let vectors = random_vectors(&mut rng, 2000, 32);
    let queries = random_vectors(&mut rng, 50, 32);

    let mut index = Hnsw::new(32, Metric::Euclidean, HnswParams::default());
    for v in &vectors {
        index.insert(v.clone()).unwrap();
    }

    let recall = measure_recall(&index, &vectors, &queries, 10, 100, Metric::Euclidean);
    println!("euclidean recall@10 (ef=100): {recall:.4}");
    assert!(recall >= 0.95, "recall@10 too low: {recall:.4}");
}

#[test]
fn recall_cosine_random_vectors() {
    let mut rng = SplitMix64::new(7777);
    let vectors = random_vectors(&mut rng, 2000, 32);
    let queries = random_vectors(&mut rng, 50, 32);

    let mut index = Hnsw::new(32, Metric::Cosine, HnswParams::default());
    for v in &vectors {
        index.insert(v.clone()).unwrap();
    }

    let recall = measure_recall(&index, &vectors, &queries, 10, 100, Metric::Cosine);
    println!("cosine recall@10 (ef=100): {recall:.4}");
    assert!(recall >= 0.95, "recall@10 too low: {recall:.4}");
}

#[test]
fn higher_ef_does_not_hurt_recall() {
    let mut rng = SplitMix64::new(31337);
    let vectors = random_vectors(&mut rng, 1000, 16);
    let queries = random_vectors(&mut rng, 30, 16);

    let mut index = Hnsw::new(16, Metric::Euclidean, HnswParams::default());
    for v in &vectors {
        index.insert(v.clone()).unwrap();
    }

    let r_low = measure_recall(&index, &vectors, &queries, 10, 10, Metric::Euclidean);
    let r_high = measure_recall(&index, &vectors, &queries, 10, 200, Metric::Euclidean);
    println!("recall@10: ef=10 -> {r_low:.4}, ef=200 -> {r_high:.4}");
    assert!(r_high >= r_low);
    assert!(r_high >= 0.99, "ef=200 on 1k vectors should be near-exact: {r_high:.4}");
}

#[test]
fn same_seed_builds_identical_index() {
    let mut rng = SplitMix64::new(555);
    let vectors = random_vectors(&mut rng, 300, 8);
    let queries = random_vectors(&mut rng, 10, 8);

    let build = || {
        let mut idx = Hnsw::new(8, Metric::Euclidean, HnswParams::default());
        for v in &vectors {
            idx.insert(v.clone()).unwrap();
        }
        idx
    };
    let (a, b) = (build(), build());
    for q in &queries {
        assert_eq!(a.search(q, 5, 50).unwrap(), b.search(q, 5, 50).unwrap());
    }
}
