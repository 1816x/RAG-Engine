use hnsw_engine::{Hnsw, HnswParams, Metric, SNAPSHOT_VERSION};

fn params() -> HnswParams {
    HnswParams {
        m: 4,
        ef_construction: 32,
        seed: 17,
    }
}
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn empty_snapshot_round_trip() {
    let index = Hnsw::new(3, Metric::Cosine, params());
    let restored = Hnsw::from_bytes(&index.to_bytes()).unwrap();
    assert_eq!(restored.len(), 0);
    assert_eq!(restored.dim(), 3);
    assert_eq!(restored.metric(), Metric::Cosine);
    assert_eq!(restored.params().m, 4);
}

#[test]
fn populated_metrics_round_trip_with_identical_searches() {
    for metric in [Metric::Cosine, Metric::Euclidean] {
        let mut index = Hnsw::new(3, metric, params());
        for vector in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]] {
            index.insert(vector.to_vec()).unwrap();
        }
        let before = index.search(&[0.9, 0.2, 0.], 4, 20).unwrap();
        let mut restored = Hnsw::from_bytes(&index.to_bytes()).unwrap();
        assert_eq!(restored.len(), index.len());
        assert_eq!(restored.dim(), index.dim());
        assert_eq!(restored.metric(), metric);
        assert_eq!(restored.params().ef_construction, params().ef_construction);
        assert_eq!(restored.search(&[0.9, 0.2, 0.], 4, 20).unwrap(), before);
        assert_eq!(restored.insert(vec![-1., 0., 0.]).unwrap(), 4);
        assert_eq!(restored.search(&[-1., 0., 0.], 1, 20).unwrap()[0].id, 4);
    }
}

#[test]
fn file_api_and_rng_continuation_are_exact() {
    let mut original = Hnsw::new(2, Metric::Euclidean, params());
    original.insert(vec![0., 0.]).unwrap();
    let path = std::env::temp_dir().join(format!("hnsw-snapshot-{}.bin", std::process::id()));
    original.save(&path).unwrap();
    let mut restored = Hnsw::load(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    for vector in [[1., 0.], [2., 0.], [3., 0.]] {
        original.insert(vector.to_vec()).unwrap();
        restored.insert(vector.to_vec()).unwrap();
    }
    assert_eq!(original.to_bytes(), restored.to_bytes());
}

#[test]
fn corrupt_truncated_and_unsupported_snapshots_fail() {
    let bytes = Hnsw::new(2, Metric::Cosine, params()).to_bytes();
    for end in 0..bytes.len() {
        assert!(Hnsw::from_bytes(&bytes[..end]).is_err());
    }
    let mut corrupt = bytes.clone();
    corrupt[20] ^= 1;
    assert!(Hnsw::from_bytes(&corrupt)
        .err()
        .unwrap()
        .to_string()
        .contains("checksum"));
    let mut unsupported = bytes;
    unsupported[8..12].copy_from_slice(&(SNAPSHOT_VERSION + 1).to_le_bytes());
    let len = unsupported.len();
    let sum = checksum(&unsupported[..len - 8]);
    unsupported[len - 8..].copy_from_slice(&sum.to_le_bytes());
    assert!(Hnsw::from_bytes(&unsupported)
        .err()
        .unwrap()
        .to_string()
        .contains("unsupported"));
}
