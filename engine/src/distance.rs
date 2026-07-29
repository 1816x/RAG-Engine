//! Distance metrics.
//!
//! Internally the index works with *comparison* distances: squared L2 for
//! Euclidean (monotonic in true L2, saves a sqrt per comparison) and
//! `1 - dot` for Cosine over vectors that were L2-normalized on the way in.
//! User-facing results are converted back to true L2 / cosine distance.

/// Distance metric for an index. Chosen at construction and fixed for the
/// index's lifetime — vectors are preprocessed per-metric on insert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine distance, `1 - cos(a, b)`. Vectors are L2-normalized on insert
    /// and queries on search, so internally this is `1 - dot`.
    Cosine,
}

pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// L2-normalize in place. Zero vectors are left untouched (their cosine
/// distance to anything is ill-defined either way).
pub(crate) fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_basic() {
        assert_eq!(l2_sq(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert_eq!(l2_sq(&[1.0, 1.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn normalize_unit_norm() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((dot(&v, &v).sqrt() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_vector_noop() {
        let mut v = vec![0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }
}
