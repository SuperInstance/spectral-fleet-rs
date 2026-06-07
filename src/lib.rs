#![allow(clippy::needless_range_loop)]

//! Spectral methods for fleet matrices.
//!
//! Provides Lanczos iteration for large sparse symmetric matrices,
//! power iteration with deflation for top-k eigenpairs, and
//! spectral clustering with k-means on eigenvectors.

pub mod kmeans;
pub mod lanczos;
pub mod power_iteration;
pub mod spectral_clustering;
pub mod scheduling;

use num_traits::{Float, NumAssign};

/// Trait for real scalars used throughout the crate.
pub trait Real: Float + NumAssign + std::fmt::Debug + Send + Sync + 'static {}
impl<T: Float + NumAssign + std::fmt::Debug + Send + Sync + 'static> Real for T {}

/// Compute the L2 norm of a vector.
pub fn l2_norm<S: Real>(v: &[S]) -> S {
    v.iter().map(|&x| x * x).fold(S::zero(), |a, b| a + b).sqrt()
}

/// Normalize a vector in-place to unit L2 norm.
pub fn normalize<S: Real>(v: &mut [S]) {
    let n = l2_norm(v);
    if n > S::zero() {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Dot product of two vectors.
pub fn dot<S: Real>(a: &[S], b: &[S]) -> S {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).fold(S::zero(), |acc, v| acc + v)
}

/// axpy: y <- a*x + y
pub fn axpy<S: Real>(a: S, x: &[S], y: &mut [S]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// Scale a vector in-place.
pub fn scale<S: Real>(v: &mut [S], s: S) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn norm_and_dot() {
        let v = vec![3.0_f64, 4.0];
        assert_relative_eq!(l2_norm(&v), 5.0, epsilon = 1e-10);
        assert_relative_eq!(dot(&v, &v), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn normalize_basic() {
        let mut v = vec![3.0_f64, 4.0];
        normalize(&mut v);
        assert_relative_eq!(l2_norm(&v), 1.0, epsilon = 1e-10);
        assert_relative_eq!(v[0], 0.6, epsilon = 1e-10);
        assert_relative_eq!(v[1], 0.8, epsilon = 1e-10);
    }
}
