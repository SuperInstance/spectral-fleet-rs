//! Lanczos iteration for large sparse symmetric matrices.
//!
//! The Lanczos algorithm builds an orthonormal basis for the Krylov subspace
//! and a corresponding tridiagonal matrix that approximates the spectrum of
//! the original symmetric operator.

use crate::{axpy, dot, normalize, Real};
use std::fmt;

/// Errors from the Lanczos process.
#[derive(Debug, Clone, PartialEq)]
pub enum LanczosError {
    ZeroStartVector,
    Breakdown { step: usize },
    DimensionMismatch,
}

impl fmt::Display for LanczosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LanczosError::ZeroStartVector => write!(f, "start vector is zero"),
            LanczosError::Breakdown { step } => write!(f, "Lanczos breakdown at step {}", step),
            LanczosError::DimensionMismatch => write!(f, "dimension mismatch in input"),
        }
    }
}

impl std::error::Error for LanczosError {}

/// Result of a Lanczos run: tridiagonal matrix and basis vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct LanczosResult<S: Real> {
    /// Diagonal entries `alpha_1 ... alpha_m`.
    pub alpha: Vec<S>,
    /// Off-diagonal entries `beta_1 ... beta_{m-1}`.
    pub beta: Vec<S>,
    /// Orthonormal Lanczos vectors (each of length `n`).
    pub q: Vec<Vec<S>>,
}

/// A symmetric linear operator `y = A*x`.
pub trait SymmetricOperator<S: Real> {
    fn apply(&self, x: &[S], y: &mut [S]);
    fn dimension(&self) -> usize;
}

/// Simple dense symmetric matrix operator.
pub struct DenseSymmetricOp<S: Real> {
    pub data: Vec<Vec<S>>,
}

impl<S: Real> SymmetricOperator<S> for DenseSymmetricOp<S> {
    fn apply(&self, x: &[S], y: &mut [S]) {
        let n = self.dimension();
        for i in 0..n {
            y[i] = S::zero();
            for j in 0..n {
                y[i] += self.data[i][j] * x[j];
            }
        }
    }

    fn dimension(&self) -> usize {
        self.data.len()
    }
}

/// Run the Lanczos iteration for `m` steps.
///
/// * `op` — symmetric operator.
/// * `q0` — initial vector (will be normalized).
/// * `m` — number of Lanczos steps.
pub fn lanczos<S: Real>(
    op: &dyn SymmetricOperator<S>,
    q0: &[S],
    m: usize,
) -> Result<LanczosResult<S>, LanczosError> {
    let n = op.dimension();
    if q0.len() != n {
        return Err(LanczosError::DimensionMismatch);
    }
    if m == 0 || n == 0 {
        return Ok(LanczosResult {
            alpha: vec![],
            beta: vec![],
            q: vec![],
        });
    }

    let mut q = Vec::with_capacity(m);
    let mut alpha = Vec::with_capacity(m);
    let mut beta = Vec::with_capacity(m.saturating_sub(1));

    let mut q_cur = q0.to_vec();
    normalize(&mut q_cur);
    if l2_norm(&q_cur) < S::epsilon() {
        return Err(LanczosError::ZeroStartVector);
    }

    let mut v = vec![S::zero(); n];
    op.apply(&q_cur, &mut v);
    let a = dot(&v, &q_cur);
    alpha.push(a);
    axpy(-a, &q_cur, &mut v);
    q.push(q_cur.clone());

    for j in 1..m {
        let b = l2_norm(&v);
        if b < S::epsilon() {
            return Err(LanczosError::Breakdown { step: j });
        }
        beta.push(b);

        let mut q_next = v.clone();
        for x in q_next.iter_mut() {
            *x /= b;
        }

        op.apply(&q_next, &mut v);
        let a = dot(&v, &q_next);
        alpha.push(a);

        axpy(-a, &q_next, &mut v);
        axpy(-b, &q_cur, &mut v);

        q.push(q_next.clone());
        q_cur = q_next;
    }

    Ok(LanczosResult { alpha, beta, q })
}

/// Reconstruct the tridiagonal matrix `T` from Lanczos coefficients.
pub fn build_tridiagonal<S: Real>(alpha: &[S], beta: &[S]) -> Vec<Vec<S>> {
    let m = alpha.len();
    let mut t = vec![vec![S::zero(); m]; m];
    for i in 0..m {
        t[i][i] = alpha[i];
    }
    for (i, &b) in beta.iter().enumerate() {
        t[i][i + 1] = b;
        t[i + 1][i] = b;
    }
    t
}

/// Compute the Rayleigh quotient `v^T A v / (v^T v)` for an approximate
/// eigenvector reconstructed from Lanczos basis.
pub fn rayleigh_quotient<S: Real>(op: &dyn SymmetricOperator<S>, v: &[S]) -> S {
    let n = op.dimension();
    let mut av = vec![S::zero(); n];
    op.apply(v, &mut av);
    dot(&av, v) / dot(v, v)
}

use crate::l2_norm;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn lanczos_on_diagonal() {
        let a = DenseSymmetricOp {
            data: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0],
                vec![0.0, 0.0, 3.0],
            ],
        };
        let q0 = vec![1.0_f64, 1.0, 1.0];
        let res = lanczos(&a, &q0, 3).unwrap();
        // For a diagonal matrix started from [1,1,1], Lanczos should
        // recover the exact tridiagonal representation in 3 steps.
        assert_eq!(res.alpha.len(), 3);
        assert_eq!(res.beta.len(), 2);
        // Eigenvalues of the built tridiagonal should match 1,2,3
        let t = build_tridiagonal(&res.alpha, &res.beta);
        // Check trace
        let trace: f64 = t.iter().enumerate().map(|(i, row)| row[i]).sum();
        assert_relative_eq!(trace, 6.0, epsilon = 1e-6);
    }

    #[test]
    fn lanczos_approximates_extreme_eigenvalue() {
        // Build a symmetric matrix with known largest eigenvalue
        let n = 20;
        let mut data = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            data[i][i] = (i + 1) as f64;
        }
        // Add some off-diagonal coupling
        for i in 0..n - 1 {
            data[i][i + 1] = 0.5;
            data[i + 1][i] = 0.5;
        }
        let a = DenseSymmetricOp { data };
        let q0: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 0.1).collect();
        let res = lanczos(&a, &q0, 5).unwrap();
        let t = build_tridiagonal(&res.alpha, &res.beta);
        // Power iterate on T to get approximate largest eigenvalue
        let mut v = vec![1.0_f64; 5];
        normalize(&mut v);
        for _ in 0..100 {
            let mut new_v = vec![0.0; 5];
            for i in 0..5 {
                for j in 0..5 {
                    new_v[i] += t[i][j] * v[j];
                }
            }
            normalize(&mut new_v);
            v = new_v;
        }
        let lambda = rayleigh_quotient(
            &DenseSymmetricOp { data: t },
            &v,
        );
        // Should be close to ~20 (the largest diagonal entry, slightly perturbed)
        assert!(lambda > 18.5, "expected lambda > 18.5, got {}", lambda);
    }

    #[test]
    fn zero_vector_fails() {
        let a = DenseSymmetricOp {
            data: vec![vec![1.0_f64]],
        };
        let q0 = vec![0.0_f64];
        assert_eq!(lanczos(&a, &q0, 1).unwrap_err(), LanczosError::ZeroStartVector);
    }
}
