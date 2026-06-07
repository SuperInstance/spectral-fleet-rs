//! Power iteration with deflation for top-k eigenpairs.
//!
//! Finds the `k` largest-magnitude eigenvalues and corresponding eigenvectors
//! of a symmetric matrix using the power method, with Hotelling deflation
//! to remove found components.

use crate::{axpy, dot, normalize, Real};
use rand::Rng;
use std::fmt;

/// Errors during power iteration.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerIterError {
    DidNotConverge { eigenpair: usize },
    ZeroVector,
}

impl fmt::Display for PowerIterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PowerIterError::DidNotConverge { eigenpair } => {
                write!(f, "power iteration did not converge for eigenpair {}", eigenpair)
            }
            PowerIterError::ZeroVector => write!(f, "zero vector encountered"),
        }
    }
}

impl std::error::Error for PowerIterError {}

/// An eigenpair `(eigenvalue, eigenvector)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Eigenpair<S: Real> {
    pub value: S,
    pub vector: Vec<S>,
}

/// A symmetric linear operator.
pub trait Operator<S: Real> {
    fn dimension(&self) -> usize;
    fn apply(&self, x: &[S], y: &mut [S]);
}

/// Simple dense symmetric operator.
pub struct DenseOp<S: Real> {
    pub matrix: Vec<Vec<S>>,
}

impl<S: Real> Operator<S> for DenseOp<S> {
    fn dimension(&self) -> usize {
        self.matrix.len()
    }

    fn apply(&self, x: &[S], y: &mut [S]) {
        let n = self.dimension();
        for i in 0..n {
            y[i] = S::zero();
            for j in 0..n {
                y[i] += self.matrix[i][j] * x[j];
            }
        }
    }
}

/// Run power iteration on `op` starting from `v0`.
///
/// Returns the dominant eigenpair.  The vector is normalized to unit norm.
pub fn power_iterate<S: Real>(
    op: &dyn Operator<S>,
    v0: &[S],
    max_iter: usize,
    tol: S,
) -> Result<Eigenpair<S>, PowerIterError> {
    let n = op.dimension();
    if v0.len() != n {
        return Err(PowerIterError::ZeroVector);
    }

    let mut v = v0.to_vec();
    normalize(&mut v);

    let mut w = vec![S::zero(); n];
    let mut lambda = S::zero();

    for _ in 0..max_iter {
        op.apply(&v, &mut w);
        let new_lambda = dot(&w, &v);
        normalize(&mut w);

        if (new_lambda - lambda).abs() < tol {
            return Ok(Eigenpair {
                value: new_lambda,
                vector: w,
            });
        }

        lambda = new_lambda;
        v = w.clone();
    }

    op.apply(&v, &mut w);
    let final_lambda = dot(&w, &v);
    normalize(&mut w);

    Ok(Eigenpair {
        value: final_lambda,
        vector: w,
    })
}

/// Find the top `k` eigenpairs using power iteration with Hotelling deflation.
///
/// After finding an eigenpair `(λ, v)`, the operator is conceptually deflated:
/// `A ← A − λ * v * v^T`.  For symmetric matrices this removes the component
/// along `v` so the next dominant eigenpair can be found.
pub fn top_k_eigenpairs<S: Real>(
    op: &dyn Operator<S>,
    k: usize,
    max_iter: usize,
    tol: S,
    rng: &mut dyn rand::RngCore,
) -> Result<Vec<Eigenpair<S>>, PowerIterError> {
    let n = op.dimension();
    let mut pairs: Vec<Eigenpair<S>> = Vec::with_capacity(k);

    for idx in 0..k {
        // Random start vector
        let mut v0: Vec<S> = (0..n)
            .map(|_| S::from(rng.gen::<f64>()).unwrap())
            .collect();

        // Deflate previously found eigenvectors
        for pair in &pairs {
            let proj = dot(&v0, &pair.vector);
            axpy(-proj, &pair.vector, &mut v0);
        }
        normalize(&mut v0);

        let mut v = v0.clone();
        let mut w = vec![S::zero(); n];
        let mut lambda = S::zero();
        let mut converged = false;

        for _ in 0..max_iter {
            op.apply(&v, &mut w);
            let new_lambda = dot(&w, &v);

            // Orthogonalize against found eigenvectors (numerically stable)
            for pair in &pairs {
                let proj = dot(&w, &pair.vector);
                axpy(-proj, &pair.vector, &mut w);
            }

            normalize(&mut w);

            if (new_lambda - lambda).abs() < tol {
                lambda = new_lambda;
                v = w;
                converged = true;
                break;
            }
            lambda = new_lambda;
            v = w.clone();
        }

        if !converged {
            return Err(PowerIterError::DidNotConverge { eigenpair: idx });
        }

        pairs.push(Eigenpair {
            value: lambda,
            vector: v,
        });
    }

    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn power_iteration_diagonal() {
        let op = DenseOp {
            matrix: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 5.0, 0.0],
                vec![0.0, 0.0, 2.0],
            ],
        };
        let mut rng = StdRng::seed_from_u64(42);
        let v0: Vec<f64> = (0..3).map(|_| rng.gen()).collect();
        let pair = power_iterate(&op, &v0, 100, 1e-10).unwrap();
        assert_relative_eq!(pair.value, 5.0, epsilon = 1e-8);
    }

    #[test]
    fn top_3_eigenpairs_diagonal() {
        let op = DenseOp {
            matrix: vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 5.0, 0.0, 0.0],
                vec![0.0, 0.0, 2.0, 0.0],
                vec![0.0, 0.0, 0.0, 8.0],
            ],
        };
        let mut rng = StdRng::seed_from_u64(123);
        let pairs = top_k_eigenpairs(&op, 3, 200, 1e-10, &mut rng).unwrap();
        let mut values: Vec<f64> = pairs.iter().map(|p| p.value).collect();
        values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_relative_eq!(values[0], 8.0, epsilon = 1e-6);
        assert_relative_eq!(values[1], 5.0, epsilon = 1e-6);
        assert_relative_eq!(values[2], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn eigenvectors_orthogonal() {
        let op = DenseOp {
            matrix: vec![
                vec![2.0, 1.0, 0.0],
                vec![1.0, 3.0, 1.0],
                vec![0.0, 1.0, 4.0],
            ],
        };
        let mut rng = StdRng::seed_from_u64(7);
        let pairs = top_k_eigenpairs(&op, 3, 200, 1e-10, &mut rng).unwrap();
        for i in 0..3 {
            for j in i + 1..3 {
                let d = dot(&pairs[i].vector, &pairs[j].vector);
                assert_relative_eq!(d, 0.0, epsilon = 1e-6);
            }
        }
    }
}
