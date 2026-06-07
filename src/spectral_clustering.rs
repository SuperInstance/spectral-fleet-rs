//! Spectral clustering using the normalized graph Laplacian.
//!
//! Algorithm (Ng–Jordan–Weiss):
//!
//! 1. Form affinity matrix `W`.
//! 2. Compute degree matrix `D`.
//! 3. Compute symmetric normalized Laplacian `L_sym = I − D^{−½} W D^{−½}`.
//! 4. Find the `k` smallest eigenvectors of `L_sym`.
//! 5. Form matrix `U` from those eigenvectors as columns.
//! 6. Normalize each row of `U` to unit length.
//! 7. Run k-means on the rows of `U`.

use crate::kmeans::kmeans;
use crate::power_iteration::{top_k_eigenpairs, DenseOp};
use crate::normalize;
use rand::RngCore;
use std::fmt;

/// Errors during spectral clustering.
#[derive(Debug, Clone, PartialEq)]
pub enum SpectralError {
    KMeansError(crate::kmeans::KMeansError),
    PowerIterError(crate::power_iteration::PowerIterError),
    InvalidAffinity,
}

impl fmt::Display for SpectralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpectralError::KMeansError(e) => write!(f, "k-means failed: {}", e),
            SpectralError::PowerIterError(e) => write!(f, "eigen computation failed: {}", e),
            SpectralError::InvalidAffinity => write!(f, "affinity matrix is invalid"),
        }
    }
}

impl std::error::Error for SpectralError {}

impl From<crate::kmeans::KMeansError> for SpectralError {
    fn from(e: crate::kmeans::KMeansError) -> Self {
        SpectralError::KMeansError(e)
    }
}

impl From<crate::power_iteration::PowerIterError> for SpectralError {
    fn from(e: crate::power_iteration::PowerIterError) -> Self {
        SpectralError::PowerIterError(e)
    }
}

/// Result of spectral clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralResult {
    /// Cluster label for each original point.
    pub labels: Vec<usize>,
    /// The embedding matrix (rows correspond to original points).
    pub embedding: Vec<Vec<f64>>,
}

/// Build a symmetric normalized Laplacian operator from an affinity matrix.
///
/// Returns an operator representing `L_sym = I − D^{−½} W D^{−½}`.
/// We want the *smallest* eigenvalues of `L_sym`, which correspond to the
/// *largest* eigenvalues of `I − L_sym = D^{−½} W D^{−½}`.
fn build_normalized_affinity_operator(affinity: &[Vec<f64>]) -> Result<DenseOp<f64>, SpectralError> {
    let n = affinity.len();
    if n == 0 || affinity.iter().any(|row| row.len() != n) {
        return Err(SpectralError::InvalidAffinity);
    }

    let mut degrees = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            degrees[i] += affinity[i][j];
        }
    }

    let mut d_inv_sqrt = vec![0.0; n];
    for i in 0..n {
        if degrees[i] > 0.0 {
            d_inv_sqrt[i] = 1.0 / degrees[i].sqrt();
        }
    }

    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            matrix[i][j] = d_inv_sqrt[i] * affinity[i][j] * d_inv_sqrt[j];
        }
    }

    Ok(DenseOp { matrix })
}

/// Perform spectral clustering.
///
/// * `affinity` — n×n symmetric affinity / similarity matrix.
/// * `k` — number of clusters.
/// * `max_iter` — max iterations for power method.
/// * `tol` — convergence tolerance.
/// * `rng` — random source.
pub fn spectral_clustering(
    affinity: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut dyn RngCore,
) -> Result<SpectralResult, SpectralError> {
    let n = affinity.len();
    if k == 0 || k > n {
        return Err(SpectralError::InvalidAffinity);
    }

    let op = build_normalized_affinity_operator(affinity)?;

    // Find top-k eigenpairs of D^{-1/2} W D^{-1/2}
    let pairs = top_k_eigenpairs(&op, k, max_iter, tol, rng)?;

    // Build embedding: each row i is [v_0[i], v_1[i], ..., v_{k-1}[i]]
    let mut embedding: Vec<Vec<f64>> = vec![vec![0.0; k]; n];
    for (cidx, pair) in pairs.iter().enumerate() {
        for i in 0..n {
            embedding[i][cidx] = pair.vector[i];
        }
    }

    // Normalize rows to unit length
    for row in embedding.iter_mut() {
        normalize(row);
    }

    // K-means on rows
    let km = kmeans(&embedding, k, 100, rng)?;

    Ok(SpectralResult {
        labels: km.labels,
        embedding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Build an affinity matrix with two clear blocks.
    fn two_block_affinity(n_per_block: usize, intra: f64, inter: f64) -> Vec<Vec<f64>> {
        let n = 2 * n_per_block;
        let mut w = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let same_block = (i < n_per_block && j < n_per_block)
                    || (i >= n_per_block && j >= n_per_block);
                w[i][j] = if same_block { intra } else { inter };
            }
        }
        w
    }

    #[test]
    fn two_clear_clusters() {
        let w = two_block_affinity(10, 1.0, 0.01);
        let mut rng = StdRng::seed_from_u64(99);
        let result = spectral_clustering(&w, 2, 200, 1e-8, &mut rng).unwrap();

        // First block should share a label
        let label_a = result.labels[0];
        for i in 0..10 {
            assert_eq!(result.labels[i], label_a);
        }
        // Second block should share a different label
        let label_b = result.labels[10];
        assert_ne!(label_a, label_b);
        for i in 10..20 {
            assert_eq!(result.labels[i], label_b);
        }
    }

    #[test]
    fn three_clear_clusters() {
        // Build three blocks manually
        let n = 15;
        let mut w = vec![vec![0.01; n]; n];
        for i in 0..5 {
            for j in 0..5 {
                w[i][j] = 1.0;
            }
        }
        for i in 5..10 {
            for j in 5..10 {
                w[i][j] = 1.0;
            }
        }
        for i in 10..15 {
            for j in 10..15 {
                w[i][j] = 1.0;
            }
        }
        let mut rng = StdRng::seed_from_u64(77);
        let result = spectral_clustering(&w, 3, 200, 1e-8, &mut rng).unwrap();

        let l0 = result.labels[0];
        let l1 = result.labels[5];
        let l2 = result.labels[10];
        assert_ne!(l0, l1);
        assert_ne!(l1, l2);
        assert_ne!(l0, l2);

        for i in 0..5 {
            assert_eq!(result.labels[i], l0);
        }
        for i in 5..10 {
            assert_eq!(result.labels[i], l1);
        }
        for i in 10..15 {
            assert_eq!(result.labels[i], l2);
        }
    }
}
