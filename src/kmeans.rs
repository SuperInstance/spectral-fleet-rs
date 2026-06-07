//! K-means clustering on vector data.
//!
//! Used by spectral clustering to group points in the eigenvector embedding.

use crate::Real;
use rand::{Rng, RngCore};
use std::fmt;

/// K-means errors.
#[derive(Debug, Clone, PartialEq)]
pub enum KMeansError {
    EmptyCluster { cluster: usize },
    MoreClustersThanPoints,
}

impl fmt::Display for KMeansError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KMeansError::EmptyCluster { cluster } => write!(f, "cluster {} became empty", cluster),
            KMeansError::MoreClustersThanPoints => {
                write!(f, "k is larger than number of data points")
            }
        }
    }
}

impl std::error::Error for KMeansError {}

/// Assignment of points to clusters and cluster centroids.
#[derive(Debug, Clone, PartialEq)]
pub struct KMeansResult<S: Real> {
    /// Cluster index for each point.
    pub labels: Vec<usize>,
    /// Centroid vectors (length `k`, each of dimension `d`).
    pub centroids: Vec<Vec<S>>,
    /// Inertia (sum of squared distances to nearest centroid).
    pub inertia: S,
}

/// Compute squared Euclidean distance between two vectors.
pub fn squared_euclidean<S: Real>(a: &[S], b: &[S]) -> S {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .fold(S::zero(), |acc, v| acc + v)
}

/// Run Lloyd's algorithm for k-means.
///
/// * `data` — list of `d`-dimensional points.
/// * `k` — number of clusters.
/// * `max_iter` — maximum iterations.
/// * `rng` — random source for centroid initialization (k-means++ style).
pub fn kmeans<S: Real>(
    data: &[Vec<S>],
    k: usize,
    max_iter: usize,
    rng: &mut dyn RngCore,
) -> Result<KMeansResult<S>, KMeansError> {
    let n = data.len();
    let d = data[0].len();
    if k > n {
        return Err(KMeansError::MoreClustersThanPoints);
    }

    // K-means++ initialization
    let mut centroids = Vec::with_capacity(k);
    let first_idx = (rng.next_u32() as usize) % n;
    centroids.push(data[first_idx].clone());

    let mut dists = vec![S::zero(); n];
    for _cidx in 1..k {
        let mut total = S::zero();
        for (i, point) in data.iter().enumerate() {
            let mut min_dist = S::max_value();
            for c in &centroids {
                let dist = squared_euclidean(point, c);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            dists[i] = min_dist;
            total += min_dist;
        }

        // Choose next centroid with probability proportional to dist^2
        let threshold = S::from(rng.gen::<f64>()).unwrap() * total;
        let mut cumulative = S::zero();
        let mut chosen = 0;
        for (i, &dist) in dists.iter().enumerate() {
            cumulative += dist;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // Assignment step
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let mut best_dist = S::max_value();
            let mut best_label = 0;
            for (cidx, c) in centroids.iter().enumerate() {
                let dist = squared_euclidean(point, c);
                if dist < best_dist {
                    best_dist = dist;
                    best_label = cidx;
                }
            }
            if labels[i] != best_label {
                labels[i] = best_label;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![vec![S::zero(); d]; k];
        for (i, point) in data.iter().enumerate() {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                new_centroids[c][j] += point[j];
            }
        }

        for c in 0..k {
            if counts[c] == 0 {
                return Err(KMeansError::EmptyCluster { cluster: c });
            }
            for j in 0..d {
                centroids[c][j] = new_centroids[c][j] / S::from(counts[c]).unwrap();
            }
        }
    }

    // Compute inertia
    let mut inertia = S::zero();
    for (i, point) in data.iter().enumerate() {
        inertia += squared_euclidean(point, &centroids[labels[i]]);
    }

    Ok(KMeansResult {
        labels,
        centroids,
        inertia,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn two_clusters() {
        let data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.0, 0.2],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.2],
        ];
        let mut rng = StdRng::seed_from_u64(1);
        let result = kmeans(&data, 2, 100, &mut rng).unwrap();
        // Points 0-2 should be in one cluster, 3-5 in another
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn inertia_decreases_with_more_clusters() {
        let data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let r2 = kmeans(&data, 2, 50, &mut rng).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let r5 = kmeans(&data, 5, 50, &mut rng).unwrap();
        assert!(r5.inertia < r2.inertia);
    }
}
