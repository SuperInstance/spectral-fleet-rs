//! Property-based tests for spectral-fleet.

use proptest::prelude::*;
use spectral_fleet::kmeans::{kmeans, squared_euclidean};
use spectral_fleet::lanczos::{lanczos, DenseSymmetricOp};
use spectral_fleet::power_iteration::{power_iterate, top_k_eigenpairs, DenseOp};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

proptest! {
    #[test]
    fn lanczos_tridiagonal_is_symmetric(
        diag in prop::collection::vec(-10.0f64..10.0, 3..6)
    ) {
        let n = diag.len();
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..n {
            data[i][i] = diag[i];
            if i + 1 < n {
                let off = (i as f64 + 1.0).sin(); // deterministic off-diagonal
                data[i][i + 1] = off;
                data[i + 1][i] = off;
            }
        }
        let op = DenseSymmetricOp { data };
        let q0: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
        let res = lanczos(&op, &q0, n).unwrap();

        // alpha and beta lengths are correct
        prop_assert_eq!(res.alpha.len(), n);
        prop_assert_eq!(res.beta.len(), n - 1);

        // Q vectors are orthonormal (at least pairwise for first few)
        for i in 0..n.min(3) {
            for j in 0..n.min(3) {
                let dot: f64 = res.q[i].iter().zip(res.q[j].iter()).map(|(&a, &b)| a * b).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!((dot - expected).abs() < 1e-6, "orthonormality failed at ({},{}): {}", i, j, dot);
            }
        }
    }

    #[test]
    fn power_iteration_eigenvalue_bounds(
        a in -5.0f64..5.0,
        b in -5.0f64..5.0,
        c in -5.0f64..5.0,
    ) {
        // Symmetric 2x2 with entries [[a, b], [b, c]]
        let op = DenseOp {
            matrix: vec![
                vec![a, b],
                vec![b, c],
            ],
        };
        let mut rng = StdRng::seed_from_u64(1);
        let v0: Vec<f64> = vec![rng.gen(), rng.gen()];
        let pair = power_iterate(&op, &v0, 200, 1e-10).unwrap();

        // Eigenvalues of symmetric 2x2 are real and bounded by trace/norm
        let trace = a + c;
        let det = a * c - b * b;
        let disc = (trace * trace - 4.0 * det).max(0.0);
        let lambda1 = (trace + disc.sqrt()) / 2.0;
        let lambda2 = (trace - disc.sqrt()) / 2.0;
        let max_ev = lambda1.abs().max(lambda2.abs());

        prop_assert!(pair.value.abs() <= max_ev + 1e-6,
            "eigenvalue {} exceeds bound {}", pair.value, max_ev);
    }

    #[test]
    fn top_k_produces_orthogonal_vectors(
        diag in prop::collection::vec(1.0f64..10.0, 3..6)
    ) {
        // Skip cases where eigenvalues are too close (power iteration struggles)
        let mut sorted = diag.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if sorted.windows(2).any(|w| (w[0] - w[1]).abs() < 0.5) {
            return Ok(());
        }

        let n = diag.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = diag[i];
        }
        let op = DenseOp { matrix };
        let mut rng = StdRng::seed_from_u64(42);
        let pairs = top_k_eigenpairs(&op, n, 500, 1e-8, &mut rng).unwrap();

        for i in 0..n {
            for j in i + 1..n {
                let d: f64 = pairs[i].vector.iter().zip(pairs[j].vector.iter())
                    .map(|(&a, &b)| a * b).sum();
                prop_assert!(d.abs() < 1e-4, "vectors not orthogonal: {}", d);
            }
        }
    }

    #[test]
    fn kmeans_inertia_non_negative(
        dim in 2usize..5,
        points in prop::collection::vec(
            prop::collection::vec(-10.0f64..10.0, 2..5),
            5..20
        ),
        k in 2usize..4
    ) {
        // Ensure uniform dimension
        let points: Vec<Vec<f64>> = points.into_iter().map(|mut p| {
            p.resize(dim, 0.0);
            p
        }).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let result = kmeans(&points, k, 50, &mut rng).unwrap();
        prop_assert!(result.inertia >= 0.0);
        prop_assert_eq!(result.centroids.len(), k);
        prop_assert!(result.labels.iter().all(|&l| l < k));
    }

    #[test]
    fn squared_euclidean_symmetric_and_zero_self(
        a in prop::collection::vec(-10.0f64..10.0, 2..5),
        b in prop::collection::vec(-10.0f64..10.0, 2..5)
    ) {
        let d_ab = squared_euclidean(&a, &b);
        let d_ba = squared_euclidean(&b, &a);
        prop_assert_eq!(d_ab, d_ba);

        let d_aa = squared_euclidean(&a, &a);
        prop_assert!(d_aa.abs() < 1e-10);
    }
}
