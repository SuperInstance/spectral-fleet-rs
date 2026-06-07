# spectral-fleet

> Spectral methods for fleet matrices — Lanczos iteration, power iteration, and spectral clustering.

## What This Does

This crate brings high-dimensional linear algebra to fleet-scale matrices without requiring dense storage. It implements Lanczos iteration for building orthonormal Krylov bases and tridiagonal approximations of large symmetric operators, power iteration with Hotelling deflation for extracting top-k eigenpairs, and end-to-end spectral clustering with k-means++ on eigenvector embeddings. Whether you need to find the dominant modes of a fleet communication graph or partition agents into natural groups, these algorithms operate with memory proportional to the number of nonzero entries rather than the square of the matrix dimension.

## Why It Matters

As agent fleets grow to hundreds or thousands of nodes, explicit matrix representations become impossible. Spectral methods let you infer global structure — connectivity clusters, principal directions of variation, low-rank approximations — from matrix-vector products alone. In the AGI trajectory, this is how you scale pattern recognition from toy problems to planetary-scale fleets: never form the full matrix, only apply it.

## Quick Start

```bash
cargo add spectral-fleet
```

```rust
use spectral_fleet::lanczos::{lanczos, DenseSymmetricOp};
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Two-block affinity: 20 agents, two tight communities
    let mut affinity = vec![vec![0.01; 20]; 20];
    for i in 0..10 { for j in 0..10 { affinity[i][j] = 1.0; } }
    for i in 10..20 { for j in 10..20 { affinity[i][j] = 1.0; } }

    // Spectral clustering into 2 groups
    let result = spectral_clustering(&affinity, 2, 200, 1e-8, &mut rng).unwrap();

    // First 10 agents should share a label
    let label_a = result.labels[0];
    for i in 0..10 {
        assert_eq!(result.labels[i], label_a);
    }
    println!("Clustered 20 agents into 2 communities");
}
```

## Architecture

| Module | Purpose |
|--------|---------|
| `lanczos` | Krylov subspace iteration, tridiagonal reduction, Rayleigh quotients |
| `power_iteration` | Dominant eigenpair extraction, Hotelling deflation for top-k |
| `spectral_clustering` | Normalized graph Laplacian, eigenvector embedding, k-means grouping |
| `kmeans` | Lloyd's algorithm with k-means++ initialization |

## API Tour

### `Real` trait

Bound on `Float + NumAssign + Debug + Send + Sync`. Used throughout the crate for generic scalar types.

```rust
pub trait Real: Float + NumAssign + std::fmt::Debug + Send + Sync + 'static {}
```

### `lanczos` function

Build an orthonormal basis and tridiagonal matrix from a symmetric operator.

```rust
pub fn lanczos<S: Real>(
    op: &dyn SymmetricOperator<S>,
    q0: &[S],
    m: usize,
) -> Result<LanczosResult<S>, LanczosError>
```

```rust
let op = DenseSymmetricOp { data: matrix };
let res = lanczos(&op, &start_vector, 10)?;
let t = build_tridiagonal(&res.alpha, &res.beta);
```

### `top_k_eigenpairs` function

Extract the `k` largest-magnitude eigenpairs with power iteration and deflation.

```rust
pub fn top_k_eigenpairs<S: Real>(
    op: &dyn Operator<S>,
    k: usize,
    max_iter: usize,
    tol: S,
    rng: &mut dyn rand::RngCore,
) -> Result<Vec<Eigenpair<S>>, PowerIterError>
```

### `spectral_clustering` function

End-to-end normalized spectral clustering (Ng–Jordan–Weiss algorithm).

```rust
pub fn spectral_clustering(
    affinity: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut dyn RngCore,
) -> Result<SpectralResult, SpectralError>
```

### `kmeans` function

Lloyd's algorithm with k-means++ initialization.

```rust
pub fn kmeans<S: Real>(
    data: &[Vec<S>],
    k: usize,
    max_iter: usize,
    rng: &mut dyn RngCore,
) -> Result<KMeansResult<S>, KMeansError>
```

## Performance

| Operation | Complexity | Memory |
|-----------|-----------|--------|
| Lanczos step | O(nnz) per iteration | O(m × n) for basis vectors |
| Power iteration | O(k × max_iter × nnz) | O(k × n) for eigenvectors |
| Spectral clustering | O(k × max_iter × n² + iter × k × n × d) | O(n²) for affinity + O(n × k) embedding |
| K-means | O(iter × k × n × d) | O(k × d) for centroids |

For sparse operators, replace n² with `nnz`. The Lanczos algorithm is the workhorse: it reduces a large symmetric eigenproblem to a small tridiagonal one in O(m) matrix-vector products.

## Ecosystem

- **[conservation-law](https://github.com/SuperInstance/conservation-law-rs)** — Model agent dynamics before clustering their trajectories
- **[fleet-warden](https://github.com/SuperInstance/fleet-warden-rs)** — Monitor fleet health and trigger re-clustering when topology changes
- **[wasserstein-agents](https://github.com/SuperInstance/wasserstein-agents-rs)** — Compare agent distributions before and after spectral partitioning
- **[categorical-agents](https://github.com/SuperInstance/categorical-agents-rs)** — Compose clustering pipelines via monadic bind chains

## Ideas for Improvement

1. **Implicitly restarted Lanczos** — Add thick-restart strategies to handle clustered eigenvalues and improve convergence.
2. **Sparse matrix formats** — Support CSR/CSC adapters so `SymmetricOperator` implementations can skip zero entries.
3. **GPU acceleration** — Offload matrix-vector products to CUDA for fleet graphs with millions of edges.
4. **Hierarchical spectral clustering** — Recursively bisect clusters using the Fiedler vector for multi-resolution fleet hierarchies.
5. **Streaming k-means** — Adapt centroids incrementally as new agents join the fleet without full recomputation.

## License

MIT OR Apache-2.0
