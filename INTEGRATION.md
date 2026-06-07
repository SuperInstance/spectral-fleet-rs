# Integration Guide: spectral-fleet

## What This Crate Provides

- **`LanczosResult<S>`** — Tridiagonal matrix + orthonormal basis from Lanczos iteration for large sparse symmetric matrices
- **`SymmetricOperator<S>`** trait — Define custom symmetric linear operators (matrix-free)
- **`DenseSymmetricOp<S>`** — Dense symmetric matrix operator
- **`top_k_eigenpairs()`** — Power iteration with deflation for top-k eigenpairs
- **`SpectralClustering`** — Ng–Jordan–Weiss spectral clustering via normalized graph Laplacian
- **`kmeans()`** — K-means clustering (used internally by spectral clustering)
- **`Deadline`** — Deadline for spectral computation tasks with propagation and expiry
- **`EigenmodeScheduler`** — Bottleneck-first scheduling using eigenvalue decomposition
- **Vector utilities**: `l2_norm`, `normalize`, `dot`, `axpy`, `scale`

This crate provides spectral methods for fleet matrix analysis: eigenvalue decomposition, spectral clustering, and eigenvalue-based task scheduling for agent fleets.

## How to Add This Crate

```bash
cargo add spectral-fleet
```

```rust
use spectral_fleet::lanczos::{DenseSymmetricOp, lanczos_iteration};
use spectral_fleet::Real;

let op = DenseSymmetricOp { data: vec![
    vec![4.0, 1.0, 0.0],
    vec![1.0, 3.0, 1.0],
    vec![0.0, 1.0, 2.0],
]};
let start = vec![1.0, 0.0, 0.0];
let result = lanczos_iteration(&op, &start, 3, 1e-10).unwrap();
println!("Eigenvalues: {:?}", result.alpha);
```

## Integration Points

### t-minus

- **Why**: t-minus provides deadline propagation and scheduling; spectral-fleet's `EigenmodeScheduler` determines task priority from eigenvalue decomposition. The slowest eigenmode (smallest eigenvalue) identifies bottlenecks and gets highest priority.
- **How**: Use `Deadline` from spectral-fleet's scheduling module, which mirrors t-minus deadline semantics. Feed eigenvalue-based priorities into t-minus's cron scheduler.

```rust
use spectral_fleet::scheduling::{Deadline, EigenmodeScheduler};

// Compute fleet eigenvalues, then schedule bottleneck-first
let eigenvalues = vec![0.0, 0.15, 0.42, 1.1, 2.3];
let scheduler = EigenmodeScheduler::new(eigenvalues);

// Create deadlines for each eigenmode task
let deadlines: Vec<Deadline> = (0..5)
    .map(|i| Deadline::new(i, scheduler.budget_for_mode(i)))
    .collect();

// Smallest eigenvalue → highest priority → tightest deadline
println!("Bottleneck mode budget: {}", deadlines[0].budget);
```

### categorical-agents

- **Why**: categorical-agents provides category-theoretic composition (adjunctions, monads, comonads); spectral-fleet provides the spectral analysis that determines HOW agents should be clustered/composed. Spectral clustering outputs groupings that become categorical compositions.
- **How**: Run spectral clustering on the fleet affinity matrix, then use the cluster assignments as inputs to categorical-agents' monadic composition.

```rust
use spectral_fleet::spectral_clustering::spectral_cluster;

// Agent affinity matrix (symmetric)
let affinity = vec![
    vec![1.0, 0.9, 0.1],
    vec![0.9, 1.0, 0.2],
    vec![0.1, 0.2, 1.0],
];
let (labels, _centers) = spectral_cluster(&affinity, 2, 100).unwrap();
// labels = [0, 0, 1] — agents 0 and 1 cluster together
```

### conservation-law

- **Why**: conservation-law defines the invariant γ + H = C; the spectral gap γ IS the smallest non-zero eigenvalue from spectral-fleet. The Fiedler value directly measures conservation health.
- **How**: Extract the Fiedler value (second-smallest eigenvalue) from the fleet Laplacian and feed it to conservation-law as the spectral gap γ.

```rust
use spectral_fleet::power_iteration::top_k_eigenpairs;
use spectral_fleet::lanczos::DenseSymmetricOp;

// Fleet Laplacian (from adjacency structure)
let laplacian = DenseSymmetricOp { data: compute_fleet_laplacian() };
let (eigenvalues, _eigenvectors) = top_k_eigenpairs(&laplacian, 3, 100, 1e-10).unwrap();

// Spectral gap = Fiedler value = second smallest eigenvalue
let gamma = eigenvalues.iter().filter(|&&e| e > 1e-10).cloned().reduce(f64::min);
println!("Spectral gap γ = {:?}", gamma);
```

## For AI Agents

- **Context needed**: Fleet adjacency/affinity matrix, number of clusters k, convergence tolerance
- **Key imports**: `spectral_fleet::lanczos::*`, `spectral_fleet::spectral_clustering::*`, `spectral_fleet::scheduling::*`
- **Integration pattern**: Build affinity matrix → run `spectral_cluster()` → extract cluster labels → feed to downstream composition/scheduling
- **Error handling**: `LanczosError` (breakdown, zero start vector), `SpectralError` (k-means failure, eigen computation failure). Always handle `Breakdown` — it means the matrix has fewer distinct eigenvalues than requested Krylov subspace dimension.

## For Humans

- **Prerequisites**: Linear algebra (eigenvalues, Laplacian matrices), basic graph theory
- **Learning path**: Start with `power_iteration.rs` (simplest eigenvalue method), then `lanczos.rs` (scalable), then `spectral_clustering.rs` (application), then `scheduling.rs` (production integration)
- **Common pitfalls**:
  - The `SymmetricOperator` trait requires the matrix to be symmetric — non-symmetric inputs produce garbage
  - Lanczos breakdown is normal for matrices with eigenvalue multiplicity; handle it gracefully
  - Spectral clustering quality depends heavily on the affinity matrix — use a suitable kernel for your data
  - The `Real` trait requires `Send + Sync` — use `f64` for most cases
