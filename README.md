# spectral-fleet-rs

Eigenvalue decomposition for agent fleet scheduling. Power iteration, Lanczos, spectral clustering, and bottleneck-first scheduling — the math that tells you which agent is your throughput limiter.

## Why Spectral?

When agents form a fleet, their performance vectors couple. The eigendecomposition of the Gram matrix PᵀP reveals:

- **Which agents dominate** (large eigenvalues → high alignment with principal modes)
- **Where bottlenecks live** (smallest eigenvalue → slowest eigenmode)
- **How to cluster agents** (spectral embedding + k-means)
- **What happens when you remove the top agent** (eigenvalue redistribution)

This crate implements all four operations with zero external dependencies beyond `rand` and `approx`.

## Architecture

```
spectral-fleet-rs
├── lib.rs                 — shared utilities: dot, normalize, axpy, Real trait
├── lanczos.rs             — Lanczos iteration for sparse symmetric operators
├── power_iteration.rs     — Power iteration + Hotelling deflation for top-k eigenpairs
├── spectral_clustering.rs — Ng–Jordan–Weiss spectral clustering
├── kmeans.rs              — Lloyd's k-means with k-means++ initialization
└── scheduling.rs          — SpectralScheduler: eigenmode-based fleet task scheduling
```

## Quick Start

```rust
use spectral_fleet::power_iteration::{DenseOp, top_k_eigenpairs, Eigenpair};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    // Agent performance matrix: 4 agents × 3 task dimensions
    // Each row is an agent's performance profile
    let agent_perfs = vec![
        vec![10.0, 8.0, 9.0],   // Agent 0: strong all-around
        vec![1.0, 0.5, 0.8],    // Agent 1: weak (the bottleneck)
        vec![5.0, 6.0, 4.0],    // Agent 2: mid-range
        vec![7.0, 7.5, 8.0],    // Agent 3: strong
    ];

    // Build the Gram matrix PᵀP
    let n = agent_perfs.len();
    let mut gram = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            gram[i][j] = agent_perfs[i].iter()
                .zip(&agent_perfs[j])
                .map(|(a, b)| a * b)
                .sum();
        }
    }

    let op = DenseOp { matrix: gram };
    let mut rng = StdRng::seed_from_u64(42);

    // Find the top 4 eigenpairs
    let pairs = top_k_eigenpairs(&op, 4, 200, 1e-10, &mut rng).unwrap();

    for (i, p) in pairs.iter().enumerate() {
        println!("Eigenvalue {}: {:.4}", i, p.value);
    }
}
```

## Eigenvalue Ranking

The eigenvalues of the Gram matrix `PᵀP` rank agents by their contribution to fleet performance:

```rust
use spectral_fleet::power_iteration::{DenseOp, top_k_eigenpairs};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn rank_fleet() {
    // 5 agents with performance vectors
    let perfs = vec![
        vec![9.0, 8.0, 7.0],   // Agent A
        vec![3.0, 2.0, 1.0],   // Agent B (weak)
        vec![6.0, 6.0, 6.0],   // Agent C (balanced mid)
        vec![10.0, 1.0, 1.0],  // Agent D (specialist)
        vec![8.0, 9.0, 8.0],   // Agent E (strong)
    ];

    let n = perfs.len();
    let mut gram = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            gram[i][j] = perfs[i].iter().zip(&perfs[j]).map(|(a, b)| a * b).sum();
        }
    }

    let op = DenseOp { matrix: gram };
    let mut rng = StdRng::seed_from_u64(99);
    let pairs = top_k_eigenpairs(&op, n, 200, 1e-10, &mut rng).unwrap();

    println!("Fleet Eigenvalue Ranking:");
    println!("Rank | Eigenvalue  | Cumulative %");
    println!("-----|-------------|-------------");

    let total: f64 = pairs.iter().map(|p| p.value).sum();
    let mut cumulative = 0.0;
    for (i, p) in pairs.iter().enumerate() {
        cumulative += p.value / total;
        println!("  {:<3} | {:<11.2} | {:.1}%", i + 1, p.value, cumulative * 100.0);
    }
}
```

The dominant eigenvalue captures the "general fleet strength" direction. The smallest eigenvalue reveals the weakest eigenmode — the bottleneck.

## Bottleneck Detection with SpectralScheduler

The `SpectralScheduler` computes eigenvalues of the Gram matrix and schedules agents bottleneck-first (smallest norm → highest priority):

```rust
use spectral_fleet::scheduling::SpectralScheduler;

fn detect_bottleneck() {
    let mut scheduler = SpectralScheduler::new();

    let agent_performances = vec![
        vec![10.0, 10.0],  // Agent 0: strong
        vec![1.0, 1.0],    // Agent 1: weak — the bottleneck
        vec![5.0, 5.0],    // Agent 2: mid
        vec![8.0, 7.0],    // Agent 3: strong
    ];

    let result = scheduler.schedule_by_eigenmode(agent_performances);

    println!("Eigenvalues (ascending): {:?}", result.eigenvalues);
    println!("Bottleneck index: {}", result.bottleneck_index);
    println!("Schedule (bottleneck-first): {:?}", result.schedule);
    println!("Deadlines expired: {}", result.deadlines_expired);

    // Agent 1 (the weak one) is scheduled first because it's the bottleneck
    assert_eq!(result.schedule[0], 1);
}
```

### With Deadlines

Real fleets have time constraints. The scheduler tracks computation deadlines and deprioritizes tasks that miss them:

```rust
use spectral_fleet::scheduling::SpectralScheduler;

fn scheduled_with_deadline() {
    // Very tight deadline — forces expiry on large fleets
    let mut scheduler = SpectralScheduler::with_deadline(0.5);

    let perfs: Vec<Vec<f64>> = (0..8)
        .map(|i| vec![i as f64 + 1.0, (i + 2) as f64])
        .collect();

    let result = scheduler.schedule_by_eigenmode(perfs);

    if result.deadlines_expired {
        println!("⚠ Deadline expired — expired tasks deprioritized");
        println!("Schedule was reversed to deprioritize: {:?}", result.schedule);
    }
}
```

### Deadline Propagation

Each `Deadline` tracks remaining budget and fraction:

```rust
use spectral_fleet::scheduling::Deadline;

fn main() {
    let mut dl = Deadline::new(1, 100.0);
    println!("Budget: {:.1}, Remaining: {:.1} ({:.0}%)",
        dl.budget, dl.remaining, dl.fraction_remaining() * 100.0);

    // Simulate work consuming time
    dl.propagate(30.0);
    println!("After 30 units: remaining={:.1}, expired={}", dl.remaining, dl.expired);

    dl.propagate(70.0);
    println!("After 100 total: remaining={:.1}, expired={}", dl.remaining, dl.expired);

    // Expired deadlines can't be un-expired
    dl.propagate(10.0); // no-op on expired
    assert!(dl.remaining == 0.0);
    assert!(dl.expired);
}
```

## What Happens When You Remove the Top Agent

This is the key question for fleet resilience. Removing the top agent redistributes eigenvalues:

```rust
use spectral_fleet::power_iteration::{DenseOp, top_k_eigenpairs};
use spectral_fleet::scheduling::SpectralScheduler;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn fleet_resilience() {
    let full_fleet = vec![
        vec![10.0, 9.0, 8.0],   // Agent 0 — top performer
        vec![4.0, 5.0, 3.0],
        vec![6.0, 7.0, 6.0],
        vec![3.0, 4.0, 5.0],
    ];

    // Full fleet analysis
    let mut sched = SpectralScheduler::new();
    let full_result = sched.schedule_by_eigenmode(full_fleet.clone());
    println!("Full fleet top eigenvalue: {:.2}", full_result.eigenvalues.last().unwrap());

    // Remove agent 0 (the top performer)
    let mut degraded_fleet = full_fleet.clone();
    degraded_fleet.remove(0);

    let mut sched2 = SpectralScheduler::new();
    let degraded_result = sched2.schedule_by_eigenmode(degraded_fleet);
    println!("Degraded fleet top eigenvalue: {:.2}",
        degraded_result.eigenvalues.last().unwrap());

    // Eigenvalue drop tells you how much resilience you lost
    let drop = full_result.eigenvalues.last().unwrap()
        - degraded_result.eigenvalues.last().unwrap();
    println!("Eigenvalue drop: {:.2} ({:.0}%)",
        drop,
        drop / full_result.eigenvalues.last().unwrap() * 100.0);
}
```

## Lanczos Iteration

For large sparse symmetric operators, the Lanczos algorithm builds a tridiagonal approximation in O(mn) time (m steps, n dimensions):

```rust
use spectral_fleet::lanczos::{
    lanczos, build_tridiagonal, rayleigh_quotient,
    DenseSymmetricOp, SymmetricOperator,
};

fn lanczos_example() {
    // 5×5 symmetric coupling matrix between agents
    let op = DenseSymmetricOp {
        data: vec![
            vec![2.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 3.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 4.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0, 3.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 2.0],
        ],
    };

    // Start from a random-ish vector
    let q0 = vec![1.0, 2.0, 3.0, 2.0, 1.0];

    // Run 5 Lanczos steps (full for a 5×5 matrix)
    let result = lanczos(&op, &q0, 5).unwrap();

    println!("Lanczos alphas (diagonal): {:?}", result.alpha);
    println!("Lanczos betas (off-diag):   {:?}", result.beta);
    println!("Basis vectors: {} vectors of dim {}", result.q.len(), result.q[0].len());

    // Reconstruct tridiagonal matrix
    let t = build_tridiagonal(&result.alpha, &result.beta);

    // Rayleigh quotient for an approximate eigenvector
    let rq = rayleigh_quotient(&op, &result.q[0]);
    println!("Rayleigh quotient of first basis vector: {:.4}", rq);
}
```

### Lanczos Error Handling

```rust
use spectral_fleet::lanczos::{lanczos, LanczosError, DenseSymmetricOp, SymmetricOperator};

fn lanczos_errors() {
    let op = DenseSymmetricOp {
        data: vec![vec![1.0_f64]],
    };

    // Zero start vector
    let err = lanczos(&op, &vec![0.0], 1).unwrap_err();
    assert_eq!(err, LanczosError::ZeroStartVector);

    // Dimension mismatch
    let err = lanczos(&op, &vec![1.0, 2.0], 1).unwrap_err();
    assert_eq!(err, LanczosError::DimensionMismatch);
}
```

## Spectral Clustering

The Ng–Jordan–Weiss algorithm clusters agents by their coupling structure:

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn cluster_fleet() {
    // 6 agents: two clear clusters
    // Cluster A: agents 0-2 (high mutual affinity)
    // Cluster B: agents 3-5 (high mutual affinity, low cross-cluster)
    let n = 6;
    let mut affinity = vec![vec![0.01; n]; n];
    for i in 0..3 {
        for j in 0..3 {
            affinity[i][j] = 1.0;
        }
    }
    for i in 3..6 {
        for j in 3..6 {
            affinity[i][j] = 1.0;
        }
    }

    let mut rng = StdRng::seed_from_u64(42);
    let result = spectral_clustering(&affinity, 2, 200, 1e-8, &mut rng).unwrap();

    println!("Cluster labels: {:?}", result.labels);
    println!("Embedding dim: {} (k eigenvectors)", result.embedding[0].len());

    // Verify: agents 0-2 share a label, agents 3-5 share a different label
    assert_eq!(result.labels[0], result.labels[1]);
    assert_eq!(result.labels[1], result.labels[2]);
    assert_eq!(result.labels[3], result.labels[4]);
    assert_ne!(result.labels[0], result.labels[3]);
}
```

### Three Clusters

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn three_clusters() {
    let n = 9;
    let mut w = vec![vec![0.01; n]; n];
    // Cluster 1: agents 0-2
    for i in 0..3 { for j in 0..3 { w[i][j] = 1.0; } }
    // Cluster 2: agents 3-5
    for i in 3..6 { for j in 3..6 { w[i][j] = 1.0; } }
    // Cluster 3: agents 6-8
    for i in 6..9 { for j in 6..9 { w[i][j] = 1.0; } }

    let mut rng = StdRng::seed_from_u64(77);
    let result = spectral_clustering(&w, 3, 200, 1e-8, &mut rng).unwrap();

    assert_ne!(result.labels[0], result.labels[3]);
    assert_ne!(result.labels[3], result.labels[6]);
    assert_ne!(result.labels[0], result.labels[6]);
}
```

## K-Means (Standalone)

The k-means implementation uses k-means++ initialization and works on any `Real` type:

```rust
use spectral_fleet::kmeans::{kmeans, squared_euclidean, KMeansResult};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn kmeans_example() {
    let data: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.2],
        vec![10.0, 10.0], vec![10.1, 10.0], vec![10.0, 10.2],
        vec![20.0, 20.0], vec![20.1, 20.1], vec![20.0, 20.2],
    ];

    let mut rng = StdRng::seed_from_u64(1);
    let result: KMeansResult<f64> = kmeans(&data, 3, 100, &mut rng).unwrap();

    println!("Labels: {:?}", result.labels);
    println!("Inertia: {:.4}", result.inertia);
    println!("Centroids: {:?}", result.centroids);

    // Distance between first two points
    let dist = squared_euclidean(&data[0], &data[3]);
    println!("Distance(0,3): {:.2}", dist);
}
```

## Real Fleet Scenarios

### Scenario: CI/CD Fleet with One Slow Runner

```rust
use spectral_fleet::scheduling::SpectralScheduler;

fn cicd_fleet() {
    // Performance profiles: [build_speed, test_speed, deploy_speed]
    let agents = vec![
        vec![95.0, 90.0, 88.0],  // runner-1: fast
        vec![92.0, 88.0, 91.0],  // runner-2: fast
        vec![45.0, 50.0, 40.0],  // runner-3: SLOW — the bottleneck
        vec![90.0, 85.0, 87.0],  // runner-4: fast
    ];

    let mut scheduler = SpectralScheduler::new();
    let result = scheduler.schedule_by_eigenmode(agents);

    println!("Bottleneck is agent index: {}", result.schedule[0]);
    // runner-3 (index 2) is the bottleneck, scheduled first
    assert_eq!(result.schedule[0], 2);

    println!("Full schedule: {:?}", result.schedule);
    println!("Eigenvalues: {:?}", result.eigenvalues);
}
```

### Scenario: Distributed Training Workers

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn training_workers() {
    // 8 GPU workers: 4 in datacenter A, 4 in datacenter B
    // Intra-datacenter bandwidth is high, cross-datacenter is low
    let n = 8;
    let mut affinity = vec![vec![0.05; n]; n];
    for i in 0..4 { for j in 0..4 { affinity[i][j] = 1.0; } }
    for i in 4..8 { for j in 4..8 { affinity[i][j] = 1.0; } }

    let mut rng = StdRng::seed_from_u64(123);
    let result = spectral_clustering(&affinity, 2, 200, 1e-8, &mut rng).unwrap();

    // Workers naturally split into their datacenter groups
    let dc_a_label = result.labels[0];
    let dc_b_label = result.labels[4];
    assert_ne!(dc_a_label, dc_b_label);

    for i in 0..4 { assert_eq!(result.labels[i], dc_a_label); }
    for i in 4..8 { assert_eq!(result.labels[i], dc_b_label); }

    println!("Datacenter A (label {}): workers 0-3", dc_a_label);
    println!("Datacenter B (label {}): workers 4-7", dc_b_label);
}
```

## API Reference

### `power_iteration` Module

| Type/Function | Description |
|---|---|
| `DenseOp<S>` | Dense symmetric matrix operator |
| `Eigenpair<S>` | Eigenvalue + eigenvector pair |
| `power_iterate(op, v0, max_iter, tol)` | Find dominant eigenpair |
| `top_k_eigenpairs(op, k, max_iter, tol, rng)` | Find top-k with Hotelling deflation |
| `Operator<S>` | Trait for symmetric linear operators |

### `lanczos` Module

| Type/Function | Description |
|---|---|
| `DenseSymmetricOp<S>` | Dense symmetric operator for Lanczos |
| `LanczosResult<S>` | Tridiagonal coefficients + basis vectors |
| `lanczos(op, q0, m)` | Run m Lanczos steps |
| `build_tridiagonal(alpha, beta)` | Reconstruct T matrix from coefficients |
| `rayleigh_quotient(op, v)` | Compute vᵀAv / vᵀv |
| `SymmetricOperator<S>` | Trait for symmetric operators |

### `scheduling` Module

| Type/Function | Description |
|---|---|
| `SpectralScheduler` | Eigenmode-based fleet scheduler |
| `ScheduleResult` | Eigenvalues + schedule + bottleneck index |
| `Deadline` | Time budget tracker with propagation |
| `SpectralScheduler::new()` | Create without deadline |
| `SpectralScheduler::with_deadline(budget)` | Create with deadline |
| `scheduler.schedule_by_eigenmode(perfs)` | Compute schedule |
| `scheduler.detect_bottleneck(eigenvalues)` | Find min eigenvalue index |
| `scheduler.gram_matrix(perfs)` | Build PᵀP from performance vectors |

### `spectral_clustering` Module

| Type/Function | Description |
|---|---|
| `spectral_clustering(affinity, k, max_iter, tol, rng)` | Ng–Jordan–Weiss clustering |
| `SpectralResult` | Labels + spectral embedding |

### `kmeans` Module

| Type/Function | Description |
|---|---|
| `kmeans(data, k, max_iter, rng)` | Lloyd's k-means with k-means++ init |
| `KMeansResult<S>` | Labels, centroids, inertia |
| `squared_euclidean(a, b)` | ‖a−b‖² |

## Building and Testing

```bash
cargo build
cargo test
```

All tests run without external services or network access.

## License

MIT
