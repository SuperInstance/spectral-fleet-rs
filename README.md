# spectral-fleet-rs

Spectral methods for fleet matrices. Lanczos iteration, power iteration with Hotelling deflation, spectral clustering (Ng–Jordan–Weiss), and eigenmode-based task scheduling for agent fleets.

## What It Does

Given a fleet of agents whose interactions or performance data form a matrix, this crate decomposes that matrix spectrally to reveal structure: clusters of agents, ranking via eigenvector centrality, and bottleneck identification for scheduling.

Core modules:

- **`power_iteration`** — Dominant eigenpairs via the power method with Hotelling deflation
- **`lanczos`** — Krylov subspace eigenvalue approximation for large sparse symmetric matrices
- **`spectral_clustering`** — Normalized graph Laplacian → k-means clustering (Ng–Jordan–Weiss)
- **`scheduling`** — Eigenmode-based task scheduling with deadline propagation and bottleneck-first ordering
- **`kmeans`** — Lloyd's algorithm with k-means++ initialization for the clustering step

## Quick Start

```toml
[dependencies]
spectral-fleet = { git = "https://github.com/SuperInstance/spectral-fleet-rs" }
rand = "0.8"
```

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    // 4 agents, 2 clear clusters: {0,1} and {2,3}
    let affinity = vec![
        vec![1.0, 0.9, 0.01, 0.01],
        vec![0.9, 1.0, 0.01, 0.01],
        vec![0.01, 0.01, 1.0, 0.9],
        vec![0.01, 0.01, 0.9, 1.0],
    ];

    let mut rng = StdRng::seed_from_u64(42);
    let result = spectral_clustering(&affinity, 2, 200, 1e-8, &mut rng).unwrap();

    println!("Cluster labels: {:?}", result.labels);
    // [0, 0, 1, 1] — agents 0,1 in cluster 0; agents 2,3 in cluster 1
}
```

## Power Iteration with Deflation

Find the top-k eigenvalues and eigenvectors of a symmetric matrix. After finding each eigenpair, Hotelling deflation removes that component so the next dominant one emerges.

```rust
use spectral_fleet::power_iteration::{DenseOp, top_k_eigenpairs};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    // Symmetric matrix with eigenvalues 8, 5, 2, 1
    let matrix = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 5.0, 0.0, 0.0],
        vec![0.0, 0.0, 2.0, 0.0],
        vec![0.0, 0.0, 0.0, 8.0],
    ];

    let op = DenseOp { matrix };
    let mut rng = StdRng::seed_from_u64(123);

    let pairs = top_k_eigenpairs(&op, 3, 200, 1e-10, &mut rng).unwrap();

    for (i, pair) in pairs.iter().enumerate() {
        println!("Eigenpair {}: λ = {:.6}", i, pair.value);
    }
    // Eigenpair 0: λ ≈ 8.0
    // Eigenpair 1: λ ≈ 5.0
    // Eigenpair 2: λ ≈ 2.0
}
```

### Single Eigenpair

```rust
use spectral_fleet::power_iteration::{DenseOp, power_iterate};

let op = DenseOp {
    matrix: vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 4.0],
    ],
};
let v0 = vec![1.0, 1.0, 1.0];
let pair = power_iterate(&op, &v0, 100, 1e-10).unwrap();
println!("Dominant eigenvalue: {:.6}", pair.value);
// ≈ 4.414
```

## Lanczos Iteration

For large sparse symmetric matrices where full decomposition is too expensive. Builds a tridiagonal matrix in Krylov subspace whose eigenvalues approximate the extremes of the original matrix.

```rust
use spectral_fleet::lanczos::{DenseSymmetricOp, lanczos, build_tridiagonal, rayleigh_quotient};

let op = DenseSymmetricOp {
    data: vec![
        vec![4.0, 1.0, 0.0, 0.0],
        vec![1.0, 3.0, 1.0, 0.0],
        vec![0.0, 1.0, 2.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
    ],
};

let q0 = vec![1.0, 1.0, 1.0, 1.0];
let result = lanczos(&op, &q0, 4).unwrap();

println!("Alpha (diagonal): {:?}", result.alpha);
println!("Beta (off-diagonal): {:?}", result.beta);

// Reconstruct tridiagonal T and compute Rayleigh quotient
let t = build_tridiagonal(&result.alpha, &result.beta);
let v = &result.q[0]; // first Lanczos vector
let rq = rayleigh_quotient(&op, v);
println!("Rayleigh quotient of first basis vector: {:.6}", rq);
```

## Spectral Clustering (Ng–Jordan–Weiss)

Cluster agents by their interaction patterns. Steps:

1. Form affinity matrix `W`
2. Compute symmetric normalized Laplacian `L_sym = I − D^{−½} W D^{−½}`
3. Find k smallest eigenvectors of `L_sym` (≡ k largest of `D^{−½} W D^{−½}`)
4. Normalize rows to unit length
5. Run k-means on the rows

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn cluster_agents() {
    // 6 agents in 3 clear groups
    let n = 6;
    let mut affinity = vec![vec![0.01; n]; n];

    // Group A: agents 0, 1
    affinity[0][0] = 1.0; affinity[0][1] = 0.9;
    affinity[1][0] = 0.9; affinity[1][1] = 1.0;

    // Group B: agents 2, 3
    affinity[2][2] = 1.0; affinity[2][3] = 0.9;
    affinity[3][2] = 0.9; affinity[3][3] = 1.0;

    // Group C: agents 4, 5
    affinity[4][4] = 1.0; affinity[4][5] = 0.9;
    affinity[5][4] = 0.9; affinity[5][5] = 1.0;

    let mut rng = StdRng::seed_from_u64(77);
    let result = spectral_clustering(&affinity, 3, 200, 1e-8, &mut rng).unwrap();

    for (agent, label) in result.labels.iter().enumerate() {
        println!("Agent {} → cluster {}", agent, label);
    }
}
```

## Eigenmode Scheduling

Schedule fleet tasks by analyzing the eigendecomposition of agent performance matrices. The bottleneck eigenmode (smallest eigenvalue) identifies the slowest agent and gets scheduled first.

```rust
use spectral_fleet::scheduling::{SpectralScheduler, Deadline};

fn schedule_fleet() {
    let mut scheduler = SpectralScheduler::new();

    // Each agent has a performance vector [tasks_completed, accuracy, latency_score]
    let performances = vec![
        vec![10.0, 0.95, 8.0],  // Agent 0: strong
        vec![1.0, 0.60, 2.0],   // Agent 1: weak (bottleneck)
        vec![5.0, 0.80, 5.0],   // Agent 2: medium
    ];

    let result = scheduler.schedule_by_eigenmode(performances);

    println!("Eigenvalues: {:?}", result.eigenvalues);
    println!("Schedule order: {:?}", result.schedule);
    println!("Bottleneck eigenmode: {}", result.bottleneck_index);
    println!("Deadlines expired: {}", result.deadlines_expired);

    // Schedule order: [1, 2, 0] — weakest agent first
}
```

### With Deadline

```rust
use spectral_fleet::scheduling::SpectralScheduler;

let mut scheduler = SpectralScheduler::with_deadline(5.0); // 5 time units

let performances = vec![
    vec![8.0, 7.0],
    vec![3.0, 2.0],
    vec![6.0, 5.0],
];

let result = scheduler.schedule_by_eigenmode(performances);
if result.deadlines_expired {
    println!("Warning: scheduling deadline expired, priorities adjusted");
}
```

### Deadline Propagation

```rust
use spectral_fleet::scheduling::Deadline;

let mut dl = Deadline::new(1, 100.0); // task 1, 100 unit budget
dl.propagate(30.0); // 30 units used
println!("Remaining: {:.1}%", dl.fraction_remaining() * 100.0); // 70.0%
println!("Expired: {}", dl.expired); // false

dl.propagate(70.0); // remaining budget used up
println!("Expired: {}", dl.expired); // true
```

## Fleet-Wide Ranking via Eigenvector Centrality

Use the dominant eigenvector of an agent interaction matrix as a PageRank-style ranking.

```rust
use spectral_fleet::power_iteration::{DenseOp, power_iterate};

fn rank_agents(interactions: &[Vec<f64>]) -> Vec<(usize, f64)> {
    let op = DenseOp { matrix: interactions.to_vec() };
    let v0 = vec![1.0; interactions.len()];
    let pair = power_iterate(&op, &v0, 200, 1e-10).unwrap();

    let mut ranking: Vec<(usize, f64)> = pair.vector
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();

    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranking
}

fn main() {
    // Agent interaction matrix: who defers to whom
    let interactions = vec![
        vec![0.0, 0.5, 0.3, 0.2],  // Agent 0 defers to 1, 2, 3
        vec![0.1, 0.0, 0.2, 0.7],  // Agent 1 mostly defers to 3
        vec![0.3, 0.2, 0.0, 0.5],  // Agent 2 defers to 0, 3
        vec![0.1, 0.2, 0.1, 0.0],  // Agent 3 is the authority
    ];

    let ranking = rank_agents(&interactions);
    for (agent, score) in &ranking {
        println!("Agent {} — centrality: {:.6}", agent, score);
    }
    // Agent 3 should rank highest
}
```

## Eigenvector Centrality for Fleet Ranking

The dominant eigenvector of an adjacency/interaction matrix gives each agent a centrality score. Agents with high centrality are those connected to other high-centrality agents — the PageRank principle.

```rust
use spectral_fleet::power_iteration::{DenseOp, top_k_eigenpairs};
use spectral_fleet::normalize;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn eigenvector_centrality(adjacency: &[Vec<f64>]) -> Vec<f64> {
    let op = DenseOp { matrix: adjacency.to_vec() };
    let mut rng = StdRng::seed_from_u64(42);
    let pairs = top_k_eigenpairs(&op, 1, 200, 1e-10, &mut rng).unwrap();
    pairs[0].vector.clone()
}
```

## Conservation Law Integration

When the fleet stack operates under conservation laws (see `entropy-conservation-rs`), spectral methods enforce the constraint that total fleet energy is conserved. The Gram matrix computed by `SpectralScheduler::gram_matrix` is positive semi-definite by construction, meaning its eigenvalues are non-negative — a natural fit for conserved quantities.

```rust
use spectral_fleet::scheduling::SpectralScheduler;

let scheduler = SpectralScheduler::new();

// Agent resource consumption vectors under conservation budget
let consumptions = vec![
    vec![0.3, 0.2, 0.1],
    vec![0.1, 0.1, 0.1],
    vec![0.5, 0.4, 0.3],
];

let gram = scheduler.gram_matrix(&consumptions);
// gram[i][j] = dot(consumption[i], consumption[j])
// trace(gram) = total squared consumption — must not exceed budget²
let total_energy: f64 = (0..gram.len()).map(|i| gram[i][i]).sum();
println!("Total fleet energy: {:.4}", total_energy);
```

## si-cli Integration

The `si-cli` tool calls `spectral_fleet::spectral_clustering::spectral_clustering` to partition a fleet into working groups based on agent similarity matrices stored in Supabase.

Typical workflow:

```bash
# Pull agent interaction data from si-fleet-api
si fleet matrix --format json > affinity.json

# Run spectral clustering
si fleet cluster --k 3 --max-iter 200 --tol 1e-8

# Get eigenmode schedule
si fleet schedule --method eigenmode --deadline 30s
```

Under the hood, `si fleet cluster` deserializes the affinity matrix, calls `spectral_clustering`, and writes cluster assignments back to Supabase.

## si-fleet-api Integration

The fleet API exposes spectral methods over HTTP:

```
POST /v1/fleet/cluster
{
  "affinity": [[1.0, 0.9], [0.9, 1.0]],
  "k": 2,
  "max_iter": 200,
  "tol": 1e-8
}

→ { "labels": [0, 1], "embedding": [[0.999], [-0.999]] }
```

```
POST /v1/fleet/schedule
{
  "agent_performances": [[10.0, 8.0], [2.0, 1.0]],
  "deadline_budget": 5.0
}

→ { "schedule": [1, 0], "eigenvalues": [5.0, 164.0], "bottleneck": 0 }
```

## Supabase Integration

Agent affinity matrices and scheduling results are stored in Supabase tables:

```sql
-- Fleet affinity matrix
CREATE TABLE fleet_affinity (
    fleet_id UUID REFERENCES fleets(id),
    agent_a UUID REFERENCES agents(id),
    agent_b UUID REFERENCES agents(id),
    affinity FLOAT NOT NULL,
    PRIMARY KEY (fleet_id, agent_a, agent_b)
);

-- Spectral clustering results
CREATE TABLE fleet_clusters (
    fleet_id UUID REFERENCES fleets(id),
    agent_id UUID REFERENCES agents(id),
    cluster_label INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Scheduling results
CREATE TABLE fleet_schedules (
    fleet_id UUID REFERENCES fleets(id),
    schedule_order INT[] NOT NULL,
    eigenvalues FLOAT[] NOT NULL,
    bottleneck_index INT NOT NULL,
    deadlines_expired BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Query cluster assignments:

```rust
// In si-fleet-api handler
async fn get_clusters(pool: &PgPool, fleet_id: Uuid) -> Vec<ClusterRow> {
    sqlx::query_as!(
        ClusterRow,
        "SELECT agent_id, cluster_label FROM fleet_clusters WHERE fleet_id = $1",
        fleet_id
    )
    .fetch_all(pool)
    .await
    .unwrap()
}
```

## API Reference

### `spectral_clustering`

```rust
pub fn spectral_clustering(
    affinity: &[Vec<f64>],     // n×n symmetric affinity matrix
    k: usize,                   // number of clusters
    max_iter: usize,            // power method iterations
    tol: f64,                   // convergence tolerance
    rng: &mut dyn RngCore,      // random source
) -> Result<SpectralResult, SpectralError>
```

Returns `SpectralResult { labels: Vec<usize>, embedding: Vec<Vec<f64>> }`.

### `top_k_eigenpairs`

```rust
pub fn top_k_eigenpairs<S: Real>(
    op: &dyn Operator<S>,
    k: usize,
    max_iter: usize,
    tol: S,
    rng: &mut dyn RngCore,
) -> Result<Vec<Eigenpair<S>>, PowerIterError>
```

Returns `Vec<Eigenpair>` where each `Eigenpair { value: S, vector: Vec<S> }`.

### `lanczos`

```rust
pub fn lanczos<S: Real>(
    op: &dyn SymmetricOperator<S>,
    q0: &[S],                   // initial vector
    m: usize,                    // number of Lanczos steps
) -> Result<LanczosResult<S>, LanczosError>
```

Returns `LanczosResult { alpha: Vec<S>, beta: Vec<S>, q: Vec<Vec<S>> }`.

### `SpectralScheduler`

```rust
impl SpectralScheduler {
    pub fn new() -> Self;
    pub fn with_deadline(budget: f64) -> Self;
    pub fn detect_bottleneck(&self, eigenvalues: &[f64]) -> usize;
    pub fn gram_matrix(&self, agent_performances: &[Vec<f64>]) -> Vec<Vec<f64>>;
    pub fn schedule_by_eigenmode(&mut self, agent_performances: Vec<Vec<f64>>) -> ScheduleResult;
}
```

### `Deadline`

```rust
impl Deadline {
    pub fn new(task_id: usize, budget: f64) -> Self;
    pub fn propagate(&mut self, dt: f64);
    pub fn fraction_remaining(&self) -> f64;
}
```

### `kmeans`

```rust
pub fn kmeans<S: Real>(
    data: &[Vec<S>],            // n points of dimension d
    k: usize,                    // number of clusters
    max_iter: usize,            // Lloyd iterations
    rng: &mut dyn RngCore,      // for k-means++ init
) -> Result<KMeansResult<S>, KMeansError>
```

Returns `KMeansResult { labels: Vec<usize>, centroids: Vec<Vec<S>>, inertia: S }`.

## Low-Level Utilities

```rust
use spectral_fleet::{l2_norm, normalize, dot, axpy, scale};

let mut v = vec![3.0_f64, 4.0];
assert_eq!(l2_norm(&v), 5.0);
normalize(&mut v);
assert!((l2_norm(&v) - 1.0).abs() < 1e-10);

let d = dot(&[1.0, 2.0], &[3.0, 4.0]); // 11.0

let mut y = vec![1.0, 2.0];
axpy(2.0, &[3.0, 4.0], &mut y); // y = [7.0, 10.0]

let mut v = vec![1.0, 2.0, 3.0];
scale(&mut v, 2.0); // [2.0, 4.0, 6.0]
```

## Testing

```bash
cargo test
```

Tests cover:
- Two-block and three-block spectral clustering
- Eigenvalue accuracy on diagonal matrices
- Eigenvector orthogonality
- Lanczos on diagonal and tridiagonal matrices
- Rayleigh quotient accuracy
- Bottleneck detection
- Deadline propagation and expiry
- Gram matrix properties
- k-means convergence and inertia

## Architecture

```
src/
├── lib.rs                 — Real trait, l2_norm, normalize, dot, axpy, scale
├── power_iteration.rs     — DenseOp, power_iterate, top_k_eigenpairs
├── lanczos.rs             — SymmetricOperator, DenseSymmetricOp, lanczos, build_tridiagonal, rayleigh_quotient
├── spectral_clustering.rs — spectral_clustering, SpectralResult, SpectralError
├── kmeans.rs              — kmeans, KMeansResult, squared_euclidean
└── scheduling.rs          — SpectralScheduler, Deadline, ScheduleResult
```

## License

MIT OR Apache-2.0
