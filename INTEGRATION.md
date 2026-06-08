# INTEGRATION.md — spectral-fleet-rs × conservation-law-rs × entropy-conservation-rs

**Spectral fleet methods** use eigenvalue decomposition and clustering to
rank and group agents. They connect to Lagrangian dynamics for evolving
agent states and to entropy conservation for analyzing information flow in
fleet matrices.

## Synergy Map

```
conservation-law-rs           spectral-fleet-rs              entropy-conservation-rs
┌──────────────────┐         ┌──────────────────────┐       ┌─────────────────────┐
│ SymplecticIntegr  │◄──────►│ lanczos              │◄─────►│ decompose           │
│ AgentState        │         │ power_iteration      │       │ gradient_energy     │
│ verify_noether    │         │ top_k_eigenpairs     │       │ curl_energy         │
│ total_energy      │         │ spectral_clustering  │       │ harmonic_energy     │
└──────────────────┘         │ kmeans               │       └─────────────────────┘
                             │ l2_norm              │
                             └──────────────────────┘
```

## Key Insight

Agent fleets form affinity graphs. Spectral-fleet extracts structure from
these graphs via eigen decomposition. Conservation-law ensures the
underlying dynamics are symplectic (energy-preserving), and
entropy-conservation decomposes the fleet's information flows into
gradient (hierarchical), curl (cyclic), and harmonic (lossy) components.

## Example 1: Symplectic Evolution of Agent Affinities

Evolve an agent fleet under a potential while periodically re-computing
its spectral ranking.

```rust
use conservation_law::lagrangian::{AgentState, MechanicalLagrangian, SymplecticIntegrator};
use spectral_fleet::power_iteration::{power_iterate, DenseOp, Operator};
use spectral_fleet::normalize;

fn evolve_and_rank() {
    let potential = |q: &[f64; 2]| 0.5 * (q[0] * q[0] + q[1] * q[1]);
    let integrator = SymplecticIntegrator::new(0.01).unwrap();
    let mut state = AgentState::new([1.0, 0.0], [0.0, 1.0]);

    // Affinity matrix evolves with state distance
    let mut affinity = vec![vec![0.0; 2]; 2];
    for step in 0..100 {
        state = integrator.step(1.0, &potential, &state).unwrap();
        let d2 = state.q[0] * state.q[0] + state.q[1] * state.q[1];
        affinity[0][0] = 1.0;
        affinity[1][1] = 1.0;
        affinity[0][1] = (-d2).exp();
        affinity[1][0] = affinity[0][1];

        if step % 20 == 0 {
            let op = DenseOp { matrix: affinity.clone() };
            let mut v0 = vec![1.0, 1.0];
            normalize(&mut v0);
            let pair = power_iterate(&op, &v0, 100, 1e-8).unwrap();
            println!("step {}: eigenvalue = {:.4}", step, pair.value);
        }
    }
}
```

## Example 2: Spectral Clustering with Entropy Decomposition

Cluster agents by affinity, then decompose the inter-cluster entropy flow.

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use entropy_conservation::hodge_decomposition::decompose;
use rand::thread_rng;

fn cluster_and_decompose(affinity: &[Vec<f64>]) {
    let mut rng = thread_rng();
    let result = spectral_clustering(affinity, 2, 100, 1e-8, &mut rng).unwrap();
    println!("Cluster labels: {:?}", result.labels);

    // Build a coarse entropy-flow matrix between clusters
    let mut cluster_flow = vec![vec![0.0; 2]; 2];
    for (i, row) in affinity.iter().enumerate() {
        for (j, &a) in row.iter().enumerate() {
            let ci = result.labels[i];
            let cj = result.labels[j];
            cluster_flow[ci][cj] += a;
        }
    }

    let hodge = decompose(&cluster_flow);
    println!("Gradient (hierarchical flow): {:.4}", hodge.gradient_energy());
    println!("Curl (cyclic alliance): {:.4}", hodge.curl_energy());
    println!("Harmonic (information loss): {:.4}", hodge.harmonic_energy());
}
```

## Example 3: Lanczos Iteration for Large Fleet Matrices

For fleets too large for dense power iteration, use Lanczos to find the
extremal eigenvalues of the affinity operator.

```rust
use spectral_fleet::lanczos::{lanczos, DenseSymmetricOp, SymmetricOperator};

fn large_fleet_spectral_gap(affinity: &[Vec<f64>]) {
    let op = DenseSymmetricOp { data: affinity.to_vec() };
    let n = affinity.len();
    let q0 = vec![1.0; n];
    let result = lanczos(&op, &q0, 20).unwrap();
    println!("Lanczos alphas: {:?}", result.alpha);
    println!("Lanczos betas: {:?}", result.beta);
    // The smallest eigenvalue of the tridiagonal approximates the spectral gap
}
```

## Cargo.toml Wiring

```toml
[dependencies]
spectral-fleet = { git = "https://github.com/SuperInstance/spectral-fleet-rs" }
conservation-law = { git = "https://github.com/SuperInstance/conservation-law-rs" }
entropy-conservation = { git = "https://github.com/SuperInstance/entropy-conservation-rs" }
```

## Design Patterns

### Pattern: Dynamic Fleet Rebalancing

Recompute spectral rankings after each agent join/leave to maintain
optimal workload distribution:

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::thread_rng;

fn rebalance_on_change(affinity: &mut Vec<Vec<f64>>, joined: usize, workload: f64) {
    let n = affinity.len();
    for i in 0..n {
        affinity[i].push((-workload).exp());
    }
    affinity.push(vec![0.0; n + 1]);
    affinity[n][n] = 1.0;

    let mut rng = thread_rng();
    let result = spectral_clustering(affinity, 3, 50, 1e-6, &mut rng).unwrap();
    println!("Agent {} assigned to cluster {}", joined, result.labels[n]);
}
```
