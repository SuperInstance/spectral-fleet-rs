# Integration Guide: spectral-fleet-rs

## What This Crate Provides

Spectral methods for fleet matrices — eigenvalue decomposition, spectral clustering, and eigenvalue-based task scheduling.

- **`spectral_clustering::spectral_clustering()`** — Ng–Jordan–Weiss spectral clustering via normalized graph Laplacian. Returns `SpectralResult` with cluster `labels` and `embedding` matrix.
- **`power_iteration::top_k_eigenpairs()`** — Power iteration with Hotelling deflation for top-k eigenpairs of symmetric matrices. Returns `Vec<Eigenpair<S>>`.
- **`power_iteration::power_iterate()`** — Single dominant eigenpair via power method with convergence tolerance.
- **`lanczos::lanczos_iteration()`** — Lanczos iteration for large sparse symmetric matrices, producing tridiagonal `LanczosResult`.
- **`kmeans::kmeans()`** — K-means clustering (used internally by spectral clustering).
- **`scheduling::Deadline`** — Deadline for spectral computation tasks with `propagate()`, `fraction_remaining()`, and expiry tracking.
- **`scheduling::ScheduleResult`** — Eigenmode scheduling output: eigenvalues sorted ascending (smallest = bottleneck) and agent indices in bottleneck-first order.
- **`scheduling::eigenmode_schedule()`** — Bottleneck-first scheduling using eigendecomposition of agent performance matrices.
- **Vector utilities**: `l2_norm()`, `normalize()`, `dot()`, `axpy()`, `scale()`.

## How to Add This Crate

```bash
cargo add spectral-fleet
```

```rust
use spectral_fleet::{
    spectral_clustering::spectral_clustering,
    power_iteration::top_k_eigenpairs,
    scheduling::{Deadline, eigenmode_schedule},
};
```

## Cross-Repo Connections

### With `conservation-law-rs`: Spectral Energy Dissipation Monitoring

Use eigenvalue spectra to monitor how energy dissipates across a fleet, treating the smallest eigenvalue as a conservation bottleneck:

```rust
use spectral_fleet::power_iteration::top_k_eigenpairs;
use spectral_fleet::power_iteration::DenseOp;
use conservation_law::lagrangian::total_energy;

fn fleet_energy_dissipation(performance_matrix: &Vec<Vec<f64>>) -> f64 {
    let op = DenseOp { matrix: performance_matrix.clone() };
    let eigenpairs = top_k_eigenpairs(&op, 3, 100, 1e-6).unwrap();
    
    // The smallest eigenvalue encodes the slowest mode = tightest bottleneck
    let spectral_gap = eigenpairs[1].value - eigenpairs[0].value;
    println!("Spectral gap (dissipation rate): {:.6}", spectral_gap);
    spectral_gap
}
```

### With `si-cli`: CLI Spectral Clustering Command

The si-cli discovery layer exposes `spectral-fleet` as a subcommand for fleet partitioning:

```rust
use spectral_fleet::spectral_clustering::spectral_clustering;
use rand::thread_rng;

fn cli_cluster_command(affinity: Vec<Vec<f64>>, k: usize) {
    let mut rng = thread_rng();
    let result = spectral_clustering(&affinity, k, 50, 1e-6, &mut rng).unwrap();
    
    for (i, label) in result.labels.iter().enumerate() {
        println!("Agent {} → cluster {}", i, label);
    }
    
    // JSON output for si-cli piping
    println!("{}", serde_json::to_string_pretty(&result.embedding).unwrap());
}
```

### With `si-fleet-api`: REST Endpoint for Fleet Ranking

Expose spectral ranking via the fleet REST API:

```rust
use spectral_fleet::scheduling::{eigenmode_schedule, Deadline};
use si_fleet_api::{AgentEntry, HttpResponse};

fn post_rank_agents(agents: Vec<AgentEntry>) -> HttpResponse {
    let perf_matrix: Vec<Vec<f64>> = agents.iter()
        .map(|a| a.performance_vector.clone())
        .collect();
    
    let schedule = eigenmode_schedule(&perf_matrix, agents.len()).unwrap();
    let mut ranked: Vec<(usize, f64)> = schedule.schedule.iter()
        .zip(schedule.eigenvalues.iter())
        .map(|(&idx, &ev)| (idx, ev))
        .collect();
    
    // Bottleneck-first: smallest eigenvalue = highest priority
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    HttpResponse::json(ranked)
}
```

### With Supabase: Persist Spectral Results to PostgreSQL

Store clustering results and eigenvalue histories in Supabase for longitudinal fleet analysis:

```rust
use spectral_fleet::spectral_clustering::SpectralResult;
use supabase_rs::SupabaseClient;

async fn persist_spectral_result(
    client: &SupabaseClient,
    fleet_id: &str,
    result: &SpectralResult,
) {
    let labels_json = serde_json::to_string(&result.labels).unwrap();
    let embedding_json = serde_json::to_string(&result.embedding).unwrap();
    
    client.from("spectral_results")
        .insert(json!({
            "fleet_id": fleet_id,
            "labels": labels_json,
            "embedding": embedding_json,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        }))
        .execute()
        .await
        .unwrap();
}

async fn load_historical_clusters(client: &SupabaseClient, fleet_id: &str) -> Vec<SpectralResult> {
    let rows = client.from("spectral_results")
        .select("*")
        .eq("fleet_id", fleet_id)
        .order("timestamp.desc")
        .limit(10)
        .execute()
        .await
        .unwrap();
    
    rows.into_iter()
        .map(|r| SpectralResult {
            labels: serde_json::from_str(r.get("labels").unwrap()).unwrap(),
            embedding: serde_json::from_str(r.get("embedding").unwrap()).unwrap(),
        })
        .collect()
}
```

## Design Patterns

### Pattern: Deadline-Aware Spectral Computation

Propagate deadlines while running spectral computations, aborting if the deadline expires:

```rust
use spectral_fleet::scheduling::Deadline;
use spectral_fleet::power_iteration::power_iterate;

fn compute_with_deadline(op: &dyn Operator<f64>, v0: &[f64], budget_ms: f64) -> Option<Eigenpair<f64>> {
    let mut deadline = Deadline::new(0, budget_ms);
    let start = std::time::Instant::now();
    
    let result = power_iterate(op, v0, 100, 1e-6).ok()?;
    
    let elapsed = start.elapsed().as_millis() as f64;
    deadline.propagate(elapsed);
    
    if deadline.expired {
        println!("Spectral computation missed deadline (remaining: {})", deadline.remaining);
        return None;
    }
    
    println!("Completed with {:.1}% budget remaining", deadline.fraction_remaining() * 100.0);
    Some(result)
}
```

### Pattern: Dynamic Fleet Rebalancing on Agent Join

Recompute spectral rankings after each agent join/leave to maintain optimal workload distribution:

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

### Pattern: Bottleneck-First Task Scheduling

Use eigenmode scheduling to prioritize agents that form the tightest coupling bottleneck:

```rust
use spectral_fleet::scheduling::eigenmode_schedule;

fn schedule_maintenance_tasks(performance: &Vec<Vec<f64>>, n_agents: usize) -> Vec<usize> {
    let schedule = eigenmode_schedule(performance, n_agents).unwrap();
    // schedule.schedule[0] is the bottleneck agent
    schedule.schedule
}
```
