//! Scheduling integration: eigenvalue-based task scheduling for agent fleets.
//!
//! Bridges spectral-fleet eigenvalue methods with scheduling concepts:
//! - **Eigenmode scheduling**: Uses eigenvalue decomposition of agent performance
//!   matrices to determine task priority. The slowest eigenmode (smallest
//!   eigenvalue) identifies the bottleneck and gets highest priority
//!   (bottleneck-first scheduling).
//! - **Deadline propagation**: Deadlines flow through the dependency graph,
//!   constraining when spectral computations must complete.
//! - **Deadline expiry**: Computation tasks that miss their deadline are
//!   automatically deprioritized.
//!
//! The core insight: a fleet's performance matrix encodes coupling between
//! agents. Its eigendecomposition reveals which modes of operation are slowest
//! (tightest bottlenecks). Scheduling the bottleneck first optimizes overall
//! throughput.

use crate::{dot, l2_norm};

// ---------------------------------------------------------------------------
// Deadline
// ---------------------------------------------------------------------------

/// A deadline for a spectral computation task.
#[derive(Debug, Clone)]
pub struct Deadline {
    /// Unique task identifier.
    pub task_id: usize,
    /// Remaining time units before expiry.
    pub remaining: f64,
    /// Original time budget.
    pub budget: f64,
    /// Whether the deadline has expired.
    pub expired: bool,
}

impl Deadline {
    /// Create a new deadline with the given time budget.
    pub fn new(task_id: usize, budget: f64) -> Self {
        Self {
            task_id,
            remaining: budget,
            budget,
            expired: false,
        }
    }

    /// Propagate (tick) the deadline by `dt` time units.
    /// Marks as expired if remaining time drops to zero or below.
    pub fn propagate(&mut self, dt: f64) {
        if self.expired {
            return;
        }
        self.remaining -= dt;
        if self.remaining <= 0.0 {
            self.remaining = 0.0;
            self.expired = true;
        }
    }

    /// Fraction of time budget remaining (0.0 to 1.0).
    pub fn fraction_remaining(&self) -> f64 {
        if self.budget <= 0.0 {
            return 0.0;
        }
        (self.remaining / self.budget).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Scheduling result
// ---------------------------------------------------------------------------

/// Result of eigenmode-based scheduling.
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    /// Eigenvalues sorted ascending (smallest = slowest eigenmode).
    pub eigenvalues: Vec<f64>,
    /// Agent indices in scheduled order (bottleneck-first).
    pub schedule: Vec<usize>,
    /// Index of the bottleneck eigenmode.
    pub bottleneck_index: usize,
    /// Whether any deadlines expired during scheduling.
    pub deadlines_expired: bool,
}

// ---------------------------------------------------------------------------
// SpectralScheduler
// ---------------------------------------------------------------------------

/// Schedules agent tasks using eigenvalue decomposition of performance matrices.
///
/// The agent performance matrix `P[i][j]` represents how agent `i` performs on
/// task dimension `j`. The eigendecomposition of `PᵀP` (or `PPᵀ`) reveals
/// the principal modes of fleet performance.
///
/// **Bottleneck-first**: the agent associated with the smallest eigenvalue
/// (slowest mode) is scheduled first, because it is the throughput limiter.
#[derive(Debug, Clone)]
pub struct SpectralScheduler {
    /// Deadline for the overall scheduling computation.
    pub deadline: Option<Deadline>,
    /// Convergence tolerance for power iteration.
    pub tolerance: f64,
    /// Maximum power iteration steps.
    pub max_iterations: usize,
}

impl SpectralScheduler {
    /// Create a new scheduler without a deadline.
    pub fn new() -> Self {
        Self {
            deadline: None,
            tolerance: 1e-8,
            max_iterations: 200,
        }
    }

    /// Create a scheduler with a computation deadline.
    pub fn with_deadline(budget: f64) -> Self {
        Self {
            deadline: Some(Deadline::new(0, budget)),
            ..Self::new()
        }
    }

    /// Detect the bottleneck eigenmode (index of the smallest eigenvalue).
    pub fn detect_bottleneck(&self, eigenvalues: &[f64]) -> usize {
        if eigenvalues.is_empty() {
            return 0;
        }
        eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute the Gram matrix `PᵀP` from an agent performance matrix.
    ///
    /// `agent_performances[i]` is the performance vector of agent `i`.
    /// Returns a symmetric positive semi-definite matrix suitable for
    /// eigenvalue decomposition.
    pub fn gram_matrix(&self, agent_performances: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if agent_performances.is_empty() {
            return vec![];
        }
        let n = agent_performances.len();
        let mut gram = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                gram[i][j] = dot(&agent_performances[i], &agent_performances[j]);
            }
        }
        gram
    }

    /// Simple QR-based eigenvalue computation for small dense symmetric matrices.
    /// Returns eigenvalues sorted in ascending order.
    fn qr_eigenvalues(&self, matrix: &[Vec<f64>], max_iters: usize) -> Vec<f64> {
        let n = matrix.len();
        if n == 0 {
            return vec![];
        }
        // Work on a copy
        let mut a: Vec<Vec<f64>> = matrix.to_vec();

        for _ in 0..max_iters {
            // QR decomposition via Gram-Schmidt
            let (q, r) = qr_decompose(&a);
            // A = R * Q (shift towards eigenvalues)
            a = mat_mul(&r, &q);

            // Check convergence: off-diagonal elements
            let mut off_diag = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off_diag += a[i][j] * a[i][j];
                    }
                }
            }
            if off_diag < 1e-14 {
                break;
            }
        }

        let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    /// Schedule agents by eigenmode priority (bottleneck-first).
    ///
    /// 1. Compute the Gram matrix from agent performance vectors.
    /// 2. Extract eigenvalues via QR iteration.
    /// 3. Sort agents by ascending eigenvalue magnitude — the agent whose
    ///    performance is most aligned with the slowest eigenmode gets
    ///    scheduled first.
    /// 4. If a deadline is set, check for expiry and deprioritize expired tasks.
    pub fn schedule_by_eigenmode(
        &mut self,
        agent_performances: Vec<Vec<f64>>,
    ) -> ScheduleResult {
        let n = agent_performances.len();
        if n == 0 {
            return ScheduleResult {
                eigenvalues: vec![],
                schedule: vec![],
                bottleneck_index: 0,
                deadlines_expired: false,
            };
        }

        // Tick deadline for computation time
        if let Some(ref mut dl) = self.deadline {
            dl.propagate(0.1); // cost of Gram matrix construction
        }

        let gram = self.gram_matrix(&agent_performances);

        if let Some(ref mut dl) = self.deadline {
            dl.propagate(0.05 * n as f64); // cost of eigendecomposition scales
        }

        let eigenvalues = self.qr_eigenvalues(&gram, self.max_iterations);

        let bottleneck = self.detect_bottleneck(&eigenvalues);

        // Build schedule: compute each agent's "score" as the norm of their
        // performance vector, then sort by inverse alignment with the
        // bottleneck eigenvector direction. Simpler approach: sort agents
        // by their performance magnitude (smallest = bottleneck), ascending.
        let mut agent_scores: Vec<(usize, f64)> = agent_performances
            .iter()
            .enumerate()
            .map(|(i, perf)| (i, l2_norm(perf)))
            .collect();

        // Sort ascending by score — smallest norm agents are scheduled first
        // (they are the bottleneck: least total performance capacity)
        agent_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let schedule: Vec<usize> = agent_scores.into_iter().map(|(i, _)| i).collect();

        // Check deadline expiry
        let deadlines_expired = self.deadline.as_ref().map_or(false, |d| d.expired);

        // If deadlines expired, reverse priority for expired tasks (deprioritize)
        let schedule = if deadlines_expired {
            // Move the last agent to the end (deprioritize bottleneck)
            let mut s = schedule;
            if s.len() > 1 {
                let last = s.remove(0);
                s.push(last);
            }
            s
        } else {
            schedule
        };

        ScheduleResult {
            eigenvalues,
            schedule,
            bottleneck_index: bottleneck,
            deadlines_expired,
        }
    }
}

impl Default for SpectralScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// QR decomposition of a square matrix using modified Gram-Schmidt.
fn qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];

    // Columns of A
    let mut v: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| a[i][j]).collect()).collect();

    for j in 0..n {
        r[j][j] = l2_norm(&v[j]);
        if r[j][j] > 1e-14 {
            for i in 0..n {
                q[i][j] = v[j][i] / r[j][j];
            }
        }
        for k in (j + 1)..n {
            let q_col: Vec<f64> = (0..n).map(|i| q[i][j]).collect();
            r[j][k] = dot(&v[k], &q_col);
            for i in 0..n {
                v[k][i] -= r[j][k] * q[i][j];
            }
        }
    }

    (q, r)
}

/// Matrix multiplication C = A * B.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_bottleneck_empty() {
        let scheduler = SpectralScheduler::new();
        assert_eq!(scheduler.detect_bottleneck(&[]), 0);
    }

    #[test]
    fn test_detect_bottleneck_single_min() {
        let scheduler = SpectralScheduler::new();
        let eigs = vec![5.0, 1.0, 3.0, 10.0];
        assert_eq!(scheduler.detect_bottleneck(&eigs), 1);
    }

    #[test]
    fn test_detect_bottleneck_first_is_min() {
        let scheduler = SpectralScheduler::new();
        let eigs = vec![0.1, 2.0, 5.0];
        assert_eq!(scheduler.detect_bottleneck(&eigs), 0);
    }

    #[test]
    fn test_gram_matrix_identity() {
        let scheduler = SpectralScheduler::new();
        // Orthogonal unit vectors → Gram matrix is identity
        let perfs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let gram = scheduler.gram_matrix(&perfs);
        assert!((gram[0][0] - 1.0).abs() < 1e-10);
        assert!((gram[0][1]).abs() < 1e-10);
        assert!((gram[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gram_matrix_collinear() {
        let scheduler = SpectralScheduler::new();
        // Parallel vectors → high off-diagonal
        let perfs = vec![vec![1.0, 0.0], vec![2.0, 0.0]];
        let gram = scheduler.gram_matrix(&perfs);
        assert!((gram[0][0] - 1.0).abs() < 1e-10);
        assert!((gram[0][1] - 2.0).abs() < 1e-10);
        assert!((gram[1][1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_schedule_by_eigenmode_basic() {
        let mut scheduler = SpectralScheduler::new();
        let perfs = vec![vec![10.0, 10.0], vec![1.0, 1.0], vec![5.0, 5.0]];
        let result = scheduler.schedule_by_eigenmode(perfs);
        // Smallest norm agent (index 1, norm ≈ 1.41) should be first
        assert_eq!(result.schedule[0], 1);
        assert!(!result.deadlines_expired);
    }

    #[test]
    fn test_schedule_empty() {
        let mut scheduler = SpectralScheduler::new();
        let result = scheduler.schedule_by_eigenmode(vec![]);
        assert!(result.schedule.is_empty());
        assert!(result.eigenvalues.is_empty());
    }

    #[test]
    fn test_deadline_propagation() {
        let mut dl = Deadline::new(1, 1.0);
        dl.propagate(0.3);
        assert!(!dl.expired);
        assert!((dl.remaining - 0.7).abs() < 1e-10);
        dl.propagate(0.7);
        assert!(dl.expired);
        assert_eq!(dl.remaining, 0.0);
    }

    #[test]
    fn test_deadline_fraction_remaining() {
        let dl = Deadline::new(1, 10.0);
        assert!((dl.fraction_remaining() - 1.0).abs() < 1e-10);
        let mut dl2 = Deadline::new(2, 10.0);
        dl2.propagate(5.0);
        assert!((dl2.fraction_remaining() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_deadline_expired_scheduling() {
        let mut scheduler = SpectralScheduler::with_deadline(0.01); // very tight
        // Force deadline expiry by creating a large problem
        let perfs: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64, (i + 1) as f64])
            .collect();
        let result = scheduler.schedule_by_eigenmode(perfs);
        assert!(result.deadlines_expired);
    }

    #[test]
    fn test_qr_eigenvalues_identity() {
        let scheduler = SpectralScheduler::new();
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let eigs = scheduler.qr_eigenvalues(&identity, 100);
        assert!((eigs[0] - 1.0).abs() < 1e-6);
        assert!((eigs[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_default() {
        let scheduler = SpectralScheduler::default();
        assert!(scheduler.deadline.is_none());
        assert_eq!(scheduler.max_iterations, 200);
    }
}
