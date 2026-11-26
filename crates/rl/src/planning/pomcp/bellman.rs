/// Empirical chance law at an action edge, including outcomes that terminate
/// or leave the represented tree. Only completed trajectories update it.
#[derive(Default)]
pub(super) struct BellmanEdge {
    direct_mean: f64,
    branches: Vec<Branch>,
}

struct Branch {
    child: usize,
    visits: u32,
    mean_discount: f64,
    mean_frontier: f64,
}

impl BellmanEdge {
    pub(super) fn update(
        &mut self,
        visits: u32,
        direct: f64,
        continuation: Option<(usize, f64, f64)>,
        child_value: impl Fn(usize) -> Option<f64>,
    ) -> f64 {
        self.direct_mean = running_mean(self.direct_mean, direct, visits);
        if let Some((child, discount, fallback)) = continuation {
            let index = self
                .branches
                .iter()
                .position(|branch| branch.child == child)
                .unwrap_or_else(|| {
                    self.branches.push(Branch {
                        child,
                        visits: 0,
                        mean_discount: 0.,
                        mean_frontier: 0.,
                    });
                    self.branches.len() - 1
                });
            let branch = &mut self.branches[index];
            branch.visits += 1;
            branch.mean_discount = running_mean(branch.mean_discount, discount, branch.visits);
            if child_value(child).is_none() {
                branch.mean_frontier =
                    running_mean(branch.mean_frontier, discount * fallback, branch.visits);
            }
        }

        self.branches.iter().fold(self.direct_mean, |sum, branch| {
            let mass = f64::from(branch.visits) / f64::from(visits);
            let continuation = child_value(branch.child)
                .map_or(branch.mean_frontier, |best| branch.mean_discount * best);
            sum + mass * continuation
        })
    }
}

pub(super) fn running_mean(mean: f64, value: f64, visits: u32) -> f64 {
    let count = f64::from(visits);
    ((count - 1.0) / count) * mean + value / count
}
