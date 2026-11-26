//! Function-space PCA over complete declared fixed populations.
use super::history::checked as ensure;
use rl::{
    Error, Result,
    core::action::ActionSpace,
    diagnostics::{
        policy_batch::{PolicyBatch, gaussian_log_std, numeric_row},
        policy_numerics::project,
        representations::{Pca, RepresentationLimits, match_reference_sets},
        schema::{Chunk, PolicyParameters},
    },
};

const MAX_CHECKPOINTS: usize = 80;
const MAX_FEATURES: usize = 4096;
const MATRIX_ELEMENTS: usize = 128 * 1024;
const PCA_MULTIPLIES: usize = 12 * 1024 * 1024;

#[derive(Debug)]
pub struct PolicySpacePoint {
    pub revision: u64,
    pub coordinates: [f64; 2],
}

#[derive(Debug)]
pub struct PolicySpaceProjection {
    pub points: Vec<PolicySpacePoint>,
    pub population_rows: usize,
    pub representation: &'static str,
    pub explained: [f64; 2],
    pub omitted_checkpoints: usize,
}

#[derive(Default)]
pub(super) struct PolicySpaceSamples {
    values: Vec<f64>,
    revisions: Vec<u64>,
    width: usize,
    rows: usize,
    label: &'static str,
    omitted_checkpoints: usize,
}

impl PolicySpaceSamples {
    pub(super) fn add(
        &mut self,
        revision: u64,
        policy: &Chunk,
        probes: &Chunk,
        anchor_probes: &Chunk,
        space: &ActionSpace,
        scales: &[f64],
    ) -> Result<()> {
        // The history caller has already certified compatible policy families,
        // identical effective masks and full fixed inputs with policy_change.
        let aligned = match_reference_sets(anchor_probes, probes)?;
        let batch = PolicyBatch::new(policy, Some(probes), space)?;
        let width = match space {
            ActionSpace::Discrete { count } => *count,
            _ => space.dimensions(),
        };
        let twice = matches!(batch.parameters(), PolicyParameters::Gaussian { .. });
        let dimension = aligned.len().saturating_mul(width) * if twice { 2 } else { 1 };
        ensure(
            (1..=MAX_FEATURES).contains(&dimension) && (self.width == 0 || self.width == dimension),
            "policy space requires a complete fixed output vector within the feature budget",
        )?;
        let cap = (2..=MAX_CHECKPOINTS)
            .take_while(|n| dimension * n <= MATRIX_ELEMENTS && dimension * n * n <= PCA_MULTIPLIES)
            .last()
            .ok_or_else(|| Error::Config("Policy PCA work exceeds admission".into()))?;
        let mut vector = Vec::with_capacity(dimension);
        let label = match batch.parameters() {
            PolicyParameters::Gaussian { mean, log_std, .. } => {
                for row in aligned.iter().copied() {
                    let a = batch
                        .parameter_array(mean)
                        .ok_or_else(|| Error::Shape("Policy mean unavailable".into()))?;
                    vector.extend(numeric_row(a, row)?);
                    vector.extend(gaussian_log_std(policy, log_std, row)?);
                }
                "Gaussian latent means and log scales"
            }
            _ => {
                for row in aligned.iter().copied() {
                    for component in 0..width {
                        let scale = scales.get(component).copied().unwrap_or(1.);
                        vector.push(project(&batch, row, component)? / scale);
                    }
                }
                match space {
                    ActionSpace::Discrete { .. } => "Legal-action probability vectors",
                    _ => "Normalized executed actions",
                }
            }
        };
        ensure(
            vector.len() == dimension && vector.iter().all(|v| v.is_finite()),
            "policy PCA output dimensions or values are invalid",
        )?;
        self.width = dimension;
        self.rows = aligned.len();
        self.label = label;
        if self.revisions.last() == Some(&revision) {
            self.values.truncate(self.values.len() - dimension);
        } else if self.revisions.len() >= cap {
            self.omitted_checkpoints += 1;
            self.revisions.pop();
            self.values.truncate(self.values.len() - dimension);
            self.revisions.push(revision);
        } else {
            self.revisions.push(revision);
        }
        self.values.extend(vector);
        Ok(())
    }

    pub(super) fn fit(self) -> Result<PolicySpaceProjection> {
        let count = self.revisions.len();
        ensure(
            count >= 2,
            "policy space needs two compatible recorded checkpoints",
        )?;
        let pca = Pca::fit(
            &self.values,
            count,
            self.width,
            2,
            RepresentationLimits {
                max_samples: MAX_CHECKPOINTS,
                max_features: MAX_FEATURES,
                max_work_elements: 520 * 1024,
            },
        )?;
        let projected = pca.project(&self.values, count)?;
        let explained = pca.explained_fraction();
        Ok(PolicySpaceProjection {
            points: self
                .revisions
                .into_iter()
                .zip(projected.chunks_exact(pca.components()))
                .map(|(revision, row)| PolicySpacePoint {
                    revision,
                    coordinates: [row[0], row.get(1).copied().unwrap_or(0.)],
                })
                .collect(),
            population_rows: self.rows,
            representation: self.label,
            explained: [explained[0], explained.get(1).copied().unwrap_or(0.)],
            omitted_checkpoints: self.omitted_checkpoints,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_coordinates_share_one_basis_and_preserve_order() {
        let data = PolicySpaceSamples {
            values: vec![1., 4., 3., 4., 7., 4.],
            revisions: vec![6, 18, 27],
            width: 2,
            rows: 2,
            label: "two coordinates",
            omitted_checkpoints: 0,
        };
        let fitted = data.fit().unwrap();
        assert_eq!(fitted.explained, [1., 0.]);
        assert_eq!(
            fitted.points.iter().map(|p| p.revision).collect::<Vec<_>>(),
            [6, 18, 27]
        );
        assert!(
            (fitted.points[2].coordinates[0] - fitted.points[0].coordinates[0] - 6.).abs() < 1e-12
        );
        assert!(fitted.points.iter().all(|p| p.coordinates[1] == 0.));
    }
}
