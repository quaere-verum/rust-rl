//! Opt-in observations of a fresh recurrent learning objective.
use super::RecurrentDdqn;
use crate::{
    Error, Result,
    core::{ensure, temporal::TemporalExecutionMode},
    data::temporal::SequenceBatch,
    nn::{finite_loss, to_vec},
};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::Tensor,
};
use serde::{Deserialize, Serialize};

/// Raw, pre-clipping gradients from the current checkpoint model and the caller's
/// diagnostic batch. These summarize a fresh sample, rather than gradients from
/// earlier optimizer updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentDiagnostics {
    pub raw_gradient_l2: f64,
    pub gradient_elements: usize,
    pub nonzero_gradient_elements: usize,
    pub max_abs_gradient: f64,
    pub encoder_gradient_l2: f64,
    pub head_gradient_l2: f64,
    pub critic_loss: f64,
    pub target_mean: f64,
    pub learning_positions: usize,
    pub approximate_online_anchor_rows: usize,
    pub approximate_target_anchor_rows: usize,
    /// Largest current-generation minus stored-anchor-generation difference.
    /// This is direct anchor age, not a bound on accumulated ancestry error.
    pub max_direct_generation_lag: u64,
}

#[derive(Default)]
struct GradientSummary {
    squared_l2: f64,
    elements: usize,
    nonzero_elements: usize,
    max_abs: f64,
}

fn summarize_gradients(
    model: &impl Module,
    gradients: &GradientsParams,
) -> Result<GradientSummary> {
    struct Collect<'a> {
        gradients: &'a GradientsParams,
        summary: GradientSummary,
        error: Option<Error>,
    }
    impl ModuleVisitor for Collect<'_> {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            if self.error.is_some() {
                return;
            }
            let Some(gradient) = self.gradients.get::<D>(parameter.id) else {
                self.error = Some(Error::Numerical(
                    "missing recurrent diagnostic parameter gradient".into(),
                ));
                return;
            };
            let values = match to_vec(gradient) {
                Ok(values) => values,
                Err(error) => {
                    self.error = Some(error);
                    return;
                }
            };
            for value in values {
                if !value.is_finite() {
                    self.error = Some(Error::Numerical(
                        "nonfinite recurrent diagnostic gradient".into(),
                    ));
                    return;
                }
                let value = f64::from(value);
                self.summary.squared_l2 += value * value;
                self.summary.elements += 1;
                self.summary.nonzero_elements += usize::from(value != 0.);
                self.summary.max_abs = self.summary.max_abs.max(value.abs());
            }
        }
    }
    let mut collector = Collect {
        gradients,
        summary: GradientSummary::default(),
        error: None,
    };
    model.visit(&mut collector);
    if let Some(error) = collector.error {
        return Err(error);
    }
    ensure(
        collector.summary.squared_l2.is_finite(),
        "nonfinite recurrent diagnostic gradient norm",
    )?;
    Ok(collector.summary)
}

impl RecurrentDdqn {
    /// Explicitly compute one fresh objective and backward pass using the
    /// selected update mode, without changing parameters, optimizer moments,
    /// update counters, or random streams. Callers choose monitoring checkpoints
    /// and sample diagnostic batches independently of the training RNG.
    ///
    /// Anchored reconstruction may populate the same derived actor caches as a
    /// read-only Q-value query; those caches are absent from checkpoints.
    pub fn diagnose_update(&self, batch: &SequenceBatch) -> Result<RecurrentDiagnostics> {
        self.config.validate()?;
        batch.validate()?;
        ensure(
            batch.config.n_step == self.config.n_step,
            "recurrent batch n-step contract mismatch",
        )?;
        let decisions = self.reconstruction_decisions(batch)?;
        let objective = match self.critic.update_mode {
            TemporalExecutionMode::Reference => self.critic.reference_objective(batch)?,
            TemporalExecutionMode::OptimizedExact => self.critic.optimized_objective(batch)?,
        };
        let critic_loss = finite_loss(&objective.loss)?;
        let learning_positions = objective.targets.len();
        let target_mean = objective
            .targets
            .iter()
            .map(|point| f64::from(point.value))
            .sum::<f64>()
            / learning_positions as f64;
        ensure(
            learning_positions > 0 && target_mean.is_finite(),
            "invalid recurrent diagnostic targets",
        )?;
        // This graph is consumed exactly once. No optimizer is constructed or
        // advanced, and the normal recurrent update path remains independent.
        let gradients = GradientsParams::from_grads(objective.loss.backward(), &self.critic.online);
        let encoder = summarize_gradients(&self.critic.online.encoder, &gradients)?;
        let head = summarize_gradients(&self.critic.online.head, &gradients)?;
        Ok(RecurrentDiagnostics {
            raw_gradient_l2: (encoder.squared_l2 + head.squared_l2).sqrt(),
            gradient_elements: encoder.elements + head.elements,
            nonzero_gradient_elements: encoder.nonzero_elements + head.nonzero_elements,
            max_abs_gradient: encoder.max_abs.max(head.max_abs),
            encoder_gradient_l2: encoder.squared_l2.sqrt(),
            head_gradient_l2: head.squared_l2.sqrt(),
            critic_loss,
            target_mean,
            learning_positions,
            approximate_online_anchor_rows: decisions
                .iter()
                .filter(|(online, _)| online.approximate)
                .count(),
            approximate_target_anchor_rows: decisions
                .iter()
                .filter(|(_, target)| target.approximate)
                .count(),
            max_direct_generation_lag: decisions
                .iter()
                .flat_map(|(online, target)| [online.generation_lag, target.generation_lag])
                .max()
                .unwrap_or(0),
        })
    }
}
