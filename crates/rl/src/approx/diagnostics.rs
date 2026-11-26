//! Evaluate action predictions and complete policies against independent evidence.
//!
//! The caller supplies the reference and binds it to the same public histories,
//! return units, objective and action identities as the reported predictions.
//! These measurements are independent of fitting and publication.

use crate::{Result, core::ensure};
use serde::Serialize;
use std::{cmp::Ordering, num::NonZeroUsize};

/// Lower and upper values supplied by an independent decision reference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ReferenceInterval {
    pub lower: f64,
    pub upper: f64,
}

impl ReferenceInterval {
    pub fn point(self) -> Option<f64> {
        (self.lower == self.upper).then_some(self.lower)
    }

    fn validate(self) -> Result<()> {
        ensure(
            self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper,
            "invalid reference value interval",
        )
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize)]
pub struct ActionValueErrors {
    pub count: usize,
    pub squared_error: f64,
}

impl ActionValueErrors {
    /// Pooled point-value RMSE. Aggregate the count and squared error before
    /// computing this across decision rows.
    pub fn rmse(self) -> Option<f64> {
        (self.count > 0).then(|| (self.squared_error / self.count as f64).sqrt())
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PairwiseRanking {
    pub possible_pairs: usize,
    /// Pairs with strictly separated reference intervals.
    pub ordered_pairs: usize,
    pub reversed_pairs: usize,
    /// Equal predicted values for a strictly ordered reference pair.
    pub tied_predictions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DecisionDiagnostics {
    pub errors: ActionValueErrors,
    pub ranking: PairwiseRanking,
    pub best_reference_value: ReferenceInterval,
    pub chosen_reference_value: ReferenceInterval,
    /// Regret relative to the best legal action under the supplied reference.
    /// The continuation is whatever the reference uses to define its values.
    pub chosen_regret: ReferenceInterval,
}

/// Compare one decision using complete ordered legal support in both rows.
/// RMSE uses point references. Overlapping or equal intervals provide no
/// strict ranking information, and the chosen action cannot regret itself.
pub fn compare_action_values<A: PartialEq>(
    predictions: &[(A, f64)],
    reference: &[(A, ReferenceInterval)],
    chosen: &A,
) -> Result<DecisionDiagnostics> {
    ensure(
        !predictions.is_empty() && predictions.len() == reference.len(),
        "decision reference requires matching nonempty action support",
    )?;
    let mut errors = ActionValueErrors::default();
    let mut chosen_reference_value = None;
    let mut alternative = ReferenceInterval {
        lower: f64::NEG_INFINITY,
        upper: f64::NEG_INFINITY,
    };
    for (index, ((key, prediction), (reference_key, value))) in
        predictions.iter().zip(reference).enumerate()
    {
        ensure(
            key == reference_key && !predictions[..index].iter().any(|(old, _)| old == key),
            "decision reference action identities differ or repeat",
        )?;
        ensure(prediction.is_finite(), "nonfinite action prediction")?;
        value.validate()?;
        if let Some(point) = value.point() {
            errors.count += 1;
            errors.squared_error += (prediction - point).powi(2);
        }
        if key == chosen {
            chosen_reference_value = Some(*value);
        } else {
            alternative.lower = alternative.lower.max(value.lower);
            alternative.upper = alternative.upper.max(value.upper);
        }
    }
    ensure(
        errors.squared_error.is_finite(),
        "nonfinite action error sum",
    )?;
    let chosen_reference_value = chosen_reference_value.ok_or_else(|| {
        crate::Error::Action("chosen action is outside the reference support".into())
    })?;
    let chosen_regret = ReferenceInterval {
        lower: (alternative.lower - chosen_reference_value.upper).max(0.0),
        upper: (alternative.upper - chosen_reference_value.lower).max(0.0),
    };
    chosen_regret.validate()?;
    let best_reference_value = ReferenceInterval {
        lower: alternative.lower.max(chosen_reference_value.lower),
        upper: alternative.upper.max(chosen_reference_value.upper),
    };
    let mut ranking = PairwiseRanking::default();
    for (left, (_, a)) in reference.iter().enumerate() {
        for (right, (_, b)) in reference.iter().enumerate().skip(left + 1) {
            ranking.possible_pairs += 1;
            let ordering = if a.upper < b.lower {
                Ordering::Less
            } else if b.upper < a.lower {
                Ordering::Greater
            } else {
                continue;
            };
            ranking.ordered_pairs += 1;
            let predicted = predictions[left]
                .1
                .partial_cmp(&predictions[right].1)
                .expect("validated finite predictions");
            if predicted == Ordering::Equal {
                ranking.tied_predictions += 1;
            } else if predicted != ordering {
                ranking.reversed_pairs += 1;
            }
        }
    }
    Ok(DecisionDiagnostics {
        errors,
        ranking,
        best_reference_value,
        chosen_reference_value,
        chosen_regret,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PolicyReturnSource {
    /// One complete physical execution.
    FactualEpisode,
    /// Average of complete physical executions under the evaluation protocol.
    FactualMean { episodes: NonZeroUsize },
    /// Complete policy value relative to the identified model and its initial law.
    ModelRelative { model: [u8; 32] },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PolicyValueComparison {
    pub fitted_value: f64,
    pub deployed_return: f64,
    pub fitted_minus_deployed: f64,
    pub source: PolicyReturnSource,
}

impl PolicyValueComparison {
    /// Compare a fitted prediction for the chosen deployment to the observed
    /// complete return or independently evaluated value in the same units.
    /// A factual sample carries its own sampling uncertainty.
    pub fn new(
        fitted_value: f64,
        deployed_return: f64,
        source: PolicyReturnSource,
    ) -> Result<Self> {
        let fitted_minus_deployed = fitted_value - deployed_return;
        ensure(
            fitted_minus_deployed.is_finite(),
            "nonfinite complete policy value comparison",
        )?;
        Ok(Self {
            fitted_value,
            deployed_return,
            fitted_minus_deployed,
            source,
        })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PolicyChanges {
    pub decisions: usize,
    pub changed: usize,
}

/// Compare two frozen policies' choices on the same public histories.
/// Each pair contains their decisions at one shared history.
pub fn policy_changes<A: PartialEq>(
    observations: impl IntoIterator<Item = (A, A)>,
) -> PolicyChanges {
    let mut report = PolicyChanges::default();
    for (before, after) in observations {
        report.decisions += 1;
        report.changed += usize::from(before != after);
    }
    report
}

/// Compare the executed and reference actions at the same factual history.
/// Both value intervals must come from the same comparison objective and law.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ComparedDecision<A> {
    pub decision: u64,
    pub physical_tick: u64,
    pub chosen: A,
    pub reference: A,
    pub chosen_value: ReferenceInterval,
    pub reference_value: ReferenceInterval,
}

/// Find the first observed disagreement exceeding the declared lower bound
/// on reference advantage. Gaps in reference coverage are allowed, so the
/// answer only covers the supplied, chronologically ordered decision history.
pub fn first_consequential_divergence<A: PartialEq>(
    observations: impl IntoIterator<Item = ComparedDecision<A>>,
    minimum_advantage: f64,
) -> Result<Option<ComparedDecision<A>>> {
    ensure(
        minimum_advantage.is_finite() && minimum_advantage >= 0.0,
        "invalid decision advantage threshold",
    )?;
    let mut previous = None;
    for row in observations {
        ensure(
            previous
                .is_none_or(|(decision, tick)| row.decision > decision && row.physical_tick > tick),
            "decision reference observations are not chronological",
        )?;
        previous = Some((row.decision, row.physical_tick));
        row.chosen_value.validate()?;
        row.reference_value.validate()?;
        if row.chosen != row.reference {
            let advantage = row.reference_value.lower - row.chosen_value.upper;
            ensure(advantage.is_finite(), "nonfinite reference advantage")?;
            if advantage > minimum_advantage {
                return Ok(Some(row));
            }
        }
    }
    Ok(None)
}
