//! Shared statistical Q design and legal public successor feature rows.

use crate::{
    Result,
    approx::batch::FactualBatch,
    approx::{FittedQ, regression::Predictor},
    core::{action::Action, ensure, env::PublicHistory},
    planning::guidance::{PolicyGuide, validate_prior},
};
use nalgebra::DMatrix;
use std::{collections::HashMap, num::NonZeroUsize};

pub(super) type Policy<'a> = dyn PolicyGuide<PublicHistory, Action> + 'a;

pub(super) struct Successor<'a> {
    pub history: &'a PublicHistory,
    pub actions: &'a [Action],
    pub features: Vec<&'a [f64]>,
}

impl Successor<'_> {
    pub fn probabilities(&self, policy: &Policy<'_>) -> Result<Vec<f64>> {
        let probabilities = policy.prior(self.history, self.actions)?;
        validate_prior(self.actions, &probabilities)?;
        Ok(probabilities)
    }

    pub fn values(&self, predictor: Option<&impl Predictor>) -> Result<Vec<f64>> {
        self.features
            .iter()
            .map(|features| {
                let value = predictor.map_or(Ok(0.), |model| model.predict(features))?;
                ensure(value.is_finite(), "nonfinite successor action value")?;
                Ok(value)
            })
            .collect()
    }
}

pub(super) struct PreparedRow {
    pub reward: f64,
    pub discount: f64,
    pub successor: Option<usize>,
    pub monte_carlo: Option<f64>,
    pub source: usize,
}

pub(super) struct PreparedBatch<'a> {
    pub design: DMatrix<f64>,
    pub weights: Vec<f64>,
    pub successors: Vec<Successor<'a>>,
    pub rows: Vec<PreparedRow>,
}

impl<'a> PreparedBatch<'a> {
    pub fn new(
        batch: &'a FactualBatch<'_, '_>,
        horizon: NonZeroUsize,
        use_td: bool,
        require_mc: bool,
    ) -> Result<Self> {
        batch.validate_feature_revision()?;
        let mut successor_slots = HashMap::new();
        let mut successors = Vec::new();
        let mut rows = Vec::new();
        let mut weights = Vec::new();
        for (source, factual) in batch.rows().iter().enumerate() {
            let monte_carlo = if require_mc {
                batch.monte_carlo(source)?
            } else {
                None
            };
            if require_mc && monte_carlo.is_none() {
                continue;
            }
            let (reward, discount, successor) = if use_td {
                let target = batch.n_step(source, horizon)?;
                let successor = if target.discount == 0. {
                    None
                } else {
                    let key = (factual.episode_id, target.successor.timestep());
                    let index = match successor_slots.get(&key) {
                        Some(index) => *index,
                        None => {
                            let index = successors.len();
                            let (actions, features) =
                                batch.successor_features(source, target.successor)?;
                            successors.push(Successor {
                                history: target.successor,
                                actions,
                                features,
                            });
                            successor_slots.insert(key, index);
                            index
                        }
                    };
                    Some(index)
                };
                (target.reward, target.discount, successor)
            } else {
                (0., 0., None)
            };
            rows.push(PreparedRow {
                reward,
                discount,
                successor,
                monte_carlo,
                source,
            });
            weights.push(factual.weight);
        }
        ensure(!rows.is_empty(), "no eligible factual fitting rows")?;
        batch.validate_feature_revision()?;
        let width = batch.feature_spec().dimension();
        let design = DMatrix::from_fn(rows.len(), width, |i, j| {
            batch.rows()[rows[i].source].features[j]
        });
        Ok(Self {
            design,
            weights,
            successors,
            rows,
        })
    }
}

pub(super) fn bind<M: Predictor>(batch: &FactualBatch<'_, '_>, model: M) -> Result<FittedQ<M>> {
    batch.validate_feature_revision()?;
    FittedQ::new(
        batch.feature_spec().clone(),
        batch.spaces().clone(),
        batch.objective(),
        model,
    )
}

pub(super) fn check_initial<M: Predictor>(
    batch: &FactualBatch<'_, '_>,
    initial: &FittedQ<M>,
) -> Result<()> {
    initial.validate()?;
    ensure(
        batch.feature_spec() == initial.feature_spec()
            && batch.spaces() == initial.spaces()
            && batch.objective() == initial.objective(),
        "initial predictor has incompatible features, spaces, or return semantics",
    )
}
