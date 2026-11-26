//! Freeze the existing recurrent candidate representation for a statistical head.

mod cache;
mod checkpoint;

use super::{
    RecurrentLightweightMemory, RecurrentLightweightQ, RecurrentLightweightState,
    StructuredHistoryInput, q::candidate_features,
};
use crate::{
    Error, Result,
    approx::{FeatureMap, FeatureSpec},
    core::{
        action::Action,
        ensure,
        env::{Observation, PublicHistory},
        stateful::HistoryInput,
        structured::{FeatureIdentity, StructuredObservation},
        temporal::TemporalSpaces,
    },
    nn::{checkpoint::hash_weights, structured::PackLimits, to_vec},
};
use burn::module::Module;
use cache::{CachedFrame, Incremental, Lookup};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, num::NonZeroUsize, sync::Arc};

/// Causal candidate features immediately before the existing recurrent Q head.
/// The frozen fingerprint binds the structured schema, architecture and all
/// representation weights. A learned head is neither run nor updated here.
///
/// The default path reconstructs from the genuine reset. An opt-in cache
/// shares recurrent state and complete candidate features across public frames.
/// Serde uses the existing frozen Flex model record and the encoder settings.
#[derive(Debug, Clone)]
pub struct FrozenRecurrentFeatures {
    model: RecurrentLightweightQ,
    limits: PackLimits,
    spec: FeatureSpec,
    incremental: Option<Incremental>,
}

impl FrozenRecurrentFeatures {
    pub fn new(model: RecurrentLightweightQ, limits: PackLimits) -> Result<Self> {
        model.config().validate()?;
        ensure(
            limits.max_elements > 0,
            "frozen feature pack limit must be positive",
        )?;
        let model = model.valid().freeze();
        let config = serde_json::to_string(model.config())?;
        let mut digest = Sha256::new();
        digest.update(b"rust_rl.recurrent_lightweight.pre_q.v1");
        digest.update((config.len() as u64).to_le_bytes());
        digest.update(config.as_bytes());
        hash_weights(&mut digest, &model.encoder)?;
        hash_weights(&mut digest, &model.operations)?;
        hash_weights(&mut digest, &model.event_projection)?;
        hash_weights(&mut digest, &model.decision_projection)?;
        hash_weights(&mut digest, &model.memory)?;
        let spec = FeatureSpec {
            namespace: "rust_rl.recurrent_lightweight.pre_q".into(),
            version: 1,
            terms: (0..model.config().joint_width())
                .map(|i| {
                    FeatureIdentity::new(
                        format!("coordinate_{i}"),
                        "recurrent candidate pre-head tensor",
                    )
                })
                .collect(),
            configuration: vec![config, format!("pack_max_elements={}", limits.max_elements)],
            representation: Some(digest.finalize().into()),
        };
        spec.validate()?;
        Ok(Self {
            model,
            limits,
            spec,
            incremental: None,
        })
    }

    /// Retain up to `capacity` recently encoded frames and use the nearest
    /// retained public ancestor for incremental inference. A cache miss
    /// reconstructs from reset, replaying individual frames when the prefix
    /// cannot be packed together. Clones and restored adapters start with an
    /// empty cache. Weak history references avoid retaining factual tapes.
    pub fn with_incremental(mut self, capacity: NonZeroUsize) -> Self {
        self.incremental = Some(Incremental::new(capacity));
        self
    }

    fn candidate_budget(&self, count: usize) -> Result<usize> {
        let width = self.spec.dimension();
        self.model
            .config()
            .model
            .check_shape(&[count.max(1), width])?;
        count
            .checked_mul(width)
            .ok_or_else(|| Error::Shape("frozen candidate feature allocation overflow".into()))
    }

    fn exact_values(
        &self,
        history: &PublicHistory,
        retain_state: bool,
    ) -> Result<(Vec<f32>, Option<RecurrentLightweightState>)> {
        let inputs = history.inputs();
        let last = inputs.len() - 1;
        let encoded = self.model.history_encoder().encode_at(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: last,
            }],
            &[(0, last)].into_iter().collect(),
            self.limits,
            |_, count| self.candidate_budget(count),
        )?;
        let values = to_vec(candidate_features(&encoded, &self.model.operations))?;
        let state = if retain_state {
            let event_start = last
                .checked_sub(1)
                .map_or(0, |previous| encoded.observations[0][previous].events.len());
            // Own only the final memory row. Tensor views can hold an entire prefix.
            let memory = RecurrentLightweightMemory::from_state(RecurrentLightweightState {
                memory: encoded.states.narrow(0, last, 1),
                event_start,
            })?;
            Some(memory.on_device(&self.model.model_device()))
        } else {
            None
        };
        Ok((values, state))
    }

    fn incremental_values(
        &self,
        cache: &Incremental,
        history: &PublicHistory,
        observation: &StructuredObservation,
    ) -> Result<Arc<CachedFrame>> {
        let (values, state) = match cache.lookup(history)? {
            Lookup::Complete(values) => return Ok(values),
            Lookup::Cold => match self.exact_values(history, true) {
                Ok((values, state)) => (values, state.expect("retained exact prefix state")),
                // A warm adapter may advance past the aggregate packing budget.
                // Apply the same per-call budget when replaying a cold history.
                Err(packed_error) => self
                    .model
                    .initial_state()
                    .and_then(|state| {
                        self.values_after_inputs(state, history.inputs(), observation)
                    })
                    .map_err(|_| packed_error)?,
            },
            Lookup::Ancestor { frame, suffix } => self.values_after_inputs(
                frame.state.clone(),
                suffix.into_iter().rev().map(|frame| HistoryInput {
                    observation: frame.observation(),
                    previous: frame.previous_input(),
                }),
                observation,
            )?,
        };
        self.validate_values(&values, observation.candidates.len())?;
        cache.remember(history, state, values)
    }

    fn values_after_inputs<'a>(
        &self,
        mut state: RecurrentLightweightState,
        inputs: impl IntoIterator<Item = HistoryInput<'a>>,
        observation: &StructuredObservation,
    ) -> Result<(Vec<f32>, RecurrentLightweightState)> {
        let encoder = self.model.history_encoder();
        for input in inputs {
            state = encoder.assimilate(&state, input, self.limits)?;
        }
        let encoded = encoder.decision(&state, observation, self.limits, |_, count| {
            self.candidate_budget(count)
        })?;
        let values = to_vec(candidate_features(&encoded, &self.model.operations))?;
        Ok((values, state))
    }

    fn validate_values(&self, values: &[f32], candidates: usize) -> Result<()> {
        ensure(
            candidates.checked_mul(self.spec.dimension()) == Some(values.len())
                && values.iter().all(|value| value.is_finite()),
            "invalid encoded recurrent candidate features",
        )
    }

    fn select_rows(&self, values: &[f32], rows: Vec<usize>) -> Vec<Vec<f64>> {
        let width = self.spec.dimension();
        rows.into_iter()
            .map(|index| {
                values[index * width..(index + 1) * width]
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect()
            })
            .collect()
    }
}

impl FeatureMap for FrozenRecurrentFeatures {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn validate_spaces(&self, spaces: &TemporalSpaces) -> Result<()> {
        self.model
            .config()
            .model
            .input
            .validate_spaces(&spaces.observations, &spaces.actions)
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        if actions.is_empty() {
            return Ok(Vec::new());
        }
        let Observation::Structured(observation) = history.observation() else {
            return Err(Error::Shape(
                "recurrent candidate features require a structured history".into(),
            ));
        };
        let indices: HashMap<_, _> = observation
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, row)| row.enabled)
            .map(|(index, row)| (row.action, index))
            .collect();
        let rows = actions
            .iter()
            .map(|action| match action {
                Action::Conditional(key) => indices.get(key.as_ref()).copied().ok_or_else(|| {
                    Error::Action("requested recurrent feature has no legal candidate".into())
                }),
                _ => Err(Error::Action(
                    "recurrent candidate features require complete conditional actions".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure(
            history.timestep() < self.model.config().max_observations as u64,
            "frozen recurrent feature history exceeds the observation budget",
        )?;
        if let Some(cache) = &self.incremental {
            let cached = self.incremental_values(cache, history, observation)?;
            Ok(self.select_rows(&cached.values, rows))
        } else {
            let (values, _) = self.exact_values(history, false)?;
            self.validate_values(&values, observation.candidates.len())?;
            Ok(self.select_rows(&values, rows))
        }
    }
}
