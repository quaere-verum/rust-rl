//! Convert explicit neural probes into compact, independent recorded arrays.
pub use super::schema::actions_array;
use super::schema::*;
use super::typed::{
    DiagnosticInput, MAX_TYPED_BYTES, MAX_TYPED_ROWS, TypedBudget, TypedPayload, TypedProbes,
};
#[cfg(feature = "neural")]
use crate::core::actor::ActMode;
#[cfg(feature = "neural")]
use crate::nn::{
    self,
    distributions::Distribution,
    policy::{DeterministicPolicy, StochasticPolicy},
    probes::{ActivationProbe, CriticPrediction, PolicyProbe},
    target::frozen_copy,
};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::{Observation, ObservationSpace},
    },
};
#[cfg(feature = "neural")]
use burn::tensor::{Tensor, TensorData, activation::softmax};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct ProbeSet {
    id: String,
    population: ProbePopulation,
    methodology: String,
    observations: Vec<Observation>,
    fixed_actions: Option<Vec<Action>>,
    row_ids: Vec<u64>,
    space: ActionSpace,
    observation_space: ObservationSpace,
    typed_inputs: Option<Vec<DiagnosticInput>>,
    histories: BTreeMap<String, Array>,
}
impl ProbeSet {
    pub fn new(
        id: &str,
        population: ProbePopulation,
        methodology: &str,
        observations: Vec<Observation>,
        fixed_actions: Option<Vec<Action>>,
        space: ActionSpace,
    ) -> Result<Self> {
        ensure(
            !id.is_empty() && !methodology.is_empty() && !observations.is_empty(),
            "probe set needs identity, methodology and inputs",
        )?;
        space.validate()?;
        let width = observations[0].dense_values().len();
        ensure(width > 0, "probe observation width must be positive")?;
        for observation in &observations {
            observation.validate(width, &space, true)?;
        }
        if let Some(actions) = &fixed_actions {
            ensure(
                actions.len() == observations.len(),
                "probe actions and observations differ in length",
            )?;
            for (action, observation) in actions.iter().zip(&observations) {
                space.check(action, observation.dense_mask())?;
            }
        }
        Ok(Self {
            id: id.into(),
            population,
            methodology: methodology.into(),
            row_ids: (0..observations.len() as u64).collect(),
            observations,
            fixed_actions,
            space,
            observation_space: ObservationSpace::Dense(width),
            typed_inputs: None,
            histories: BTreeMap::new(),
        })
    }

    /// Construct a probe population with lossless observations or complete histories.
    pub fn new_in(
        id: &str,
        population: ProbePopulation,
        methodology: &str,
        inputs: Vec<DiagnosticInput>,
        fixed_actions: Option<Vec<Action>>,
        observation_space: ObservationSpace,
        space: ActionSpace,
    ) -> Result<Self> {
        ensure(
            !id.is_empty()
                && id.len() <= 256
                && methodology.len() <= 4096
                && (1..=MAX_TYPED_ROWS).contains(&inputs.len()),
            "invalid typed probe identity",
        )?;
        observation_space.validate()?;
        space.validate()?;
        let probes = TypedProbes {
            population,
            methodology: methodology.into(),
            row_ids: (0..inputs.len() as u64).collect(),
            inputs,
            fixed_actions,
        };
        probes.validate_in(&observation_space, &space)?;
        let observations = probes
            .inputs
            .iter()
            .map(|input| input.observation().clone())
            .collect();
        Ok(Self {
            id: id.into(),
            population: probes.population,
            methodology: probes.methodology,
            observations,
            fixed_actions: probes.fixed_actions,
            row_ids: probes.row_ids,
            space,
            observation_space,
            typed_inputs: Some(probes.inputs),
            histories: BTreeMap::new(),
        })
    }
    /// Samples without replacement using the diagnostic RNG, never the collector
    /// RNG. The pool must come from a separately recorded evaluation trajectory.
    pub fn visited(
        id: &str,
        methodology: &str,
        pool: &[Observation],
        count: usize,
        space: ActionSpace,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let width = pool.first().map_or(0, |obs| obs.dense_values().len());
        Self::visited_in(
            id,
            methodology,
            pool,
            count,
            (width.into(), space),
            rng,
            MAX_TYPED_BYTES,
        )
    }

    pub(crate) fn visited_in(
        id: &str,
        methodology: &str,
        pool: &[Observation],
        count: usize,
        spaces: (ObservationSpace, ActionSpace),
        rng: &mut ChaCha8Rng,
        max_chunk_bytes: usize,
    ) -> Result<Self> {
        ensure(
            count > 0
                && count <= pool.len()
                && (spaces.0.dense_width().is_some() || count <= MAX_TYPED_ROWS),
            "invalid visited probe sample size",
        )?;
        let indices = rand::seq::index::sample(rng, pool.len(), count).into_vec();
        let mut set = Self::from_observations(
            id,
            ProbePopulation::PolicyVisited,
            methodology,
            indices.iter().map(|&i| &pool[i]),
            spaces.0,
            spaces.1,
            max_chunk_bytes,
        )?;
        set.row_ids = indices.into_iter().map(|i| i as u64).collect();
        Ok(set)
    }

    pub(crate) fn from_observations<'a>(
        id: &str,
        population: ProbePopulation,
        methodology: &str,
        observations: impl ExactSizeIterator<Item = &'a Observation> + Clone,
        observation_space: ObservationSpace,
        space: ActionSpace,
        max_chunk_bytes: usize,
    ) -> Result<Self> {
        if observation_space.dense_width().is_some() {
            Self::new(
                id,
                population,
                methodology,
                observations.cloned().collect(),
                None,
                space,
            )
        } else {
            ensure(
                (1..=MAX_TYPED_ROWS).contains(&observations.len()),
                "invalid typed observation count",
            )?;
            let mut budget = TypedBudget::new(max_chunk_bytes);
            for observation in observations.clone() {
                budget.observation(observation)?;
                budget.charge(size_of::<DiagnosticInput>().saturating_add(size_of::<u64>()))?;
            }
            Self::new_in(
                id,
                population,
                methodology,
                observations.cloned().map(DiagnosticInput::from).collect(),
                None,
                observation_space,
                space,
            )
        }
    }
    /// Optional imported observation histories or recurrent hidden states. The
    /// leading dimension must match probe rows; the backend need not be recurrent.
    pub fn with_history(mut self, name: &str, array: Array) -> Result<Self> {
        ensure(
            self.typed_inputs.is_none()
                && !name.is_empty()
                && array.shape()[0] == self.observations.len(),
            "invalid imported probe history",
        )?;
        self.histories.insert(name.into(), array);
        Ok(self)
    }
    /// Supply durable caller-owned row IDs, one per observation.
    pub fn with_row_ids(mut self, ids: Vec<u64>) -> Result<Self> {
        let mut unique = std::collections::BTreeSet::new();
        ensure(
            ids.len() == self.observations.len() && ids.iter().all(|id| unique.insert(*id)),
            "diagnostic probe IDs must be unique and match the row count",
        )?;
        self.row_ids = ids;
        Ok(self)
    }

    pub fn population(&self) -> &ProbePopulation {
        &self.population
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }
    pub fn typed_inputs(&self) -> Option<&[DiagnosticInput]> {
        self.typed_inputs.as_deref()
    }
    pub fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }
    pub fn observation_space(&self) -> &ObservationSpace {
        &self.observation_space
    }
    pub fn fixed_actions(&self) -> Option<&[Action]> {
        self.fixed_actions.as_deref()
    }
    pub fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    pub fn record(&self, context: RecordContext) -> Result<Chunk> {
        self.record_with_limit(context, MAX_TYPED_BYTES)
    }

    pub(crate) fn record_with_limit(
        &self,
        mut context: RecordContext,
        max_chunk_bytes: usize,
    ) -> Result<Chunk> {
        if self.typed_inputs.is_some() {
            return self.record_as_typed(context, max_chunk_bytes);
        }
        context.probe_set = Some(self.id.clone());
        let n = self.observations.len();
        let width = self.observations[0].dense_values().len();
        let mut arrays = BTreeMap::from([
            (
                "observations".into(),
                Array::f32(
                    vec![n, width],
                    self.observations
                        .iter()
                        .flat_map(|o| o.dense_values().iter().copied())
                        .collect(),
                )?,
            ),
            ("row_ids".into(), Array::u64(vec![n], self.row_ids.clone())?),
        ]);
        let action_masks = masks(self, &mut arrays)?;
        let fixed_actions = if let Some(actions) = &self.fixed_actions {
            arrays.insert("fixed_actions".into(), actions_array(actions)?);
            Some("fixed_actions".into())
        } else {
            None
        };
        let mut histories = BTreeMap::new();
        for (name, array) in &self.histories {
            let key = format!("history/{name}");
            arrays.insert(key.clone(), array.clone());
            histories.insert(name.clone(), key);
        }
        Ok(Chunk {
            context,
            record: Record::ProbeSet(ProbeSetRecord {
                id: self.id.clone(),
                population: self.population.clone(),
                methodology: self.methodology.clone(),
                observations: "observations".into(),
                action_masks,
                fixed_actions,
                row_ids: "row_ids".into(),
                histories,
            }),
            arrays,
        })
    }

    pub(crate) fn record_as_typed(
        &self,
        mut context: RecordContext,
        limit: usize,
    ) -> Result<Chunk> {
        ensure(
            (1..=MAX_TYPED_ROWS).contains(&self.observations.len()),
            "invalid typed probe row count",
        )?;
        let mut budget = TypedBudget::new(limit);
        budget.charge(
            self.methodology
                .len()
                .saturating_add(size_of_val(self.row_ids.as_slice())),
        )?;
        if let Some(inputs) = &self.typed_inputs {
            for input in inputs {
                budget.input(input)?;
            }
        } else {
            for observation in &self.observations {
                budget.observation(observation)?;
                budget.charge(size_of::<DiagnosticInput>())?;
            }
        }
        for action in self.fixed_actions.iter().flatten() {
            budget.action(action)?;
        }
        context.probe_set = Some(self.id.clone());
        let inputs = self.typed_inputs.clone().unwrap_or_else(|| {
            self.observations
                .iter()
                .cloned()
                .map(DiagnosticInput::from)
                .collect()
        });
        TypedPayload::Probes(TypedProbes {
            population: self.population.clone(),
            methodology: self.methodology.clone(),
            row_ids: self.row_ids.clone(),
            inputs,
            fixed_actions: self.fixed_actions.clone(),
        })
        .record(context, limit)
    }
}
fn masks(set: &ProbeSet, arrays: &mut BTreeMap<String, Array>) -> Result<Option<ArrayKey>> {
    if let ActionSpace::Discrete { count } = set.space {
        arrays.insert(
            "action_masks".into(),
            Array::u8(
                vec![set.observations.len(), count],
                set.observations
                    .iter()
                    .flat_map(|o| (0..count).map(|i| u8::from(o.dense_mask().is_none_or(|m| m[i]))))
                    .collect(),
            )?,
        );
        Ok(Some("action_masks".into()))
    } else {
        Ok(None)
    }
}
#[cfg(feature = "neural")]
pub fn continuous_actions(actions: Tensor<2>) -> Result<Vec<Action>> {
    let [rows, width] = actions.dims();
    ensure(rows > 0 && width > 0, "empty action tensor")?;
    Ok(nn::to_vec(actions)?
        .chunks(width)
        .map(|v| Action::Continuous(v.into()))
        .collect())
}
#[cfg(feature = "neural")]
fn tensor<const D: usize>(value: Tensor<D>) -> Result<Array> {
    Array::f32(value.dims().to_vec(), nn::to_vec(value.without_autodiff())?)
}
#[cfg(feature = "neural")]
pub fn policy<P: StochasticPolicy>(
    mut context: RecordContext,
    model: &P,
    set: &ProbeSet,
    mode: ActMode,
    rng: &mut ChaCha8Rng,
) -> Result<Chunk> {
    ensure(
        model.action_space() == set.action_space(),
        "policy/probe action spaces differ",
    )?;
    context.probe_set = Some(set.id.clone());
    let prediction = PolicyProbe(model).predict(set.observations().into(), mode, rng)?;
    let mut arrays = BTreeMap::from([
        ("actions".into(), actions_array(&prediction.actions)?),
        ("entropy".into(), tensor(prediction.entropy)?),
    ]);
    let entropy_method = match &prediction.distribution {
        Distribution::Categorical { .. } => "exact categorical entropy",
        Distribution::Gaussian { bounds: None, .. } => "exact diagonal Gaussian differential entropy",
        Distribution::Gaussian { bounds: Some(_), .. } => "one independent latent Monte Carlo sample of transformed differential entropy per state",
    }.to_string();
    let parameters = match prediction.distribution {
        Distribution::Categorical { logits, masks } => {
            let invalid = Tensor::from_data(
                TensorData::new(masks.iter().map(|m| !*m).collect::<Vec<_>>(), logits.dims()),
                &logits.device(),
            );
            arrays.insert(
                "probabilities".into(),
                tensor(softmax(logits.mask_fill(invalid, f32::NEG_INFINITY), 1))?,
            );
            PolicyParameters::Categorical {
                probabilities: "probabilities".into(),
            }
        }
        Distribution::Gaussian {
            mean,
            log_std,
            bounds,
        } => {
            arrays.insert("mean".into(), tensor(mean)?);
            arrays.insert("log_std".into(), tensor(log_std)?);
            PolicyParameters::Gaussian {
                mean: "mean".into(),
                log_std: "log_std".into(),
                tanh_bounded: bounds.is_some(),
            }
        }
    };
    let masks = masks(set, &mut arrays)?;
    Ok(Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters,
            entropy: Some("entropy".into()),
            entropy_method: Some(entropy_method),
            masks,
            stochastic: mode == ActMode::Sample,
        }),
        arrays,
    })
}
#[cfg(feature = "neural")]
pub fn deterministic_policy<P: DeterministicPolicy>(
    mut context: RecordContext,
    model: &P,
    set: &ProbeSet,
) -> Result<Chunk> {
    ensure(
        model.action_space() == set.action_space(),
        "policy/probe action spaces differ",
    )?;
    context.probe_set = Some(set.id.clone());
    let actions = frozen_copy(model).actions(set.observations().into())?;
    Ok(Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: false,
        }),
        arrays: BTreeMap::from([("actions".into(), tensor(actions)?)]),
    })
}
#[cfg(feature = "neural")]
pub(crate) fn categorical_return_bytes(rows: usize, width: usize) -> Result<usize> {
    rows.checked_add(1)
        .and_then(|cells| cells.checked_mul(width))
        .and_then(|cells| cells.checked_mul(4))
        .ok_or_else(|| crate::Error::Shape("categorical return prediction size overflow".into()))
}

#[cfg(feature = "neural")]
pub fn critic(
    context: RecordContext,
    prediction: CriticPrediction,
    action_condition: ActionCondition,
    actions: Option<&[Action]>,
) -> Result<Chunk> {
    ensure(
        !matches!(
            action_condition,
            ActionCondition::Fixed | ActionCondition::CurrentPolicy
        ) || actions.is_some(),
        "action-conditioned predictions must retain their actual evaluated actions",
    )?;
    let mut arrays = BTreeMap::new();
    let representation = match prediction {
        CriticPrediction::Scalar(values) => {
            arrays.insert("values".into(), tensor(values)?);
            CriticRepresentation::Scalar {
                values: "values".into(),
            }
        }
        CriticPrediction::Discrete(values) => {
            arrays.insert("values".into(), tensor(values)?);
            CriticRepresentation::Discrete {
                values: "values".into(),
            }
        }
        CriticPrediction::Quantiles {
            values,
            locations,
            masses,
        } => {
            arrays.insert("values".into(), tensor(values)?);
            arrays.insert(
                "locations".into(),
                Array::f32(vec![locations.len()], locations)?,
            );
            arrays.insert("masses".into(), Array::f32(vec![masses.len()], masses)?);
            CriticRepresentation::Quantiles {
                values: "values".into(),
                locations: "locations".into(),
                masses: "masses".into(),
            }
        }
        CriticPrediction::Moments { mean, raw_second } => {
            arrays.insert("mean".into(), tensor(mean)?);
            arrays.insert("raw_second".into(), tensor(raw_second)?);
            CriticRepresentation::RawMoments {
                mean: "mean".into(),
                raw_second: "raw_second".into(),
            }
        }
        CriticPrediction::CategoricalReturn {
            atoms,
            probabilities,
        } => {
            arrays.insert("atoms".into(), Array::f32(vec![atoms.len()], atoms)?);
            arrays.insert("probabilities".into(), tensor(probabilities)?);
            CriticRepresentation::CategoricalReturn {
                atoms: "atoms".into(),
                probabilities: "probabilities".into(),
            }
        }
    };
    let actions = if let Some(actions) = actions {
        let predicted_rows = match &representation {
            CriticRepresentation::Scalar { values }
            | CriticRepresentation::Discrete { values }
            | CriticRepresentation::Quantiles { values, .. }
            | CriticRepresentation::CategoricalRegression { values, .. } => {
                arrays[values].shape()[0]
            }
            CriticRepresentation::RawMoments { mean, .. } => arrays[mean].shape()[0],
            CriticRepresentation::CategoricalReturn { probabilities, .. } => {
                arrays[probabilities].shape()[0]
            }
        };
        ensure(
            actions.len() == predicted_rows,
            "critic probe actions and predictions differ in length",
        )?;
        arrays.insert("actions".into(), actions_array(actions)?);
        Some("actions".into())
    } else {
        None
    };
    Ok(Chunk {
        context,
        record: Record::Critic(CriticRecord {
            representation,
            action_condition,
            actions,
            continuation_policy: None,
            continuation_stochastic: None,
        }),
        arrays,
    })
}
/// Attach an explicit continuation-policy identity for conditional comparison.
/// Imported records still require the producer to label their policy honestly.
#[cfg(feature = "neural")]
pub fn critic_for_policy(
    context: RecordContext,
    prediction: CriticPrediction,
    action_condition: ActionCondition,
    actions: Option<&[Action]>,
    policy_id: &str,
    mode: ActMode,
) -> Result<Chunk> {
    ensure(
        !policy_id.is_empty(),
        "empty critic continuation-policy identity",
    )?;
    let mut chunk = critic(context, prediction, action_condition, actions)?;
    if let Record::Critic(record) = &mut chunk.record {
        record.continuation_policy = Some(policy_id.into());
        record.continuation_stochastic = Some(mode == ActMode::Sample);
    }
    Ok(chunk)
}
pub fn activation_summary(values: &[f32]) -> ActivationSummary {
    let mut count = 0u64;
    let mut mean = 0.;
    let mut m2 = 0.;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for &x in values.iter().filter(|x| x.is_finite()) {
        let x = x as f64;
        count += 1;
        let delta = x - mean;
        mean += delta / count as f64;
        m2 += delta * (x - mean);
        minimum = minimum.min(x);
        maximum = maximum.max(x);
    }
    ActivationSummary {
        count: values.len() as u64,
        finite_count: count,
        mean: (count > 0).then_some(mean),
        standard_deviation: (count > 0).then(|| (m2 / count as f64).max(0.).sqrt()),
        minimum: (count > 0).then_some(minimum),
        maximum: (count > 0).then_some(maximum),
        zero_fraction: values.iter().filter(|&&x| x == 0.).count() as f64
            / values.len().max(1) as f64,
    }
}
#[cfg(feature = "neural")]
pub fn activations(
    context: RecordContext,
    network: &str,
    model: &impl ActivationProbe,
    input: Tensor<2>,
    names: &[&str],
) -> Result<Chunk> {
    let outputs = model.activations(input, names)?;
    let mut arrays = BTreeMap::new();
    let mut layers = BTreeMap::new();
    let mut summaries = BTreeMap::new();
    for output in outputs {
        let array = tensor(output.tensor)?;
        summaries.insert(output.name.clone(), activation_summary(array.as_f32()?));
        layers.insert(output.name.clone(), output.name.clone());
        arrays.insert(output.name, array);
    }
    Ok(Chunk {
        context,
        record: Record::Activations(ActivationRecord {
            network: network.into(),
            layers,
            summaries,
            ..Default::default()
        }),
        arrays,
    })
}
