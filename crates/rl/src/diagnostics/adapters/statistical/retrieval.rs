use super::{
    FiniteRow, RootActor, dense_policy, empty_metrics, evaluate_rows, metric, metric_chunk,
    snapshot, typed_policy,
};
use crate::{
    Error, Result,
    approx::{FeatureSpec, ReturnWeightedRetrieval, greedy_index},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        ensure,
        env::{Observation, PublicHistory},
        temporal::TemporalSpaces,
    },
    diagnostics::{
        codec,
        models::ModelLimits,
        schema::{Chunk, RecordContext},
        session::{CaptureRequest, DiagnosticLearner, LearnerInfo},
        typed::FiniteValueKind,
    },
};
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

const ARCHITECTURE: &str = "rust-rl/return-weighted-retrieval/1";
const MODEL: &str = "retrieval_population";

/// The frozen state and semantic action transformations used to populate the bank.
/// Specify the ordering and scaling of both coordinate systems in their feature contracts.
pub trait RetrievalFeatureMap {
    fn state_spec(&self) -> &FeatureSpec;
    fn action_spec(&self) -> &FeatureSpec;
    fn encode_state(&self, history: &PublicHistory) -> Result<Vec<f64>>;
    fn encode_actions(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>>;
}

type RetrievalEncoder<F> = fn(&RetrievalDiagnostics<'_, F>, usize) -> Result<Vec<u8>>;

/// A completed retrieval publication with its original feature transformations.
/// Observation-only evaluation assumes those transformations are observation-sufficient.
pub struct RetrievalDiagnostics<'a, F> {
    bank: &'a ReturnWeightedRetrieval,
    features: &'a F,
    spaces: TemporalSpaces,
    revision: u64,
    portable: Option<RetrievalEncoder<F>>,
}

impl<'a, F: RetrievalFeatureMap> RetrievalDiagnostics<'a, F> {
    pub fn new(
        bank: &'a ReturnWeightedRetrieval,
        features: &'a F,
        spaces: TemporalSpaces,
        revision: u64,
    ) -> Result<Self> {
        validate_contract(bank, features, &spaces)?;
        Ok(Self {
            bank,
            features,
            spaces,
            revision,
            portable: None,
        })
    }

    /// Store the feature transformations alongside the frozen retrieval population.
    pub fn with_portable_model(mut self) -> Self
    where
        F: Serialize + DeserializeOwned,
    {
        self.portable = Some(|adapter, limit| {
            codec::bounded_json(
                &PortableBank {
                    model: adapter.bank_payload(),
                    features: adapter.features,
                },
                limit,
            )
        });
        self
    }

    fn bank_payload(&self) -> BankPayload<'_, &ReturnWeightedRetrieval> {
        BankPayload {
            bank: self.bank,
            spaces: &self.spaces,
            revision: self.revision,
            state: self.features.state_spec(),
            action: self.features.action_spec(),
        }
    }

    fn predict(&self, history: &PublicHistory) -> Result<FiniteRow> {
        predictions(self.bank, self.features, &self.spaces, history)
    }
}

impl<F: RetrievalFeatureMap> DiagnosticLearner for RetrievalDiagnostics<'_, F> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        Ok(LearnerInfo {
            algorithm: "Return-weighted retrieval",
            model: "return-weighted neighbor population",
            configuration: serde_json::json!({
                "capacity": self.bank.capacity(),
                "neighbors": self.bank.neighbors(),
                "state_features": self.features.state_spec().namespace,
                "action_features": self.features.action_spec().namespace,
                "policy_outputs": "kernel weights normalized over legal actions",
                "evaluation_inputs": "observation roots",
            }),
            discount: None,
        })
    }

    fn diagnostic_revision(&self) -> u64 {
        self.revision
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(RootActor {
            space: &self.spaces.actions,
            decide: |observation: &Observation, mode, rng: &mut ChaCha8Rng| {
                let mut row = self.predict(&PublicHistory::new(observation.clone()))?;
                let index = match mode {
                    ActMode::Deterministic => row.selected,
                    ActMode::Sample => WeightedIndex::new(&row.probabilities)
                        .map_err(|error| Error::Numerical(error.to_string()))?
                        .sample(rng),
                };
                Ok(row.actions.swap_remove(index))
            },
        }))
    }

    fn diagnostic_metrics(&self, context: &RecordContext) -> Result<Option<Chunk>> {
        let mut metrics = empty_metrics();
        let bank = self.bank;
        for (name, value) in [
            ("retrieval.retained_decisions", bank.len()),
            ("retrieval.capacity", bank.capacity()),
            ("retrieval.requested_neighbors", bank.neighbors()),
            (
                "retrieval.available_neighbors",
                bank.len().min(bank.neighbors()),
            ),
            (
                "retrieval.state_feature_dimension",
                self.features.state_spec().dimension(),
            ),
            (
                "retrieval.action_feature_dimension",
                self.features.action_spec().dimension(),
            ),
        ] {
            metric(&mut metrics, name, value as f64);
        }
        metric(
            &mut metrics,
            "retrieval.occupancy",
            bank.len() as f64 / bank.capacity() as f64,
        );
        Ok(Some(metric_chunk(context, metrics)))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        if request.predictions {
            for set in request.probes {
                let mut rows = evaluate_rows(
                    set,
                    &self.spaces.actions,
                    request.max_chunk_bytes,
                    |history| self.predict(history),
                )?;
                if set.typed_inputs().is_none() {
                    if matches!(set.action_space(), ActionSpace::Discrete { .. }) {
                        chunks.push(dense_policy(request, set, &rows)?);
                    }
                    chunks.push(
                        set.record_as_typed(request.context.clone(), request.max_chunk_bytes)?,
                    );
                }
                for row in &mut rows {
                    row.values = Some(row.probabilities.clone());
                }
                chunks.push(typed_policy(
                    request,
                    set,
                    rows,
                    Some(FiniteValueKind::PreferenceScore),
                )?);
            }
        }
        if request.snapshots {
            let mode = if self.portable.is_some() {
                snapshot::Reconstruction::Portable
            } else {
                snapshot::Reconstruction::NeedsFeatures
            };
            chunks.push(snapshot::capture(
                request,
                ARCHITECTURE,
                MODEL,
                mode,
                |limit| match self.portable {
                    Some(encode) => encode(self, limit),
                    None => codec::bounded_json(&self.bank_payload(), limit),
                },
            )?);
        }
        Ok(chunks)
    }
}

#[derive(Serialize)]
struct BankPayload<'a, B> {
    bank: B,
    spaces: &'a TemporalSpaces,
    revision: u64,
    state: &'a FeatureSpec,
    action: &'a FeatureSpec,
}

#[derive(Deserialize)]
struct FrozenBank {
    bank: ReturnWeightedRetrieval,
    spaces: TemporalSpaces,
    revision: u64,
    state: FeatureSpec,
    action: FeatureSpec,
}

#[derive(Serialize, Deserialize)]
struct PortableBank<B, F> {
    model: B,
    features: F,
}

pub struct RestoredRetrieval<F> {
    model: FrozenBank,
    features: F,
}

impl<F: RetrievalFeatureMap> RestoredRetrieval<F> {
    pub fn probabilities(&self, history: &PublicHistory) -> Result<Vec<(Action, f64)>> {
        let row = predictions(
            &self.model.bank,
            &self.features,
            &self.model.spaces,
            history,
        )?;
        Ok(row.actions.into_iter().zip(row.probabilities).collect())
    }

    pub fn revision(&self) -> u64 {
        self.model.revision
    }
}

/// Restore the frozen retrieval population together with its serialized transformations.
pub fn restore_retrieval<F>(chunk: &Chunk, limits: ModelLimits) -> Result<RestoredRetrieval<F>>
where
    F: RetrievalFeatureMap + DeserializeOwned,
{
    let portable: PortableBank<FrozenBank, F> = snapshot::load(
        chunk,
        ARCHITECTURE,
        MODEL,
        snapshot::Reconstruction::Portable,
        limits,
    )?;
    restore(chunk, portable.model, portable.features, limits)
}

/// Restore a population against feature transformations supplied by the caller.
pub fn restore_retrieval_with_features<F: RetrievalFeatureMap>(
    chunk: &Chunk,
    features: F,
    limits: ModelLimits,
) -> Result<RestoredRetrieval<F>> {
    let model = snapshot::load(
        chunk,
        ARCHITECTURE,
        MODEL,
        snapshot::Reconstruction::NeedsFeatures,
        limits,
    )?;
    restore(chunk, model, features, limits)
}

fn restore<F: RetrievalFeatureMap>(
    chunk: &Chunk,
    model: FrozenBank,
    features: F,
    limits: ModelLimits,
) -> Result<RestoredRetrieval<F>> {
    ensure(
        model.revision == chunk.context.identity.learner_update_revision
            && &model.state == features.state_spec()
            && &model.action == features.action_spec(),
        "retrieval snapshot revision or feature contract differs",
    )?;
    let row_width = model
        .state
        .dimension()
        .saturating_add(model.action.dimension());
    ensure(
        model.bank.len().saturating_mul(row_width) <= limits.max_parameters,
        "retrieval snapshot exceeds parameter limit",
    )?;
    validate_contract(&model.bank, &features, &model.spaces)?;
    Ok(RestoredRetrieval { model, features })
}

fn validate_contract(
    bank: &ReturnWeightedRetrieval,
    features: &impl RetrievalFeatureMap,
    spaces: &TemporalSpaces,
) -> Result<()> {
    spaces.observations.validate()?;
    spaces.actions.validate()?;
    features.state_spec().validate()?;
    features.action_spec().validate()?;
    ensure(
        matches!(
            spaces.actions,
            ActionSpace::Discrete { .. } | ActionSpace::Conditional(_)
        ) && bank.feature_dimensions().is_none_or(|(state, action)| {
            state == features.state_spec().dimension()
                && action == features.action_spec().dimension()
        }),
        "retrieval dimensions or action space differ from the diagnostic feature contract",
    )
}

fn predictions(
    bank: &ReturnWeightedRetrieval,
    features: &impl RetrievalFeatureMap,
    spaces: &TemporalSpaces,
    history: &PublicHistory,
) -> Result<FiniteRow> {
    let observation = history.observation();
    observation.validate_in(&spaces.observations, &spaces.actions, true)?;
    let actions = observation.finite_actions(&spaces.actions)?;
    let state = features.encode_state(history)?;
    let encoded = features.encode_actions(history, &actions)?;
    let valid =
        |row: &[f64], width| row.len() == width && row.iter().all(|value| value.is_finite());
    ensure(
        valid(&state, features.state_spec().dimension())
            && encoded.len() == actions.len()
            && encoded
                .iter()
                .all(|row| valid(row, features.action_spec().dimension())),
        "retrieval features differ from their declared dimensions",
    )?;
    let probabilities = bank.probabilities(&state, &encoded)?;
    let selected = greedy_index(&actions, &probabilities)?;
    Ok(FiniteRow {
        actions,
        selected,
        probabilities,
        values: None,
    })
}
