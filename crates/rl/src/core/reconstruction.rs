//! Optional typed state estimation and codecs, independent of policy/action heads.
use super::{
    Result,
    anchors::{
        ANCHOR_VERSION, AnchorPhase, AnchorSchema, AnchorSource, RepresentationId, StateAnchor,
    },
    ensure,
    stateful::HistoryInput,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub trait ReconstructionModel: Clone {
    type State: Clone;
    fn model_identity(&self) -> [u8; 32];
    fn generation(&self) -> u64;
    fn schema(&self) -> &AnchorSchema;
    fn initial_state(&self) -> Result<Self::State>;
    fn validate_state(&self, state: &Self::State) -> Result<()>;
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State>;
    fn encode_state(&self, state: &Self::State) -> Result<Vec<u8>>;
    fn decode_state(&self, payload: &[u8]) -> Result<Self::State>;
}

/// Sources need not be the behavior actor, and are not required to select actions.
pub trait ReplayRepresentations {
    type Model: ReconstructionModel;
    fn replay_representations(&self) -> Result<Vec<(RepresentationId, Self::Model)>>;
}

pub trait ReplayReconstruction: ReplayRepresentations {
    fn reconstruction_config(&self) -> ReconstructionConfig;
}

/// Optional codec/provenance capability for explicit runtime reconciliation.
/// Parameter generations are independent of behavior-publication revisions.
pub trait ReconstructibleActor: super::stateful::StatefulActor {
    type Model: ReconstructionModel<State = Self::State>;
    fn representation(&self) -> Result<Self::Model>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ReconstructionMode {
    #[default]
    Exact,
    Anchored {
        max_generation_lag: Option<u64>,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionConfig {
    pub version: u32,
    pub mode: ReconstructionMode,
}
impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            version: 1,
            mode: ReconstructionMode::Exact,
        }
    }
}
impl ReconstructionConfig {
    pub fn validate(self) -> Result<()> {
        ensure(
            self.version == 1,
            "unsupported reconstruction configuration version",
        )
    }
    pub fn is_exact_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionDecision {
    pub version: u32,
    pub approximate: bool,
    pub generation_lag: u64,
    pub source_model_identity: [u8; 32],
    pub current_model_identity: [u8; 32],
    pub observation_index: usize,
}

pub fn reset_anchor<R: ReconstructionModel>(
    model: &R,
    id: RepresentationId,
    episode: u64,
) -> Result<StateAnchor> {
    let state = model.initial_state()?;
    model.validate_state(&state)?;
    let anchor = StateAnchor {
        version: ANCHOR_VERSION,
        episode_id: episode,
        observation_index: 0,
        phase: AnchorPhase::BeforeObservation,
        representation_id: id,
        model_identity: model.model_identity(),
        generation: model.generation(),
        schema: model.schema().clone(),
        payload: model.encode_state(&state)?,
        exact_ancestry: true,
        source: AnchorSource::ResetReconstruction,
    };
    anchor.validate()?;
    Ok(anchor)
}

pub fn decode_anchor<R: ReconstructionModel>(
    config: ReconstructionConfig,
    model: &R,
    id: RepresentationId,
    episode: u64,
    index: usize,
    anchor: &StateAnchor,
) -> Result<(R::State, ReconstructionDecision)> {
    config.validate()?;
    anchor.validate()?;
    ensure(
        anchor.representation_id == id
            && anchor.episode_id == episode
            && anchor.observation_index == index
            && anchor.phase == AnchorPhase::BeforeObservation,
        "anchor representation or pre-observation chronology mismatch",
    )?;
    ensure(
        &anchor.schema == model.schema(),
        "incompatible recurrent-state codec/schema",
    )?;
    ensure(
        anchor.generation <= model.generation(),
        "anchor generation is in the model's future",
    )?;
    let lag = model.generation() - anchor.generation;
    let approximate = !anchor.exact_ancestry || anchor.model_identity != model.model_identity();
    match config.mode {
        ReconstructionMode::Exact => ensure(
            !approximate,
            "exact reconstruction rejects stale or approximate anchor ancestry",
        )?,
        ReconstructionMode::Anchored { max_generation_lag } => ensure(
            max_generation_lag.is_none_or(|maximum| lag <= maximum),
            "anchor exceeds the declared parameter-generation lag",
        )?,
    }
    let state = model.decode_state(&anchor.payload)?;
    model.validate_state(&state)?;
    if index == 0 && !approximate {
        ensure(
            anchor.payload == model.encode_state(&model.initial_state()?)?,
            "exact reset anchor differs from the model's initial prior",
        )?;
    }
    Ok((
        state,
        ReconstructionDecision {
            version: 1,
            approximate,
            generation_lag: lag,
            source_model_identity: anchor.model_identity,
            current_model_identity: model.model_identity(),
            observation_index: index,
        },
    ))
}

/// Assimilate exactly the anchor's named observation. The returned post-state is
/// represented as a prior for index+1. This is also the streaming recording clock.
pub fn advance_anchor<R: ReconstructionModel>(
    config: ReconstructionConfig,
    model: &R,
    anchor: &StateAnchor,
    input: HistoryInput<'_>,
) -> Result<(StateAnchor, ReconstructionDecision)> {
    ensure(
        input.previous.is_none() == (anchor.observation_index == 0),
        "anchor previous-action/reward presence is shifted by one timestep",
    )?;
    let (prior, decision) = decode_anchor(
        config,
        model,
        anchor.representation_id,
        anchor.episode_id,
        anchor.observation_index,
        anchor,
    )?;
    let state = model.assimilate(&prior, input)?;
    model.validate_state(&state)?;
    let index = anchor
        .observation_index
        .checked_add(1)
        .ok_or_else(|| super::Error::State("anchor clock overflow".into()))?;
    let next = StateAnchor {
        version: ANCHOR_VERSION,
        episode_id: anchor.episode_id,
        observation_index: index,
        phase: AnchorPhase::BeforeObservation,
        representation_id: anchor.representation_id,
        model_identity: model.model_identity(),
        generation: model.generation(),
        schema: model.schema().clone(),
        payload: model.encode_state(&state)?,
        exact_ancestry: !decision.approximate,
        source: AnchorSource::Continued {
            parent_identity: anchor_identity(anchor)?,
            parent_model_identity: anchor.model_identity,
            parent_generation: anchor.generation,
            parent_exact: anchor.exact_ancestry,
        },
    };
    next.validate()?;
    Ok((next, decision))
}

/// Canonical binary identity of the bounded envelope; no JSON float boundary or
/// growing ancestor list. Source/shape metadata and payload are all included.
pub fn anchor_identity(anchor: &StateAnchor) -> Result<[u8; 32]> {
    anchor.validate()?;
    let mut h = Sha256::new();
    h.update(b"rust-rl-state-anchor-v2");
    h.update(anchor.version.to_le_bytes());
    h.update(anchor.episode_id.to_le_bytes());
    h.update((anchor.observation_index as u64).to_le_bytes());
    h.update([0]);
    h.update(anchor.representation_id.0.to_le_bytes());
    h.update(anchor.model_identity);
    h.update(anchor.generation.to_le_bytes());
    h.update(anchor.schema.codec.to_le_bytes());
    h.update(anchor.schema.codec_version.to_le_bytes());
    h.update(anchor.schema.representation_identity);
    h.update((anchor.schema.dimensions.len() as u64).to_le_bytes());
    for &d in &anchor.schema.dimensions {
        h.update((d as u64).to_le_bytes());
    }
    h.update([u8::from(anchor.exact_ancestry)]);
    match &anchor.source {
        AnchorSource::ResetReconstruction => h.update([0]),
        AnchorSource::Continued {
            parent_identity,
            parent_model_identity,
            parent_generation,
            parent_exact,
        } => {
            h.update([1]);
            h.update(parent_identity);
            h.update(parent_model_identity);
            h.update(parent_generation.to_le_bytes());
            h.update([u8::from(*parent_exact)]);
        }
        AnchorSource::BehaviorSnapshot { behavior_revision } => {
            h.update([2]);
            h.update(behavior_revision.to_le_bytes());
        }
    }
    h.update((anchor.payload.len() as u64).to_le_bytes());
    h.update(&anchor.payload);
    Ok(h.finalize().into())
}
