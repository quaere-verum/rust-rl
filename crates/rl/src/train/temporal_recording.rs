//! Optional bounded chronological recording from independent reconstruction models.
use crate::{
    Error, Result,
    core::{
        anchors::{ANCHOR_VERSION, AnchorSet, MAX_ANCHOR_KEYS, RepresentationId},
        ensure,
        env::EpisodeStart,
        reconstruction::{
            ReconstructionConfig, ReconstructionMode, ReconstructionModel, advance_anchor,
            decode_anchor, reset_anchor,
        },
        stateful::{HistoryInput, PreviousInput},
        temporal::{TemporalSink, TemporalTransition},
    },
    data::temporal::{
        AnchorRequirement, AnchorRequirements, EpisodeStore, SampleLimits, SequenceConfig,
        TemporalStorage,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalReplayConfig {
    pub version: u32,
    pub storage: TemporalStorage,
    pub reconstruction: ReconstructionConfig,
    pub sample_limits: SampleLimits,
    pub anchor_interval: usize,
}
impl TemporalReplayConfig {
    pub fn validate(&self, capacity: usize, lanes: usize, sequence: &SequenceConfig) -> Result<()> {
        sequence.validate()?;
        self.reconstruction.validate()?;
        ensure(
            self.version == 1 && self.anchor_interval > 0 && self.anchor_interval <= capacity,
            "invalid temporal recording version or anchor interval",
        )?;
        ensure(
            self.sample_limits.max_sample_tokens > 0
                && self.sample_limits.max_batch_tokens > 0
                && self.sample_limits.max_sample_payload_bytes > 0
                && self.sample_limits.max_batch_payload_bytes > 0,
            "temporal recording sample limits must be positive",
        )?;
        if let TemporalStorage::Segmented(config) = &self.storage {
            ensure(
                matches!(
                    self.reconstruction.mode,
                    ReconstructionMode::Anchored { .. }
                ),
                "segmented temporal replay requires explicit anchored reconstruction",
            )?;
            ensure(
                self.anchor_interval <= config.segment_steps
                    && config.minimum_context
                        > sequence.burn_in.saturating_add(sequence.n_step - 1)
                    && config
                        .minimum_context
                        .checked_mul(lanes)
                        .is_some_and(|required| required <= capacity),
                "segmented temporal context or interval cannot serve every lane",
            )?;
        }
        Ok(())
    }
    pub fn is_segmented(&self) -> bool {
        matches!(self.storage, TemporalStorage::Segmented(_))
    }
}

pub(crate) fn validate_models<M: ReconstructionModel>(
    models: &[(RepresentationId, M)],
) -> Result<()> {
    ensure(
        !models.is_empty() && models.len() <= MAX_ANCHOR_KEYS,
        "invalid reconstruction source count",
    )?;
    for (index, (id, model)) in models.iter().enumerate() {
        ensure(
            models[..index].iter().all(|(old, _)| old != id),
            "duplicate reconstruction source",
        )?;
        model.schema().validate()?;
        model.validate_state(&model.initial_state()?)?;
    }
    Ok(())
}

pub(crate) fn validate_stream_set(config: &TemporalReplayConfig, set: &AnchorSet) -> Result<()> {
    set.validate()?;
    let limits = match &config.storage {
        TemporalStorage::Segmented(segmented) => {
            ensure(
                segmented
                    .required_representations
                    .iter()
                    .all(|id| set.anchors.contains_key(id)),
                "reconstruction stream lacks a configured representation",
            )?;
            &segmented.limits
        }
        _ => {
            return Err(Error::State(
                "raw recording must not retain reconstruction streams".into(),
            ));
        }
    };
    ensure(
        set.anchors.len() <= limits.max_anchors && set.payload_bytes() <= limits.max_anchor_bytes,
        "reconstruction stream exceeds anchor resource limit",
    )?;
    for anchor in set.anchors.values() {
        ensure(
            anchor.payload.len() <= limits.max_anchor_payload_bytes
                && anchor.schema.dimensions.len() <= limits.max_dimension_rank,
            "reconstruction stream exceeds payload or dimension limit",
        )?;
    }
    Ok(())
}

pub(crate) fn validate_source_keys<M: ReconstructionModel>(
    config: &TemporalReplayConfig,
    models: &[(RepresentationId, M)],
) -> Result<()> {
    validate_models(models)?;
    if let TemporalStorage::Segmented(segmented) = &config.storage {
        ensure(
            segmented
                .required_representations
                .iter()
                .all(|key| models.iter().any(|(id, _)| id == key)),
            "reconstruction models lack configured representation",
        )?;
    }
    Ok(())
}

pub(crate) fn requirements<M: ReconstructionModel>(
    config: ReconstructionConfig,
    models: &[(RepresentationId, M)],
) -> Result<AnchorRequirements> {
    config.validate()?;
    validate_models(models)?;
    Ok(AnchorRequirements {
        required: models
            .iter()
            .map(|(id, model)| {
                let (identity, exact, minimum) = match config.mode {
                    ReconstructionMode::Exact => (Some(model.model_identity()), true, None),
                    ReconstructionMode::Anchored { max_generation_lag } => (
                        None,
                        false,
                        max_generation_lag.map(|lag| model.generation().saturating_sub(lag)),
                    ),
                };
                AnchorRequirement {
                    representation_id: *id,
                    schema: Some(model.schema().clone()),
                    model_identity: identity,
                    require_exact: exact,
                    min_generation: minimum,
                    max_generation: Some(model.generation()),
                }
            })
            .collect(),
    })
}

pub(crate) fn audit_streams<M: ReconstructionModel>(
    config: ReconstructionConfig,
    models: &[(RepresentationId, M)],
    streams: &BTreeMap<u64, AnchorSet>,
) -> Result<()> {
    validate_models(models)?;
    for (&episode, set) in streams {
        set.validate()?;
        ensure(
            set.episode_id() == episode && set.anchors.len() == models.len(),
            "recording stream episode/key count differs",
        )?;
        for (id, model) in models {
            let anchor = set.anchors.get(id).ok_or_else(|| {
                Error::State("recording stream lacks reconstruction source".into())
            })?;
            decode_anchor(config, model, *id, episode, set.observation_index(), anchor)?;
        }
    }
    Ok(())
}

pub(crate) struct StreamingSink<'a, M: ReconstructionModel> {
    pub store: &'a mut EpisodeStore,
    pub streams: &'a mut BTreeMap<u64, AnchorSet>,
    pub models: &'a [(RepresentationId, M)],
    pub config: &'a TemporalReplayConfig,
    pub max_streams: usize,
}
impl<M: ReconstructionModel> TemporalSink for StreamingSink<'_, M> {
    fn begin_episode(&mut self, id: u64, start: &EpisodeStart) -> Result<()> {
        ensure(
            self.streams.len() < self.max_streams && !self.streams.contains_key(&id),
            "recording stream capacity or identity exhausted",
        )?;
        let history = start
            .context
            .as_ref()
            .map(|context| context.history.inputs());
        let fresh = [HistoryInput {
            observation: &start.observation,
            previous: None,
        }];
        let inputs = history.as_deref().unwrap_or(&fresh);
        let mut next = BTreeMap::new();
        for (key, model) in self.models {
            let mut anchor = reset_anchor(model, *key, id)?;
            for &input in inputs {
                (anchor, _) = advance_anchor(self.config.reconstruction, model, &anchor, input)?;
            }
            next.insert(*key, anchor);
        }
        let mut set = AnchorSet {
            version: ANCHOR_VERSION,
            anchors: next,
        };
        validate_stream_set(self.config, &set)?;
        set.canonicalize();
        self.store.begin_episode(id, start)?;
        self.streams.insert(id, set);
        Ok(())
    }

    fn push(&mut self, transition: &TemporalTransition) -> Result<()> {
        let current = self.streams.get(&transition.episode_id).ok_or_else(|| {
            Error::State("temporal transition has no reconstruction stream".into())
        })?;
        let index = transition
            .timestep
            .checked_add(1)
            .ok_or_else(|| Error::State("recording stream clock overflow".into()))?;
        ensure(
            current.observation_index() == index,
            "recording pre-observation clock differs",
        )?;
        let mut next = BTreeMap::new();
        for (key, model) in self.models {
            let anchor = current
                .anchors
                .get(key)
                .ok_or_else(|| Error::State("recording stream lacks source".into()))?;
            let (anchor, _) = advance_anchor(
                self.config.reconstruction,
                model,
                anchor,
                HistoryInput {
                    observation: &transition.next_observation,
                    previous: Some(PreviousInput::from_transition(
                        &transition.action,
                        &transition.reward,
                        &transition.observation,
                    )),
                },
            )?;
            next.insert(*key, anchor);
        }
        let mut next = AnchorSet {
            version: ANCHOR_VERSION,
            anchors: next,
        };
        validate_stream_set(self.config, &next)?;
        next.canonicalize();
        let recorded = (index % self.config.anchor_interval == 0).then(|| current.clone());
        self.store.push_with_anchors(transition, recorded)?;
        if transition.ended() {
            self.streams.remove(&transition.episode_id);
        } else {
            self.streams.insert(transition.episode_id, next);
        }
        Ok(())
    }
    fn abandon_episode(&mut self, id: u64) -> Result<()> {
        self.store.abandon_episode(id)?;
        self.streams.remove(&id);
        Ok(())
    }
}
