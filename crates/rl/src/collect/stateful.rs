//! Opt-in temporal interaction using the existing vector lane driver.
use super::stateful_recording::BasicRecording;
use super::{
    CollectionDelta,
    driver::{LaneDriver, ResumeContexts},
    vector_env::VectorEnv,
};
use crate::{
    Result,
    core::{
        actor::ActMode,
        anchors::{RepresentationId, StateAnchor},
        checkpoint::Snapshot,
        ensure,
        env::{EpisodeStart, Observation, ObservationSpace, ResumeContext},
        reconstruction::{
            ReconstructibleActor, ReconstructionConfig, ReconstructionDecision,
            ReconstructionModel, anchor_identity, decode_anchor,
        },
        recording::{RecordingKey, StatefulRecording},
        stateful::{
            HistoryInput, PreviousInput, PreviousStep, StatefulActor, StatefulDecision,
            replay_actor_history,
        },
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::cell::Cell;

mod history;
mod reset;
pub use history::{AdoptionRecord, AdoptionSource, RetainedHistory};
use history::{actions_equal_bits, previous_input, public_history_identity};
use reset::{boundary_previous, reset_states};

#[path = "stateful_revision.rs"]
mod revisions;
pub use revisions::AdoptionBase;

pub const MAX_LANE_ADOPTIONS: usize = 64;

// Ordinary snapshots historically omitted this field or stored null.
fn reject_selection_streams<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<(), D::Error> {
    if Option::<serde::de::IgnoredAny>::deserialize(deserializer)?.is_some() {
        return Err(serde::de::Error::custom(
            "delegated actor selection streams are unsupported",
        ));
    }
    Ok(())
}

/// Scalar evaluation is the historical reference. Batched evaluation is opt-in;
/// actors without an explicit compatibility capability retain scalar calls.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceMode {
    #[default]
    Scalar,
    Batched,
    /// Versioned scalar-row arithmetic. Choose at construction; legacy snapshots
    /// keep their original `batched` execution and are never silently reinterpreted.
    #[serde(rename = "canonical_batched_v1")]
    CanonicalBatched,
}
impl InferenceMode {
    fn is_scalar(&self) -> bool {
        *self == Self::Scalar
    }
    fn snapshot_version(self) -> u32 {
        match self {
            Self::Scalar => 1,
            Self::Batched => 2,
            Self::CanonicalBatched => 5,
        }
    }
}

struct Scratch<S> {
    all_active: Vec<bool>,
    reset: Vec<bool>,
    groups: Vec<Vec<usize>>,
    actions: Vec<crate::core::action::Action>,
    rngs: Vec<ChaCha8Rng>,
    metadata: Vec<BehaviorMetadata>,
    decisions: Vec<Option<StatefulDecision>>,
    keys: Vec<RecordingKey>,
    states: Vec<Option<S>>,
}
impl<S> Default for Scratch<S> {
    fn default() -> Self {
        Self {
            all_active: Vec::new(),
            reset: Vec::new(),
            groups: Vec::new(),
            actions: Vec::new(),
            rngs: Vec::new(),
            metadata: Vec::new(),
            decisions: Vec::new(),
            keys: Vec::new(),
            states: Vec::new(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "A: Serialize, A::State: Serialize",
    deserialize = "A: DeserializeOwned, A::State: DeserializeOwned"
))]
struct AgentLane<A: StatefulActor> {
    actor: A,
    state: A::State,
    rng: ChaCha8Rng,
    episode_id: u64,
    previous: Option<PreviousStep>,
    begun: bool,
    #[serde(default, skip_serializing_if = "zero_timestep")]
    learning_start: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending_context: Option<ResumeContext>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    adoptions: Vec<AdoptionRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    adoption_base: Option<AdoptionBase>,
}

fn zero_timestep(value: &usize) -> bool {
    *value == 0
}

/// Each lane pins a complete behavior actor until genuine episode reset. Publishing
/// a newer revision affects only subsequent episodes, never an old model's memory.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "V: Serialize, A: Serialize, A::State: Serialize",
    deserialize = "V: DeserializeOwned, A: DeserializeOwned, A::State: DeserializeOwned"
))]
pub struct StatefulCollector<V, A: StatefulActor> {
    version: u32,
    driver: LaneDriver<V>,
    latest: A,
    agents: Vec<AgentLane<A>>,
    next_episode_id: u64,
    #[serde(
        default,
        rename = "selection_rngs",
        skip_serializing,
        deserialize_with = "reject_selection_streams"
    )]
    _legacy_selection_rngs: (),
    #[serde(default, skip_serializing_if = "InferenceMode::is_scalar")]
    inference_mode: InferenceMode,
    #[serde(default, skip_serializing_if = "zero_adoptions")]
    adoption_count: u64,
    #[serde(default, skip_serializing_if = "zero_adoptions")]
    compaction_count: u64,
    // Deserialization cannot establish trust. The first operational use audits
    // the complete graph before mutation; only private validated paths maintain it.
    #[serde(skip)]
    audited: Cell<bool>,
    #[serde(skip)]
    needs_recovery: bool,
    #[serde(skip)]
    scratch: Scratch<A::State>,
}

impl<V: VectorEnv, A: StatefulActor + Clone> StatefulCollector<V, A> {
    pub fn new(env: V, actor: A, seed: u64) -> Result<Self> {
        Self::new_with_inference_mode(env, actor, seed, InferenceMode::Scalar)
    }
    pub fn new_with_inference_mode(
        env: V,
        actor: A,
        seed: u64,
        inference_mode: InferenceMode,
    ) -> Result<Self> {
        Self::initialize(env, actor, seed, inference_mode, None)
    }
    /// Use explicit per-lane initial environment seeds independently of action
    /// sampling. Later `reset(None)` calls retain environment-owned streams;
    /// the existing stateful per-lane action RNG derivation is unchanged.
    pub fn new_with_reset_seeds(
        env: V,
        actor: A,
        action_seed: u64,
        inference_mode: InferenceMode,
        reset_seeds: &[u64],
    ) -> Result<Self> {
        Self::initialize(env, actor, action_seed, inference_mode, Some(reset_seeds))
    }
    fn initialize(
        env: V,
        actor: A,
        seed: u64,
        inference_mode: InferenceMode,
        reset_seeds: Option<&[u64]>,
    ) -> Result<Self> {
        ensure(
            actor.observation_space() == env.observation_space()
                && actor.action_space() == env.action_space(),
            "stateful actor/environment contract mismatch",
        )?;
        let (mut driver, contexts) = match reset_seeds {
            Some(seeds) => LaneDriver::new_with_reset_seeds(env, seed, seeds)?,
            None => LaneDriver::new(env, seed)?,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5354_4154_4546_554c);
        let lanes: Vec<_> = (0..driver.len()).collect();
        let states = reset_states(
            &actor,
            inference_mode,
            &driver.observations,
            &lanes,
            &contexts,
        )?;
        let mut contexts = contexts.into_iter().peekable();
        let mut agents = Vec::new();
        for (i, state) in states.into_iter().enumerate() {
            actor.validate_state(&state)?;
            let context = contexts
                .next_if(|(lane, _)| *lane == i)
                .map(|(_, context)| context);
            let clock = context
                .as_ref()
                .map_or(0, |context| context.history.timestep());
            driver.episode_steps[i] = clock;
            agents.push(AgentLane {
                actor: actor.clone(),
                state,
                rng: ChaCha8Rng::seed_from_u64(rng.random()),
                episode_id: i as u64,
                previous: boundary_previous(context.as_ref()),
                begun: false,
                learning_start: clock as usize,
                pending_context: context,
                adoptions: Vec::new(),
                adoption_base: None,
            });
        }
        let next_episode_id = agents.len() as u64;
        let mut collector = Self {
            version: inference_mode.snapshot_version(),
            driver,
            latest: actor,
            agents,
            next_episode_id,
            _legacy_selection_rngs: (),
            inference_mode,
            adoption_count: 0,
            compaction_count: 0,
            audited: Cell::new(false),
            needs_recovery: false,
            scratch: Scratch::default(),
        };
        collector.version = collector.snapshot_version();
        collector.validate_state()?;
        Ok(collector)
    }
    pub fn inference_mode(&self) -> InferenceMode {
        self.inference_mode
    }
    pub fn set_inference_mode(&mut self, mode: InferenceMode) -> Result<()> {
        self.validate_local()?;
        ensure(
            mode == self.inference_mode
                || (mode != InferenceMode::CanonicalBatched
                    && self.inference_mode != InferenceMode::CanonicalBatched),
            "canonical numerical execution must be selected at construction",
        )?;
        self.inference_mode = mode;
        self.version = self.snapshot_version();
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.agents.len()
    }
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }
    pub fn dense_observation_dim(&self) -> Option<usize> {
        self.observation_space().dense_width()
    }
    pub fn observation_space(&self) -> ObservationSpace {
        self.driver.env.observation_space()
    }
    pub fn action_space(&self) -> &crate::core::action::ActionSpace {
        self.driver.env.action_space()
    }
    /// Borrow the owned vector environment for external diagnostics and physical
    /// duration accounting, matching the ordinary collector's inspection API.
    /// This does not advance/reset lanes or expose data to the actor contract.
    pub fn environment(&self) -> &V {
        &self.driver.env
    }
    pub fn lane_state(&self, lane: usize) -> Option<&A::State> {
        self.agents.get(lane).map(|a| &a.state)
    }
    /// Only the enclosing complete-run importer may install a memory already
    /// audited against this lane's current immutable actor and factual history.
    /// No publication, reset, RNG consumption or provenance event occurs here.
    pub(crate) fn install_imported_memory(&mut self, lane: usize, state: A::State) -> Result<()> {
        let agent = self
            .agents
            .get_mut(lane)
            .ok_or_else(|| crate::Error::State("invalid imported lane".into()))?;
        agent.actor.validate_state(&state)?;
        agent.state = state;
        self.audited.set(false);
        Ok(())
    }
    pub fn lane_revision(&self, lane: usize) -> Option<u64> {
        self.agents.get(lane).map(|a| a.actor.revision())
    }
    pub fn lane_actor(&self, lane: usize) -> Option<&A> {
        self.agents.get(lane).map(|agent| &agent.actor)
    }
    pub fn latest_actor(&self) -> &A {
        &self.latest
    }
    pub fn adoption_count(&self) -> u64 {
        self.adoption_count
    }
    pub fn lane_adoptions(&self, lane: usize) -> Option<&[AdoptionRecord]> {
        self.agents
            .get(lane)
            .map(|agent| agent.adoptions.as_slice())
    }
    /// Expected behavior revision for an absolute transition clock, including a
    /// retained suffix whose earlier steps have already been compacted away.
    pub fn lane_behavior_revision_at(&self, lane: usize, timestep: usize) -> Option<u64> {
        let agent = self.agents.get(lane)?;
        if timestep < agent.learning_start
            || u64::try_from(timestep).ok()? > *self.driver.episode_steps.get(lane)?
        {
            return None;
        }
        let mut revision = if let Some(base) = &agent.adoption_base {
            if timestep < base.observation_index {
                return None;
            }
            base.revision
        } else {
            agent
                .adoptions
                .first()
                .map_or_else(|| agent.actor.revision(), |record| record.from_revision)
        };
        for record in &agent.adoptions {
            if record.observation_index > timestep {
                break;
            }
            revision = record.to_revision;
        }
        Some(revision)
    }
    pub fn lane_episode(&self, lane: usize) -> Option<u64> {
        self.agents.get(lane).map(|a| a.episode_id)
    }
    pub fn lane_timestep(&self, lane: usize) -> Option<u64> {
        self.driver.episode_steps.get(lane).copied()
    }
    pub fn lane_learning_start(&self, lane: usize) -> Option<usize> {
        self.agents.get(lane).map(|agent| agent.learning_start)
    }
    pub(crate) fn lane_pending_context(&self, lane: usize) -> Option<&ResumeContext> {
        self.agents.get(lane)?.pending_context.as_ref()
    }
    /// A resumed start can precede the first publication for a collection.
    /// No behavior has been recorded at this boundary.
    pub(crate) fn reconcile_pending_context(&mut self, lane: usize) -> Result<bool> {
        let Some(context) = self.lane_pending_context(lane) else {
            return Ok(false);
        };
        self.validate_local()?;
        let actor = self.latest.clone();
        let state = actor.reconstruct_history(&context.history.inputs())?;
        actor.validate_state(&state)?;
        self.agents[lane].actor = actor;
        self.agents[lane].state = state;
        Ok(true)
    }
    pub fn lane_begun(&self, lane: usize) -> Option<bool> {
        self.agents.get(lane).map(|a| a.begun)
    }
    pub fn lane_needs_reset(&self, lane: usize) -> Option<bool> {
        self.driver.needs_reset.get(lane).copied()
    }
    pub fn latest_revision(&self) -> u64 {
        self.latest.revision()
    }
    pub fn observations(&self) -> &[Observation] {
        &self.driver.observations
    }
    pub fn publish(&mut self, actor: A) -> Result<()> {
        self.validate_local()?;
        ensure(
            actor.observation_space() == self.observation_space()
                && actor.action_space() == self.driver.env.action_space()
                && actor.revision() > self.latest.revision(),
            "published actor must have compatible dimensions and a newer revision",
        )?;
        actor.validate_state(&actor.initial_state()?)?;
        self.latest = actor;
        Ok(())
    }
    /// Alternative to episode pinning for future on-policy consumers: rebuild a
    /// lane under the latest published actor from its complete actual public tape.
    /// No environment reset, likelihood rewrite, or stored-state approximation.
    pub fn reconcile_lane(
        &mut self,
        lane: usize,
        history: crate::core::stateful::ReconstructionHistory<'_>,
    ) -> Result<()> {
        self.reconcile_lane_with(lane, history, |_, _| Ok(None))
    }
    pub(crate) fn reconcile_lane_with(
        &mut self,
        lane: usize,
        full_history: crate::core::stateful::ReconstructionHistory<'_>,
        reconstruct: impl FnOnce(
            &A,
            crate::core::stateful::ReconstructionHistory<'_>,
        ) -> Result<Option<A::State>>,
    ) -> Result<()> {
        let clock = self.adoption_preflight(lane)?;
        let history = RetainedHistory {
            base_timestep: 0,
            previous: None,
            observations: full_history.observations,
            actions: full_history.actions,
            rewards: full_history.rewards,
        };
        self.validate_adoption_history(lane, history, clock)?;
        let actor = self.latest.clone();
        let state = {
            #[cfg(feature = "probes")]
            let _profile = crate::profiling::enter(crate::profiling::Phase::LaneReconstruction);
            if let Some(state) = reconstruct(&actor, full_history)? {
                actor.validate_state(&state)?;
                state
            } else {
                replay_actor_history(
                    &actor,
                    history
                        .observations
                        .iter()
                        .enumerate()
                        .map(|(t, observation)| HistoryInput {
                            observation,
                            previous: previous_input(history, t),
                        }),
                )?
            }
        };
        let record = AdoptionRecord {
            version: 1,
            episode_id: self.agents[lane].episode_id,
            observation_index: clock,
            from_revision: self.agents[lane].actor.revision(),
            to_revision: actor.revision(),
            history_identity: public_history_identity(self.agents[lane].episode_id, history),
            source: AdoptionSource::FullHistory {
                origin_index: 0,
                observations: history.observations.len(),
            },
        };
        self.commit_adoption(lane, actor, state, record)
    }
    fn ordinary_snapshot_version(&self) -> u32 {
        if self.inference_mode == InferenceMode::CanonicalBatched {
            5
        } else if self.compaction_count > 0 {
            4
        } else if self.adoption_count > 0 {
            3
        } else {
            self.inference_mode.snapshot_version()
        }
    }
    fn snapshot_version(&self) -> u32 {
        let ordinary = self.ordinary_snapshot_version();
        if self
            .agents
            .iter()
            .any(|a| a.learning_start != 0 || a.pending_context.is_some())
        {
            ordinary + 5
        } else {
            ordinary
        }
    }
    fn adoption_preflight(&self, lane: usize) -> Result<usize> {
        self.validate_local()?;
        let old = self
            .agents
            .get(lane)
            .ok_or_else(|| crate::Error::State("reconciliation lane out of range".into()))?;
        ensure(
            !self.driver.needs_reset[lane],
            "cannot adopt a policy on an ended lane",
        )?;
        ensure(
            self.latest.revision() > old.actor.revision(),
            "reconciliation requires a newer published actor",
        )?;
        if old.adoptions.len() >= MAX_LANE_ADOPTIONS {
            return Err(crate::Error::State(
                "temporal resource limit: per-episode adoption ledger exhausted".into(),
            ));
        }
        self.adoption_count
            .checked_add(1)
            .ok_or_else(|| crate::Error::State("adoption counter overflow".into()))?;
        usize::try_from(self.driver.episode_steps[lane])
            .map_err(|_| crate::Error::Shape("reconstruction clock is not representable".into()))
    }
    fn validate_adoption_history(
        &self,
        lane: usize,
        history: RetainedHistory<'_>,
        clock: usize,
    ) -> Result<()> {
        self.validate_public_history(lane, history, clock, true)
    }
    fn validate_public_history(
        &self,
        lane: usize,
        history: RetainedHistory<'_>,
        clock: usize,
        final_live: bool,
    ) -> Result<()> {
        ensure(
            !history.observations.is_empty()
                && history.actions.len().checked_add(1) == Some(history.observations.len())
                && history.actions.len() == history.rewards.len()
                && history.base_timestep.checked_add(history.actions.len()) == Some(clock)
                && (history.base_timestep == 0) == history.previous.is_none()
                && history.observations.last() == Some(&self.driver.observations[lane]),
            "reconstruction history does not match the current episode clock/observation",
        )?;
        if let Some(previous) = history.previous {
            self.action_space().check(previous.action, None)?;
            ensure(
                previous.reward.is_finite(),
                "nonfinite reconstruction preceding reward",
            )?;
        }
        let actual_previous = previous_input(history, history.actions.len());
        match (&self.agents[lane].previous, actual_previous) {
            (None, None) => (),
            (Some(actual), Some(provided)) => ensure(
                actions_equal_bits(&actual.action, provided.action)
                    && actual.reward.to_bits() == provided.reward.to_bits(),
                "reconstruction previous transition mismatch",
            )?,
            _ => {
                return Err(crate::Error::State(
                    "reconstruction preceding transition is missing or shifted".into(),
                ));
            }
        }
        for (t, observation) in history.observations.iter().enumerate() {
            observation.validate_in(
                &self.observation_space(),
                self.action_space(),
                t + 1 < history.observations.len() || final_live,
            )?;
            if let Some(action) = history.actions.get(t) {
                observation.check_action(self.action_space(), action)?;
                let input = history.rewards.input(t, action, observation);
                ensure(input.reward.is_finite(), "nonfinite reconstruction reward")?;
                if let Some(factual) = input.factual {
                    factual.reward.validate()?;
                }
            }
        }
        Ok(())
    }
    fn commit_adoption(
        &mut self,
        lane: usize,
        actor: A,
        state: A::State,
        record: AdoptionRecord,
    ) -> Result<()> {
        record.validate()?;
        let next_count = self
            .adoption_count
            .checked_add(1)
            .ok_or_else(|| crate::Error::State("adoption counter overflow".into()))?;
        self.agents[lane]
            .adoptions
            .try_reserve_exact(1)
            .map_err(|error| {
                crate::Error::State(format!("cannot reserve adoption provenance: {error}"))
            })?;
        // No environment, RNG, clock, observation or sink operation occurs here.
        // Every fallible candidate/validation/allocation operation precedes commit.
        self.agents[lane].actor = actor;
        self.agents[lane].state = state;
        self.agents[lane].adoptions.push(record);
        self.adoption_count = next_count;
        self.version = self.snapshot_version();
        Ok(())
    }
    pub fn validate_state(&self) -> Result<()> {
        self.audited.set(false);
        ensure(
            !self.needs_recovery,
            "stateful collector requires full recovery",
        )?;
        ensure(
            self.version == self.snapshot_version(),
            "incompatible stateful runtime snapshot version/inference mode",
        )?;
        self.driver.validate_state()?;
        ensure(
            self.agents.len() == self.driver.len()
                && self.latest.observation_space() == self.observation_space()
                && self.latest.action_space() == self.driver.env.action_space(),
            "invalid stateful runtime dimensions",
        )?;
        let mut ids = std::collections::BTreeSet::new();
        let mut retained_adoptions = 0u64;
        let mut retained_compactions = 0u64;
        for (i, a) in self.agents.iter().enumerate() {
            ensure(
                a.actor.observation_space() == self.observation_space()
                    && a.actor.action_space() == self.driver.env.action_space()
                    && a.actor.revision() <= self.latest.revision()
                    && a.episode_id < self.next_episode_id
                    && ids.insert(a.episode_id),
                "invalid lane actor revision or episode identity",
            )?;
            ensure(
                (self.driver.episode_steps[i] == 0) == a.previous.is_none(),
                "history clock/previous transition mismatch",
            )?;
            let clock = self.driver.episode_steps[i];
            ensure(
                a.learning_start as u64 <= clock
                    && (a.begun || a.learning_start as u64 == clock)
                    && (a.begun || a.learning_start == 0 || a.pending_context.is_some()),
                "stateful lane learning boundary differs from its clock",
            )?;
            if let Some(context) = &a.pending_context {
                let expected = context.history.previous_input();
                ensure(
                    !a.begun
                        && context.history.timestep() == clock
                        && context
                            .history
                            .matches(&self.driver.observations[i], context.physical_timestep)
                        && a.adoptions.is_empty()
                        && a.adoption_base.is_none()
                        && match (expected, a.previous.as_ref()) {
                            (Some(expected), Some(previous)) => {
                                expected.action == &previous.action
                                    && expected.reward.to_bits() == previous.reward.to_bits()
                            }
                            (None, None) => true,
                            _ => false,
                        },
                    "pending resume context differs from stateful lane boundary",
                )?;
                context
                    .history
                    .validate_in(&self.observation_space(), self.action_space())?;
            }
            if let Some(p) = &a.previous {
                a.actor.action_space().check(&p.action, None)?;
                ensure(p.reward.is_finite(), "nonfinite previous reward")?;
            }
            let preceding = a
                .pending_context
                .as_ref()
                .and_then(|context| context.history.previous_input())
                .and_then(|input| input.factual)
                .map(|input| input.observation);
            a.actor
                .validate_state_at(&a.state, &self.driver.observations[i], preceding)?;
            ensure(
                a.adoptions.len() <= MAX_LANE_ADOPTIONS,
                "adoption ledger exceeds its per-episode limit",
            )?;
            if let Some(base) = &a.adoption_base {
                base.validate()?;
                ensure(
                    self.compaction_count > 0
                        && base.episode_id == a.episode_id
                        && u64::try_from(base.observation_index)
                            .is_ok_and(|clock| clock <= self.driver.episode_steps[i])
                        && base.revision <= a.actor.revision(),
                    "invalid adoption base episode, clock or revision",
                )?;
                retained_adoptions = retained_adoptions
                    .checked_add(base.retired_count)
                    .ok_or_else(|| crate::Error::State("adoption count overflow".into()))?;
                retained_compactions = retained_compactions
                    .checked_add(base.retired_count.div_ceil(MAX_LANE_ADOPTIONS as u64))
                    .ok_or_else(|| crate::Error::State("compaction count overflow".into()))?;
            }
            let mut previous_adoption: Option<&AdoptionRecord> = None;
            for record in &a.adoptions {
                record.validate()?;
                ensure(
                    record.episode_id == a.episode_id
                        && u64::try_from(record.observation_index)
                            .is_ok_and(|clock| clock <= self.driver.episode_steps[i])
                        && previous_adoption.is_none_or(|previous| {
                            previous.to_revision == record.from_revision
                                && previous.observation_index <= record.observation_index
                        })
                        && (previous_adoption.is_some()
                            || a.adoption_base.as_ref().is_none_or(|base| {
                                base.revision == record.from_revision
                                    && base.observation_index <= record.observation_index
                            })),
                    "adoption episode, clock or revision chain differs",
                )?;
                previous_adoption = Some(record);
            }
            ensure(
                previous_adoption.map_or_else(
                    || {
                        a.adoption_base
                            .as_ref()
                            .is_none_or(|base| base.revision == a.actor.revision())
                    },
                    |record| record.to_revision == a.actor.revision(),
                ),
                "adoption ledger does not end at the pinned actor revision",
            )?;
            retained_adoptions = retained_adoptions
                .checked_add(a.adoptions.len() as u64)
                .ok_or_else(|| crate::Error::State("adoption count overflow".into()))?;
        }
        ensure(
            retained_adoptions <= self.adoption_count
                && retained_compactions <= self.compaction_count
                && self.compaction_count <= self.adoption_count,
            "adoption or compaction counters differ from retained provenance",
        )?;
        self.audited.set(true);
        Ok(())
    }
    /// Cheap operational check after a complete initial audit. No caller can
    /// mutate lanes, models or driver state through the public read accessors.
    /// Untrusted deserialization always takes the complete audit before use.
    pub fn validate_local(&self) -> Result<()> {
        ensure(
            !self.needs_recovery,
            "stateful collector requires full recovery",
        )?;
        if !self.audited.get() {
            return self.validate_state();
        }
        let ordinary = self.ordinary_snapshot_version();
        ensure(
            (self.version == ordinary || self.version == ordinary + 5)
                && !self.agents.is_empty()
                && self.agents.len() == self.driver.len()
                && self.driver.observations.len() == self.len()
                && self.driver.needs_reset.len() == self.len()
                && self.driver.episode_steps.len() == self.len()
                && self.latest.observation_space() == self.observation_space()
                && self.latest.action_space() == self.driver.env.action_space(),
            "invalid stateful runtime local dimensions",
        )
    }
    pub fn collect<S: TemporalSink>(
        &mut self,
        vector_steps: usize,
        mode: ActMode,
        sink: &mut S,
    ) -> Result<CollectionDelta> {
        let mut all_active = std::mem::take(&mut self.scratch.all_active);
        all_active.resize(self.len(), true);
        all_active.fill(true);
        let result = (|| {
            let mut total = CollectionDelta::default();
            for _ in 0..vector_steps {
                total.add(self.tick(&all_active, mode, sink)?);
            }
            Ok(total)
        })();
        self.scratch.all_active = all_active;
        result
    }
    pub fn tick<S: TemporalSink>(
        &mut self,
        active: &[bool],
        mode: ActMode,
        sink: &mut S,
    ) -> Result<CollectionDelta> {
        self.tick_recorded(active, mode, &mut BasicRecording(sink))
    }

    /// Opt-in metadata recording using the same lane clock, staging and recovery
    /// rules as ordinary temporal collection. The strategy never owns live memory.
    pub fn tick_recorded<P: StatefulRecording<A>>(
        &mut self,
        active: &[bool],
        mode: ActMode,
        sink: &mut P,
    ) -> Result<CollectionDelta> {
        self.validate_local()?;
        ensure(
            active.len() == self.len(),
            "stateful active mask width mismatch",
        )?;
        let resets = active
            .iter()
            .zip(&self.driver.needs_reset)
            .filter(|(active, reset)| **active && **reset)
            .count() as u64;
        self.next_episode_id
            .checked_add(resets)
            .ok_or_else(|| crate::Error::State("episode identity overflow".into()))?;
        ensure(
            active.iter().enumerate().all(|(i, active)| {
                !active || self.driver.needs_reset[i] || self.driver.episode_steps[i] < u64::MAX
            }),
            "stateful episode clock overflow",
        )?;
        let mut scratch = std::mem::take(&mut self.scratch);
        let result = self.tick_inner(active, mode, sink, &mut scratch);
        self.scratch = scratch;
        if result.is_err() {
            self.audited.set(false);
            self.needs_recovery = true;
            self.driver.fail();
        }
        result
    }
    fn tick_inner<S: StatefulRecording<A>>(
        &mut self,
        active: &[bool],
        mode: ActMode,
        sink: &mut S,
        scratch: &mut Scratch<A::State>,
    ) -> Result<CollectionDelta> {
        scratch.reset.clear();
        scratch.reset.extend(
            active
                .iter()
                .zip(&self.driver.needs_reset)
                .map(|(a, b)| *a && *b),
        );
        self.driver.begin(false)?;
        self.driver.active.clone_from_slice(active);
        let contexts = self.driver.prepare()?;
        self.initialize_reset_lanes(&scratch.reset, contexts)?;
        for i in 0..self.len() {
            if !self.agents[i].begun {
                // A sink may persist a tape before reporting an I/O failure.
                // Recovery must abandon every attempted begin, not just successes.
                self.agents[i].begun = true;
                let start = EpisodeStart {
                    observation: std::mem::replace(
                        &mut self.driver.observations[i],
                        Observation::new(Vec::new()),
                    ),
                    context: self.agents[i].pending_context.take(),
                };
                let zero_depth_context = start.context.is_some() && start.decision_timestep() == 0;
                let lane = &self.agents[i];
                let result = sink.begin_episode(lane.episode_id, &start);
                self.driver.observations[i] = start.observation;
                result?;
                if zero_depth_context {
                    self.version = self.snapshot_version();
                }
            }
        }
        self.make_groups(&mut scratch.groups);
        scratch.rngs.clear();
        scratch.rngs.extend(
            self.driver
                .lanes
                .iter()
                .map(|&i| self.agents[i].rng.clone()),
        );
        scratch.decisions.clear();
        scratch
            .decisions
            .resize_with(self.driver.lanes.len(), || None);
        for group in &scratch.groups {
            let i = self.driver.lanes[group[0]];
            let actor = &self.agents[i].actor;
            if group.len() == 1 {
                let j = group[0];
                scratch.decisions[j] = Some(sink.decide(
                    RecordingKey {
                        episode_id: self.agents[i].episode_id,
                        timestep: self.driver.episode_steps[i] as usize,
                    },
                    actor,
                    &self.agents[i].state,
                    &self.driver.observations[i],
                    mode,
                    &mut scratch.rngs[j],
                )?);
            } else {
                let states = group
                    .iter()
                    .map(|&j| &self.agents[self.driver.lanes[j]].state)
                    .collect::<Vec<_>>();
                let observations = group
                    .iter()
                    .map(|&j| &self.driver.observations[self.driver.lanes[j]])
                    .collect::<Vec<_>>();
                let mut rngs = group
                    .iter()
                    .map(|&j| scratch.rngs[j].clone())
                    .collect::<Vec<_>>();
                scratch.keys.clear();
                if S::NEEDS_KEYS {
                    scratch.keys.extend(group.iter().map(|&j| {
                        let lane = self.driver.lanes[j];
                        RecordingKey {
                            episode_id: self.agents[lane].episode_id,
                            timestep: self.driver.episode_steps[lane] as usize,
                        }
                    }));
                }
                let decisions = if self.inference_mode == InferenceMode::CanonicalBatched {
                    sink.decide_canonical_batch(
                        &scratch.keys,
                        actor,
                        &states,
                        &observations,
                        mode,
                        &mut rngs,
                    )?
                } else {
                    sink.decide_batch(
                        &scratch.keys,
                        actor,
                        &states,
                        &observations,
                        mode,
                        &mut rngs,
                    )?
                };
                ensure(
                    decisions.len() == group.len(),
                    "stateful decision batch output count mismatch",
                )?;
                for ((&j, decision), rng) in group.iter().zip(decisions).zip(rngs) {
                    scratch.decisions[j] = Some(decision);
                    scratch.rngs[j] = rng;
                }
            }
        }
        scratch.actions.clear();
        scratch.metadata.clear();
        for (&i, choice) in self.driver.lanes.iter().zip(&mut scratch.decisions) {
            let choice = choice.take().expect("all grouped decisions are staged");
            ensure(
                choice.log_prob.is_none_or(|x| x.is_finite()),
                "nonfinite behavior likelihood",
            )?;
            let lane = &self.agents[i];
            scratch.actions.push(choice.action);
            scratch.metadata.push(BehaviorMetadata {
                revision: lane.actor.revision(),
                log_prob: choice.log_prob,
            });
        }
        self.driver.step(&mut scratch.actions)?;
        scratch.states.clear();
        scratch.states.resize_with(self.driver.lanes.len(), || None);
        for group in &scratch.groups {
            let i = self.driver.lanes[group[0]];
            let actor = &self.agents[i].actor;
            if group.len() == 1 {
                let j = group[0];
                let step = self.driver.results[i]
                    .as_ref()
                    .expect("validated active result");
                let state = actor.assimilate(
                    &self.agents[i].state,
                    HistoryInput {
                        observation: &step.observation,
                        previous: Some(PreviousInput::from_transition(
                            &scratch.actions[j],
                            &step.reward,
                            &self.driver.observations[i],
                        )),
                    },
                )?;
                actor.validate_state(&state)?;
                scratch.states[j] = Some(state);
                continue;
            }
            let states = group
                .iter()
                .map(|&j| &self.agents[self.driver.lanes[j]].state)
                .collect::<Vec<_>>();
            let inputs = group
                .iter()
                .map(|&j| {
                    let lane = self.driver.lanes[j];
                    let step = self.driver.results[lane]
                        .as_ref()
                        .expect("validated active result");
                    HistoryInput {
                        observation: &step.observation,
                        previous: Some(PreviousInput::from_transition(
                            &scratch.actions[j],
                            &step.reward,
                            &self.driver.observations[lane],
                        )),
                    }
                })
                .collect::<Vec<_>>();
            let candidates = if self.inference_mode == InferenceMode::CanonicalBatched {
                actor.assimilate_canonical_batch(&states, &inputs)?
            } else {
                actor.assimilate_batch(&states, &inputs)?
            };
            ensure(
                candidates.len() == group.len(),
                "stateful assimilation batch output count mismatch",
            )?;
            for (&j, state) in group.iter().zip(candidates) {
                self.agents[self.driver.lanes[j]]
                    .actor
                    .validate_state(&state)?;
                scratch.states[j] = Some(state);
            }
        }
        if S::NEEDS_KEYS {
            for group in &scratch.groups {
                let actor = &self.agents[self.driver.lanes[group[0]]].actor;
                scratch.keys.clear();
                scratch.keys.extend(group.iter().map(|&j| {
                    let lane = self.driver.lanes[j];
                    RecordingKey {
                        episode_id: self.agents[lane].episode_id,
                        timestep: self.driver.episode_steps[lane] as usize,
                    }
                }));
                let states = group
                    .iter()
                    .map(|&j| {
                        scratch.states[j]
                            .as_ref()
                            .expect("validated successor candidate")
                    })
                    .collect::<Vec<_>>();
                let observations = group
                    .iter()
                    .map(|&j| {
                        &self.driver.results[self.driver.lanes[j]]
                            .as_ref()
                            .expect("validated result")
                            .observation
                    })
                    .collect::<Vec<_>>();
                let terminated = group
                    .iter()
                    .map(|&j| {
                        self.driver.results[self.driver.lanes[j]]
                            .as_ref()
                            .expect("validated result")
                            .terminated
                    })
                    .collect::<Vec<_>>();
                if self.inference_mode == InferenceMode::CanonicalBatched {
                    sink.prepare_canonical_successors(
                        &scratch.keys,
                        actor,
                        &states,
                        &observations,
                        &terminated,
                    )?;
                } else {
                    sink.prepare_successors(
                        &scratch.keys,
                        actor,
                        &states,
                        &observations,
                        &terminated,
                    )?;
                }
            }
        }
        // Sink errors may leave a persisted prefix; poison prevents pretending the
        // environment/memory/tape transaction completed. Recovery abandons that tape.
        for (j, &i) in self.driver.lanes.iter().enumerate() {
            let step = self.driver.results[i]
                .as_ref()
                .expect("validated active result");
            sink.push_with_state(
                &TemporalTransition {
                    episode_id: self.agents[i].episode_id,
                    timestep: self.driver.episode_steps[i] as usize,
                    observation: self.driver.observations[i].clone(),
                    action: scratch.actions[j].clone(),
                    reward: step.reward.clone(),
                    next_observation: step.observation.clone(),
                    terminated: step.terminated,
                    truncated: step.truncated,
                    behavior: scratch.metadata[j].clone(),
                },
                &self.agents[i].actor,
                scratch.states[j]
                    .as_ref()
                    .expect("validated successor candidate"),
            )?;
        }
        for (((&i, state), rng), action) in self
            .driver
            .lanes
            .iter()
            .zip(scratch.states.drain(..))
            .zip(scratch.rngs.drain(..))
            .zip(scratch.actions.drain(..))
        {
            let step = self.driver.results[i]
                .as_ref()
                .expect("validated active result");
            self.agents[i].state = state.expect("all grouped states are staged");
            self.agents[i].rng = rng;
            self.agents[i].previous = Some(PreviousStep {
                action,
                reward: step.reward.total(),
            });
        }
        Ok(self.driver.commit_step(false))
    }
    fn make_groups(&self, groups: &mut Vec<Vec<usize>>) {
        for group in groups.iter_mut() {
            group.clear();
        }
        let mut count = 0;
        for (position, &lane) in self.driver.lanes.iter().enumerate() {
            let actor = &self.agents[lane].actor;
            let compatible =
                if self.inference_mode != InferenceMode::Scalar && actor.batch_compatible(actor) {
                    groups[..count].iter().position(|group| {
                        let other = &self.agents[self.driver.lanes[group[0]]].actor;
                        actor.batch_compatible(other) && other.batch_compatible(actor)
                    })
                } else {
                    None
                };
            if let Some(group) = compatible {
                groups[group].push(position);
            } else {
                if count == groups.len() {
                    groups.push(Vec::new());
                }
                groups[count].push(position);
                count += 1;
            }
        }
        groups.truncate(count);
    }
    fn initialize_reset_lanes(&mut self, reset: &[bool], contexts: ResumeContexts) -> Result<()> {
        let lanes = reset
            .iter()
            .enumerate()
            .filter_map(|(i, reset)| reset.then_some(i))
            .collect::<Vec<_>>();
        if lanes.is_empty() {
            return Ok(());
        }
        let states = reset_states(
            &self.latest,
            self.inference_mode,
            &self.driver.observations,
            &lanes,
            &contexts,
        )?;
        for state in &states {
            self.latest.validate_state(state)?;
        }
        let mut contexts = contexts.into_iter().peekable();
        for (i, state) in lanes.into_iter().zip(states) {
            let context = contexts
                .next_if(|(lane, _)| *lane == i)
                .map(|(_, context)| context);
            let clock = context
                .as_ref()
                .map_or(0, |context| context.history.timestep());
            self.driver.episode_steps[i] = clock;
            self.agents[i].actor = self.latest.clone();
            self.agents[i].state = state;
            self.agents[i].episode_id = self.next_episode_id;
            self.agents[i].previous = boundary_previous(context.as_ref());
            self.agents[i].begun = false;
            self.agents[i].learning_start = clock as usize;
            self.agents[i].pending_context = context;
            self.agents[i].adoptions.clear();
            self.agents[i].adoption_base = None;
            self.next_episode_id += 1;
        }
        self.version = self.snapshot_version();
        Ok(())
    }
    /// Recover after a failed interaction/sink operation. Old partial tapes are
    /// explicitly abandoned and all environments/states begin fresh episodes.
    pub fn reset_all<S: TemporalSink>(&mut self, sink: &mut S) -> Result<()> {
        // Only an in-memory failure can set needs_recovery. A deserialized or
        // otherwise untrusted graph must pass its audit before even a full reset.
        if !self.needs_recovery {
            self.validate_local()?;
        }
        self.next_episode_id
            .checked_add(self.len() as u64)
            .ok_or_else(|| crate::Error::State("episode identity overflow".into()))?;
        let result = (|| {
            for a in &self.agents {
                if a.begun {
                    sink.abandon_episode(a.episode_id)?;
                }
            }
            let contexts = self.driver.reset_all()?;
            self.initialize_reset_lanes(&vec![true; self.len()], contexts)?;
            self.needs_recovery = false;
            self.validate_state()
        })();
        if result.is_err() {
            self.audited.set(false);
            self.needs_recovery = true;
            self.driver.fail();
        }
        result
    }
}

impl<V: VectorEnv, A: ReconstructibleActor + Clone> StatefulCollector<V, A> {
    /// Reconcile from an explicit pre-observation boundary and its actual retained
    /// public suffix. No zero-state substitution or older-anchor fallback occurs.
    /// A reconstruction failure is pure candidate work and does not poison or
    /// alter the live lane. The returned decision states any approximation.
    pub fn reconcile_lane_anchored(
        &mut self,
        lane: usize,
        id: RepresentationId,
        config: ReconstructionConfig,
        anchor: &StateAnchor,
        history: RetainedHistory<'_>,
    ) -> Result<ReconstructionDecision> {
        ensure(
            self.inference_mode != InferenceMode::CanonicalBatched,
            "canonical inference requires full-history reconciliation; anchors lack numerical execution provenance",
        )?;
        let clock = self.adoption_preflight(lane)?;
        self.validate_adoption_history(lane, history, clock)?;
        let actor = self.latest.clone();
        let model = actor.representation()?;
        let (mut state, decision) = decode_anchor(
            config,
            &model,
            id,
            self.agents[lane].episode_id,
            history.base_timestep,
            anchor,
        )?;
        for (t, observation) in history.observations.iter().enumerate() {
            state = model.assimilate(
                &state,
                HistoryInput {
                    observation,
                    previous: previous_input(history, t),
                },
            )?;
            model.validate_state(&state)?;
        }
        actor.validate_state(&state)?;
        let record = AdoptionRecord {
            version: 1,
            episode_id: self.agents[lane].episode_id,
            observation_index: clock,
            from_revision: self.agents[lane].actor.revision(),
            to_revision: actor.revision(),
            history_identity: public_history_identity(self.agents[lane].episode_id, history),
            source: AdoptionSource::Anchored {
                representation_id: id,
                anchor_identity: anchor_identity(anchor)?,
                config,
                decision: decision.clone(),
                anchor_generation: anchor.generation,
                current_generation: model.generation(),
            },
        };
        self.commit_adoption(lane, actor, state, record)?;
        Ok(decision)
    }
}

fn zero_adoptions(count: &u64) -> bool {
    *count == 0
}
impl<V, A> Snapshot for StatefulCollector<V, A>
where
    V: VectorEnv + Snapshot,
    A: StatefulActor + Clone + Snapshot,
    A::State: Serialize + DeserializeOwned,
{
    fn validate_snapshot(&self) -> Result<()> {
        let result = (|| {
            self.validate_state()?;
            self.driver.env.validate_snapshot()?;
            self.latest.validate_snapshot()?;
            for a in &self.agents {
                a.actor.validate_snapshot()?;
            }
            Ok(())
        })();
        self.audited.set(result.is_ok());
        result
    }
}
