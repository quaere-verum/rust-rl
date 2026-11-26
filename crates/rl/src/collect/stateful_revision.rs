//! Opt-in bounded revision provenance, committed to an externally retained tape.
use super::{
    AdoptionSource, RetainedHistory, StatefulActor, StatefulCollector, VectorEnv,
    public_history_identity,
};
use crate::{
    Error, Result,
    core::{ensure, env::Observation, temporal::BehaviorMetadata},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Revisions before this absolute boundary are committed by the retained tape's
/// identity rather than individually retained adoption records. The named
/// revision applies at the boundary, until a later suffix adoption supersedes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdoptionBase {
    pub version: u32,
    pub episode_id: u64,
    pub observation_index: usize,
    pub revision: u64,
    pub retired_count: u64,
    pub history_identity: [u8; 32],
}
impl AdoptionBase {
    pub(super) fn validate(&self) -> Result<()> {
        ensure(
            self.version == 1 && self.retired_count > 0 && self.revision >= self.retired_count,
            "invalid adoption base version, revision or retired count",
        )
    }
}

impl<V: VectorEnv, A: StatefulActor + Clone> StatefulCollector<V, A> {
    pub fn compaction_count(&self) -> u64 {
        self.compaction_count
    }

    pub fn lane_adoption_base(&self, lane: usize) -> Option<&AdoptionBase> {
        self.agents.get(lane)?.adoption_base.as_ref()
    }

    /// Audit the genuine reset prefix against compacted provenance and every
    /// remaining adoption. This is read-only and also accepts an ended lane's
    /// actual final observation; the temporal store validates its end flags.
    pub fn validate_lane_revision_prefix(
        &self,
        lane: usize,
        history: RetainedHistory<'_>,
        behavior: &[BehaviorMetadata],
    ) -> Result<()> {
        self.validate_local()?;
        let agent = self
            .agents
            .get(lane)
            .ok_or_else(|| Error::State("revision audit lane out of range".into()))?;
        let clock = usize::try_from(self.driver.episode_steps[lane])
            .map_err(|_| Error::Shape("revision audit clock is not representable".into()))?;
        ensure(
            history.base_timestep == 0 && history.previous.is_none(),
            "revision provenance requires the complete genuine reset prefix",
        )?;
        self.validate_public_history(lane, history, clock, !self.driver.needs_reset[lane])?;
        ensure(
            history
                .observations
                .last()
                .is_some_and(|last| observations_equal_bits(last, &self.driver.observations[lane]))
                && behavior.len() == history.actions.len(),
            "revision prefix observation bits or behavior count differs",
        )?;
        for metadata in behavior {
            metadata.validate()?;
        }

        let first = if let Some(base) = &agent.adoption_base {
            let prefix = history_slice(history, 0, base.observation_index);
            ensure(
                base.history_identity
                    == base_history_identity(base, prefix, &behavior[..base.observation_index]),
                "compacted adoption prefix identity differs from the actual tape",
            )?;
            base.observation_index
        } else {
            0
        };
        for (timestep, metadata) in behavior
            .iter()
            .enumerate()
            .skip(first.max(agent.learning_start))
        {
            ensure(
                self.lane_behavior_revision_at(lane, timestep) == Some(metadata.revision),
                "temporal behavior revision differs from adoption provenance",
            )?;
        }
        for record in &agent.adoptions {
            let origin = match &record.source {
                AdoptionSource::FullHistory { .. } => 0,
                AdoptionSource::Anchored { decision, .. } => decision.observation_index,
            };
            ensure(
                record.history_identity
                    == public_history_identity(
                        agent.episode_id,
                        history_slice(history, origin, record.observation_index),
                    ),
                "adoption public-history identity differs from the actual tape",
            )?;
        }
        Ok(())
    }

    /// Retire pending adoption records into a bounded commitment without changing
    /// the actor, post-observation memory, environment, clock or inference RNG.
    /// The caller must retain the complete raw episode prefix independently.
    /// All validation and overflow checks precede the infallible commit.
    pub fn compact_lane_revisions(
        &mut self,
        lane: usize,
        history: RetainedHistory<'_>,
        behavior: &[BehaviorMetadata],
    ) -> Result<()> {
        self.validate_lane_revision_prefix(lane, history, behavior)?;
        let agent = &self.agents[lane];
        ensure(
            !self.driver.needs_reset[lane] && !agent.adoptions.is_empty(),
            "revision compaction requires a continuing lane with pending adoptions",
        )?;
        let count = u64::try_from(agent.adoptions.len())
            .map_err(|_| Error::State("retired adoption count is not representable".into()))?;
        let retired_count = agent
            .adoption_base
            .as_ref()
            .map_or(0, |base| base.retired_count)
            .checked_add(count)
            .ok_or_else(|| Error::State("retired adoption count overflow".into()))?;
        let compaction_count = self
            .compaction_count
            .checked_add(1)
            .ok_or_else(|| Error::State("compaction counter overflow".into()))?;
        let mut base = AdoptionBase {
            version: 1,
            episode_id: agent.episode_id,
            observation_index: history.actions.len(),
            revision: agent.actor.revision(),
            retired_count,
            history_identity: [0; 32],
        };
        base.validate()?;
        base.history_identity = base_history_identity(&base, history, behavior);

        self.agents[lane].adoption_base = Some(base);
        self.agents[lane].adoptions.clear();
        self.compaction_count = compaction_count;
        self.version = self.snapshot_version();
        Ok(())
    }
}

fn history_slice(history: RetainedHistory<'_>, start: usize, end: usize) -> RetainedHistory<'_> {
    RetainedHistory {
        base_timestep: start,
        previous: start.checked_sub(1).map(|index| {
            history
                .rewards
                .input(index, &history.actions[index], &history.observations[index])
        }),
        observations: &history.observations[start..=end],
        actions: &history.actions[start..end],
        rewards: history.rewards.slice(start..end),
    }
}

fn base_history_identity(
    base: &AdoptionBase,
    history: RetainedHistory<'_>,
    behavior: &[BehaviorMetadata],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"rust-rl-adoption-base-v1\0");
    hash.update(base.version.to_le_bytes());
    hash.update(base.episode_id.to_le_bytes());
    hash.update((base.observation_index as u64).to_le_bytes());
    hash.update(base.revision.to_le_bytes());
    hash.update(base.retired_count.to_le_bytes());
    hash.update(public_history_identity(base.episode_id, history));
    hash.update((behavior.len() as u64).to_le_bytes());
    for metadata in behavior {
        hash.update(metadata.revision.to_le_bytes());
        match metadata.log_prob {
            None => hash.update([0]),
            Some(log_prob) => {
                hash.update([1]);
                hash.update(log_prob.to_bits().to_le_bytes());
            }
        }
    }
    hash.finalize().into()
}

fn observations_equal_bits(left: &Observation, right: &Observation) -> bool {
    left.matches(right)
}
