use super::{CollectionDelta, driver::LaneDriver, vector_env::VectorEnv};
use crate::{
    Result,
    core::{
        action::Action,
        actor::{ActMode, Actor},
        ensure,
        env::ObservationBatch,
        transition::TransitionView,
    },
    data::replay::ReplayWriter,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffPolicyCollector<V> {
    pub(super) driver: LaneDriver<V>,
    #[serde(skip)]
    actions: Vec<Action>,
}
impl<V: VectorEnv> OffPolicyCollector<V> {
    pub fn new(env: V, seed: u64) -> Result<Self> {
        Ok(Self {
            driver: LaneDriver::new(env, seed)?.0,
            actions: vec![],
        })
    }
    pub fn environment(&self) -> &V {
        &self.driver.env
    }
    /// Reset after a failed collection. Also clear any caller-owned n-step accumulator;
    /// its unfinished trajectories must not cross this explicit reset.
    pub fn reset_all(&mut self) -> Result<()> {
        self.driver.reset_all().map(|_| ())
    }
    pub(crate) fn validate_state(&self) -> Result<()> {
        self.driver.validate_state()
    }
    pub fn collect_into(
        &mut self,
        actor: &dyn Actor,
        steps: usize,
        writer: &mut ReplayWriter<'_>,
    ) -> Result<CollectionDelta> {
        let result = self.collect(actor, steps, writer);
        if result.is_err() {
            self.driver.fail();
        }
        result
    }
    fn collect(
        &mut self,
        actor: &dyn Actor,
        steps: usize,
        writer: &mut ReplayWriter<'_>,
    ) -> Result<CollectionDelta> {
        ensure(
            steps > 0 && actor.action_space() == self.driver.env.action_space(),
            "invalid collection budget or action space",
        )?;
        writer.check_shape(
            self.driver.env.observation_space().require_dense_width()?,
            self.driver.env.action_space(),
            self.driver.len(),
        )?;
        self.driver.begin(false)?;
        let mut delta = CollectionDelta::default();
        for _ in 0..steps {
            self.driver.prepare()?;
            let obs = ObservationBatch::selected(&self.driver.observations, &self.driver.lanes)?;
            actor.act_into(
                obs,
                ActMode::Sample,
                &mut self.driver.rng,
                &mut self.actions,
            )?;
            self.driver.step(&mut self.actions)?;
            for (j, &lane) in self.driver.lanes.iter().enumerate() {
                let s = self.driver.results[lane]
                    .as_ref()
                    .expect("validated active step");
                writer.push(
                    lane,
                    TransitionView {
                        observation: &self.driver.observations[lane],
                        action: &self.actions[j],
                        reward: &s.reward,
                        next_observation: &s.observation,
                        terminated: s.terminated,
                        truncated: s.truncated,
                    },
                )?;
            }
            delta.add(self.driver.commit_step(false));
        }
        Ok(delta)
    }
}
impl<V: VectorEnv + crate::core::checkpoint::Snapshot> crate::core::checkpoint::Snapshot
    for OffPolicyCollector<V>
{
    fn validate_snapshot(&self) -> Result<()> {
        self.driver.env.validate_snapshot()?;
        self.validate_state()
    }
}
