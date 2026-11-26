use rl::{
    Error, Result,
    algorithms::tabular::{QTable, TabularActor, TdConfig},
    core::{
        action::FiniteActionKey,
        learner::UpdateReport,
        state_key::StateKeyBuilder,
        stateful::{BehaviorPublisher, StatefulLearner},
        temporal::TemporalSpaces,
    },
    data::temporal::SequenceBatch,
};

use crate::hall::{GAMMA, RememberCue, State};

/// Q-learning with a bounded change per factual decision. The TD residual
/// is clipped by adjusting the table's step size; the target stays factual.
pub struct BoundedTd {
    pub table: QTable<State, FiniteActionKey>,
    spaces: TemporalSpaces,
}

impl BoundedTd {
    pub fn new(spaces: TemporalSpaces) -> Result<Self> {
        Ok(Self {
            table: QTable::new(0.)?,
            spaces,
        })
    }
}

impl StatefulLearner for BoundedTd {
    type Batch = SequenceBatch;

    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        batch.validate_in(&self.spaces.observations, &self.spaces.actions)?;
        if batch.config.learn_len != 1 || batch.config.n_step != 1 {
            return Err(Error::Config("bounded TD expects one-step windows".into()));
        }
        let mut table = self.table.clone();
        for sample in &batch.samples {
            let position = sample.learn_start;
            let mut inputs = sample.exact_history_inputs()?;
            let memory = inputs
                .by_ref()
                .take(position + 1)
                .try_fold(RememberCue.initial_memory()?, |memory, input| {
                    RememberCue.observe(&memory, input)
                })?;
            let state = RememberCue.key(&memory)?;
            let step = &sample.steps[position];
            let observation = &sample.observations[position];
            let action = observation.finite_action_key(&step.action)?;
            let (reward, discount) = step.reward.discounted_f64(GAMMA)?;
            let mut residual = reward - table.value(&state, &action);
            let next = if step.terminated {
                None
            } else {
                let input = inputs
                    .next()
                    .ok_or_else(|| Error::State("missing factual successor".into()))?;
                let next_state = RememberCue.key(&RememberCue.observe(&memory, input)?)?;
                let legal = input
                    .observation
                    .finite_actions(&self.spaces.actions)?
                    .iter()
                    .map(|action| input.observation.finite_action_key(action))
                    .collect::<Result<Vec<_>>>()?;
                let best = table.greedy_index(&next_state, &legal)?;
                residual += discount * table.value(&next_state, &legal[best]);
                Some((next_state, legal))
            };
            table.q_learning(
                &state,
                &action,
                &step.reward,
                next.as_ref()
                    .map(|(state, legal)| (state, legal.as_slice())),
                TdConfig {
                    gamma: GAMMA,
                    step_size: 1. / residual.abs().max(1.),
                },
            )?;
        }
        self.table = table;
        Ok(UpdateReport {
            updates: 1,
            ..UpdateReport::default()
        })
    }
}

impl BehaviorPublisher for BoundedTd {
    type Actor = TabularActor<RememberCue>;
    type Settings = f64;

    fn behavior_actor(&self, epsilon: &f64, revision: u64) -> Result<Self::Actor> {
        TabularActor::new(
            RememberCue,
            self.table.clone(),
            self.spaces.clone(),
            revision,
            *epsilon,
        )
    }
}
