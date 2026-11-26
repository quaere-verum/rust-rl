use super::RecurrentGradientMode;
use crate::{
    Error, Result,
    algorithms::candidate_ddqn::{
        CandidateCritic, CandidateDdqnConfig, candidate_frame, double_q_target, maximum_target,
        policy_target,
    },
    core::{
        action::{Action, ActionSpace},
        checkpoint::LearnerRecord,
        ensure,
        env::ObservationSpace,
        learner::UpdateReport,
        stateful::HistoryInput,
        structured::ActionCandidate,
        temporal::TemporalSpaces,
    },
    data::temporal::{NStepTarget, SequenceBatch},
    nn::{
        structured::{RecurrentLightweightQ, StructuredHistoryInput, StructuredHistoryValues},
        to_vec,
    },
};
use burn::{module::Module, tensor::Tensor};

pub(super) fn new_critic(
    online: RecurrentLightweightQ,
    config: CandidateDdqnConfig,
    observations: ObservationSpace,
    actions: ActionSpace,
) -> Result<CandidateCritic<RecurrentLightweightQ>> {
    online.config().validate()?;
    online
        .config()
        .model
        .input
        .validate_spaces(&observations, &actions)?;
    CandidateCritic::new_with_checked_spaces(online, config, observations, actions)
}

pub(super) struct LearningPosition<'a> {
    pub history: usize,
    pub step: usize,
    pub candidates: &'a [ActionCandidate],
    action_index: usize,
    pub target: NStepTarget,
}

pub(super) struct RecurrentBatch<'a> {
    histories: Vec<Vec<HistoryInput<'a>>>,
    learn_starts: Vec<usize>,
    pub positions: Vec<LearningPosition<'a>>,
}

impl<'a> RecurrentBatch<'a> {
    pub fn new(
        batch: &'a SequenceBatch,
        critic: &CandidateCritic<RecurrentLightweightQ>,
    ) -> Result<Self> {
        critic.check_update()?;
        ensure(
            batch.config.n_step == 1,
            "recurrent candidate learning requires one-decision returns",
        )?;
        batch.validate_in(&critic.observation_space, &critic.action_space)?;
        let observations = batch.samples.iter().fold(0usize, |total, sample| {
            total.saturating_add(sample.observations.len())
        });
        ensure(
            observations <= critic.online.config().max_observations,
            "recurrent candidate batch exceeds observation budget",
        )?;
        let mut histories = Vec::with_capacity(batch.samples.len());
        let mut learn_starts = Vec::with_capacity(batch.samples.len());
        let mut positions = Vec::new();
        for (history, sample) in batch.samples.iter().enumerate() {
            histories.push(sample.exact_history_inputs()?.collect());
            learn_starts.push(sample.learn_start);
            for (step, &learn) in sample.loss_mask.iter().enumerate() {
                if !learn {
                    continue;
                }
                let frame = candidate_frame(&sample.observations[step])?;
                let Action::Conditional(action) = &sample.steps[step].action else {
                    return Err(Error::Action(
                        "recurrent candidate learning needs a complete action".into(),
                    ));
                };
                let action_index = frame
                    .candidates
                    .iter()
                    .position(|candidate| candidate.enabled && &candidate.action == action.as_ref())
                    .ok_or_else(|| {
                        Error::Action("recorded recurrent candidate is unavailable".into())
                    })?;
                positions.push(LearningPosition {
                    history,
                    step,
                    candidates: &frame.candidates,
                    action_index,
                    target: sample.target_at(step, critic.config.gamma, 1)?,
                });
            }
        }
        Ok(Self {
            histories,
            learn_starts,
            positions,
        })
    }

    pub fn inputs(&self, gradient: RecurrentGradientMode) -> Vec<StructuredHistoryInput<'_>> {
        self.histories
            .iter()
            .zip(&self.learn_starts)
            .map(|(inputs, &learn_start)| {
                let learn_start = match gradient {
                    RecurrentGradientMode::LearningWindow => learn_start,
                    RecurrentGradientMode::FullHistory => 0,
                };
                StructuredHistoryInput {
                    inputs,
                    learn_start,
                }
            })
            .collect()
    }

    pub fn next_candidates(&self, position: &LearningPosition<'_>) -> Result<&[ActionCandidate]> {
        Ok(&candidate_frame(
            self.histories[position.history][position.target.bootstrap_index].observation,
        )?
        .candidates)
    }
}

pub(super) fn decision_values(
    history: &[StructuredHistoryValues],
    lane: usize,
    step: usize,
    candidates: usize,
) -> Result<Tensor<1>> {
    let values = history
        .get(lane)
        .and_then(|h| h.decisions.get(step))
        .and_then(Option::as_ref)
        .ok_or_else(|| Error::Shape("recurrent Q omitted a required candidate row".into()))?;
    ensure(
        values.dims()[0] == candidates,
        "recurrent Q candidate count differs",
    )?;
    Ok(values.clone())
}

pub(super) enum RecurrentBackup<'a> {
    Maximum,
    DoubleQ(&'a [StructuredHistoryValues]),
    Policy(&'a [Vec<f64>]),
}

impl CandidateCritic<RecurrentLightweightQ> {
    pub(super) fn recurrent_record(&self) -> Result<LearnerRecord> {
        let mut record = self.checkpoint_model(self.online.config())?;
        record.temporal_spaces = Some(TemporalSpaces {
            observations: self.observation_space.clone(),
            actions: self.action_space.clone(),
        });
        Ok(record)
    }

    pub(super) fn validate_recurrent_record(&self, record: &LearnerRecord) -> Result<()> {
        self.config
            .returns()
            .check(record.temporal_returns.ok_or_else(|| {
                Error::State("recurrent candidate checkpoint is missing its return contract".into())
            })?)?;
        record
            .temporal_spaces
            .as_ref()
            .ok_or_else(|| {
                Error::State(
                    "recurrent candidate checkpoint is missing its observation/action spaces"
                        .into(),
                )
            })?
            .check(&self.observation_space, &self.action_space)
    }

    pub(super) fn online_history_values(
        &self,
        batch: &RecurrentBatch<'_>,
    ) -> Result<Vec<StructuredHistoryValues>> {
        self.online.valid().evaluate_histories(
            &batch.inputs(RecurrentGradientMode::FullHistory),
            self.config.pack_limits(),
        )
    }

    pub(super) fn fit_recurrent(
        &mut self,
        batch: &RecurrentBatch<'_>,
        backup: RecurrentBackup<'_>,
        gradient: RecurrentGradientMode,
    ) -> Result<UpdateReport> {
        let limits = self.config.pack_limits();
        let inputs = batch.inputs(gradient);
        let target_values = if batch.positions.iter().any(|p| p.target.discount > 0.) {
            self.target.evaluate_histories(&inputs, limits)?
        } else {
            Vec::new()
        };
        let mut targets = Vec::with_capacity(batch.positions.len());
        for (index, position) in batch.positions.iter().enumerate() {
            let NStepTarget {
                reward,
                discount,
                bootstrap_index,
            } = position.target;
            if discount == 0. {
                targets.push(reward);
                continue;
            }
            let candidates = batch.next_candidates(position)?;
            let target = to_vec(decision_values(
                &target_values,
                position.history,
                bootstrap_index,
                candidates.len(),
            )?)?;
            let value = match &backup {
                RecurrentBackup::DoubleQ(_) | RecurrentBackup::Maximum => {
                    let enabled: Vec<_> = candidates
                        .iter()
                        .map(|candidate| candidate.enabled)
                        .collect();
                    if let RecurrentBackup::DoubleQ(online) = &backup {
                        let online = to_vec(decision_values(
                            online,
                            position.history,
                            bootstrap_index,
                            candidates.len(),
                        )?)?;
                        double_q_target(reward, discount, &online, &target, &enabled)?
                    } else {
                        maximum_target(reward, discount, &target, &enabled)?
                    }
                }
                RecurrentBackup::Policy(probabilities) => {
                    let probabilities = &probabilities[index];
                    ensure(
                        probabilities.len() == candidates.len()
                            && candidates
                                .iter()
                                .zip(probabilities)
                                .all(|(c, &p)| c.enabled || p == 0.),
                        "recurrent policy backup support differs",
                    )?;
                    policy_target(reward, discount, &target, probabilities)?
                }
            };
            targets.push(value);
        }
        let values = self.online.evaluate_histories(&inputs, limits)?;
        let mut predictions = Vec::with_capacity(batch.positions.len());
        for position in &batch.positions {
            let row = decision_values(
                &values,
                position.history,
                position.step,
                position.candidates.len(),
            )?;
            ensure(
                to_vec(row.clone())?.iter().all(|value| value.is_finite()),
                "nonfinite recurrent candidate Q",
            )?;
            predictions.push(row.narrow(0, position.action_index, 1));
        }
        let candidates = batch
            .positions
            .iter()
            .map(|position| position.candidates.len())
            .sum();
        self.fit_predictions(Tensor::cat(predictions, 0), &targets, candidates)
    }
}
