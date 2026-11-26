#![cfg(feature = "neural")]

//! Independent temporal numerical oracles, deliberately separate from learner losses.
use burn::{
    module::Param,
    tensor::{Tensor, TensorData},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    collect::{
        stateful::{InferenceMode, StatefulCollector},
        temporal_on_policy::OnPolicyRecording,
        vector_env::SyncVectorEnv,
    },
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, Observation, ObservationBatch, ObservationSpace, Step},
        stateful::{HistoryInput, HistorySpec, PreviousInput, StatefulActor},
        stateful_on_policy::DistributionSnapshot,
    },
    data::{
        temporal::{BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep},
        temporal_on_policy::{TemporalBatchKind, TemporalRollout},
    },
    nn::{
        device,
        temporal::GruEncoder,
        temporal_actor::{TemporalActor, TemporalActorState},
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_value::TemporalValue,
        to_vec, vector,
    },
};
use serde::{Deserialize, Serialize};

fn categorical_policy() -> TemporalPolicy {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut policy =
        TemporalPolicy::new(spec, Some(3), &[], &mut ChaCha8Rng::seed_from_u64(151)).unwrap();
    policy.head.layers[0].weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(vec![1.1, -0.8, -0.7, 0.9, 0.6, -0.5], [3, 2]),
        &device(),
    ));
    policy
}
fn history() -> SequenceBatch {
    SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 4,
            n_step: 1,
        },
        samples: vec![SequenceSample {
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: Default::default(),
            episode_id: 7,
            learn_start: 0,
            burn_start: 0,
            observations: [0.9, -0.4, 0.9, 0.9, 0.]
                .map(|x| Observation::new(vec![x]))
                .to_vec(),
            steps: (0..4)
                .map(|i| TemporalStep {
                    action: Action::Discrete([0, 1, 1, 0][i]),
                    reward: [0.3_f32, -0.6, 0.8, 0.2][i].into(),
                    terminated: i == 3,
                    truncated: false,
                    behavior: BehaviorMetadata {
                        revision: 0,
                        log_prob: None,
                    },
                })
                .collect(),
            loss_mask: vec![true; 4],
        }],
    }
}

fn logits_from_state(
    actor: &TemporalActor,
    state: &TemporalActorState,
    observation: &Observation,
) -> Vec<f32> {
    let features = Tensor::from_data(
        TensorData::new(state.policy.clone(), [1, state.policy.len()]),
        &actor.policy().device(),
    );
    let distribution = actor
        .policy()
        .distribution_from_features(
            features,
            ObservationBatch::all(std::slice::from_ref(observation)),
        )
        .unwrap();
    let DistributionSnapshot::Categorical { logits, .. } =
        DistributionSnapshot::from_distribution(&distribution, 0).unwrap()
    else {
        panic!("categorical oracle expected")
    };
    logits
}

// This scalar state-machine path does not use SequenceBatch encoding, masks,
// recurrent unroll, token gathering, or tensor likelihood reduction.
fn scalar_states_and_logits(
    actor: &TemporalActor,
    history: &SequenceBatch,
) -> (Vec<TemporalActorState>, Vec<Vec<f32>>) {
    let sample = &history.samples[0];
    let mut state = actor.initial_state().unwrap();
    let mut states = Vec::new();
    let mut logits = Vec::new();
    for t in 0..sample.steps.len() {
        state = actor
            .assimilate(
                &state,
                HistoryInput {
                    observation: &sample.observations[t],
                    previous: t.checked_sub(1).map(|previous| PreviousInput {
                        factual: None,
                        action: &sample.steps[previous].action,
                        reward: sample.steps[previous].reward.total(),
                    }),
                },
            )
            .unwrap();
        logits.push(logits_from_state(actor, &state, &sample.observations[t]));
        states.push(state.clone());
    }
    (states, logits)
}
fn scalar_log_prob(logits: &[f32], action: usize) -> f64 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum = logits
        .iter()
        .map(|&x| (x as f64 - maximum).exp())
        .sum::<f64>()
        .ln();
    logits[action] as f64 - maximum - log_sum
}
fn scalar_weighted_loss(policy: TemporalPolicy, history: &SequenceBatch) -> f64 {
    let actor = TemporalActor::new(policy, None, 0).unwrap();
    let (_, logits) = scalar_states_and_logits(&actor, history);
    logits
        .iter()
        .zip([0, 1, 1, 0])
        .zip([0.4, -0.7, 1.3, 0.5])
        .map(|((logits, action), weight)| -weight * scalar_log_prob(logits, action))
        .sum()
}

#[test]
fn recurrent_likelihood_gradient_matches_independent_scalar_central_difference() {
    let policy = categorical_policy();
    let history = history();
    let tokens = vec![(0, 0), (0, 1), (0, 2), (0, 3)];
    let actions: Vec<_> = history.samples[0]
        .steps
        .iter()
        .map(|step| step.action.clone())
        .collect();
    let logp = policy
        .distribution(&history, &tokens)
        .unwrap()
        .log_prob(&actions, &[None, None, None, None])
        .unwrap();
    let loss = (logp * vector(&[0.4, -0.7, 1.3, 0.5])).sum().neg();
    let gradients = loss.backward();
    let weight = policy
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .hidden_transform
        .weight
        .val();
    let analytic =
        to_vec(weight.grad(&gradients).expect("missing recurrent gradient")).unwrap()[0] as f64;
    assert!(
        analytic.abs() > 1e-4,
        "fixture must exercise a material history-dependent gradient: {analytic}"
    );
    let original = to_vec(weight).unwrap();
    for step in [0.01f32, 0.005] {
        let perturbed = |delta| {
            let mut candidate = policy.clone();
            let mut values = original.clone();
            values[0] += delta;
            let actual = values[0];
            candidate
                .encoder
                .gru
                .as_mut()
                .unwrap()
                .gru
                .new_gate
                .hidden_transform
                .weight = Param::from_tensor(Tensor::from_data(
                TensorData::new(values, [3, 3]),
                &policy.device(),
            ));
            (scalar_weighted_loss(candidate, &history), actual)
        };
        let (positive, plus) = perturbed(step);
        let (negative, minus) = perturbed(-step);
        let numerical = (positive - negative) / (plus as f64 - minus as f64);
        // The oracle accumulates f64 log-softmax, but GRU outputs are f32. At
        // h=0.005..0.01 the finite-difference roundoff is O(f32 epsilon / h),
        // while smooth central truncation is O(h^2). Check both perturbations.
        let tolerance = 6e-5 + 0.003 * analytic.abs();
        assert!(
            (numerical - analytic).abs() <= tolerance,
            "h={step}: autodiff={analytic:.9}, scalar central difference={numerical:.9}, tolerance={tolerance:.9}"
        );
        eprintln!(
            "recurrent scalar finite difference: h={step}, autodiff={analytic:.9}, central={numerical:.9}"
        );
    }
}

#[test]
fn distinct_behavior_and_current_grus_reconstruct_their_own_actual_histories() {
    let old = categorical_policy();
    let mut current = old.clone();
    current.encoder.gru = Some(
        GruEncoder::new(
            current.spec().input_dim(),
            3,
            &mut ChaCha8Rng::seed_from_u64(219),
        )
        .unwrap(),
    );
    let old_actor = TemporalActor::new(old.clone(), None, 0).unwrap();
    let current_actor = TemporalActor::new(current.clone(), None, 1).unwrap();
    let mut history = history();
    // Include detached prefix reconstruction as well as a complete learned tail.
    history.samples[0].learn_start = 2;
    history.samples[0].burn_start = 2;
    history.samples[0].loss_mask = vec![false, false, true, true];
    let (old_states, old_logits) = scalar_states_and_logits(&old_actor, &history);
    let (current_states, current_logits) = scalar_states_and_logits(&current_actor, &history);
    assert_ne!(old_states[3].policy, current_states[3].policy);
    for (model, expected) in [(&old, &old_logits), (&current, &current_logits)] {
        let distribution = model.distribution(&history, &[(0, 3), (0, 2)]).unwrap();
        let actual = to_vec(
            distribution
                .log_prob(&[Action::Discrete(0), Action::Discrete(1)], &[None, None])
                .unwrap(),
        )
        .unwrap();
        for ((&value, t), action) in actual.iter().zip([3, 2]).zip([0, 1]) {
            let oracle = scalar_log_prob(&expected[t], action);
            assert!(
                (value as f64 - oracle).abs() < 2e-6,
                "own-history likelihood mismatch at token {t}: tensor={value}, scalar={oracle}"
            );
        }
    }
    // Heads are identical. Reusing old post-observation memory in the current
    // head must visibly differ from the independently reconstructed current state.
    let wrong = logits_from_state(
        &current_actor,
        &old_states[3],
        &history.samples[0].observations[3],
    );
    let wrong_logp = scalar_log_prob(&wrong, 0);
    let correct_logp = scalar_log_prob(&current_logits[3], 0);
    assert!(
        (wrong_logp - correct_logp).abs() > 0.01,
        "fixture does not distinguish stale hidden state: old-memory={wrong_logp}, current-memory={correct_logp}"
    );
}

#[derive(Clone, Serialize, Deserialize)]
struct Staggered {
    lane: usize,
    tick: usize,
    episode: usize,
    horizon: usize,
    space: ActionSpace,
}
impl Staggered {
    fn observation(&self, influence: f32) -> Observation {
        let mut observation = Observation::new(vec![
            0.1 * self.lane as f32 + 0.2 * self.tick as f32 - 0.03 * self.episode as f32
                + 0.05 * influence,
        ]);
        if matches!(self.space, ActionSpace::Discrete { .. }) {
            observation.as_dense_mut().unwrap().action_mask =
                Some(vec![true, !self.tick.is_multiple_of(3)]);
        }
        observation
    }
}
impl Env for Staggered {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.tick = 0;
        self.episode += 1;
        Ok(EpisodeStart::fresh(self.observation(0.)))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.tick >= self.horizon {
            return Err(Error::State("staggered episode already ended".into()));
        }
        self.space.check(action, None)?;
        let influence = match action {
            Action::Discrete(a) => *a as f32,
            Action::Continuous(a) => a[0],
            Action::Conditional(_) => unreachable!("validated fixed-width action"),
        };
        self.tick += 1;
        let ended = self.tick == self.horizon;
        let truncated = ended && self.lane % 2 == 1;
        let terminated = ended && !truncated;
        let mut observation = self.observation(influence);
        if terminated && matches!(self.space, ActionSpace::Discrete { .. }) {
            observation.as_dense_mut().unwrap().action_mask = Some(vec![false, false]);
        }
        Ok(Step {
            observation,
            reward: (0.2 * influence - 0.1 * self.tick as f32).into(),
            terminated,
            truncated,
        })
    }
}
impl Snapshot for Staggered {
    fn validate_snapshot(&self) -> Result<()> {
        self.space.validate()?;
        if self.horizon < 2 || self.tick > self.horizon {
            return Err(Error::State("invalid staggered clock".into()));
        }
        Ok(())
    }
}

#[test]
fn rich_scalar_and_batched_collection_preserve_lane_streams_at_1_4_16_lanes() {
    for lanes in [1, 4, 16] {
        for space in [
            ActionSpace::Discrete { count: 2 },
            ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
        ] {
            let spec = HistorySpec::new(1, space.clone()).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(303);
            let actor = TemporalActor::new(
                TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
                Some(TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap()),
                0,
            )
            .unwrap();
            let make = |mode| {
                StatefulCollector::new_with_inference_mode(
                    SyncVectorEnv::new(
                        (0..lanes)
                            .map(|lane| Staggered {
                                lane,
                                tick: 0,
                                episode: 0,
                                horizon: 2 + lane % 4,
                                space: space.clone(),
                            })
                            .collect(),
                    )
                    .unwrap(),
                    actor.clone(),
                    304,
                    mode,
                )
                .unwrap()
            };
            let mut scalar = make(InferenceMode::Scalar);
            let mut batched = make(InferenceMode::Batched);
            let mut left = TemporalRollout::new(1024, 1, space.clone(), 1024, 4 << 20).unwrap();
            let mut right = TemporalRollout::new(1024, 1, space, 1024, 4 << 20).unwrap();
            for tick in 0..24 {
                let active: Vec<_> = (0..lanes)
                    .map(|lane| !(tick + lane).is_multiple_of(7))
                    .collect();
                scalar
                    .tick_recorded(
                        &active,
                        ActMode::Sample,
                        &mut OnPolicyRecording::new(&mut left),
                    )
                    .unwrap();
                batched
                    .tick_recorded(
                        &active,
                        ActMode::Sample,
                        &mut OnPolicyRecording::new(&mut right),
                    )
                    .unwrap();
                for lane in 0..lanes {
                    assert_eq!(
                        scalar.lane_state(lane),
                        batched.lane_state(lane),
                        "lanes={lanes}, tick={tick}, lane={lane}"
                    );
                    assert_eq!(scalar.lane_episode(lane), batched.lane_episode(lane));
                    assert_eq!(scalar.lane_timestep(lane), batched.lane_timestep(lane));
                }
            }
            // Per-lane RNG is intentionally private operational state; portable
            // snapshots expose its exact counter/key representation for this audit.
            let scalar_wire = serde_json::to_value(&scalar).unwrap();
            let batched_wire = serde_json::to_value(&batched).unwrap();
            for lane in 0..lanes {
                assert!(!scalar_wire["agents"][lane]["rng"].is_null());
                assert_eq!(
                    scalar_wire["agents"][lane]["rng"],
                    batched_wire["agents"][lane]["rng"]
                );
                assert_eq!(
                    scalar_wire["agents"][lane]["previous"],
                    batched_wire["agents"][lane]["previous"]
                );
            }
            let config = SequenceConfig {
                burn_in: 1,
                learn_len: 3,
                n_step: 1,
            };
            let a = left
                .seal(
                    actor.clone(),
                    TemporalBatchKind::Bootstrapped,
                    0.97,
                    Some(0.9),
                    false,
                    &config,
                )
                .unwrap();
            let b = right
                .seal(
                    actor,
                    TemporalBatchKind::Bootstrapped,
                    0.97,
                    Some(0.9),
                    false,
                    &config,
                )
                .unwrap();
            assert!(a.rows == b.rows, "rich rows differ for {lanes} lanes");
            assert!(
                a.sequences == b.sequences,
                "actions or factual histories differ for {lanes} lanes"
            );
        }
    }
}
