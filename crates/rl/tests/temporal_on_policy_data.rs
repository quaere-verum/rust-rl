use rand_chacha::ChaCha8Rng;
use rl::{
    core::recording::RecordingKey,
    core::{
        action::{ActionArgument, ActionGrammar, ConditionalAction, Operation},
        checkpoint::Snapshot,
        env::{PublicHistory, ResumeContext},
        reward::Reward,
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        stateful_on_policy::{
            ConditionalOutcome, DistributionSnapshot, OnPolicySample, StatefulOnPolicyActor,
        },
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, FeatureIdentity, ObservationSchema,
            StructuredObservation,
        },
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
    data::{
        temporal::{EpisodeStore, SampleLimits, SequenceConfig},
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch, TemporalRollout},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Behavior {
    revision: u64,
    space: ActionSpace,
    observations: ObservationSpace,
}
impl StatefulActor for Behavior {
    type State = ();
    fn observation_space(&self) -> ObservationSpace {
        self.observations.clone()
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> Result<()> {
        Ok(())
    }
    fn validate_state(&self, _: &()) -> Result<()> {
        Ok(())
    }
    fn assimilate(&self, _: &(), _: HistoryInput<'_>) -> Result<()> {
        Ok(())
    }
    fn decide(
        &self,
        _: &(),
        _: &Observation,
        _: ActMode,
        _: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        Ok(sample(None).decision)
    }
}
impl StatefulOnPolicyActor for Behavior {
    fn sample_on_policy(
        &self,
        _: &(),
        _: &Observation,
        _: ActMode,
        _: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample> {
        Ok(sample(None))
    }
    fn state_value(&self, _: &(), _: &Observation) -> Result<Option<f32>> {
        Ok(None)
    }
}
impl Snapshot for Behavior {
    fn validate_snapshot(&self) -> Result<()> {
        self.space.validate()
    }
}

fn behavior(revision: u64) -> Behavior {
    Behavior {
        revision,
        space: ActionSpace::Discrete { count: 2 },
        observations: ObservationSpace::Dense(1),
    }
}
fn sample(value: Option<f32>) -> OnPolicySample {
    OnPolicySample {
        decision: StatefulDecision {
            action: Action::Discrete(0),
            log_prob: Some(-std::f32::consts::LN_2),
        },
        latent: None,
        distribution: DistributionSnapshot::Categorical {
            logits: vec![0., 0.],
            masks: vec![true, true],
        },
        value,
    }
}
fn sequence(learn_len: usize) -> SequenceConfig {
    SequenceConfig {
        burn_in: 1,
        learn_len,
        n_step: 1,
    }
}
fn rollout(capacity: usize, max_fresh: usize) -> TemporalRollout {
    TemporalRollout::new(
        capacity,
        1,
        ActionSpace::Discrete { count: 2 },
        max_fresh,
        1 << 20,
    )
    .unwrap()
}
fn transition(
    id: u64,
    t: usize,
    reward: f32,
    terminated: bool,
    truncated: bool,
    revision: u64,
) -> TemporalTransition {
    TemporalTransition {
        episode_id: id,
        timestep: t,
        observation: Observation::new(vec![t as f32]),
        action: Action::Discrete(0),
        reward: reward.into(),
        next_observation: Observation::new(vec![(t + 1) as f32]),
        terminated,
        truncated,
        behavior: BehaviorMetadata {
            revision,
            log_prob: Some(-std::f32::consts::LN_2),
        },
    }
}
fn push(
    out: &mut TemporalRollout,
    transition: TemporalTransition,
    value: Option<f32>,
    next: Option<f32>,
) {
    let key = RecordingKey {
        episode_id: transition.episode_id,
        timestep: transition.timestep,
    };
    out.stage_decision(key, transition.behavior.revision, &sample(value))
        .unwrap();
    out.stage_successors(&[key], &[next], &[transition.terminated])
        .unwrap();
    out.push_recorded(&transition).unwrap();
}
fn fixture() -> TemporalOnPolicyBatch<Behavior> {
    let mut out = rollout(20, 10);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    out.begin_episode(1, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(
        &mut out,
        transition(0, 0, 1., false, false, 7),
        Some(0.5),
        Some(0.6),
    );
    push(
        &mut out,
        transition(1, 0, 4., true, false, 7),
        Some(1.),
        Some(777.),
    );
    push(
        &mut out,
        transition(0, 1, 2., false, false, 7),
        Some(0.6),
        Some(0.7),
    );
    push(
        &mut out,
        transition(0, 2, 3., false, true, 7),
        Some(0.7),
        Some(9.),
    );
    out.seal(
        behavior(7),
        TemporalBatchKind::Bootstrapped,
        0.9,
        Some(0.8),
        false,
        &sequence(2),
    )
    .unwrap()
}
fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
}

#[test]
fn gae_uses_full_fresh_fragments_before_window_partitioning() {
    let batch = fixture();
    batch.validate().unwrap();
    assert_eq!(batch.len(), 4);
    assert_eq!(batch.sequences.samples.len(), 3);
    for (row, (advantage, target)) in
        batch
            .rows
            .iter()
            .zip([(7.89296, 8.39296), (9.518, 10.118), (10.4, 11.1), (3., 4.)])
    {
        close(row.advantage.unwrap(), advantage);
        close(row.return_.unwrap(), target);
    }
    assert_eq!(batch.rows[3].next_value, Some(0.));
    assert_eq!(
        batch.rows[2].key,
        RecordingKey {
            episode_id: 0,
            timestep: 2
        }
    );
    assert_eq!(
        batch.sequences.samples[1].loss_mask,
        vec![false, false, true]
    );
    assert_eq!(batch.tokens(), vec![(0, 0), (0, 1), (1, 2), (2, 0)]);
    assert_eq!(batch.actions(), vec![Action::Discrete(0); 4]);
    assert_eq!(batch.observations()[2].dense_values(), vec![2.]);
}

#[test]
fn extended_actions_discount_primitive_rewards_and_decay_gae_per_decision() {
    let mut out = rollout(20, 10);
    for id in [0, 1] {
        out.begin_episode(id, &EpisodeStart::fresh(Observation::new(vec![0.])))
            .unwrap();
    }
    let mut first = transition(0, 0, 0., false, false, 7);
    first.reward = Reward::from_primitives(vec![1., 4.]).unwrap();
    push(&mut out, first, Some(0.5), Some(2.));
    let mut truncated = transition(0, 1, 0., false, true, 7);
    truncated.reward = Reward::from_primitives(vec![2., 0., 8.]).unwrap();
    push(&mut out, truncated, Some(2.), Some(8.));
    let mut terminal = transition(1, 0, 0., true, false, 7);
    terminal.reward = Reward::from_primitives(vec![3., -2.]).unwrap();
    push(&mut out, terminal, Some(1.), Some(777.));

    let batch = out
        .seal(
            behavior(7),
            TemporalBatchKind::Bootstrapped,
            0.5,
            Some(0.5),
            false,
            &sequence(1),
        )
        .unwrap();
    assert_eq!(batch.advantages().unwrap(), [3.375, 3., 1.]);
    assert_eq!(batch.returns().unwrap(), [3.875, 5., 2.]);
    assert_eq!(batch.rows[1].discount_weight, 0.25);
    assert_eq!(batch.rows[2].next_value, Some(0.));

    let returns = out
        .seal(
            behavior(7),
            TemporalBatchKind::Bootstrapped,
            0.5,
            None,
            false,
            &sequence(1),
        )
        .unwrap();
    assert_eq!(returns.returns().unwrap(), [4.25, 5., 2.]);
}

#[test]
fn restored_episodic_suffix_excludes_context_loss_and_keeps_its_physical_clock() {
    let mut out = rollout(20, 10);
    let initial = Observation::new(vec![0.]);
    let observation = Observation::new(vec![1.]);
    let history = PublicHistory::new(initial)
        .append(
            Action::Discrete(0),
            Reward::from_primitives(vec![100., 0., 0.]).unwrap(),
            observation.clone(),
        )
        .unwrap();
    out.begin_episode(
        0,
        &EpisodeStart {
            observation,
            context: Some(ResumeContext {
                physical_timestep: history.physical_timestep(),
                history,
            }),
        },
    )
    .unwrap();
    let mut terminal = transition(0, 1, 0., true, false, 7);
    terminal.reward = Reward::from_primitives(vec![2., 4., 8.]).unwrap();
    push(&mut out, terminal, None, None);
    let batch = out
        .seal(
            behavior(7),
            TemporalBatchKind::Episodic,
            0.5,
            None,
            false,
            &sequence(1),
        )
        .unwrap();
    assert_eq!(batch.sequences.samples[0].loss_mask, [false, true]);
    assert_eq!(batch.returns().unwrap(), [6.]);
    assert_eq!(batch.rows[0].discount_weight, 0.125);
}

#[test]
fn conditional_recording_checks_semantic_support_and_joint_likelihood() {
    let mut behavior = behavior(11);
    behavior.observations = ObservationSpace::Structured(Arc::new(ObservationSchema {
        namespace: "conditional-fixture".into(),
        version: 1,
        spatial: vec![],
        entities: vec![FeatureIdentity::new("public_signal", "unit_interval")],
        events: vec![],
        scalars: vec![],
        vocabularies: vec![],
        conventions: vec![],
    }));
    behavior.space = ActionSpace::Conditional(Arc::new(ActionGrammar {
        namespace: "conditional-fixture".into(),
        version: 1,
        operations: vec![
            Operation {
                name: "wait".into(),
                argument: ArgumentKind::None,
            },
            Operation {
                name: "inspect".into(),
                argument: ArgumentKind::Entity,
            },
        ],
        candidate_features: vec![],
    }));
    let entity = |id| EntityId { kind: 1, id };
    let action = |id| ConditionalAction {
        operation: usize::from(id != 0),
        argument: (id != 0).then(|| ActionArgument::Entity(entity(id))),
    };
    let observation = Observation::Structured(Arc::new(StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        entities: [2, 1]
            .map(|id| EntityRow {
                id: entity(id),
                features: vec![0.],
            })
            .to_vec(),
        events: vec![],
        scalars: vec![],
        candidates: [2, 1, 0]
            .map(|operation| ActionCandidate {
                action: action(operation),
                features: vec![],
                enabled: operation != 2,
            })
            .to_vec(),
    }));
    let mut out = TemporalRollout::new(
        20,
        behavior.observation_space(),
        behavior.space.clone(),
        10,
        1 << 20,
    )
    .unwrap();
    out.begin_episode(0, &EpisodeStart::fresh(observation.clone()))
        .unwrap();
    let key = RecordingKey {
        episode_id: 0,
        timestep: 0,
    };
    let snapshot = |disabled| {
        DistributionSnapshot::conditional(vec![
            ConditionalOutcome {
                action: action(disabled),
                log_prob: None,
            },
            ConditionalOutcome {
                action: action(1),
                log_prob: Some(0.25_f32.ln()),
            },
            ConditionalOutcome {
                action: action(0),
                log_prob: Some(0.75_f32.ln()),
            },
        ])
        .unwrap()
    };
    let log_prob = 0.25_f32.ln();
    let selected = Action::Conditional(Arc::new(action(1)));
    let mut sample = OnPolicySample {
        decision: StatefulDecision {
            action: selected.clone(),
            log_prob: Some(log_prob),
        },
        latent: None,
        distribution: snapshot(3),
        value: None,
    };
    assert!(out.stage_decision(key, 11, &sample).is_err());
    sample.distribution = snapshot(2);
    sample.decision.log_prob = Some(0.75_f32.ln());
    assert!(out.stage_decision(key, 11, &sample).is_err());
    sample.decision.log_prob = Some(log_prob);
    out.stage_decision(key, 11, &sample).unwrap();
    out.stage_successors(&[key], &[None], &[true]).unwrap();
    out.push_recorded(&TemporalTransition {
        episode_id: 0,
        timestep: 0,
        observation: observation.clone(),
        action: selected,
        reward: 1.0.into(),
        next_observation: observation,
        terminated: true,
        truncated: false,
        behavior: BehaviorMetadata {
            revision: 11,
            log_prob: Some(log_prob),
        },
    })
    .unwrap();
    let batch = out
        .seal(
            behavior,
            TemporalBatchKind::Episodic,
            1.,
            None,
            false,
            &sequence(1),
        )
        .unwrap();
    assert_eq!(batch.rows[0].distribution, sample.distribution);
}

#[test]
fn repeated_episode_prefixes_reject_contradictory_context_facts() {
    let original = fixture();
    assert_eq!(
        original.sequences.samples[0].episode_id,
        original.sequences.samples[1].episode_id
    );
    assert!(!original.sequences.samples[1].loss_mask[0]);
    for corruption in 0..7 {
        let mut invalid = original.clone();
        let copy = &mut invalid.sequences.samples[1];
        match corruption {
            0 => copy.observations[0].as_dense_mut().unwrap().values[0] = 0.125,
            1 => copy.observations[0].as_dense_mut().unwrap().values[0] = -0.0,
            2 => copy.observations[0].as_dense_mut().unwrap().action_mask = Some(vec![true, true]),
            3 => copy.steps[0].action = Action::Discrete(1),
            4 => copy.steps[0].reward = 1.5_f32.into(),
            5 => copy.steps[0].behavior.revision = 6,
            _ => copy.steps[0].behavior.log_prob = Some(-3.),
        }
        invalid.sequences.validate().unwrap();
        assert!(
            invalid.validate().is_err(),
            "contradictory factual prefix variant {corruption} was accepted"
        );
    }
}

#[test]
fn overlapping_prefix_end_flags_and_optional_likelihood_must_agree() {
    let original = fixture();
    for terminated in [false, true] {
        let mut invalid = original.clone();
        let final_step = invalid.sequences.samples[0].steps.last_mut().unwrap();
        final_step.terminated = terminated;
        final_step.truncated = !terminated;
        if terminated {
            invalid.rows[1].next_value = Some(0.);
        }
        invalid.sequences.validate().unwrap();
        assert!(
            invalid.validate().is_err(),
            "one prefix cannot end an episode that another continues"
        );
    }
    let mut invalid = original;
    invalid.sequences.samples[1].steps[0].behavior.log_prob = None;
    invalid.sequences.validate().unwrap();
    assert!(
        invalid.validate().is_err(),
        "missing likelihood in only one context copy was accepted"
    );
}

#[test]
fn repeated_continuous_actions_and_rewards_compare_float_bits() {
    let mut original = fixture();
    original.behavior.space = ActionSpace::Unbounded { dimensions: 1 };
    let log_prob = -0.5 * std::f32::consts::TAU.ln();
    for sample in &mut original.sequences.samples {
        for step in &mut sample.steps {
            step.action = Action::Continuous(vec![0.]);
            step.behavior.log_prob = Some(log_prob);
        }
        sample.steps[0].reward = 0.0_f32.into();
    }
    for row in &mut original.rows {
        row.log_prob = log_prob;
        row.distribution = DistributionSnapshot::Gaussian {
            mean: vec![0.],
            log_std: vec![0.],
            bounds: None,
        };
    }
    original.validate().unwrap();
    let mut action = original.clone();
    action.sequences.samples[1].steps[0].action = Action::Continuous(vec![-0.]);
    assert!(
        action.validate().is_err(),
        "action signed-zero mismatch was accepted"
    );
    let mut reward = original;
    reward.sequences.samples[1].steps[0].reward = (-0.0_f32).into();
    assert!(
        reward.validate().is_err(),
        "reward signed-zero mismatch was accepted"
    );
}

#[test]
fn consistent_reordered_prefixes_allow_window_masks_and_custom_finite_targets() {
    let mut batch = fixture();
    batch.rows[0].advantage = Some(-31.);
    batch.rows[0].return_ = Some(72.);
    batch.validate().unwrap();
    let reordered = batch.select_windows(&[1, 2, 0]).unwrap();
    reordered.validate_snapshot().unwrap();
    assert_eq!(reordered.rows[0].advantage, Some(-31.));
    assert_eq!(reordered.rows[0].return_, Some(72.));
    assert_ne!(
        reordered.sequences.samples[0].loss_mask,
        reordered.sequences.samples[2].loss_mask
    );
}

#[test]
fn fragment_cuts_stop_gae_and_acknowledgement_keeps_exact_active_prefix() {
    let mut out = rollout(20, 10);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(
        &mut out,
        transition(0, 0, 1., false, false, 1),
        Some(0.5),
        Some(0.6),
    );
    push(
        &mut out,
        transition(0, 1, 2., false, false, 1),
        Some(0.6),
        Some(0.7),
    );
    let first = out
        .seal(
            behavior(1),
            TemporalBatchKind::Bootstrapped,
            0.9,
            Some(0.5),
            false,
            &sequence(1),
        )
        .unwrap();
    close(first.rows[0].advantage.unwrap(), 1.9535);
    close(first.rows[1].advantage.unwrap(), 2.03);
    let immutable = serde_json::to_value(&first).unwrap();
    out.acknowledge().unwrap();
    assert_eq!(out.store().episode(0).unwrap().steps.len(), 2);
    push(
        &mut out,
        transition(0, 2, 3., false, false, 2),
        Some(0.7),
        Some(0.8),
    );
    push(
        &mut out,
        transition(0, 3, 4., false, false, 2),
        Some(0.8),
        Some(0.9),
    );
    let next = out
        .seal(
            behavior(2),
            TemporalBatchKind::Bootstrapped,
            0.9,
            Some(0.5),
            false,
            &sequence(1),
        )
        .unwrap();
    assert_eq!(
        next.rows.iter().map(|r| r.key.timestep).collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(
        next.sequences.samples[0].loss_mask,
        vec![false, false, true]
    );
    assert_eq!(
        next.sequences.samples[1].loss_mask,
        vec![false, false, false, true]
    );
    close(next.rows[0].discount_weight, 0.81);
    close(next.rows[1].discount_weight, 0.729);
    assert_eq!(immutable, serde_json::to_value(&first).unwrap());
}

#[test]
fn window_minibatches_cover_each_fresh_key_once_and_reindex_tokens() {
    let batch = fixture();
    let mut seen = BTreeSet::new();
    for index in [2, 0, 1] {
        let mini = batch.select_windows(&[index]).unwrap();
        mini.validate().unwrap();
        assert!(mini.rows.iter().all(|row| row.sample == 0));
        for row in mini.rows {
            assert!(seen.insert(row.key));
        }
    }
    assert_eq!(seen, batch.rows.iter().map(|row| row.key).collect());
    assert!(batch.select_windows(&[0, 0]).is_err());
    assert!(batch.select_windows(&[99]).is_err());
    let permuted = batch.select_windows(&[2, 0]).unwrap();
    assert_eq!(permuted.rows[0].sample, 1);
    assert_eq!(permuted.rows.last().unwrap().sample, 0);
}

#[test]
fn monte_carlo_requires_complete_terminal_episodes_and_preserves_episode_reduction() {
    let mut out = rollout(20, 10);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(&mut out, transition(0, 0, 1., false, false, 3), None, None);
    assert!(
        out.seal(
            behavior(3),
            TemporalBatchKind::Episodic,
            0.5,
            None,
            false,
            &sequence(1)
        )
        .is_err()
    );
    push(&mut out, transition(0, 1, 2., true, false, 3), None, None);
    out.begin_episode(1, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(&mut out, transition(1, 0, 5., true, false, 3), None, None);
    let batch = out
        .seal(
            behavior(3),
            TemporalBatchKind::Episodic,
            0.5,
            None,
            false,
            &sequence(1),
        )
        .unwrap();
    assert_eq!(batch.episode_count, 2);
    assert_eq!(batch.returns().unwrap(), vec![2., 2., 5.]);
    assert_eq!(batch.advantages().unwrap(), vec![2., 2., 5.]);
    assert_eq!(batch.select_windows(&[0]).unwrap().episode_count, 2);
    out.acknowledge().unwrap();
    assert!(out.store().is_empty());
    let mut cut = rollout(20, 10);
    cut.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(&mut cut, transition(0, 0, 1., false, false, 3), None, None);
    cut.acknowledge().unwrap();
    push(&mut cut, transition(0, 1, 2., true, false, 3), None, None);
    assert!(
        cut.seal(
            behavior(3),
            TemporalBatchKind::Episodic,
            0.5,
            None,
            false,
            &sequence(2)
        )
        .is_err()
    );
    let mut truncated = rollout(5, 5);
    truncated
        .begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(
        &mut truncated,
        transition(0, 0, 1., false, true, 3),
        None,
        None,
    );
    assert!(
        truncated
            .seal(
                behavior(3),
                TemporalBatchKind::Episodic,
                0.5,
                None,
                false,
                &sequence(2)
            )
            .is_err()
    );
}

#[test]
fn metadata_staging_and_capacity_failures_are_atomic() {
    let mut out = rollout(1, 4);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    assert!(
        out.push_recorded(&transition(0, 0, 1., false, false, 1))
            .is_err()
    );
    push(
        &mut out,
        transition(0, 0, 1., true, false, 1),
        Some(1.),
        Some(0.),
    );
    out.begin_episode(1, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    let key = RecordingKey {
        episode_id: 1,
        timestep: 0,
    };
    out.stage_decision(key, 1, &sample(Some(1.))).unwrap();
    let before = serde_json::to_value(&out).unwrap();
    assert!(out.stage_decision(key, 1, &sample(Some(1.))).is_err());
    assert!(out.stage_successors(&[key], &[None], &[false]).is_err());
    assert_eq!(before, serde_json::to_value(&out).unwrap());
    out.stage_successors(&[key], &[Some(2.)], &[false]).unwrap();
    let before = serde_json::to_value(&out).unwrap();
    assert!(
        out.push_recorded(&transition(1, 0, 1., false, false, 1))
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&out).unwrap());
    assert!(out.store().episode(0).is_some());
    assert!(out.acknowledge().is_err());
    out.abandon_episode(1).unwrap();
    out.acknowledge().unwrap();
    assert!(out.store().is_empty());
    out.validate_snapshot().unwrap();

    let mut tiny = TemporalRollout::new(5, 1, ActionSpace::Discrete { count: 2 }, 4, 1).unwrap();
    tiny.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    let before = serde_json::to_value(&tiny).unwrap();
    assert!(
        tiny.stage_decision(
            RecordingKey {
                episode_id: 0,
                timestep: 0
            },
            1,
            &sample(None)
        )
        .is_err()
    );
    assert_eq!(before, serde_json::to_value(&tiny).unwrap());
    let mut one = rollout(5, 1);
    one.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(&mut one, transition(0, 0, 1., false, false, 1), None, None);
    let before = serde_json::to_value(&one).unwrap();
    assert!(
        one.stage_decision(
            RecordingKey {
                episode_id: 0,
                timestep: 1
            },
            1,
            &sample(None)
        )
        .is_err()
    );
    assert_eq!(before, serde_json::to_value(&one).unwrap());
}

#[test]
fn mixed_revision_and_invalid_continuous_sampling_metadata_are_rejected() {
    let mut out = rollout(8, 8);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(
        &mut out,
        transition(0, 0, 1., false, false, 1),
        Some(0.),
        Some(0.),
    );
    push(
        &mut out,
        transition(0, 1, 1., false, false, 2),
        Some(0.),
        Some(0.),
    );
    assert!(
        out.seal(
            behavior(2),
            TemporalBatchKind::Bootstrapped,
            0.9,
            Some(0.9),
            false,
            &sequence(2)
        )
        .is_err()
    );
    let mut continuous = TemporalRollout::new(
        5,
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        5,
        4096,
    )
    .unwrap();
    continuous
        .begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    let key = RecordingKey {
        episode_id: 0,
        timestep: 0,
    };
    let sample = OnPolicySample {
        decision: StatefulDecision {
            action: Action::Continuous(vec![1.]),
            log_prob: Some(
                20. - 0.5 * std::f32::consts::TAU.ln()
                    - 2. * (std::f32::consts::LN_2 - 40. - (-80f32).exp().ln_1p()),
            ),
        },
        latent: None,
        distribution: DistributionSnapshot::Gaussian {
            mean: vec![40.],
            log_std: vec![-20.],
            bounds: Some((vec![-1.], vec![1.])),
        },
        value: None,
    };
    let before = serde_json::to_value(&continuous).unwrap();
    assert!(continuous.stage_decision(key, 1, &sample).is_err());
    assert_eq!(before, serde_json::to_value(&continuous).unwrap());
    let mut correct = sample.clone();
    correct.latent = Some(vec![40.]);
    continuous.stage_decision(key, 1, &correct).unwrap();
    let mut invalid = sample;
    invalid.latent = Some(vec![40.]);
    if let DistributionSnapshot::Gaussian { bounds, .. } = &mut invalid.distribution {
        *bounds = Some((vec![-2.], vec![2.]));
    }
    let mut other = TemporalRollout::new(
        5,
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        5,
        4096,
    )
    .unwrap();
    other
        .begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    assert!(other.stage_decision(key, 1, &invalid).is_err());
}

#[test]
fn snapshot_audits_store_sidecar_and_batch_learning_token_provenance() {
    let mut out = rollout(10, 10);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    push(
        &mut out,
        transition(0, 0, 1., false, false, 1),
        Some(0.),
        Some(0.),
    );
    let serialized = serde_json::to_value(&out).unwrap();
    let mut ordinary = serialized.clone();
    assert!(ordinary["store"].get("episode_behaviors").is_none());
    ordinary["store"]["episode_behaviors"] = serde_json::json!({});
    let restored: TemporalRollout = serde_json::from_value(ordinary.clone()).unwrap();
    let mut bound = ordinary;
    bound["store"]["episode_behaviors"]["0"] = serde_json::json!({"revision": 1});
    assert!(serde_json::from_value::<TemporalRollout>(bound).is_err());
    restored.validate_snapshot().unwrap();
    let batch = restored
        .seal(
            behavior(1),
            TemporalBatchKind::Bootstrapped,
            0.9,
            Some(0.95),
            false,
            &sequence(2),
        )
        .unwrap();
    let snapshot = serde_json::to_value(&batch).unwrap();
    for corruption in 0..5 {
        let mut invalid = snapshot.clone();
        match corruption {
            0 => invalid["revision"] = 2.into(),
            1 => invalid["rows"][0]["key"]["timestep"] = 99.into(),
            2 => invalid["rows"][0]["log_prob"] = (-8.).into(),
            3 => invalid["rows"][0]["time"] = 1.into(),
            _ => invalid["sequences"]["samples"][0]["loss_mask"][0] = false.into(),
        }
        let loaded: TemporalOnPolicyBatch<Behavior> = serde_json::from_value(invalid).unwrap();
        assert!(loaded.validate_snapshot().is_err());
    }
    let mut missing = serialized;
    missing["fresh"] = serde_json::json!([]);
    let loaded: TemporalRollout = serde_json::from_value(missing).unwrap();
    assert!(loaded.validate_snapshot().is_err());
}

#[test]
fn explicit_completed_retirement_validates_all_ids_before_mutation() {
    let mut store = EpisodeStore::new(8, 1, ActionSpace::Discrete { count: 2 }).unwrap();
    store
        .begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    store.push(&transition(0, 0, 1., true, false, 1)).unwrap();
    store
        .begin_episode(1, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    store.push(&transition(1, 0, 1., false, false, 1)).unwrap();
    let before = serde_json::to_value(&store).unwrap();
    assert!(store.retire_completed(&[0, 1]).is_err());
    assert!(store.retire_completed(&[0, 0]).is_err());
    assert_eq!(before, serde_json::to_value(&store).unwrap());
    store.retire_completed(&[0]).unwrap();
    assert!(store.episode(0).is_none());
    assert_eq!(store.episode(1).unwrap().steps.len(), 1);
    assert_eq!(store.len(), 1);
    assert_eq!(store.highest_episode_id(), Some(1));
    store.validate_snapshot().unwrap();
}

#[test]
fn sealing_preflights_padded_tokens_and_copied_prefix_payload() {
    let mut out = rollout(32, 16);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    for t in 0..8 {
        push(
            &mut out,
            transition(0, t, 1., false, false, 1),
            Some(0.),
            Some(0.),
        );
    }
    let allowed = SampleLimits {
        max_sample_tokens: 32,
        max_batch_tokens: 100,
        max_sample_payload_bytes: 1 << 20,
        max_batch_payload_bytes: 1 << 20,
    };
    let before = serde_json::to_value(&out).unwrap();
    for limits in [
        SampleLimits {
            max_sample_tokens: 5,
            ..allowed
        },
        SampleLimits {
            max_batch_tokens: 8,
            ..allowed
        },
        SampleLimits {
            max_sample_payload_bytes: 1,
            ..allowed
        },
        SampleLimits {
            max_batch_payload_bytes: 1,
            ..allowed
        },
    ] {
        assert!(
            out.seal_with_limits(
                behavior(1),
                TemporalBatchKind::Bootstrapped,
                0.9,
                Some(0.95),
                false,
                &sequence(1),
                &limits
            )
            .is_err()
        );
        assert_eq!(before, serde_json::to_value(&out).unwrap());
    }
    let batch = out
        .seal_with_limits(
            behavior(1),
            TemporalBatchKind::Bootstrapped,
            0.9,
            Some(0.95),
            false,
            &sequence(1),
            &allowed,
        )
        .unwrap();
    assert_eq!(batch.len(), 8);
    assert_eq!(batch.sequences.samples.len(), 8);
}

#[test]
fn fresh_recording_audit_borrows_only_committed_rows_until_acknowledgement() {
    let mut out = rollout(8, 8);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))
        .unwrap();
    let key = RecordingKey {
        episode_id: 0,
        timestep: 0,
    };
    let decision = sample(Some(0.5));
    out.stage_decision(key, 3, &decision).unwrap();
    assert_eq!(out.fresh_recordings().count(), 0);
    out.stage_successors(&[key], &[Some(0.8)], &[false])
        .unwrap();
    assert_eq!(out.fresh_recordings().count(), 0);
    out.push_recorded(&transition(0, 0, 1., false, false, 3))
        .unwrap();
    let recorded = out.fresh_recordings().next().unwrap();
    assert_eq!(recorded.key, key);
    assert_eq!(recorded.revision, 3);
    assert_eq!(recorded.action, decision.decision.action);
    assert_eq!(
        recorded.log_prob.to_bits(),
        decision.decision.log_prob.unwrap().to_bits()
    );
    assert_eq!(recorded.latent, decision.latent);
    assert_eq!(recorded.distribution, decision.distribution);
    assert_eq!(recorded.value, Some(0.5));
    assert_eq!(recorded.next_value, Some(0.8));
    out.acknowledge().unwrap();
    assert_eq!(out.fresh_recordings().count(), 0);
    assert_eq!(out.store().episode(0).unwrap().steps.len(), 1);
}

#[test]
fn owned_rollout_restores_through_its_historical_run_identifier() -> rl::Result<()> {
    use rl::{
        algorithms::tabular::QTable,
        train::checkpoint::{load_run, load_run_portable, save_run},
    };

    let mut out = rollout(8, 4);
    out.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![0.])))?;
    push(
        &mut out,
        transition(0, 0, 1., false, false, 1),
        Some(0.),
        Some(0.),
    );
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("owned-rollout");
    let mut learner = QTable::<u8, u8>::new(0.)?;
    save_run(&path, &learner, &out)?;
    let envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("checkpoint.json"))?)?;
    assert_eq!(
        envelope["snapshot_type"],
        "rl::data::temporal_on_policy::TemporalRollout<rl::data::temporal::EpisodeStore>"
    );
    assert_eq!(envelope["snapshot"]["version"], 1);
    let native: TemporalRollout = load_run(&path, &mut learner)?;
    let (portable, _): (TemporalRollout, _) = load_run_portable(&path, &mut learner)?;
    let expected = serde_json::to_value(out)?;
    assert_eq!(serde_json::to_value(native)?, expected);
    assert_eq!(serde_json::to_value(portable)?, expected);
    Ok(())
}
