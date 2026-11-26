#![cfg(feature = "neural")]

#[path = "support/resume_fixture.rs"]
mod resume_fixture;

use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::{Tensor, TensorData},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use resume_fixture::ResumeFixture;
use rl::{
    algorithms::{recurrent_reppo::RecurrentReppo, reppo::ReppoConfig},
    core::{
        checkpoint::{Checkpoint, Snapshot},
        stateful::{BehaviorPublisher, HistoryInput, HistorySpec, PreviousInput, StatefulLearner},
    },
    data::{temporal::SequenceConfig, temporal_on_policy::TemporalOnPolicyBatch},
    envs::{delayed_bit::DelayedBit, hidden_target::HiddenTarget},
    nn::{
        temporal::TemporalEncoder, temporal_pathwise_actor::TemporalPathwiseActor,
        temporal_policy::TemporalPolicy, temporal_q::TemporalQ, to_vec,
    },
    objectives::hl_gauss::HlGaussConfig,
    prelude::*,
    train::{
        checkpoint::{load_run, save_run},
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};

type Run<E> = TemporalOnPolicyTrainer<SyncVectorEnv<E>, TemporalPathwiseActor>;
fn config(aux: f32) -> ReppoConfig {
    ReppoConfig {
        epochs: 2,
        minibatch_size: 32,
        auxiliary_coefficient: aux,
        hl_gauss: HlGaussConfig {
            min: -5.,
            max: 5.,
            bins: 11,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn learner(space: ActionSpace, seed: u64, aux: f32) -> RecurrentReppo {
    let spec = HistorySpec::new(5, space).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    RecurrentReppo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[8], &mut rng).unwrap(),
        TemporalQ::new(spec, Some(5), &[8], 11, &mut rng).unwrap(),
        config(aux),
        seed ^ 0xabc,
    )
    .unwrap()
}
fn run<E: Env>(l: &RecurrentReppo, env: Vec<E>) -> Run<E> {
    TemporalOnPolicyTrainer::new(
        SyncVectorEnv::new(env).unwrap(),
        l,
        TemporalOnPolicyConfig {
            collection: OnPolicyConfig {
                rollout_steps: 4,
                gamma: 0.9,
                gae_lambda: None,
                normalize_advantages: false,
                ..Default::default()
            },
            sequence: SequenceConfig {
                burn_in: 0,
                learn_len: 4,
                n_step: 1,
            },
            ..Default::default()
        },
        71,
    )
    .unwrap()
}

#[test]
fn checkpointed_reppo_updates_from_a_reconstructed_fresh_suffix() {
    let mut model = learner(ActionSpace::Discrete { count: 2 }, 31, 0.1);
    model.set_update_mode(TemporalExecutionMode::OptimizedExact);
    let env = ResumeFixture::new(DelayedBit::new(3, None, false, 11).unwrap(), Some(3));
    let mut trainer = run(&model, vec![env]);
    for _ in 0..4 {
        trainer.collect_tick(&model).unwrap();
    }
    trainer.update(&mut model).unwrap();
    trainer.collect_tick(&model).unwrap();
    let batch = trainer.seal(&model).unwrap();
    assert_eq!(batch.rows.len(), 1);
    assert_eq!(batch.rows[0].key.timestep, 3);
    assert_eq!(
        batch.sequences.samples[0].loss_mask,
        [false, false, false, true]
    );
    assert!(trainer.update(&mut model).unwrap().updates > 0);
    trainer.validate_snapshot().unwrap();
}

#[test]
fn reppo_publishes_first_revision_before_an_unbegun_resumed_lane() {
    let mut env = ResumeFixture::new(DelayedBit::new(2, None, false, 11).unwrap(), Some(2));
    env.reset(Some(11)).unwrap();
    for _ in 0..2 {
        env.step(&Action::Discrete(0)).unwrap();
    }
    let mut model = learner(ActionSpace::Discrete { count: 2 }, 31, 0.1);
    env.resume_on_seed = true;
    let trainer = run(&model, vec![env]);
    trainer.validate_snapshot().unwrap();
    let serialized = serde_json::to_vec(&trainer).unwrap();
    let mut trainer: Run<ResumeFixture<DelayedBit>> = serde_json::from_slice(&serialized).unwrap();
    trainer.validate_snapshot().unwrap();
    assert_eq!(trainer.collector().lane_timestep(0), Some(2));

    trainer.collect_tick(&model).unwrap();
    let batch = trainer.seal(&model).unwrap();
    assert_eq!(batch.rows.len(), 1);
    assert_eq!(batch.rows[0].key.timestep, 2);
    assert_eq!(batch.sequences.samples[0].loss_mask, [false, false, true]);
    assert!(trainer.update(&mut model).unwrap().updates > 0);
    trainer.validate_snapshot().unwrap();
}

fn weights(m: &impl Module) -> Vec<f32> {
    struct W(Vec<f32>);
    impl ModuleVisitor for W {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.0.extend(to_vec(p.val()).unwrap());
        }
    }
    let mut w = W(vec![]);
    m.visit(&mut w);
    w.0
}

fn canonical_record(record: &rl::core::checkpoint::LearnerRecord) -> serde_json::Value {
    use std::collections::BTreeMap;
    fn pack(bytes: Vec<u8>) -> serde_json::Value {
        let reader =
            burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes)).unwrap();
        let scalars = reader
            .scalars()
            .iter()
            .map(|(key, value)| (key.clone(), format!("{value:?}")))
            .collect::<BTreeMap<_, _>>();
        let metadata = reader
            .metadata()
            .iter()
            .map(|(key, value)| (key.clone(), format!("{value:?}")))
            .collect::<BTreeMap<_, _>>();
        let tensors=reader.into_tensors().unwrap().into_iter().map(|tensor| {
            let (name,dtype,shape,id,bytes)=tensor.into_parts().unwrap();
            (name,serde_json::json!({"dtype":format!("{dtype:?}"),"shape":shape,"id":id,"bytes":bytes}))
        }).collect::<BTreeMap<_,_>>();
        serde_json::json!({"scalars":scalars,"metadata":metadata,"tensors":tensors})
    }
    let mut value = serde_json::to_value(record).unwrap();
    for model in value["models"].as_object_mut().unwrap().values_mut() {
        model["bytes"] = pack(serde_json::from_value(model["bytes"].clone()).unwrap());
    }
    for optimizer in value["optimizers"].as_object_mut().unwrap().values_mut() {
        *optimizer = pack(serde_json::from_value(optimizer.clone()).unwrap());
    }
    value
}

fn assert_complete_learner_state(left: &RecurrentReppo, right: &RecurrentReppo) {
    let left = left.checkpoint().unwrap();
    let right = right.checkpoint().unwrap();
    for record in [&left, &right] {
        assert_eq!(
            record
                .optimizers
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["critic", "entropy_dual", "kl_dual", "policy"]
        );
    }
    assert_eq!(
        left.metadata["rng"], right.metadata["rng"],
        "learner random stream differs"
    );
    assert!(
        canonical_record(&left) == canonical_record(&right),
        "canonical REPPO checkpoint differs (models, four Adam owners, counters or RNG)"
    );
}
fn scalar_successor_q(
    batch: &TemporalOnPolicyBatch<TemporalPathwiseActor>,
    row: usize,
    successor: bool,
    support: &HlGaussConfig,
) -> f32 {
    let r = &batch.rows[row];
    let sample = &batch.sequences.samples[r.sample];
    let q = batch.behavior.critic();
    let mut state = q.encoder.initial_state(1).unwrap();
    let last = r.time + usize::from(successor);
    for t in 0..=last {
        let input = q
            .spec()
            .encode(HistoryInput {
                observation: &sample.observations[t],
                previous: t.checked_sub(1).map(|p| PreviousInput {
                    factual: None,
                    action: &sample.steps[p].action,
                    reward: sample.steps[p].reward.total(),
                }),
            })
            .unwrap();
        state = q
            .encoder
            .step(
                Tensor::from_data(
                    TensorData::new(input, [1, q.spec().input_dim()]),
                    &q.device(),
                ),
                state,
            )
            .unwrap()
            .1;
    }
    // This fixture is before the decision: only action0 is legal, entropy=0.
    to_vec(
        support
            .decode(
                q.values_from_features(state, Tensor::from_data([[1., 0.]], &q.device()))
                    .unwrap(),
            )
            .unwrap(),
    )
    .unwrap()[0]
}

#[test]
fn fixed_soft_targets_use_successor_critic_history_and_stop_at_fragment_tail() {
    let l = learner(ActionSpace::Discrete { count: 2 }, 11, 0.);
    let mut t = run(&l, vec![DelayedBit::new(9, None, false, 47).unwrap()]);
    for _ in 0..4 {
        t.collect_tick(&l).unwrap();
    }
    let batch = t.seal(&l).unwrap().clone();
    let before = l.checkpoint().unwrap();
    let targets = l.target_preview(&batch).unwrap();
    let bootstrap: Vec<_> = (0..batch.len())
        .map(|i| scalar_successor_q(&batch, i, true, &l.config().hl_gauss))
        .collect();
    let mut expected = vec![0.; batch.len()];
    let mut following = None;
    for i in (0..batch.len()).rev() {
        let continuation = following.map_or(bootstrap[i] as f64, |n| {
            (1. - l.config().lambda as f64) * bootstrap[i] as f64 + l.config().lambda as f64 * n
        });
        let value = batch.gamma as f64 * continuation;
        expected[i] = value as f32;
        following = Some(value);
    }
    for (a, b) in targets.iter().zip(expected) {
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
    let wrong = scalar_successor_q(&batch, 3, false, &l.config().hl_gauss) * batch.gamma;
    assert!(
        (targets[3] - wrong).abs() > 1e-4,
        "fixture failed to distinguish shifted Q memory"
    );
    assert!(
        serde_json::to_value(before).unwrap()
            == serde_json::to_value(l.checkpoint().unwrap()).unwrap(),
        "target preview consumed state"
    );
}

#[test]
fn both_recurrent_encoders_train_with_exact_masks_and_continuous_pathwise_actions() {
    let mut discrete = learner(ActionSpace::Discrete { count: 2 }, 23, 0.25);
    let mut dt = run(
        &discrete,
        vec![DelayedBit::new(2, Some(1), false, 47).unwrap()],
    );
    let mut continuous = learner(
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.],
        },
        23,
        0.25,
    );
    let mut ct = run(
        &continuous,
        vec![HiddenTarget::new(2, Some(1), false, 47).unwrap()],
    );
    for (l, p0, q0, report) in [
        {
            let p = weights(&discrete.policy().encoder);
            let q = weights(&discrete.critic().encoder);
            let r = dt.cycle(&mut discrete).unwrap();
            (&discrete, p, q, r)
        },
        {
            let p = weights(&continuous.policy().encoder);
            let q = weights(&continuous.critic().encoder);
            let r = ct.cycle(&mut continuous).unwrap();
            (&continuous, p, q, r)
        },
    ] {
        assert!(report.updates > 0 && report.metrics.values().all(|x| x.is_finite()));
        assert_ne!(p0, weights(&l.policy().encoder));
        assert_ne!(q0, weights(&l.critic().encoder));
        assert!(report.metrics["auxiliary_loss"] > 0.);
    }
    dt.validate_snapshot().unwrap();
    ct.validate_snapshot().unwrap();
}

#[test]
fn collection_and_update_revisions_reject_stale_and_foreign_batches_atomically() {
    let mut l = learner(ActionSpace::Discrete { count: 2 }, 47, 0.);
    let mut t = run(&l, vec![DelayedBit::new(9, None, false, 47).unwrap()]);
    assert_eq!(l.revision(), 0);
    t.collect_tick(&l).unwrap();
    assert_eq!(l.revision(), 1);
    let input = t.seal(&l).unwrap().clone();
    let old = serde_json::to_value(&input.behavior).unwrap();
    let mut bad = input.clone();
    bad.sequences.samples[0].steps[0].action = Action::Discrete(1);
    let before = serde_json::to_value(l.checkpoint().unwrap()).unwrap();
    assert!(l.update(&bad).is_err());
    assert!(serde_json::to_value(l.checkpoint().unwrap()).unwrap() == before);
    t.update(&mut l).unwrap();
    assert_eq!(l.revision(), 2);
    let before = serde_json::to_value(l.checkpoint().unwrap()).unwrap();
    assert!(l.update(&input).is_err());
    assert!(serde_json::to_value(l.checkpoint().unwrap()).unwrap() == before);
    assert!(
        serde_json::to_value(&input.behavior).unwrap() == old,
        "frozen behavior changed"
    );
    t.collect_tick(&l).unwrap();
    assert_eq!(l.revision(), 3);
    assert_eq!(t.collector().lane_timestep(0), Some(2));
    assert_eq!(t.collector().lane_episode(0), Some(0));
    let foreign = learner(ActionSpace::Discrete { count: 2 }, 99, 0.);
    let mut forged = t.seal(&l).unwrap().clone();
    forged.behavior =
        TemporalPathwiseActor::new(l.policy(), foreign.critic(), l.revision()).unwrap();
    let before = serde_json::to_value(l.checkpoint().unwrap()).unwrap();
    assert!(l.update(&forged).is_err());
    assert!(serde_json::to_value(l.checkpoint().unwrap()).unwrap() == before);
    l.behavior_actor(&(), l.revision()).unwrap();
}

#[test]
fn complete_run_restores_partial_and_sealed_pathwise_rollouts_exactly() {
    fn continuation<E: Env + Snapshot>(space: ActionSpace, environments: Vec<E>, sealed: bool) {
        let mut l = learner(space.clone(), 11, 0.1);
        let mut t = run(&l, environments);
        for _ in 0..3 {
            t.collect_tick(&l).unwrap();
        }
        if sealed {
            t.seal(&l).unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run");
        save_run(&path, &l, &t).unwrap();
        let mut restored = learner(space, 99, 0.);
        let mut rt: Run<E> = load_run(&path, &mut restored).unwrap();
        assert_complete_learner_state(&l, &restored);
        for _ in 0..4 {
            let a = t.update(&mut l).unwrap();
            let b = rt.update(&mut restored).unwrap();
            assert_eq!(a.updates, b.updates);
            assert_eq!(a.metrics, b.metrics);
            assert_eq!(weights(&l.policy()), weights(&restored.policy()));
            assert_eq!(weights(&l.critic()), weights(&restored.critic()));
            assert_eq!(l.multipliers(), restored.multipliers());
            assert_eq!(t.counters(), rt.counters());
            assert_complete_learner_state(&l, &restored);
            for lane in 0..2 {
                assert_eq!(
                    t.collector().lane_state(lane),
                    rt.collector().lane_state(lane)
                );
            }
            for _ in 0..3 {
                t.collect_tick(&l).unwrap();
                rt.collect_tick(&restored).unwrap();
            }
            let a = t.seal(&l).unwrap();
            let b = rt.seal(&restored).unwrap();
            assert_eq!(a.rows, b.rows);
            assert_eq!(a.sequences, b.sequences);
            assert_complete_learner_state(&l, &restored);
        }
    }
    for sealed in [false, true] {
        continuation(
            ActionSpace::Discrete { count: 2 },
            vec![
                DelayedBit::new(5, Some(2), false, 47).unwrap(),
                DelayedBit::new(9, Some(4), false, 71).unwrap(),
            ],
            sealed,
        );
        continuation(
            ActionSpace::Box {
                low: vec![-2.],
                high: vec![2.],
            },
            vec![
                HiddenTarget::new(5, Some(2), false, 47).unwrap(),
                HiddenTarget::new(9, Some(4), false, 71).unwrap(),
            ],
            sealed,
        );
    }
}

#[test]
fn exact_reppo_matches_targets_duals_and_checkpoint_continuation() {
    fn paired<E: Env + Snapshot>(space: ActionSpace, environments: impl Fn() -> Vec<E>) {
        let mut reference = learner(space.clone(), 121, 0.25);
        let mut optimized = learner(space.clone(), 137, 0.);
        optimized.restore(&reference.checkpoint().unwrap()).unwrap();
        optimized.set_update_mode(TemporalExecutionMode::OptimizedExact);
        let mut left = run(&reference, environments());
        let mut right = run(&optimized, environments());
        for cycle in 0..4 {
            for _ in 0..4 {
                left.collect_tick(&reference).unwrap();
                right.collect_tick(&optimized).unwrap();
            }
            let a = left.seal(&reference).unwrap();
            let b = right.seal(&optimized).unwrap();
            assert_eq!(a.rows, b.rows);
            assert_eq!(a.sequences, b.sequences);
            let expected = reference.target_preview(a).unwrap();
            let actual = optimized.target_preview(b).unwrap();
            assert!(
                expected
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(actual.iter().map(|x| x.to_bits()))
            );
            let expected = left.update(&mut reference).unwrap();
            let actual = right.update(&mut optimized).unwrap();
            assert_eq!(expected.updates, actual.updates);
            assert!(expected.metrics.keys().eq(actual.metrics.keys()));
            for (name, number) in expected.metrics {
                assert_eq!(number.to_bits(), actual.metrics[&name].to_bits(), "{name}");
            }
            let mut record = optimized.checkpoint().unwrap();
            let mode = record
                .metadata
                .as_object_mut()
                .unwrap()
                .remove("update_mode")
                .unwrap();
            assert_eq!(mode, serde_json::json!("OptimizedExact"));
            assert!(
                canonical_record(&reference.checkpoint().unwrap()) == canonical_record(&record),
                "REPPO models, four optimizers, or random stream differ"
            );
            assert_eq!(left.counters(), right.counters());
            for lane in 0..2 {
                assert_eq!(
                    serde_json::to_vec(&left.collector().lane_state(lane)).unwrap(),
                    serde_json::to_vec(&right.collector().lane_state(lane)).unwrap()
                );
            }
            if cycle == 1 {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("exact-reppo");
                save_run(&path, &optimized, &right).unwrap();
                optimized = learner(space.clone(), 139, 0.);
                right = load_run(&path, &mut optimized).unwrap();
                assert_eq!(
                    optimized.update_mode(),
                    TemporalExecutionMode::OptimizedExact
                );
            }
        }
    }
    paired(ActionSpace::Discrete { count: 2 }, || {
        vec![
            DelayedBit::new(23, Some(2), false, 47).unwrap(),
            DelayedBit::new(31, Some(4), false, 71).unwrap(),
        ]
    });
    paired(
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.],
        },
        || {
            vec![
                HiddenTarget::new(17, Some(2), false, 47).unwrap(),
                HiddenTarget::new(25, Some(4), false, 71).unwrap(),
            ]
        },
    );
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Cut {
    tick: usize,
    ended: bool,
}
impl Env for Cut {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(5)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.tick = 0;
        self.ended = false;
        Ok(EpisodeStart::fresh(Observation::dense(
            vec![0.; 5],
            Some(vec![true, false]),
        )))
    }
    fn step(&mut self, a: &Action) -> Result<Step> {
        assert!(!self.ended);
        assert_eq!(*a, Action::Discrete(0));
        self.tick += 1;
        self.ended = true;
        Ok(Step {
            observation: Observation::dense(vec![0.7; 5], Some(vec![true, false])),
            reward: 0.25_f32.into(),
            terminated: false,
            truncated: true,
        })
    }
}
#[test]
fn external_truncation_bootstraps_from_final_memory_without_crossing_reset() {
    let l = learner(ActionSpace::Discrete { count: 2 }, 71, 0.);
    let mut t = run(
        &l,
        vec![Cut {
            tick: 0,
            ended: true,
        }],
    );
    for _ in 0..3 {
        t.collect_tick(&l).unwrap();
    }
    let b = t.seal(&l).unwrap();
    let targets = l.target_preview(b).unwrap();
    for (i, &target) in targets.iter().enumerate() {
        let expected = 0.25 + b.gamma * scalar_successor_q(b, i, true, &l.config().hl_gauss);
        assert!((target - expected).abs() < 2e-5);
        assert!(b.sequences.samples[b.rows[i].sample].steps[b.rows[i].time].truncated);
    }
    assert_eq!(
        b.rows.iter().map(|r| r.key.episode_id).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn schema_architecture_and_capability_failures_do_not_install_partial_state() {
    let mut l = learner(ActionSpace::Discrete { count: 2 }, 11, 0.1);
    let mut t = run(&l, vec![DelayedBit::new(2, None, false, 47).unwrap()]);
    t.cycle(&mut l).unwrap();
    let record = l.checkpoint().unwrap();
    let before = serde_json::to_value(&record).unwrap();
    for field in ["version", "revision"] {
        let mut bad = record.clone();
        bad.metadata[field] = serde_json::json!(u64::MAX);
        assert!(l.restore(&bad).is_err());
        assert!(serde_json::to_value(l.checkpoint().unwrap()).unwrap() == before);
    }
    let mut bad = record.clone();
    bad.metadata["critic"]["model"]["head_input_extra"] = 0.into();
    assert!(l.restore(&bad).is_err());
    assert!(serde_json::to_value(l.checkpoint().unwrap()).unwrap() == before);
}
