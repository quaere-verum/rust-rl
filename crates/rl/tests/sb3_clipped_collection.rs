#![cfg(feature = "neural")]

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
};

use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{OnPolicyCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, RolloutActor, RolloutBehavior},
        env::{Env, EpisodeStart, Observation, ObservationBatch, ObservationSpace, Step},
    },
    data::rollout::RolloutBuffer,
    nn::{distributions::Distribution, matrix, to_vec},
};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
struct Event {
    observation: f32,
    physical_action: f32,
    raw_action: f32,
    log_prob: f32,
    reward: f32,
    next_observation: f32,
    truncated: bool,
    advantage: f32,
    value: f32,
    buffer_reward: f32,
    #[serde(rename = "return")]
    target: f32,
}

#[derive(Deserialize)]
struct Fixture {
    low: f32,
    high: f32,
    mean_weight: f32,
    mean_bias: f32,
    log_std: f32,
    value_weight: f32,
    value_bias: f32,
    terminal_bootstrap: f32,
    events: Vec<Event>,
    after_policy_weight: Vec<f32>,
    after_policy_bias: Vec<f32>,
    after_log_std: Vec<f32>,
    after_value_weight: Vec<f32>,
    after_value_bias: Vec<f32>,
    policy_loss: f32,
    value_loss: f32,
    approx_kl: f32,
    moments: BTreeMap<String, Vec<f32>>,
}

#[derive(Clone)]
struct RecordingEnv {
    t: usize,
    space: ActionSpace,
    physical_actions: Rc<RefCell<Vec<f32>>>,
}
impl Env for RecordingEnv {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> rl::Result<EpisodeStart> {
        self.t = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, action: &Action) -> rl::Result<Step> {
        self.space.check(action, None)?;
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        self.physical_actions.borrow_mut().push(action[0]);
        self.t += 1;
        Ok(Step {
            observation: Observation::new(vec![self.t as f32]),
            reward: (action[0] * 0.5).into(),
            terminated: false,
            truncated: self.t == 3,
        })
    }
}

struct ReferenceActor {
    space: ActionSpace,
    events: Vec<Event>,
    index: Cell<usize>,
    weight: f32,
    bias: f32,
    log_std: f32,
    value_weight: f32,
    value_bias: f32,
}
impl Actor for ReferenceActor {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        _: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        _: &mut Vec<Action>,
    ) -> rl::Result<()> {
        unreachable!("this fixture exercises collection")
    }
}
impl RolloutActor for ReferenceActor {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        _: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> rl::Result<()> {
        let event = &self.events[self.index.get()];
        assert_eq!(obs.len(), 1);
        assert_eq!(obs.get(0).unwrap().dense_values(), [event.observation]);
        let mean = event.observation * self.weight + self.bias;
        let dist = Distribution::Gaussian {
            mean: matrix(vec![mean], 1, 1),
            log_std: matrix(vec![self.log_std], 1, 1),
            bounds: None,
        };
        out.actions = vec![Action::Continuous(vec![event.raw_action])];
        out.latents = vec![None];
        out.log_probs = to_vec(dist.log_prob(&out.actions, &out.latents)?)?;
        out.values = vec![event.observation * self.value_weight + self.value_bias];
        self.index.set(self.index.get() + 1);
        Ok(())
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> rl::Result<()> {
        *out = obs
            .iter()
            .map(|o| o.dense_values()[0] * self.value_weight + self.value_bias)
            .collect();
        Ok(())
    }
}

#[test]
fn collection_keeps_raw_sb3_actions_and_sends_clipped_actions_to_the_environment() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sb3_clipped_actions.json");
    let f: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let physical = Rc::new(RefCell::new(Vec::new()));
    let inner = RecordingEnv {
        t: 0,
        space: ActionSpace::Box {
            low: vec![f.low],
            high: vec![f.high],
        },
        physical_actions: physical.clone(),
    };
    let actor = ReferenceActor {
        space: ActionSpace::Unbounded { dimensions: 1 },
        events: f.events.clone(),
        index: Cell::new(0),
        weight: f.mean_weight,
        bias: f.mean_bias,
        log_std: f.log_std,
        value_weight: f.value_weight,
        value_bias: f.value_bias,
    };
    let env = rl::core::env::ClipAction::new(inner).unwrap();
    let mut collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 1).unwrap();
    let mut rollout = RolloutBuffer::default();
    collector
        .collect_steps_into(&actor, f.events.len(), &mut rollout)
        .unwrap();
    for (i, event) in f.events.iter().enumerate() {
        let row = rollout.row(i).unwrap();
        assert_eq!(
            row.transition.action,
            Action::Continuous(vec![event.raw_action])
        );
        assert_eq!(physical.borrow()[i], event.physical_action);
        assert_eq!(row.transition.reward.total(), event.reward);
        assert_eq!(
            row.transition.next_observation.dense_values(),
            [event.next_observation]
        );
        assert_eq!(row.transition.truncated, event.truncated);
        assert!(!row.transition.terminated);
        if event.truncated {
            assert!(f.terminal_bootstrap.abs() > 0.1);
            assert!((rollout.next_value(i).unwrap() - f.terminal_bootstrap).abs() < 2e-6);
            assert!(
                (event.buffer_reward - (event.reward + 0.99 * f.terminal_bootstrap)).abs() < 2e-6
            );
        } else {
            assert_eq!(event.buffer_reward, event.reward);
        }
    }
    let batch = rollout.to_batch(0.99, Some(0.95), false).unwrap();
    let metadata = batch.bootstrapped().unwrap();
    let mut collection_max = [0.0_f32; 3];
    for (i, event) in f.events.iter().enumerate() {
        for (k, (label, a, e)) in [
            ("log probability", metadata.log_probs[i], event.log_prob),
            ("advantage", metadata.advantages[i], event.advantage),
            ("return", batch.returns[i], event.target),
            ("value", metadata.values[i], event.value),
        ]
        .into_iter()
        .enumerate()
        {
            let error = (a - e).abs();
            if k < collection_max.len() {
                collection_max[k] = collection_max[k].max(error);
            }
            assert!(error < 2e-6, "{label}[{i}]: Rust {a}, SB3 {e}");
        }
    }
    println!(
        "collection maximum errors: logp={:.9} advantage={:.9} return={:.9}",
        collection_max[0], collection_max[1], collection_max[2]
    );
    use burn::module::Param;
    use burn_pack::{Bytes, Reader};
    use rand::SeedableRng;
    use rl::{
        algorithms::ppo::{OptimizerConfig, Ppo, PpoConfig},
        core::{checkpoint::Checkpoint, learner::Learner},
        nn::{
            critics::scalar::Value,
            policy::{ActorCritic, Policy},
            vector,
        },
    };
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut policy =
        Policy::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[], &mut rng).unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![f.mean_weight], 1, 1));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[f.mean_bias])));
    policy.log_std = Param::from_tensor(vector(&[f.log_std]));
    let mut value = Value::new(1, &[], &mut rng).unwrap();
    value.network.layers[0].weight = Param::from_tensor(matrix(vec![f.value_weight], 1, 1));
    value.network.layers[0].bias = Some(Param::from_tensor(vector(&[f.value_bias])));
    let opt = OptimizerConfig {
        learning_rate: 0.0003,
        max_grad_norm: None,
    };
    let config = PpoConfig {
        policy: opt.clone(),
        value: opt,
        epochs: 1,
        minibatch_size: 6,
        clip: 0.2,
        entropy: 0.,
        target_kl: None,
    };
    let mut learner =
        Ppo::new_with_joint_gradient_clipping(ActorCritic { policy, value }, config, 77, 0.5, 0.5)
            .unwrap();
    let report = learner.update(&batch).unwrap();
    let mut metric_max = 0.0_f32;
    for (name, actual, expected) in [
        (
            "policy loss",
            report.metrics["policy_loss"] as f32,
            f.policy_loss,
        ),
        (
            "value loss",
            report.metrics["value_loss"] as f32,
            f.value_loss,
        ),
        ("approx KL", report.metrics["approx_kl"] as f32, f.approx_kl),
    ] {
        let error = (actual - expected).abs();
        metric_max = metric_max.max(error);
        assert!(error < 2e-6, "{name}: Rust {actual}, SB3 {expected}");
    }
    let policy = learner.policy();
    let value = learner.value();
    let mut parameter_max = 0.0_f32;
    for (name, actual, expected) in [
        (
            "policy weight",
            to_vec(policy.network.layers[0].weight.val()).unwrap(),
            f.after_policy_weight,
        ),
        (
            "policy bias",
            to_vec(policy.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
            f.after_policy_bias,
        ),
        (
            "log std",
            to_vec(policy.log_std.val()).unwrap(),
            f.after_log_std,
        ),
        (
            "value weight",
            to_vec(value.network.layers[0].weight.val()).unwrap(),
            f.after_value_weight,
        ),
        (
            "value bias",
            to_vec(value.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
            f.after_value_bias,
        ),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            let error = (a - b).abs();
            parameter_max = parameter_max.max(error);
            assert!(error < 2e-6, "{name}: Rust {a}, SB3 {b}");
        }
    }
    let mut observed = BTreeMap::new();
    for (name, data) in learner.checkpoint().unwrap().optimizers {
        let reader = Reader::from_bytes(Bytes::from_bytes_vec(data)).unwrap();
        let paths = reader.metadata().clone();
        for tensor in reader.into_tensors().unwrap() {
            let (tensor_name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
            let param = paths[&id.unwrap().to_string()].rsplit('.').next().unwrap();
            let moment = tensor_name.rsplit('.').next().unwrap();
            observed.insert(
                format!("{name}.{param}.{moment}"),
                burn::tensor::TensorData::from_bytes(bytes, shape, dtype)
                    .try_to_vec::<f32>()
                    .unwrap(),
            );
        }
    }
    assert_eq!(observed.len(), f.moments.len());
    let mut moment_max = 0.0_f32;
    for (name, expected) in f.moments {
        assert_eq!(observed[&name].len(), expected.len());
        for (a, b) in observed[&name].iter().zip(expected) {
            let error = (a - b).abs();
            moment_max = moment_max.max(error);
            assert!(error < 2e-6, "{name}: Rust {a}, SB3 {b}");
        }
    }
    println!(
        "update maximum errors: loss/KL={metric_max:.9} parameters={parameter_max:.9} moments={moment_max:.9}"
    );
}

#[test]
fn clipped_adapter_rejects_invalid_actions_without_advancing_inner_environment() {
    use rl::core::env::ClipAction;
    let physical = Rc::new(RefCell::new(Vec::new()));
    let inner = RecordingEnv {
        t: 0,
        space: ActionSpace::Box {
            low: vec![-1., 2.],
            high: vec![2., 5.],
        },
        physical_actions: physical.clone(),
    };
    let mut wrapped = ClipAction::new(inner).unwrap();
    assert_eq!(
        wrapped.action_space(),
        &ActionSpace::Unbounded { dimensions: 2 }
    );
    assert_eq!(
        wrapped.physical_action_space(),
        &ActionSpace::Box {
            low: vec![-1., 2.],
            high: vec![2., 5.]
        }
    );
    let raw = Action::Continuous(vec![-4., 7.]);
    assert_eq!(
        wrapped.clipped_action(&raw).unwrap(),
        Action::Continuous(vec![-1., 5.])
    );
    assert_eq!(raw, Action::Continuous(vec![-4., 7.]));
    assert_eq!(
        wrapped
            .clipped_action(&Action::Continuous(vec![0.5, 3.]))
            .unwrap(),
        Action::Continuous(vec![0.5, 3.])
    );
    wrapped.reset(None).unwrap();
    for invalid in [
        Action::Discrete(0),
        Action::Continuous(vec![0.]),
        Action::Continuous(vec![f32::NAN, 3.]),
        Action::Continuous(vec![0., f32::INFINITY]),
    ] {
        assert!(wrapped.step(&invalid).is_err());
        assert_eq!(wrapped.inner().t, 0);
        assert!(physical.borrow().is_empty());
    }
    assert!(
        ClipAction::new(RecordingEnv {
            t: 0,
            space: ActionSpace::Discrete { count: 2 },
            physical_actions: physical.clone()
        })
        .is_err()
    );
    assert!(
        ClipAction::new(RecordingEnv {
            t: 0,
            space: ActionSpace::Unbounded { dimensions: 1 },
            physical_actions: physical
        })
        .is_err()
    );
}

#[test]
fn clipped_adapter_snapshot_preserves_rng_and_rejects_invalid_space() {
    use rl::{
        core::{checkpoint::Snapshot, env::ClipAction},
        envs::lqg::{Lqg, LqgConfig},
    };
    let config = LqgConfig::scalar(3, 0.9, 1., 1., 0.2, 1., 1., 0.1, 2.).unwrap();
    let mut original = ClipAction::new(Lqg::new(config, 7).unwrap()).unwrap();
    original.reset(Some(91)).unwrap();
    original.step(&Action::Continuous(vec![4.])).unwrap();
    let json = serde_json::to_value(&original).unwrap();
    let mut restored: ClipAction<Lqg> = serde_json::from_value(json.clone()).unwrap();
    restored.validate_snapshot().unwrap();
    let a = original.step(&Action::Continuous(vec![-3.])).unwrap();
    let b = restored.step(&Action::Continuous(vec![-3.])).unwrap();
    assert_eq!(a.observation, b.observation);
    assert_eq!(a.reward, b.reward);
    assert_eq!((a.terminated, a.truncated), (b.terminated, b.truncated));
    let mut invalid = json;
    invalid["policy_space"] = serde_json::json!({"Unbounded": {"dimensions": 2}});
    let mut restored: ClipAction<Lqg> = serde_json::from_value(invalid).unwrap();
    assert!(restored.validate_snapshot().is_err());
    assert!(restored.reset(None).is_err());
    assert!(restored.step(&Action::Continuous(vec![0.])).is_err());
}

#[test]
fn clipped_adapter_preserves_time_limit_and_finite_horizon_boundaries() {
    use rl::{
        core::env::{ClipAction, TimeLimit},
        envs::lqg::{Lqg, LqgConfig},
    };
    let config = LqgConfig::scalar(3, 0.9, 1., 1., 0.2, 1., 1., 0., 2.).unwrap();
    let inner = Lqg::new(config.clone(), 7).unwrap();
    let mut timed = ClipAction::new(TimeLimit::new(inner, 2).unwrap()).unwrap();
    timed.reset(Some(17)).unwrap();
    let first = timed.step(&Action::Continuous(vec![5.])).unwrap();
    assert!(!first.terminated && !first.truncated);
    let second = timed.step(&Action::Continuous(vec![-5.])).unwrap();
    assert!(!second.terminated && second.truncated);
    assert!(timed.step(&Action::Continuous(vec![0.])).is_err());
    timed.reset(Some(17)).unwrap();
    assert_eq!(
        timed
            .step(&Action::Continuous(vec![5.]))
            .unwrap()
            .observation,
        first.observation
    );

    let mut finite = ClipAction::new(Lqg::new(config, 7).unwrap()).unwrap();
    finite.reset(Some(17)).unwrap();
    for i in 0..3 {
        let step = finite.step(&Action::Continuous(vec![0.])).unwrap();
        assert_eq!(step.terminated, i == 2);
        assert!(!step.truncated);
    }
    assert!(finite.step(&Action::Continuous(vec![0.])).is_err());
}

#[cfg(feature = "diagnostics")]
#[test]
fn clipped_adapter_preserves_branching_and_physical_frames() {
    use rl::{
        core::env::ClipAction,
        diagnostics::evaluation::{BranchingEnv, FrameSource},
        envs::lqg::{Lqg, LqgConfig},
    };
    let config = LqgConfig::scalar(3, 0.9, 1., 1., 0.2, 1., 1., 0.1, 2.).unwrap();
    let mut wrapped = ClipAction::new(Lqg::new(config, 7).unwrap()).unwrap();
    wrapped.reset(Some(17)).unwrap();
    let mut branch = wrapped.branch(71).unwrap();
    let mut physical_branch = wrapped.inner().branch(71).unwrap();
    assert_eq!(
        branch.branch_observation().unwrap(),
        physical_branch.branch_observation().unwrap()
    );
    assert_eq!(
        branch.branch_state().unwrap(),
        physical_branch.branch_state().unwrap()
    );
    assert!(branch.branch_description().unwrap().contains("clipp"));
    let frame = branch.diagnostic_frame().unwrap();
    let physical_frame = physical_branch.diagnostic_frame().unwrap();
    assert_eq!(frame.environment, physical_frame.environment);
    assert_eq!(frame.channels["state"], physical_frame.channels["state"]);
    assert_eq!(frame.channels["clip_action.low"], vec![-2.]);
    assert_eq!(frame.channels["clip_action.high"], vec![2.]);
    let a = branch.step(&Action::Continuous(vec![5.])).unwrap();
    let b = physical_branch.step(&Action::Continuous(vec![2.])).unwrap();
    assert_eq!(a.observation, b.observation);
    assert_eq!(a.reward, b.reward);
}

#[cfg(feature = "diagnostics")]
#[test]
fn clipped_evaluation_records_raw_actions_and_physical_bounds() {
    use rl::{
        core::env::ClipAction,
        diagnostics::{
            evaluation::{EvaluationConfig, evaluate_with_frames},
            schema::{Record, RecordContext, RunManifest},
            storage::{ReadLimits, Recorder, Recording, StorageConfig},
        },
        envs::lqg::{Lqg, LqgConfig},
    };
    struct FixedRaw(ActionSpace);
    impl Actor for FixedRaw {
        fn action_space(&self) -> &ActionSpace {
            &self.0
        }
        fn act_into(
            &self,
            _: ObservationBatch<'_>,
            _: ActMode,
            _: &mut ChaCha8Rng,
            out: &mut Vec<Action>,
        ) -> rl::Result<()> {
            out.push(Action::Continuous(vec![5.]));
            Ok(())
        }
    }
    let config = LqgConfig::scalar(2, 0., 1., 0., 1., 0., 0., 0., 2.).unwrap();
    let actor = FixedRaw(ActionSpace::Unbounded { dimensions: 1 });
    let evaluation = evaluate_with_frames(
        |seed| ClipAction::new(Lqg::new(config.clone(), seed)?),
        &actor,
        &EvaluationConfig {
            seeds: vec![7],
            trajectories: 1,
            max_steps: 2,
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    let chunk = evaluation
        .trajectory_record(0, RecordContext::new("clip", 0), "trajectory", "lqg")
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recording");
    let manifest =
        RunManifest::new("clip", "ppo", "clipped-lqg", 2, actor.0.clone(), 0.99).unwrap();
    let mut recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    recorder.append(&chunk).unwrap();
    recorder.finish().unwrap();
    let recording = Recording::open(&path, ReadLimits::default()).unwrap();
    let chunk = recording.load(0).unwrap();
    let Record::Trajectory(record) = &chunk.record else {
        panic!("trajectory")
    };
    assert_eq!(chunk.arrays[&record.actions].as_f32().unwrap(), &[5., 5.]);
    assert_eq!(
        chunk.arrays[&record.frames["clip_action.low"]]
            .as_f64()
            .unwrap(),
        &[-2., -2., -2.]
    );
    assert_eq!(
        chunk.arrays[&record.frames["clip_action.high"]]
            .as_f64()
            .unwrap(),
        &[2., 2., 2.]
    );
    assert_eq!(
        chunk.arrays[&record.frames["state"]].as_f64().unwrap(),
        &[0., 2., 2.]
    );
    assert_eq!(chunk.arrays[&record.rewards].as_f32().unwrap(), &[-4., -4.]);
}

#[cfg(feature = "diagnostics")]
#[test]
fn clipped_adapter_rejects_reserved_frame_channel_collision() {
    use rl::{
        core::env::ClipAction,
        diagnostics::evaluation::{EnvironmentFrame, FrameSource},
    };
    struct Colliding(RecordingEnv);
    impl Env for Colliding {
        fn observation_space(&self) -> ObservationSpace {
            self.0.observation_space()
        }
        fn action_space(&self) -> &ActionSpace {
            self.0.action_space()
        }
        fn reset(&mut self, seed: Option<u64>) -> rl::Result<EpisodeStart> {
            self.0.reset(seed)
        }
        fn step(&mut self, action: &Action) -> rl::Result<Step> {
            self.0.step(action)
        }
    }
    impl FrameSource for Colliding {
        fn diagnostic_frame(&self) -> rl::Result<EnvironmentFrame> {
            Ok(EnvironmentFrame {
                environment: "collision".into(),
                simulation_time: None,
                channels: BTreeMap::from([("clip_action.low".into(), vec![999.])]),
            })
        }
    }
    let inner = RecordingEnv {
        t: 0,
        space: ActionSpace::Box {
            low: vec![-1.],
            high: vec![2.],
        },
        physical_actions: Rc::new(RefCell::new(Vec::new())),
    };
    assert!(
        ClipAction::new(Colliding(inner))
            .unwrap()
            .diagnostic_frame()
            .is_err()
    );
}

#[test]
fn unbounded_policy_rollout_keeps_raw_samples_and_clips_at_the_environment() {
    use burn::module::Param;
    use rand::SeedableRng;
    use rl::{
        core::actor::ActMode,
        core::env::ClipAction,
        nn::{
            critics::scalar::Value,
            policy::{ActorCritic, Policy, StochasticPolicy},
            vector,
        },
    };
    let physical = Rc::new(RefCell::new(Vec::new()));
    let inner = RecordingEnv {
        t: 0,
        space: ActionSpace::Box {
            low: vec![-1.],
            high: vec![2.],
        },
        physical_actions: physical.clone(),
    };
    let env = ClipAction::new(inner).unwrap();
    let mut init = ChaCha8Rng::seed_from_u64(13);
    let mut policy = Policy::new(1, env.action_space().clone(), &[], &mut init).unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![3.], 1, 1));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[-3.])));
    policy.log_std = Param::from_tensor(vector(&[0.1]));
    let models = ActorCritic {
        policy,
        value: Value::new(1, &[], &mut init).unwrap(),
    };
    let mut collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 77).unwrap();
    let mut rollout = RolloutBuffer::default();
    collector
        .collect_steps_into(&models, 6, &mut rollout)
        .unwrap();
    let mut saturated = 0;
    for i in 0..6 {
        let row = rollout.row(i).unwrap();
        let Action::Continuous(raw) = &row.transition.action else {
            unreachable!()
        };
        assert_eq!(physical.borrow()[i], raw[0].clamp(-1., 2.));
        saturated += usize::from(raw[0] != physical.borrow()[i]);
        assert!(row.latent.is_none());
    }
    assert!(saturated >= 2);
    let batch = rollout.to_batch(0.99, Some(0.95), false).unwrap();
    let recomputed = models
        .policy
        .distribution((&batch.observations).into())
        .unwrap()
        .log_prob(&batch.actions, &batch.latents)
        .unwrap();
    for (a, b) in to_vec(recomputed)
        .unwrap()
        .iter()
        .zip(&batch.bootstrapped().unwrap().log_probs)
    {
        assert!((a - b).abs() < 2e-6);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let deterministic = models
        .act(
            &[Observation::new(vec![0.])],
            ActMode::Deterministic,
            &mut rng,
        )
        .unwrap();
    assert_eq!(deterministic, vec![Action::Continuous(vec![-3.])]);
}

#[test]
fn clipped_adapter_detects_physical_space_changes_during_reset_and_step() {
    use rl::core::env::ClipAction;
    #[derive(Clone)]
    struct Changes {
        space: ActionSpace,
        on_reset: bool,
    }
    impl Env for Changes {
        fn observation_space(&self) -> ObservationSpace {
            ObservationSpace::Dense(1)
        }
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }
        fn reset(&mut self, _: Option<u64>) -> rl::Result<EpisodeStart> {
            if self.on_reset {
                self.space = ActionSpace::Box {
                    low: vec![-2.],
                    high: vec![2.],
                };
            }
            Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
        }
        fn step(&mut self, _: &Action) -> rl::Result<Step> {
            self.space = ActionSpace::Box {
                low: vec![-2.],
                high: vec![2.],
            };
            Ok(Step {
                observation: Observation::new(vec![0.]),
                reward: 0.0_f32.into(),
                terminated: false,
                truncated: false,
            })
        }
    }
    let base = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let mut reset_change = ClipAction::new(Changes {
        space: base.clone(),
        on_reset: true,
    })
    .unwrap();
    assert!(reset_change.reset(None).is_err());
    let mut step_change = ClipAction::new(Changes {
        space: base,
        on_reset: false,
    })
    .unwrap();
    step_change.reset(None).unwrap();
    assert!(step_change.step(&Action::Continuous(vec![0.])).is_err());
}
