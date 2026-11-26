#![cfg(feature = "neural")]

use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::Tensor,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::nn::temporal_actor::TemporalActor;
use rl::{
    algorithms::{
        mdpo::{MdpoConfig, MdpoWeighting, MirrorStepSchedule},
        ppo::OptimizerConfig,
        recurrent_mdpo::RecurrentMdpo,
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        checkpoint::{Checkpoint, LearnerRecord},
        env::Observation,
        recording::RecordingKey,
        stateful::{BehaviorPublisher, HistorySpec, StatefulLearner},
        stateful_on_policy::DistributionSnapshot,
        temporal::TemporalExecutionMode,
    },
    data::{
        temporal::{BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep},
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch, TemporalRow},
    },
    envs::delayed_bit::DelayedBit,
    nn::{
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_value::{TemporalStateValue, TemporalValue},
        to_vec, vector,
    },
    train::{
        checkpoint::{load_run, save_run},
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};
use std::collections::BTreeMap;

fn config(inner_steps: usize) -> MdpoConfig {
    let optimizer = OptimizerConfig {
        learning_rate: 0.02,
        max_grad_norm: None,
    };
    MdpoConfig {
        policy: optimizer.clone(),
        value: optimizer,
        inner_steps,
        value_epochs: 2,
        value_minibatch_size: 4,
        mirror_step: MirrorStepSchedule::Linear {
            start: 0.5,
            end: 1.5,
            duration: 4,
        },
        weighting: MdpoWeighting::Uniform,
    }
}
fn learner(seed: u64, config: MdpoConfig) -> RecurrentMdpo {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    RecurrentMdpo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
        config,
        seed + 1,
    )
    .unwrap()
}
fn weights(model: &impl Module) -> Vec<f32> {
    struct Collect(Vec<f32>);
    impl ModuleVisitor for Collect {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.0.extend(to_vec(param.val()).unwrap());
        }
    }
    let mut out = Collect(Vec::new());
    model.visit(&mut out);
    out.0
}
fn batch(learner: &RecurrentMdpo, gamma: f32) -> TemporalOnPolicyBatch<TemporalActor> {
    let behavior = learner.behavior_actor(&(), learner.revision()).unwrap();
    let observations: Vec<_> = [0.8, -0.3, 0.4, 0.9, -0.6, 0.2, 0.]
        .into_iter()
        .map(|x| Observation::new(vec![x]))
        .collect();
    let latent_values = [0.2_f32, -0.4, 0.6, 0.1, -0.2, 0.3];
    let actions: Vec<_> = (0..6)
        .map(|i| match behavior.policy().action_space() {
            ActionSpace::Discrete { .. } => Action::Discrete(i % 2),
            ActionSpace::Unbounded { .. } => Action::Continuous(vec![latent_values[i]]),
            ActionSpace::Box { low, high } => Action::Continuous(vec![
                (low[0] + high[0]) * 0.5 + (high[0] - low[0]) * 0.5 * latent_values[i].tanh(),
            ]),
            ActionSpace::Conditional(_) => unreachable!("fixed-width test policy"),
        })
        .collect();
    let latents: Vec<_> = (0..6)
        .map(|i| {
            matches!(behavior.policy().action_space(), ActionSpace::Box { .. })
                .then(|| vec![latent_values[i]])
        })
        .collect();
    let steps: Vec<_> = (0..6)
        .map(|i| TemporalStep {
            action: actions[i].clone(),
            reward: [0.2_f32, -0.3, 0.5, 0., 1.3, -0.2][i].into(),
            terminated: i == 5,
            truncated: false,
            behavior: BehaviorMetadata {
                revision: learner.revision(),
                log_prob: None,
            },
        })
        .collect();
    let mut sequences = SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 2,
            n_step: 1,
        },
        samples: (0..3)
            .map(|window| {
                let end = 2 * (window + 1);
                SequenceSample {
                    episode_id: 1,
                    trajectory_start: 0,
                    learn_start: 2 * window,
                    burn_start: 2 * window,
                    observations: observations[..=end].to_vec(),
                    steps: steps[..end].to_vec(),
                    loss_mask: (0..end).map(|time| time >= 2 * window).collect(),
                    history_start: 0,
                    previous: None,
                    origin: Default::default(),
                }
            })
            .collect(),
    };
    let tokens: Vec<_> = (0..6).map(|time| (time / 2, time)).collect();
    let distribution = behavior.policy().distribution(&sequences, &tokens).unwrap();
    let log_probs = to_vec(distribution.log_prob(&actions, &latents).unwrap()).unwrap();
    let values = to_vec(
        behavior
            .value()
            .unwrap()
            .values(&sequences, &tokens)
            .unwrap(),
    )
    .unwrap();
    for sample in &mut sequences.samples {
        for (time, step) in sample.steps.iter_mut().enumerate() {
            step.behavior.log_prob = Some(log_probs[time]);
        }
    }
    let rows = (0..6)
        .map(|time| TemporalRow {
            key: RecordingKey {
                episode_id: 1,
                timestep: time,
            },
            sample: time / 2,
            time,
            log_prob: log_probs[time],
            latent: latents[time].clone(),
            distribution: DistributionSnapshot::from_distribution(&distribution, time).unwrap(),
            value: Some(values[time]),
            next_value: Some(if time == 5 { 0. } else { values[time + 1] }),
            advantage: Some([0.5, -0.7, 1.2, -0.4, 0.9, -0.3][time]),
            return_: Some([1., 0.3, -0.2, 1.2, 0.7, -0.4][time]),
            discount_weight: (f64::from(gamma)).powf(time as f64) as f32,
        })
        .collect();
    let batch = TemporalOnPolicyBatch {
        version: 1,
        behavior,
        revision: learner.revision(),
        kind: TemporalBatchKind::Bootstrapped,
        gamma,
        gae_lambda: Some(0.95),
        episode_count: 1,
        sequences,
        rows,
    };
    batch.validate().unwrap();
    batch
}
fn canonical_record(record: &LearnerRecord) -> serde_json::Value {
    fn pack(bytes: &[u8]) -> serde_json::Value {
        let reader =
            burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes.to_vec()))
                .unwrap();
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
        let tensors=reader.into_tensors().unwrap().into_iter().map(|tensor|{let (name,dtype,shape,id,bytes)=tensor.into_parts().unwrap();(name,serde_json::json!({"dtype":format!("{dtype:?}"),"shape":shape,"id":id,"bytes":bytes}))}).collect::<BTreeMap<_,_>>();
        serde_json::json!({"scalars":scalars,"metadata":metadata,"tensors":tensors})
    }
    let models = serde_json::to_value(&record.models)
        .unwrap()
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, model)| {
            let bytes: Vec<u8> = serde_json::from_value(model["bytes"].clone()).unwrap();
            (name.clone(), pack(&bytes))
        })
        .collect::<BTreeMap<_, _>>();
    let optimizers = record
        .optimizers
        .iter()
        .map(|(name, bytes)| (name.clone(), pack(bytes)))
        .collect::<BTreeMap<_, _>>();
    serde_json::json!({"algorithm":record.algorithm,"metadata":record.metadata,"models":models,"optimizers":optimizers})
}

#[test]
fn full_actor_steps_train_both_grus_and_preserve_one_frozen_behavior() {
    let mut learner = learner(71, config(3));
    let input = batch(&learner, 0.9);
    let old = weights(input.behavior.policy());
    let old_actor = weights(&learner.policy().encoder);
    let old_critic = weights(&learner.value().encoder);
    let report = learner.update(&input).unwrap();
    assert_eq!(
        (
            report.updates,
            learner.updates(),
            learner.actor_updates(),
            learner.critic_updates(),
            learner.outer_iterations(),
            learner.revision()
        ),
        (7, 7, 3, 4, 1, 1)
    );
    assert_eq!(report.metrics["mirror_step"], 0.5);
    assert_ne!(old_actor, weights(&learner.policy().encoder));
    assert_ne!(old_critic, weights(&learner.value().encoder));
    assert_eq!(old, weights(input.behavior.policy()));
    assert!(report.metrics["kl_new_behavior"] > 0.);
}

#[test]
fn recurrent_mdpo_supports_unbounded_and_bounded_gaussians_with_the_same_temporal_contract() {
    for space in [
        ActionSpace::Unbounded { dimensions: 1 },
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![3.],
        },
    ] {
        let mut rng = ChaCha8Rng::seed_from_u64(311);
        let spec = HistorySpec::new(1, space).unwrap();
        let mut configuration = config(2);
        configuration.value_epochs = 1;
        configuration.weighting = MdpoWeighting::DiscountedOccupancy;
        let mut learner = RecurrentMdpo::new(
            TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
            TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
            configuration,
            313,
        )
        .unwrap();
        let input = batch(&learner, 0.9);
        let before = weights(&learner.policy().encoder);
        let report = learner.update(&input).unwrap();
        assert_eq!(report.updates, 4);
        assert!(report.metrics.values().all(|x| x.is_finite()));
        assert_ne!(before, weights(&learner.policy().encoder));
    }
}

#[test]
fn inner_loop_matches_manual_adam_with_fixed_absolute_occupancy_and_renewed_current_gru() {
    let mut configuration = config(3);
    configuration.weighting = MdpoWeighting::DiscountedOccupancy;
    let mut learner = learner(73, configuration.clone());
    let input = batch(&learner, 0.5).select_windows(&[0, 2]).unwrap();
    let mut policy = learner
        .policy()
        .valid()
        .fork(&rl::nn::device())
        .unfreeze()
        .train();
    let mut optimizer = configuration.policy.build().unwrap();
    let behavior = input
        .behavior
        .policy()
        .distribution(&input.sequences, &input.tokens())
        .unwrap();
    let actions = input.actions();
    let latents = input.latents();
    let old_log_probs = behavior
        .log_prob(&actions, &latents)
        .unwrap()
        .without_autodiff();
    let mass = 1. + 0.5 + 0.0625 + 0.03125;
    let occupancy = vector(&[1. / mass, 0.5 / mass, 0.0625 / mass, 0.03125 / mass]);
    let advantages = vector(&input.advantages().unwrap());
    for _ in 0..3 {
        let current = policy
            .distribution(&input.sequences, &input.tokens())
            .unwrap();
        let ratio = (current.log_prob(&actions, &latents).unwrap() - old_log_probs.clone()).exp();
        let surrogate = (occupancy.clone() * ratio * advantages.clone()).sum();
        let kl = (occupancy.clone() * current.kl(&behavior).unwrap()).sum();
        let loss = surrogate.neg() + kl * 2.;
        let grads = GradientsParams::from_grads(loss.backward(), &policy);
        policy = optimizer.step(configuration.policy.learning_rate, policy, grads);
    }
    learner.update(&input).unwrap();
    for (actual, expected) in weights(&learner.policy()).iter().zip(weights(&policy)) {
        assert!((actual - expected).abs() < 2e-6);
    }
}

#[test]
fn value_passes_targets_and_minibatches_cannot_change_finished_actor_improvement() {
    let mut left = learner(79, config(3));
    let mut right_config = config(3);
    right_config.value_epochs = 3;
    right_config.value_minibatch_size = 2;
    let mut right = learner(79, right_config);
    let input = batch(&left, 0.9);
    let mut altered = input.clone();
    for row in &mut altered.rows {
        row.return_ = Some(row.return_.unwrap() + 100.);
    }
    left.update(&input).unwrap();
    right.update(&altered).unwrap();
    assert_eq!(weights(&left.policy()), weights(&right.policy()));
    assert_ne!(weights(&left.value()), weights(&right.value()));
}

#[test]
fn mirror_strength_has_no_first_step_effect_but_regularizes_later_improvement() {
    use rl::{
        algorithms::mdpo::Mdpo,
        core::learner::Learner,
        core::stateful::{HistoryInput, PreviousInput, StatefulActor},
        data::rollout::{BootstrappedBatch, OnPolicyBatch, OnPolicyMetadata},
        nn::{
            critics::scalar::Value,
            policy::{ActorCritic, Policy, StochasticPolicy},
        },
    };
    let make = |inner, step| {
        let mut configuration = config(inner);
        configuration.mirror_step = MirrorStepSchedule::Constant(step);
        learner(83, configuration)
    };
    let mut weak = make(1, 100.);
    let mut strong = make(1, 0.01);
    let input = batch(&weak, 0.9);
    let initial_parameters = weights(&weak.policy());
    let probe = weak
        .policy()
        .valid()
        .fork(&rl::nn::device())
        .unfreeze()
        .train();
    let current = probe
        .distribution(&input.sequences, &input.tokens())
        .unwrap();
    let behavior = input
        .behavior
        .policy()
        .distribution(&input.sequences, &input.tokens())
        .unwrap();
    let initial_kl = (vector(&vec![1. / input.len() as f32; input.len()])
        * current.kl(&behavior).unwrap())
    .sum();
    let initial_kl_number = initial_kl.clone().into_scalar::<f32>();
    let gradients = GradientsParams::from_grads(initial_kl.backward(), &probe);
    struct GradientMaximum<'a> {
        gradients: &'a GradientsParams,
        maximum: f32,
    }
    impl ModuleVisitor for GradientMaximum<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if let Some(gradient) = self.gradients.get::<D>(param.id) {
                for value in to_vec(gradient).unwrap() {
                    self.maximum = self.maximum.max(value.abs());
                }
            }
        }
    }
    let mut gradient = GradientMaximum {
        gradients: &gradients,
        maximum: 0.,
    };
    probe.visit(&mut gradient);
    weak.update(&input).unwrap();
    strong.update(&input).unwrap();
    let weak_likelihood = to_vec(
        weak.policy()
            .distribution(&input.sequences, &input.tokens())
            .unwrap()
            .log_prob(&input.actions(), &input.latents())
            .unwrap(),
    )
    .unwrap();
    let strong_likelihood = to_vec(
        strong
            .policy()
            .distribution(&input.sequences, &input.tokens())
            .unwrap()
            .log_prob(&input.actions(), &input.latents())
            .unwrap(),
    )
    .unwrap();
    let likelihood_difference = weak_likelihood
        .iter()
        .zip(strong_likelihood)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    let parameter_difference = weights(&weak.policy())
        .iter()
        .zip(weights(&strong.policy()))
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    let weak_parameters = weights(&weak.policy());
    let strong_parameters = weights(&strong.policy());
    let first_index = weak_parameters
        .iter()
        .zip(&strong_parameters)
        .position(|(a, b)| (a - b).abs() >= 1e-6);
    let maximum_index = weak_parameters
        .iter()
        .zip(&strong_parameters)
        .enumerate()
        .map(|(index, (a, b))| (index, (a - b).abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap();
    // Feedforward control: the identical frozen recurrent representations enter
    // the same MLP head; this isolates the shared objective/Adam convention.
    let actor = &input.behavior;
    let sample = &input.sequences.samples[2];
    let mut state = actor.initial_state().unwrap();
    let mut features = Vec::new();
    for (time, observation) in sample.observations.iter().take(6).enumerate() {
        state = actor
            .assimilate(
                &state,
                HistoryInput {
                    observation,
                    previous: time.checked_sub(1).map(|i| PreviousInput {
                        factual: None,
                        action: &sample.steps[i].action,
                        reward: sample.steps[i].reward.total(),
                    }),
                },
            )
            .unwrap();
        features.push(Observation::new(state.policy.clone()));
    }
    let ff_policy = Policy::from_network(
        probe
            .head
            .valid()
            .fork(&rl::nn::device())
            .unfreeze()
            .train(),
        ActionSpace::Discrete { count: 2 },
    )
    .unwrap();
    let ff_value = Value::new(3, &[], &mut ChaCha8Rng::seed_from_u64(311)).unwrap();
    let ff_models = ActorCritic {
        policy: ff_policy,
        value: ff_value,
    };
    let log_probs = to_vec(
        ff_models
            .policy
            .distribution((&features).into())
            .unwrap()
            .log_prob(&input.actions(), &input.latents())
            .unwrap(),
    )
    .unwrap();
    let ff_input = OnPolicyBatch {
        episode_count: 1,
        observations: features,
        actions: input.actions(),
        latents: input.latents(),
        returns: input.returns().unwrap(),
        discount_weights: input.rows.iter().map(|row| row.discount_weight).collect(),
        metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
            log_probs,
            values: input.old_values().unwrap(),
            advantages: input.advantages().unwrap(),
        }),
    };
    let mut ff_weak = Mdpo::new(ff_models.clone(), weak.config().clone(), 7).unwrap();
    let mut ff_strong = Mdpo::new(ff_models, strong.config().clone(), 7).unwrap();
    ff_weak.update(&ff_input).unwrap();
    ff_strong.update(&ff_input).unwrap();
    let ff_weak_likelihood = to_vec(
        ff_weak
            .policy()
            .distribution((&ff_input.observations).into())
            .unwrap()
            .log_prob(&ff_input.actions, &ff_input.latents)
            .unwrap(),
    )
    .unwrap();
    let ff_strong_likelihood = to_vec(
        ff_strong
            .policy()
            .distribution((&ff_input.observations).into())
            .unwrap()
            .log_prob(&ff_input.actions, &ff_input.latents)
            .unwrap(),
    )
    .unwrap();
    let ff_likelihood_difference = ff_weak_likelihood
        .iter()
        .zip(ff_strong_likelihood)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    let ff_parameter_difference = weights(&ff_weak.policy())
        .iter()
        .zip(weights(&ff_strong.policy()))
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    println!(
        "initial KL={initial_kl_number:e}, maximum initial KL gradient={:e}, maximum first-step parameter difference={parameter_difference:e}, maximum first-step likelihood difference={likelihood_difference:e}",
        gradient.maximum
    );
    println!(
        "first index={first_index:?}, maximum index={maximum_index}, initial={:e}, weak={:e}, strong={:e}; matched feedforward head parameter difference={ff_parameter_difference:e}, likelihood difference={ff_likelihood_difference:e}",
        initial_parameters[maximum_index],
        weak_parameters[maximum_index],
        strong_parameters[maximum_index]
    );
    // Characterization (seed83, penalties0.01/100): KL=0, maximum KL
    // gradient1.8626451e-8, encoder-parameter delta1.4305115e-6 at a
    // parameter initially0.6101634, and log-likelihood delta1.1920929e-7.
    // The matched feedforward head has no such drift. Repeated GRU adjoint
    // reductions leave sub-epsilon KL-gradient residue which Adam can amplify;
    // raw parameter equality is therefore not the stationarity contract.
    // Bound the initial derivative by f32 roundoff and compare the policy
    // function, as the existing feedforward MDPO test does. The separate manual
    // Adam fixture still compares every parameter after all inner updates.
    assert!(initial_kl_number.abs() <= f32::EPSILON);
    assert!(gradient.maximum <= f32::EPSILON);
    assert!(likelihood_difference <= 1e-6);
    assert!(ff_likelihood_difference <= 1e-6);
    let mut weak = make(5, 100.);
    let mut strong = make(5, 0.01);
    let input = batch(&weak, 0.9);
    let weak_report = weak.update(&input).unwrap();
    let strong_report = strong.update(&input).unwrap();
    assert!(strong_report.metrics["kl_new_behavior"] < weak_report.metrics["kl_new_behavior"]);
}

#[test]
fn malformed_stale_zero_occupancy_and_corrupt_restore_are_atomic() {
    let mut configuration = config(2);
    configuration.weighting = MdpoWeighting::DiscountedOccupancy;
    let mut learner = learner(89, configuration);
    let zero = batch(&learner, 0.).select_windows(&[1]).unwrap();
    let before = canonical_record(&learner.checkpoint().unwrap());
    assert!(learner.update(&zero).is_err());
    assert_eq!(before, canonical_record(&learner.checkpoint().unwrap()));
    let input = batch(&learner, 0.9);
    let mut malformed = input.clone();
    malformed.rows[0].discount_weight = 0.;
    assert!(learner.update(&malformed).is_err());
    assert_eq!(before, canonical_record(&learner.checkpoint().unwrap()));
    learner.update(&input).unwrap();
    let before = canonical_record(&learner.checkpoint().unwrap());
    assert!(learner.update(&input).is_err());
    assert_eq!(before, canonical_record(&learner.checkpoint().unwrap()));
    let saved = learner.checkpoint().unwrap();
    for corrupt in ["counter", "owner", "config"] {
        let mut bad = saved.clone();
        match corrupt {
            "counter" => bad.metadata["outer_iterations"] = 999.into(),
            "owner" => {
                bad.optimizers
                    .insert("policy".into(), bad.optimizers["value"].clone());
            }
            _ => bad.metadata["config"]["policy"]["learning_rate"] = (-1.).into(),
        };
        assert!(learner.restore(&bad).is_err());
        assert_eq!(before, canonical_record(&learner.checkpoint().unwrap()));
    }
}

#[test]
fn unsafe_finite_gradients_require_explicit_checkpoint_recovery_before_reuse() {
    let mut learner = learner(109, config(2));
    let input = batch(&learner, 0.9);
    let saved = learner.checkpoint().unwrap();
    let before = weights(&learner.policy());
    let mut unsafe_input = input.clone();
    for row in &mut unsafe_input.rows {
        row.advantage = Some(1e30);
    }
    assert!(learner.update(&unsafe_input).is_err());
    assert_eq!(
        weights(&learner.policy()),
        before,
        "safe-f32 gradient guard must run before the first Adam mutation"
    );
    assert!(learner.behavior_actor(&(), learner.revision()).is_err());
    assert!(learner.checkpoint().is_err());
    assert!(learner.update(&input).is_err());
    learner.restore(&saved).unwrap();
    assert_eq!(
        canonical_record(&saved),
        canonical_record(&learner.checkpoint().unwrap())
    );
    assert!(learner.update(&input).is_ok());
}

#[test]
fn full_run_restore_repeats_future_actions_memories_schedule_optimizers_and_counters() {
    type Trainer = TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>;
    let build = |seed| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
        RecurrentMdpo::new(
            TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
            TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
            config(2),
            seed + 1,
        )
        .unwrap()
    };
    let mut learner = build(97);
    let environments = SyncVectorEnv::new(vec![
        DelayedBit::new(7, Some(3), false, 17).unwrap(),
        DelayedBit::new(9, Some(4), false, 19).unwrap(),
    ])
    .unwrap();
    let mut trainer = Trainer::new(
        environments,
        &learner,
        TemporalOnPolicyConfig {
            collection: OnPolicyConfig {
                rollout_steps: 3,
                gamma: 0.9,
                gae_lambda: Some(0.95),
                normalize_advantages: false,
                episode_guard: None,
            },
            sequence: SequenceConfig {
                burn_in: 0,
                learn_len: 2,
                n_step: 1,
            },
            ..Default::default()
        },
        101,
    )
    .unwrap();
    trainer.cycle(&mut learner).unwrap();
    trainer.collect_tick(&learner).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    save_run(&path, &learner, &trainer).unwrap();
    let mut restored = build(103);
    let mut other: Trainer = load_run(&path, &mut restored).unwrap();
    for _ in 0..4 {
        trainer.collect_tick(&learner).unwrap();
        other.collect_tick(&restored).unwrap();
        let a = trainer.update(&mut learner).unwrap();
        let b = other.update(&mut restored).unwrap();
        assert_eq!(a.metrics, b.metrics);
        assert_eq!(trainer.counters(), other.counters());
        assert_eq!(
            canonical_record(&learner.checkpoint().unwrap()),
            canonical_record(&restored.checkpoint().unwrap())
        );
        for lane in 0..2 {
            assert_eq!(
                trainer.collector().lane_state(lane),
                other.collector().lane_state(lane)
            );
            assert_eq!(
                trainer.collector().observations()[lane],
                other.collector().observations()[lane]
            );
        }
    }
}

#[test]
fn exact_mdpo_reuses_full_and_selected_histories_with_bitwise_resume() {
    type Trainer = TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>;
    let build = |seed| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
        let mut config = config(3);
        config.value_minibatch_size = 6;
        RecurrentMdpo::new(
            TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
            TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
            config,
            seed + 1,
        )
        .unwrap()
    };
    let trainer = |learner: &RecurrentMdpo| {
        Trainer::new(
            SyncVectorEnv::new(vec![
                DelayedBit::new(37, Some(3), false, 17).unwrap(),
                DelayedBit::new(39, Some(4), false, 19).unwrap(),
            ])
            .unwrap(),
            learner,
            TemporalOnPolicyConfig {
                collection: OnPolicyConfig {
                    rollout_steps: 6,
                    gamma: 0.9,
                    gae_lambda: Some(0.95),
                    normalize_advantages: false,
                    episode_guard: None,
                },
                sequence: SequenceConfig {
                    burn_in: 2,
                    learn_len: 2,
                    n_step: 1,
                },
                ..Default::default()
            },
            101,
        )
        .unwrap()
    };
    let mut reference = build(97);
    let mut optimized = build(103);
    optimized.restore(&reference.checkpoint().unwrap()).unwrap();
    optimized.set_update_mode(TemporalExecutionMode::OptimizedExact);
    let mut left = trainer(&reference);
    let mut right = trainer(&optimized);
    for cycle in 0..5 {
        let expected = left.cycle(&mut reference).unwrap();
        let actual = right.cycle(&mut optimized).unwrap();
        assert_eq!(expected.updates, actual.updates);
        assert!(expected.metrics.keys().eq(actual.metrics.keys()));
        for (name, number) in expected.metrics {
            assert_eq!(number.to_bits(), actual.metrics[&name].to_bits(), "{name}");
        }
        assert_eq!(left.counters(), right.counters());
        let mut saved = optimized.checkpoint().unwrap();
        let mode = saved
            .metadata
            .as_object_mut()
            .unwrap()
            .remove("update_mode")
            .unwrap();
        assert_eq!(
            canonical_record(&reference.checkpoint().unwrap()),
            canonical_record(&saved)
        );
        for lane in 0..2 {
            assert_eq!(
                serde_json::to_vec(&left.collector().lane_state(lane)).unwrap(),
                serde_json::to_vec(&right.collector().lane_state(lane)).unwrap()
            );
            assert_eq!(
                left.collector().observations()[lane],
                right.collector().observations()[lane]
            );
        }
        if cycle == 2 {
            assert_eq!(mode, serde_json::json!("OptimizedExact"));
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("optimized");
            save_run(&path, &optimized, &right).unwrap();
            optimized = build(107);
            right = load_run(&path, &mut optimized).unwrap();
            assert_eq!(
                optimized.update_mode(),
                TemporalExecutionMode::OptimizedExact
            );
        }
    }
}
