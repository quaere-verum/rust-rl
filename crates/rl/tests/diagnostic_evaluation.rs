#![cfg(feature = "diagnostics")]

#[path = "support/riccati_actor.rs"]
mod riccati_actor;
use riccati_actor::LqgRiccatiActor;
use rl::{
    core::{action::Action, env::Env},
    diagnostics::{analysis::*, evaluation::*},
    envs::lqg::{Lqg, LqgConfig},
    train::trainer::RandomActor,
};

#[test]
fn empirical_tail_integral_histogram_and_inverse_cdf_match_hand_calculation() {
    let empirical = EmpiricalDistribution::new(&[4., 1., 3., 2.]).unwrap();
    assert_eq!(empirical.mean(), 2.5);
    assert_eq!(empirical.quantile(0.5).unwrap(), 2.);
    assert_eq!(empirical.cdf(2.), 0.5);
    assert!((empirical.lower_cvar(0.375).unwrap() - 4. / 3.).abs() < 1e-12);
    let histogram = empirical.histogram(2).unwrap();
    assert_eq!(histogram.edges, [1., 2.5, 4.]);
    assert_eq!(histogram.counts, [2, 2]);
    assert!(empirical.lower_cvar(0.).is_err());
    assert!(EmpiricalDistribution::new(&[f64::NAN]).is_err());
}

#[test]
fn quantile_scores_use_probability_masses_and_preserve_crossing_outputs() {
    let forecast =
        QuantileForecast::new(vec![-1., 1.], vec![0.125, 0.625], vec![0.25, 0.75]).unwrap();
    assert_eq!(forecast.mean(), 0.5);
    assert!((forecast.lower_cvar(0.5).unwrap()).abs() < 1e-12);
    assert!((forecast.crps(0.).unwrap() - 0.625).abs() < 1e-12);
    let empirical = EmpiricalDistribution::new(&[0.]).unwrap();
    assert_eq!(forecast.wasserstein1(&empirical).unwrap(), 1.);
    assert_eq!(forecast.coverage(&empirical), vec![0., 1.]);
    let crossed = QuantileForecast::new(vec![1., -1.], vec![0.25, 0.75], vec![0.5, 0.5]).unwrap();
    assert_eq!(crossed.values(), &[1., -1.]);
    assert_eq!(crossed.crossing_count(), 1);
    assert!(crossed.crps(0.).is_err());
    assert!(crossed.lower_cvar(0.5).is_err());
    assert!(moment_consistency(3., 8., 1e-8).unwrap().is_some());
    assert!(moment_consistency(3., 13., 1e-8).unwrap().is_none());
    let three = EmpiricalDistribution::new(&[-2., 0., 2.]).unwrap();
    assert!((forecast.mean_crps(&three).unwrap() - 3.875 / 3.).abs() < 1e-12);
    assert!((forecast.wasserstein1(&three).unwrap() - 7. / 6.).abs() < 1e-12);
}

#[test]
fn ordered_quantiles_use_full_row_with_probability_grid_fixed() {
    let source = vec![4., -2., 1.];
    let locations = vec![0.1, 0.4, 0.8];
    let masses = vec![0.2, 0.3, 0.5];
    let raw = QuantileForecast::new(source.clone(), locations.clone(), masses.clone()).unwrap();
    let ordered =
        QuantileForecast::ordered(source.clone(), locations.clone(), masses.clone()).unwrap();
    assert_eq!(source, [4., -2., 1.]);
    assert_eq!(ordered.values(), [-2., 1., 4.]);
    assert_eq!(ordered.locations(), locations);
    assert_eq!(ordered.masses(), masses);
    assert_eq!(ordered.crossing_count(), 0);
    assert_eq!(ordered.source_crossing_count(), 1);
    let close = |actual: f64, expected: f64| {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}")
    };
    close(raw.mean(), 0.7);
    close(ordered.mean(), 1.9);
    close(ordered.lower_cvar(0.4).unwrap(), -0.5);
    assert_eq!(
        ordered.cdf_points().unwrap(),
        [
            [-2., 0.],
            [-2., 0.2],
            [1., 0.2],
            [1., 0.5],
            [4., 0.5],
            [4., 1.]
        ]
    );
    let empirical = EmpiricalDistribution::new(&[-3., 0., 2., 5.]).unwrap();
    assert_eq!(ordered.coverage(&empirical), [0.25, 0.5, 0.75]);
    assert_eq!(ordered.strict_coverage(&empirical), [0.25, 0.5, 0.75]);
    close(ordered.coverage_interval_error(&empirical), 0.1);
    close(ordered.crps(0.).unwrap(), 1.47);
    close(ordered.mean_crps(&empirical).unwrap(), 1.97);
    close(ordered.wasserstein1(&empirical).unwrap(), 1.4);
    let second =
        QuantileForecast::ordered(vec![5., 0., -1.], locations.clone(), masses.clone()).unwrap();
    assert_eq!(second.values(), [-1., 0., 5.]);
    assert_eq!(second.source_crossing_count(), 2);
    let ties =
        QuantileForecast::ordered(vec![1., -2., 1.], locations.clone(), masses.clone()).unwrap();
    assert_eq!(ties.values(), [-2., 1., 1.]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            QuantileForecast::ordered(vec![bad, 0., 1.], locations.clone(), masses.clone())
                .is_err()
        );
        assert!(
            QuantileForecast::ordered(source.clone(), vec![0.1, bad, 0.8], masses.clone()).is_err()
        );
        assert!(
            QuantileForecast::ordered(source.clone(), locations.clone(), vec![0.2, bad, 0.5])
                .is_err()
        );
    }
    assert!(
        QuantileForecast::ordered(source.clone(), vec![0.1, 0.1, 0.8], masses.clone()).is_err()
    );
    assert!(QuantileForecast::ordered(vec![], vec![], vec![]).is_err());
    assert!(QuantileForecast::ordered(source.clone(), vec![0.1, 0.4], masses.clone()).is_err());
    assert!(
        QuantileForecast::ordered(source.clone(), vec![0.4, 0.1, 0.8], masses.clone()).is_err()
    );
    assert!(QuantileForecast::ordered(source.clone(), vec![0., 0.4, 0.8], masses.clone()).is_err());
    assert!(QuantileForecast::ordered(source, locations, vec![0.2, 0.2, 0.2]).is_err());
}

#[test]
fn atomic_quantiles_are_calibrated_when_nominal_probability_lies_between_strict_and_weak_coverage()
{
    let constant = EmpiricalDistribution::new(&[0.; 20]).unwrap();
    let forecast =
        QuantileForecast::new(vec![0.; 3], vec![0.1, 0.5, 0.9], vec![0.2, 0.6, 0.2]).unwrap();
    assert_eq!(forecast.coverage_intervals(&constant), vec![[0., 1.]; 3]);
    assert_eq!(forecast.coverage_interval_error(&constant), 0.);
    let binary = EmpiricalDistribution::new(&[0., 0., 0., 1.]).unwrap();
    let correct = QuantileForecast::new(vec![0., 1.], vec![0.5, 0.9], vec![0.75, 0.25]).unwrap();
    assert_eq!(
        correct.coverage_intervals(&binary),
        vec![[0., 0.75], [0.75, 1.]]
    );
    assert_eq!(correct.coverage_interval_error(&binary), 0.);
    let wrong = QuantileForecast::new(vec![-1., 0.], vec![0.5, 0.9], vec![0.5, 0.5]).unwrap();
    assert!((wrong.coverage_interval_error(&binary) - 0.325).abs() < 1e-12);
}

#[test]
fn stochastic_policy_and_environment_draws_use_independent_domains_even_for_seed_zero() {
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};
    use rl::core::{
        action::ActionSpace,
        actor::{ActMode, Actor},
        env::ObservationBatch,
    };
    struct Gaussian(ActionSpace);
    impl Actor for Gaussian {
        fn action_space(&self) -> &ActionSpace {
            &self.0
        }
        fn act_into(
            &self,
            obs: ObservationBatch<'_>,
            _: ActMode,
            rng: &mut ChaCha8Rng,
            out: &mut Vec<Action>,
        ) -> rl::Result<()> {
            for _ in 0..obs.len() {
                out.push(Action::Continuous(vec![StandardNormal.sample(rng)]));
            }
            Ok(())
        }
    }
    let mut config = LqgConfig::scalar(1, 0., 0., 0., 1., 1., 0., 1., 3.).unwrap();
    config.action_space = ActionSpace::Unbounded { dimensions: 1 };
    let mut env = Lqg::new(config, 0).unwrap();
    env.reset(Some(0)).unwrap();
    let actor = Gaussian(env.action_space().clone());
    let config = EvaluationConfig {
        seeds: (0..2048).collect(),
        mode: ActMode::Sample,
        policy_seed: 0,
        trajectories: 2048,
        max_steps: 1,
        ..Default::default()
    };
    let result = evaluate_conditional(&env, &actor, None, &config).unwrap();
    let mut cross = 0.;
    for (i, t) in result.trajectories.iter().enumerate() {
        let Action::Continuous(a) = &t.actions[0] else {
            panic!()
        };
        let noise = t.observations[1].dense_values()[0];
        if i == 0 {
            assert_ne!(
                a[0], noise,
                "policy must not replay the environment's first normal draw"
            );
        }
        cross += a[0] as f64 * noise as f64;
    }
    assert!((cross / 2048.).abs() < 0.08);
    let repeat = evaluate_conditional(&env, &actor, None, &config).unwrap();
    assert_eq!(
        result.trajectories[0].actions,
        repeat.trajectories[0].actions
    );
}

#[test]
fn riccati_accepts_long_horizon_normalized_f32_decision_clocks() {
    use rand::SeedableRng;
    use rl::core::{
        actor::{ActMode, Actor},
        env::Observation,
    };
    let config = LqgConfig::scalar(10_000, 1., 1., 1., 1., 1., 0., 0., 3.).unwrap();
    let actor = LqgRiccatiActor::new(config, true).unwrap();
    let input = Observation::new(vec![1., 5001. / 10_000.]);
    let actions = actor
        .act(
            &[input],
            ActMode::Deterministic,
            &mut rand_chacha::ChaCha8Rng::seed_from_u64(0),
        )
        .unwrap();
    assert_eq!(actions.len(), 1);
}

fn stochastic_lqg() -> Lqg {
    // G = -Z^2 for fixed u=0 at x=0, Z standard normal. This is not Gaussian.
    let mut env = Lqg::new(
        LqgConfig::scalar(1, 1., 1., 0., 1., 1., 0., 1., 3.).unwrap(),
        9,
    )
    .unwrap();
    env.reset(Some(9)).unwrap();
    env
}

#[cfg(feature = "neural")]
#[test]
fn lqg_conditional_branches_reseed_future_noise_preserve_source_and_calibrate_known_quantiles() {
    use burn::module::Param;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::nn::{
        critics::{quantile::QuantileGrid, scalar::QValue},
        matrix,
        probes::{CriticPrediction, CriticProbe, QuantileProbe},
        to_vec, vector,
    };
    let env = stochastic_lqg();
    let before = serde_json::to_value(&env).unwrap();
    let actor = RandomActor {
        space: env.action_space().clone(),
    };
    let config = EvaluationConfig {
        seeds: (0..4096).collect(),
        policy_seed: 7,
        max_steps: 4,
        trajectories: 2,
        discount: 1.,
        ..Default::default()
    };
    let outcome =
        evaluate_conditional(&env, &actor, Some(&Action::Continuous(vec![0.])), &config).unwrap();
    let empirical = EmpiricalDistribution::new(&outcome.returns).unwrap();
    assert!((empirical.mean() + 1.).abs() < 0.08);
    assert!(
        outcome
            .returns
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count()
            > 4000
    );
    assert_eq!(before, serde_json::to_value(&env).unwrap());
    assert_eq!(outcome.trajectories.len(), 2);
    assert_eq!(outcome.trajectories[0].observations.len(), 2);
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let mut critic = QValue::new(2, 1, 3, &[], &mut rng).unwrap();
    critic.network.layers[0].weight = Param::from_tensor(matrix(vec![0.; 9], 3, 3));
    critic.network.layers[0].bias = Some(Param::from_tensor(vector(&[
        -2.7055435,
        -0.45493642,
        -0.015790775,
    ])));
    let grid = QuantileGrid {
        edges: vec![0., 0.2, 0.8, 1.],
    };
    let observation = env.branch_observation().unwrap();
    let CriticPrediction::Quantiles {
        values,
        locations,
        masses,
    } = QuantileProbe::new(&critic, &grid)
        .unwrap()
        .predict((&[observation]).into(), matrix(vec![0.], 1, 1))
        .unwrap()
    else {
        panic!()
    };
    let forecast = QuantileForecast::new(
        to_vec(values).unwrap().into_iter().map(f64::from).collect(),
        locations.into_iter().map(f64::from).collect(),
        masses.into_iter().map(f64::from).collect(),
    )
    .unwrap();
    for (coverage, target) in forecast.coverage(&empirical).iter().zip([0.1, 0.5, 0.9]) {
        assert!((coverage - target).abs() < 0.025);
    }
    assert!(forecast.mean_crps(&empirical).unwrap().is_finite());
    println!(
        "LQG G=-Z^2: n={} mean={:.6} lower_cvar_0.1={:.6} coverage={:?} three_atom_crps={:.6} three_atom_w1={:.6} elapsed={:.4}s",
        outcome.returns.len(),
        empirical.mean(),
        empirical.lower_cvar(0.1).unwrap(),
        forecast.coverage(&empirical),
        forecast.mean_crps(&empirical).unwrap(),
        forecast.wasserstein1(&empirical).unwrap(),
        outcome.elapsed_seconds
    );
    let repeat =
        evaluate_conditional(&env, &actor, Some(&Action::Continuous(vec![0.])), &config).unwrap();
    assert_eq!(outcome.returns, repeat.returns);
    let alternate =
        evaluate_conditional(&env, &actor, Some(&Action::Continuous(vec![1.])), &config).unwrap();
    assert!(
        (EmpiricalDistribution::new(&alternate.returns)
            .unwrap()
            .mean()
            + 3.)
            .abs()
            < 0.15
    );
}

#[test]
fn evaluation_rejects_partial_returns_and_keeps_trajectory_discount_and_masks() {
    let config = LqgConfig::scalar(2, 1., 0., 1., 1., 1., 1., 0., 2.).unwrap();
    let actor = RandomActor {
        space: config.action_space.clone(),
    };
    let evaluation = EvaluationConfig {
        seeds: vec![2, 3],
        policy_seed: 6,
        discount: 0.5,
        max_steps: 3,
        trajectories: 1,
        ..Default::default()
    };
    let result = evaluate(|seed| Lqg::new(config.clone(), seed), &actor, &evaluation).unwrap();
    // Deterministic RandomActor chooses zero: rewards -1, -2 (terminal cost).
    assert_eq!(result.returns, [-2., -2.]);
    assert_eq!(result.trajectories[0].rewards, [-1., -2.]);
    assert!(
        evaluate(
            |seed| Lqg::new(config.clone(), seed),
            &actor,
            &EvaluationConfig {
                max_steps: 1,
                ..evaluation.clone()
            }
        )
        .is_err()
    );
    assert!(
        evaluate(
            |seed| Lqg::new(config.clone(), seed),
            &actor,
            &EvaluationConfig {
                seeds: vec![2, 2],
                ..evaluation
            }
        )
        .is_err()
    );
}

#[cfg(feature = "neural")]
#[test]
fn conditional_comparison_refuses_mismatched_action_policy_state_discount_and_prefix_returns() {
    use rl::{
        core::actor::ActMode,
        diagnostics::{probes, schema::*, storage::*},
        nn::{probes::CriticPrediction, vector},
    };
    let env = stochastic_lqg();
    let actor = RandomActor {
        space: env.action_space().clone(),
    };
    let evaluation = EvaluationConfig {
        seeds: vec![31, 32],
        max_steps: 2,
        trajectories: 1,
        discount: 1.,
        policy_id: "fixed-policy-k0-deterministic".into(),
        ..Default::default()
    };
    let mut context = RecordContext::new("conditional", 0);
    context.probe_set = Some("state".into());
    let set = probes::ProbeSet::new(
        "state",
        ProbePopulation::FixedReference,
        "exact initial LQG state",
        vec![env.branch_observation().unwrap()],
        Some(vec![Action::Continuous(vec![0.])]),
        env.action_space().clone(),
    )
    .unwrap();
    let probe = set.record(context.clone()).unwrap();
    let prediction = probes::critic_for_policy(
        context.clone(),
        CriticPrediction::Scalar(vector(&[-1.])),
        ActionCondition::Fixed,
        set.fixed_actions(),
        &evaluation.policy_id,
        ActMode::Deterministic,
    )
    .unwrap();
    let result = evaluate_conditional_with_frames(
        &env,
        &actor,
        Some(&Action::Continuous(vec![0.])),
        &evaluation,
        |_| {},
    )
    .unwrap();
    let empirical = result.empirical_record(context.clone(), Some(0)).unwrap();
    let manifest = RunManifest::new(
        "conditional",
        "fixture",
        "known-critic",
        2,
        env.action_space().clone(),
        1.,
    )
    .unwrap();
    assert_eq!(
        match_conditional(&prediction, &probe, &empirical, &manifest)
            .unwrap()
            .probe_row,
        0
    );
    let mut wrong_role = empirical.clone();
    wrong_role.context.identity.model_role = "target".into();
    assert!(match_conditional(&prediction, &probe, &wrong_role, &manifest).is_err());
    let mut tagged_prediction = prediction.clone();
    tagged_prediction.context.identity.checkpoint_id = Some("checkpoint-a".into());
    let mut wrong_checkpoint = empirical.clone();
    wrong_checkpoint.context.identity.checkpoint_id = Some("checkpoint-b".into());
    assert!(match_conditional(&tagged_prediction, &probe, &wrong_checkpoint, &manifest).is_err());
    for field in ["initial_action", "conditioning_observation"] {
        let mut wrong = empirical.clone();
        let array = &wrong.arrays[field];
        wrong.arrays.insert(
            field.into(),
            Array::f32(array.shape().into(), vec![1.; array.len()]).unwrap(),
        );
        assert!(match_conditional(&prediction, &probe, &wrong, &manifest).is_err());
    }
    for reason in 0..4 {
        let mut wrong = empirical.clone();
        let Record::Empirical(record) = &mut wrong.record else {
            panic!()
        };
        match reason {
            0 => record.evaluation.discount = 0.5,
            1 => record.evaluation.continuation_policy = "other".into(),
            2 => record.evaluation.complete_returns = false,
            _ => record.evaluation.stochastic_policy = true,
        }
        assert!(match_conditional(&prediction, &probe, &wrong, &manifest).is_err());
    }
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("conditional");
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    for data in [
        probe,
        prediction,
        empirical,
        result
            .trajectory_record(0, context, "sample-31", "lqg")
            .unwrap(),
    ] {
        writer.append(&data).unwrap();
    }
    writer.finish().unwrap();
    let reader = Recording::open(path, ReadLimits::default()).unwrap();
    assert_eq!(reader.entries().len(), 4);
    let data = reader.load(3).unwrap();
    assert_eq!(data.arrays["simulation_times"].as_f64().unwrap(), &[0., 1.]);
    assert_eq!(data.arrays["frames/state"].shape(), &[2, 1]);
}

#[test]
fn analytic_values_include_noise_and_do_not_claim_clipped_optimality() {
    let noisy = stochastic_lqg();
    let analytic = LqgRiccatiActor::new(noisy.config().clone(), false).unwrap();
    assert_eq!(analytic.unconstrained_value(&[0.], 0).unwrap(), -1.);
    assert_eq!(
        analytic
            .unconstrained_action_value(&[0.], 0, &[1.])
            .unwrap(),
        -3.
    );
    assert_eq!(analytic.unconstrained_value(&[1.], 1).unwrap(), 0.);
    let config = LqgConfig::scalar(1, 1., 1., 1., 1., 1., 1., 0., 0.1).unwrap();
    let clipped = LqgRiccatiActor::new(config.clone(), true).unwrap();
    assert_eq!(clipped.unconstrained_value(&[1.], 0).unwrap(), -1.5);
    let evaluation = EvaluationConfig {
        seeds: vec![1],
        max_steps: 2,
        discount: 1.,
        ..Default::default()
    };
    let result = evaluate(|seed| Lqg::new(config.clone(), seed), &clipped, &evaluation).unwrap();
    assert!((result.returns[0] + 1.82).abs() < 1e-6);
    let unconstrained = LqgRiccatiActor::new(config.clone(), false).unwrap();
    assert!(
        evaluate(
            |seed| Lqg::new(config.clone(), seed),
            &unconstrained,
            &evaluation
        )
        .is_err()
    );
}

#[test]
fn external_time_limits_are_rejected_or_explicitly_recorded_as_partial_returns() {
    use rl::core::env::TimeLimit;
    let config = LqgConfig::scalar(3, 1., 0., 1., 1., 1., 1., 0., 2.).unwrap();
    let actor = RandomActor {
        space: config.action_space.clone(),
    };
    let mut evaluation = EvaluationConfig {
        seeds: vec![1, 2],
        discount: 1.,
        max_steps: 4,
        trajectories: 1,
        ..Default::default()
    };
    assert!(
        evaluate(
            |seed| TimeLimit::new(Lqg::new(config.clone(), seed)?, 1),
            &actor,
            &evaluation
        )
        .is_err()
    );
    evaluation.allow_truncated_prefix = true;
    let result = evaluate(
        |seed| TimeLimit::new(Lqg::new(config.clone(), seed)?, 1),
        &actor,
        &evaluation,
    )
    .unwrap();
    assert!(!result.metadata.complete_returns);
    assert_eq!(result.returns, [-1., -1.]);
    assert_eq!(result.trajectories[0].truncated, [true]);
}

#[test]
fn evaluation_and_recorded_suffix_discount_each_primitive_reward() {
    use rl::{
        Result,
        core::{
            action::ActionSpace,
            env::{EpisodeStart, Observation, ObservationSpace, Step},
            reward::Reward,
        },
        diagnostics::{
            schema::{RecordContext, RunManifest},
            storage::{ReadLimits, Recorder, Recording, StorageConfig},
            value::trajectory_return_to_go,
        },
    };

    struct ExtendedActions {
        space: ActionSpace,
        decision: usize,
    }

    impl Env for ExtendedActions {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn observation_space(&self) -> ObservationSpace {
            ObservationSpace::Dense(1)
        }

        fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
            self.decision = 0;
            Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
        }

        fn step(&mut self, _: &Action) -> Result<Step> {
            let rewards: [&[f32]; 2] = [&[2., 4., 8.], &[-4., 2.]];
            let reward = Reward::from_primitives(rewards[self.decision].to_vec())?;
            self.decision += 1;
            Ok(Step {
                observation: Observation::new(vec![self.decision as f32]),
                reward,
                terminated: self.decision == 2,
                truncated: false,
            })
        }
    }

    let space = ActionSpace::Discrete { count: 1 };
    let actor = RandomActor {
        space: space.clone(),
    };
    let result = evaluate(
        |_| {
            Ok(ExtendedActions {
                space: space.clone(),
                decision: 0,
            })
        },
        &actor,
        &EvaluationConfig {
            seeds: vec![1],
            discount: 0.5,
            max_steps: 2,
            trajectories: 1,
            ..Default::default()
        },
    )
    .unwrap();

    // The primitive trace is [2, 4, 8, -4, 2], split after the first three ticks.
    assert_eq!(result.returns, [5.625]);
    assert_eq!(result.environment_steps, 2);
    assert_eq!(result.physical_steps, 5);
    assert_eq!(result.trajectories[0].rewards, [14., -2.]);
    let trajectory = result
        .trajectory_record(0, RecordContext::new("eval", 0), "episode", "extended")
        .unwrap();
    let root = tempfile::tempdir().unwrap();
    let run = root.path().join("eval");
    let manifest = RunManifest::new("eval", "test", "fixture", 1, space, 0.5).unwrap();
    let mut recorder = Recorder::create(&run, manifest, StorageConfig::default()).unwrap();
    recorder.append(&trajectory).unwrap();
    recorder.finish().unwrap();
    let recording = Recording::open(run, ReadLimits::default()).unwrap();
    let reopened = recording.load(0).unwrap();
    assert_eq!(trajectory_return_to_go(&reopened, 0).unwrap(), Some(5.625));
    assert_eq!(trajectory_return_to_go(&reopened, 1).unwrap(), Some(-3.));
}

#[test]
fn truncated_scenario_does_not_invalidate_another_episode_trajectory() {
    use rl::{
        core::env::TimeLimit,
        diagnostics::{schema::RecordContext, value::trajectory_return_to_go},
    };

    let config = LqgConfig::scalar(2, 1., 0., 1., 1., 1., 1., 0., 2.).unwrap();
    let actor = RandomActor {
        space: config.action_space.clone(),
    };
    let result = evaluate(
        |seed| TimeLimit::new(Lqg::new(config.clone(), seed)?, seed as usize),
        &actor,
        &EvaluationConfig {
            seeds: vec![1, 2],
            discount: 0.5,
            max_steps: 2,
            trajectories: 2,
            allow_truncated_prefix: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!result.metadata.complete_returns);
    let record = |index| {
        result
            .trajectory_record(index, RecordContext::new("eval", 0), "episode", "lqg")
            .unwrap()
    };
    assert_eq!(trajectory_return_to_go(&record(0), 0).unwrap(), None);
    assert_eq!(trajectory_return_to_go(&record(1), 0).unwrap(), Some(-2.));
}

#[test]
fn finite_decision_objective_finishes_exactly_at_its_declared_boundary() {
    use rl::{
        core::env::TimeLimit,
        diagnostics::{
            schema::RecordContext,
            value::{trajectory_return_suffixes, trajectory_return_to_go},
        },
    };

    let config = LqgConfig::scalar(3, 1., 0., 1., 1., 1., 1., 0., 2.).unwrap();
    let actor = RandomActor {
        space: config.action_space.clone(),
    };
    let evaluation = EvaluationConfig {
        seeds: vec![1],
        discount: 0.5,
        max_steps: 2,
        trajectories: 1,
        ..Default::default()
    };
    let horizon = EvaluationHorizon::FiniteDecisions(2);
    let result = evaluate_with_horizon(
        |seed| TimeLimit::new(Lqg::new(config.clone(), seed)?, 2),
        &actor,
        &evaluation,
        horizon,
    )
    .unwrap();

    assert_eq!(result.returns, [-1.5]);
    assert!(result.metadata.complete_returns);
    assert_eq!(result.trajectories[0].terminated, [false, false]);
    assert_eq!(result.trajectories[0].truncated, [false, true]);
    let trajectory = result
        .trajectory_record(0, RecordContext::new("eval", 0), "episode", "lqg")
        .unwrap();
    let samples = result
        .empirical_record(RecordContext::new("eval", 0), None)
        .unwrap();
    samples.validate().unwrap();
    assert_eq!(evaluation_horizon(&samples).unwrap(), horizon);
    assert_eq!(evaluation_horizon(&trajectory).unwrap(), horizon);
    assert_eq!(trajectory_return_to_go(&trajectory, 0).unwrap(), None);
    assert_eq!(
        trajectory_return_suffixes(&trajectory).unwrap(),
        Some(vec![-1.5, -1.])
    );
    assert!(
        evaluate_with_horizon(
            |seed| TimeLimit::new(Lqg::new(config.clone(), seed)?, 1),
            &actor,
            &evaluation,
            horizon,
        )
        .is_err()
    );
}

#[test]
fn authored_observation_sampler_checks_every_scenario_before_scoring() {
    use rl::{
        core::env::Observation,
        diagnostics::schema::{Conditioning, RecordContext},
    };

    let config = LqgConfig::scalar(2, 1., 0., 1., 1., 1., 1., 0., 2.).unwrap();
    let actor = RandomActor {
        space: config.action_space.clone(),
    };
    let evaluation = EvaluationConfig {
        seeds: vec![3, 4],
        discount: 0.5,
        max_steps: 2,
        trajectories: 0,
        ..Default::default()
    };
    let observation = Observation::new(vec![1., 0.]);
    let first_action = Action::Continuous(vec![1.]);
    let methodology =
        "LQG reset is deterministic given known initial state; each seed sets future noise";
    let start = |seed| {
        let mut env = Lqg::new(config.clone(), seed)?;
        let observation = env.reset(Some(seed))?.observation;
        Ok((env, observation))
    };
    let result = evaluate_observation_conditioned(
        start,
        &observation,
        Some(&first_action),
        &actor,
        &evaluation,
        methodology,
    )
    .unwrap();

    assert_eq!(result.returns, [-3., -3.]);
    assert_eq!(result.initial_action.as_ref(), Some(&first_action));
    assert_eq!(result.conditioning_observation.as_ref(), Some(&observation));
    assert_eq!(result.simulator_state, None);
    assert!(result.metadata.complete_returns && result.metadata.independent_future_streams);
    assert_eq!(
        result.metadata.conditioning,
        Conditioning::Observation {
            methodology: methodology.into()
        },
    );
    result
        .empirical_record(RecordContext::new("eval", 0), Some(0))
        .unwrap()
        .validate()
        .unwrap();

    assert!(
        evaluate_observation_conditioned(
            |seed| {
                let (env, mut observation) = start(seed)?;
                if seed == 4 {
                    observation = Observation::new(vec![0., 0.]);
                }
                Ok((env, observation))
            },
            &observation,
            Some(&first_action),
            &actor,
            &evaluation,
            methodology,
        )
        .is_err()
    );
}
