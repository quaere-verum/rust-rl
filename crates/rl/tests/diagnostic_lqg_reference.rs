#![cfg(feature = "diagnostics")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        env::Observation,
    },
    diagnostics::{
        evaluation::{EvaluationConfig, evaluate},
        reference::{GradientMethod, LqgReference, ReferenceCheckpoint, ReferenceProvider},
    },
    envs::lqg::{Lqg, LqgConfig},
};

fn unbounded(noise_std: f64) -> LqgConfig {
    let mut config = LqgConfig::scalar(2, 1., 1., 1., 1., 2., 1., noise_std, 2.).unwrap();
    config.action_space = ActionSpace::Unbounded { dimensions: 1 };
    config
}

fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} versus {expected}"
    );
}

#[test]
fn discounted_riccati_values_and_gradients_include_final_cost_and_process_noise() {
    let reference = LqgReference::new(&unbounded(0.5), 0.5).unwrap();
    reference.specification().validate().unwrap();
    let actor = reference.actor().unwrap();
    let checkpoint = ReferenceCheckpoint {
        revision: 0,
        policy_id: "fixture",
        policy: actor,
        mode: ActMode::Deterministic,
    };
    let start = Observation::new(vec![1., 0.]);
    let last = Observation::new(vec![1., 0.5]);
    let terminal = Observation::new(vec![1., 1.]);
    let q = |observation, action| {
        reference
            .action_value(&checkpoint, observation, &Action::Continuous(vec![action]))
            .unwrap()
            .unwrap()
    };

    // Last-stage P=5/3, noise cost=1/2. The previous stage has P=16/11,
    // noise cost=11/24 and feedback gain=5/11.
    near(
        reference.state_value(&checkpoint, &start).unwrap().unwrap(),
        -505. / 264.,
        1e-12,
    );
    near(
        reference.state_value(&checkpoint, &last).unwrap().unwrap(),
        -13. / 6.,
        1e-12,
    );
    near(q(&start, 0.5), -43. / 12., 1e-12);
    near(q(&last, 0.), -3.5, 1e-12);
    let gradient = reference
        .action_gradient(&checkpoint, &start, &Action::Continuous(vec![0.5]))
        .unwrap()
        .unwrap();
    assert_eq!(gradient.method, GradientMethod::Analytic);
    near(gradient.values[0], -3.5, 1e-12);
    let delta = 1. / 1024.;
    let difference = (q(&start, 0.5 + delta) - q(&start, 0.5 - delta)) / f64::from(2. * delta);
    near(gradient.values[0], difference, 1e-11);
    assert_eq!(
        reference.state_value(&checkpoint, &terminal).unwrap(),
        Some(0.)
    );
    assert!(
        reference
            .action_value(&checkpoint, &terminal, &Action::Continuous(vec![0.]))
            .unwrap()
            .is_none()
    );

    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let actions = actor
        .act(&[start], ActMode::Deterministic, &mut rng)
        .unwrap();
    let [Action::Continuous(values)] = actions.as_slice() else {
        panic!("continuous action required")
    };
    near(f64::from(values[0]), -5. / 11., 1e-7);
}

#[test]
fn riccati_evaluation_uses_an_independent_actor_and_requires_unbounded_actions() {
    let config = unbounded(0.);
    let reference = LqgReference::new(&config, 0.5).unwrap();
    assert_eq!(
        reference.evaluation_mode(ActMode::Sample),
        ActMode::Deterministic
    );
    let result = evaluate(
        |seed| Lqg::new(config.clone(), seed),
        reference.actor().unwrap(),
        &EvaluationConfig {
            seeds: vec![1, 2],
            discount: 0.5,
            max_steps: 2,
            ..Default::default()
        },
    )
    .unwrap();
    for actual in result.returns {
        near(actual, -16. / 11., 1e-7);
    }
    let bounded = LqgConfig::scalar(2, 1., 1., 1., 1., 2., 1., 0., 2.).unwrap();
    assert!(LqgReference::new(&bounded, 0.5).is_err());
}

#[test]
fn session_records_optimal_decision_regret_at_declared_lqg_coordinates() -> rl::Result<()> {
    use rl::{
        core::{actor::Actor, env::ObservationBatch},
        diagnostics::{
            probes::ProbeSet,
            schema::{ProbePopulation, Record},
            session::{DiagnosticLearner, DiagnosticSession, LearnerInfo},
            storage::{ReadLimits, Recording},
        },
    };
    struct ZeroActor(ActionSpace);
    impl Actor for ZeroActor {
        fn action_space(&self) -> &ActionSpace {
            &self.0
        }
        fn act_into(
            &self,
            observations: ObservationBatch<'_>,
            _: ActMode,
            _: &mut ChaCha8Rng,
            actions: &mut Vec<Action>,
        ) -> rl::Result<()> {
            actions.clear();
            actions.resize_with(observations.len(), || Action::Continuous(vec![0.]));
            Ok(())
        }
    }
    impl DiagnosticLearner for ZeroActor {
        fn diagnostic_info(&self) -> rl::Result<LearnerInfo> {
            Ok(LearnerInfo {
                algorithm: "fixture",
                model: "zero action",
                configuration: serde_json::json!({}),
                discount: Some(0.5),
            })
        }
        fn diagnostic_revision(&self) -> u64 {
            0
        }
        fn diagnostic_evaluation_mode(&self, _: ActMode) -> ActMode {
            ActMode::Deterministic
        }
        fn diagnostic_actor(&self) -> rl::Result<Box<dyn Actor + '_>> {
            Ok(Box::new(ZeroActor(self.0.clone())))
        }
    }
    let config = unbounded(0.);
    let reference = LqgReference::new(&config, 0.5)?;
    let context = reference.specification().context.clone();
    let env = Lqg::new(config.clone(), 4)?;
    let learner = ZeroActor(config.action_space.clone());
    let probes = ProbeSet::new(
        "lifted-input",
        ProbePopulation::FixedReference,
        "Exact coordinate in the LQG model",
        vec![Observation::new(vec![1., 0.])],
        None,
        config.action_space,
    )?;
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("lqg-regret");
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .reference_context(context)
        .probe_set(probes)
        .reference(reference)
        .build(&learner)?;
    session.finish(&learner)?;
    let recording = Recording::open(path, ReadLimits::default())?;
    let mut found = None;
    for index in 0..recording.entries().len() {
        let chunk = recording.load(index)?;
        if let Record::Metrics(metrics) = chunk.record {
            found = found.or_else(|| {
                metrics
                    .values
                    .get("reference.model_coordinate_decision_regret")
                    .copied()
            });
        }
    }
    near(
        found.expect("recorded model-coordinate decision regret"),
        25. / 66.,
        1e-12,
    );
    Ok(())
}
