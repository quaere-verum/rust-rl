#![cfg(all(feature = "diagnostics", feature = "neural"))]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::categorical_dpg::{CategoricalDpg, CategoricalDpgConfig},
    core::{
        action::Action,
        env::{Env, Observation},
        learner::Learner,
        transition::Transition,
    },
    data::replay::{ReplayBatch, ReplayItem},
    diagnostics::{
        analysis::EmpiricalDistribution,
        models::{CriticInspectionRequest, InspectionRequest, ModelBundle, ModelKind, ModelLimits},
        probes::ProbeSet,
        schema::{Array, CriticRepresentation, ProbePopulation, Record, RecordKind},
        session::{DiagnosticConfig, DiagnosticSession},
        storage::{ReadLimits, Recording},
        value::{self, ReturnTarget, ValueOutput, ValuePrediction},
    },
    envs::lqg::{Lqg, LqgConfig},
    nn::{
        critics::scalar::{ActionValue, QValue},
        matrix,
        policy::Deterministic,
        to_vec,
    },
    objectives::categorical::CategoricalSupport,
    train::counters::RunCounters,
};

#[test]
fn categorical_recording_preserves_atoms_and_probabilities_at_each_revision() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("categorical-recording");
    let env = Lqg::new(
        LqgConfig::scalar(4, 0.95, 0.5, 1., 0.2, 1., 1., 0.15, 2.).unwrap(),
        18,
    )
    .unwrap();
    let space = env.action_space().clone();
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let policy = Deterministic::new(2, space.clone(), &[], &mut rng).unwrap();
    let critic = QValue::new(2, 1, 3, &[], &mut rng).unwrap();
    let support = CategoricalSupport::new(vec![-4., -1., 3.]).unwrap();
    let mut learner_config = CategoricalDpgConfig::new(support.clone());
    learner_config.training.target_noise = 0.15;
    let mut control =
        CategoricalDpg::new(policy.clone(), critic.clone(), learner_config.clone(), 20).unwrap();
    let mut learner = CategoricalDpg::new(policy, critic, learner_config, 20).unwrap();
    let observations = vec![
        Observation::new(vec![-0.3, 0.5]),
        Observation::new(vec![0.2, 0.]),
    ];
    let actions = vec![
        Action::Continuous(vec![-0.5]),
        Action::Continuous(vec![0.4]),
    ];
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .config(DiagnosticConfig::detailed())
        .probe_set(
            ProbeSet::new(
                "fixed-actions",
                ProbePopulation::FixedReference,
                "Two state-action conditions",
                observations.clone(),
                Some(actions.clone()),
                space,
            )
            .unwrap(),
        )
        .build(&learner)
        .unwrap();
    session
        .checkpoint(&learner, RunCounters::default())
        .unwrap();
    let initial = [learner.critic(), learner.target_critic()];
    let batch = ReplayBatch {
        items: vec![ReplayItem {
            transition: Transition {
                observation: observations[0].clone(),
                action: actions[0].clone(),
                next_observation: observations[1].clone(),
                reward: 0.5.into(),
                terminated: false,
                truncated: false,
            },
            discount: 0.95,
        }],
    };
    assert_eq!(
        learner.update(&batch).unwrap().metrics,
        control.update(&batch).unwrap().metrics
    );
    let updated = [learner.critic(), learner.target_critic()];
    session
        .checkpoint(
            &learner,
            RunCounters {
                updates: 1,
                ..Default::default()
            },
        )
        .unwrap();
    session.finish(&learner).unwrap();
    assert_eq!(
        learner.update(&batch).unwrap().metrics,
        control.update(&batch).unwrap().metrics
    );
    let recording = Recording::open(path, ReadLimits::default()).unwrap();
    assert!(recording.complete());

    let expected: Vec<_> = [&initial, &updated]
        .into_iter()
        .map(|models| {
            models.each_ref().map(|model| {
                let logits = model
                    .values(
                        observations.as_slice().into(),
                        matrix(vec![-0.5, 0.4], 2, 1),
                    )
                    .unwrap();
                let logits = to_vec(logits).unwrap();
                logits
                    .chunks_exact(3)
                    .map(|row| {
                        let peak = f64::from(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
                        let weights: Vec<_> =
                            row.iter().map(|&x| (f64::from(x) - peak).exp()).collect();
                        let total: f64 = weights.iter().sum();
                        weights.into_iter().map(|p| p / total).collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let assert_prediction = |chunk: &_, revision: usize, role: usize| {
        assert_eq!(
            value::return_target(chunk).unwrap(),
            Some(ReturnTarget::Environmental)
        );
        for (row, expected_row) in expected[revision][role].iter().enumerate() {
            let prediction = ValuePrediction::from_critic(chunk, row, None).unwrap();
            let ValuePrediction::CategoricalReturn {
                atoms,
                probabilities,
            } = &prediction
            else {
                panic!("expected predictive return probabilities");
            };
            assert_eq!(atoms, &[-4_f64, -1., 3.]);
            // Recorded and restored inference use f32 kernels. Compare them
            // with an independent f64 softmax of the frozen model's logits.
            for (&observed, &reference) in probabilities.iter().zip(expected_row) {
                assert!(
                    (observed - reference).abs() < 2e-7,
                    "{observed} != {reference}"
                );
            }
            let total: f64 = probabilities.iter().sum();
            let mean = atoms
                .iter()
                .zip(probabilities)
                .map(|(z, p)| z * p / total)
                .sum::<f64>();
            assert!(
                (prediction
                    .scalar(&ValueOutput::ExpectedReturn, None)
                    .unwrap()
                    - mean)
                    .abs()
                    < 1e-12
            );
        }
    };
    let mut snapshots = [[false; 2]; 2];
    let mut predictions = [[false; 2]; 2];
    for (index, entry) in recording.entries().iter().enumerate() {
        let revision = entry.revision as usize;
        let role = usize::from(entry.model_role == "target");
        match entry.kind {
            RecordKind::ModelSnapshot => {
                let restored =
                    ModelBundle::load(&recording.load(index).unwrap(), ModelLimits::default())
                        .unwrap();
                assert!(
                    matches!(&restored.configuration().models["categorical_return_q"].kind, ModelKind::CategoricalReturnQ { support: actual } if *actual == support)
                );
                let policy = if role == 0 { "policy" } else { "target_policy" };
                let mut full = InspectionRequest::new(policy, observations.clone());
                full.critics = vec!["categorical_return_q".into()];
                full.fixed_actions = Some(actions.clone());
                let all = restored.inspect(&full).unwrap();
                let only = restored
                    .inspect_critics(&CriticInspectionRequest {
                        critics: full.critics,
                        observations: observations.clone(),
                        fixed_actions: Some(actions.clone()),
                    })
                    .unwrap();
                assert_prediction(&all.critics["categorical_return_q"], revision, role);
                assert_prediction(&only.critics["categorical_return_q"], revision, role);
                snapshots[revision][role] = true;
            }
            RecordKind::Critic => {
                let recorded = recording.load(index).unwrap();
                assert_prediction(&recorded, revision, role);
                if revision == 0 && role == 0 {
                    let Record::Critic(record) = &recorded.record else {
                        unreachable!()
                    };
                    let CriticRepresentation::CategoricalReturn { probabilities, .. } =
                        &record.representation
                    else {
                        unreachable!()
                    };
                    let key = probabilities.clone();
                    let mut malformed = recorded;
                    malformed
                        .arrays
                        .insert(key, Array::f32(vec![2, 3], vec![0.; 6]).unwrap());
                    assert!(malformed.validate().is_err());
                }
                predictions[revision][role] = true;
            }
            _ => {}
        }
    }
    assert_eq!(snapshots, [[true; 2]; 2]);
    assert_eq!(predictions, [[true; 2]; 2]);
}

#[test]
fn fixed_support_statistics_use_probability_mass_and_ignore_zero_end_atoms() {
    let prediction = ValuePrediction::CategoricalReturn {
        atoms: vec![-5., -2., 1., 5., 9.],
        probabilities: vec![0., 0.25, 0.5, 0.25, 0.],
    };
    let distribution = prediction.weighted_atoms().unwrap();
    let empirical = EmpiricalDistribution::new(&[-2., 1., 1., 5.]).unwrap();
    assert_eq!(distribution.quantile(0.).unwrap(), -2.);
    assert_eq!(distribution.quantile(0.25).unwrap(), -2.);
    assert_eq!(distribution.quantile(1.).unwrap(), 5.);
    assert_eq!(distribution.mean(), 1.25);
    assert_eq!(distribution.lower_cvar(0.5).unwrap(), -0.5);
    assert_eq!(
        distribution.cdf_points().unwrap(),
        [
            [-2., 0.],
            [-2., 0.25],
            [1., 0.25],
            [1., 0.75],
            [5., 0.75],
            [5., 1.]
        ]
    );
    assert_eq!(
        prediction.scalar(&ValueOutput::Variance, None).unwrap(),
        6.1875
    );
    assert_eq!(distribution.wasserstein1(&empirical).unwrap(), 0.);
    assert_eq!(distribution.crps(1.).unwrap(), 0.4375);
    assert!(prediction.ordered_quantiles().is_err());
}
