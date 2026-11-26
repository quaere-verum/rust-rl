#![cfg(all(feature = "diagnostics", feature = "neural"))]

use burn::module::Param;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::ddpg::Ddpg,
    core::{
        action::Action,
        env::{Env, Observation},
        learner::Learner,
        transition::Transition,
    },
    data::replay::{ReplayBatch, ReplayItem},
    diagnostics::{
        models::{CriticInspectionRequest, InspectionRequest, ModelBundle, ModelKind, ModelLimits},
        probes::ProbeSet,
        schema::{ProbePopulation, RecordKind},
        session::{DiagnosticConfig, DiagnosticLearner, DiagnosticSession},
        storage::{ReadLimits, Recording},
        value::{ValueOutput, ValuePrediction},
    },
    envs::lqg::{Lqg, LqgConfig},
    nn::{
        critics::{
            residual::{ResidualQ, ResidualScale},
            scalar::{ActionValue, QValue},
        },
        matrix,
        policy::Deterministic,
        to_vec, vector,
    },
    train::counters::RunCounters,
};

#[test]
fn residual_recording_reconstructs_the_transform_for_both_roles_at_each_revision() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("residual-recording");
    let config = LqgConfig::scalar(4, 0.95, 0.5, 1., 0.2, 1., 1., 0.15, 2.).unwrap();
    let env = Lqg::new(config, 18).unwrap();
    let space = env.action_space().clone();
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let policy = Deterministic::new(2, space.clone(), &[], &mut rng).unwrap();
    let mut anchor = Deterministic::new(2, space.clone(), &[], &mut rng).unwrap();
    let mut inner = QValue::new(2, 1, 1, &[], &mut rng).unwrap();
    inner.network.layers[0].weight = Param::from_tensor(matrix(vec![0.3, 0.4, 1.2], 3, 1));
    let critic = ResidualQ::new(inner, &anchor, ResidualScale::Scalar(0.2)).unwrap();
    let mut learner = Ddpg::new(policy, critic, Default::default(), 20).unwrap();
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
                "Two explicit state-action conditions",
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
    learner
        .update(&ReplayBatch {
            items: vec![ReplayItem {
                transition: Transition {
                    observation: observations[0].clone(),
                    action: actions[0].clone(),
                    next_observation: observations[1].clone(),
                    reward: 0.5.into(),
                    terminated: true,
                    truncated: false,
                },
                discount: 0.,
            }],
        })
        .unwrap();
    anchor.network.layers[0].bias = Some(Param::from_tensor(vector(&[1.])));
    learner.refresh_residual_anchor(&anchor).unwrap();
    let updated = [learner.critic(), learner.target_critic()];
    assert_ne!(
        initial[0].anchor_identity().unwrap(),
        updated[0].anchor_identity().unwrap()
    );
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
    let recording = Recording::open(path, ReadLimits::default()).unwrap();
    assert!(recording.complete());
    let action_values = actions
        .iter()
        .map(|action| match action {
            Action::Continuous(values) => values[0],
            _ => unreachable!(),
        })
        .collect::<Vec<_>>();
    let mut covered = [[false; 2]; 2];
    for (index, entry) in recording
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.kind == RecordKind::ModelSnapshot)
    {
        let revision = entry.revision as usize;
        let role = usize::from(entry.model_role == "target");
        let expected = [&initial, &updated][revision].get(role).unwrap();
        let snapshot = recording.load(index).unwrap();
        let restored = ModelBundle::load(&snapshot, ModelLimits::default()).unwrap();
        let ModelKind::ResidualQ { anchor, .. } = &restored.configuration().models["q"].kind else {
            panic!("historical descriptor did not identify the transform");
        };
        assert_eq!(anchor.identity, expected.anchor_identity().unwrap());
        assert_eq!(anchor.scale, vec![0.2]);
        let network = if role == 0 { "policy" } else { "target_policy" };
        assert!(restored.numerical_network(network).is_ok());
        assert!(restored.numerical_network("q").is_err());
        assert!(!restored.activation_networks().contains_key("q"));
        let mut full = InspectionRequest::new(network, observations.clone());
        full.critics = vec!["q".into()];
        full.fixed_actions = Some(actions.clone());
        let prediction = restored.inspect(&full).unwrap();
        let only = restored
            .inspect_critics(&CriticInspectionRequest {
                critics: full.critics,
                observations: observations.clone(),
                fixed_actions: Some(actions.clone()),
            })
            .unwrap();
        let fitted = to_vec(
            expected
                .values(
                    observations.as_slice().into(),
                    matrix(action_values.clone(), 2, 1),
                )
                .unwrap(),
        )
        .unwrap();
        for (row, value) in fitted.into_iter().enumerate() {
            for result in [&prediction.critics["q"], &only.critics["q"]] {
                let actual = ValuePrediction::from_critic(result, row, None)
                    .unwrap()
                    .scalar(&ValueOutput::ExpectedReturn, None)
                    .unwrap();
                assert_eq!(actual, f64::from(value));
            }
            let derivative = restored
                .residual_action_gradient("q", &observations[row], &actions[row])
                .unwrap()
                .unwrap();
            let eval = |a| {
                to_vec(
                    expected
                        .values(
                            std::slice::from_ref(&observations[row]).into(),
                            matrix(vec![a], 1, 1),
                        )
                        .unwrap(),
                )
                .unwrap()[0]
            };
            let a = action_values[row];
            let numerical = f64::from((eval(a + 0.001) - eval(a - 0.001)) / 0.002);
            assert!((derivative.values[0] - numerical).abs() < 0.001);
            if revision == 1 && role == 0 {
                assert_eq!(
                    derivative.values,
                    learner
                        .diagnostic_action_gradient("q", &observations[row], &actions[row])
                        .unwrap()
                        .unwrap()
                        .values,
                );
            }
        }
        covered[revision][role] = true;
    }
    assert_eq!(covered, [[true, true]; 2]);
    assert!(
        recording
            .entries()
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kind == RecordKind::Activations)
            .any(|(index, _)| matches!(recording.load(index).unwrap().record,
            rl::diagnostics::schema::Record::Activations(record) if record.network == "q"))
    );
}
