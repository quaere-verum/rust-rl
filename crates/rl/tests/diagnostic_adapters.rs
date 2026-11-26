#![cfg(all(feature = "diagnostics", feature = "neural"))]

use burn::module::Param;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        models::{InspectionRequest, ModelBundle, ModelLimits},
        probes::ProbeSet,
        schema::*,
        session::{CaptureRequest, DiagnosticLearner},
        value,
    },
    prelude::*,
};

fn constant_head(network: &mut Mlp, outputs: &[f32]) {
    assert_eq!(network.layers.len(), 1);
    let input = network.input_dim();
    network.layers[0].weight = Param::from_tensor(rl::nn::matrix(
        vec![0.; input * outputs.len()],
        input,
        outputs.len(),
    ));
    network.layers[0].bias = Some(Param::from_tensor(rl::nn::vector(outputs)));
}
fn capture(learner: &impl DiagnosticLearner, set: ProbeSet) -> Vec<Chunk> {
    let mut context = RecordContext::new("test", learner.diagnostic_revision());
    context.identity.checkpoint_id = Some(format!("revision-{}", learner.diagnostic_revision()));
    let records = learner
        .diagnostic_records(&CaptureRequest {
            context: &context,
            probes: &[set],
            predictions: true,
            representations: true,
            snapshots: true,
            max_activation_values: 64,
            max_chunk_bytes: 1024 * 1024,
            conventions: None,
            evaluation_mode: rl::core::actor::ActMode::Sample,
            policy_seed: 11,
        })
        .unwrap();
    for chunk in &records {
        chunk.validate().unwrap();
        assert_eq!(
            chunk.context.identity.learner_update_revision,
            learner.diagnostic_revision()
        );
    }
    records
}

fn snapshot(records: &[Chunk], role: &str) -> ModelBundle {
    let snapshot = records
        .iter()
        .find(|c| {
            matches!(c.record, Record::ModelSnapshot(_)) && c.context.identity.model_role == role
        })
        .unwrap();
    ModelBundle::load(snapshot, ModelLimits::default()).unwrap()
}

#[test]
fn ddqn_adapter_records_masked_greedy_choices_and_both_training_networks() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut network = DiscreteQ::new(2, 2, &[], &mut rng).unwrap();
    constant_head(&mut network.network, &[7., 9.]);
    let learner = Ddqn::new(network, Default::default()).unwrap();
    let space = ActionSpace::Discrete { count: 2 };
    let observations = vec![
        Observation::dense(vec![0.1, -0.2], Some(vec![false, true])),
        Observation::dense(vec![0.3, 0.4], Some(vec![true, false])),
    ];
    let set = ProbeSet::new(
        "masked",
        ProbePopulation::FixedReference,
        "test data",
        observations.clone(),
        None,
        space,
    )
    .unwrap();
    assert_eq!(
        learner.diagnostic_evaluation_mode(ActMode::Sample),
        ActMode::Deterministic
    );
    let records = capture(&learner, set);
    let policy = records
        .iter()
        .find(|c| matches!(c.record, Record::Policy(_)))
        .unwrap();
    let Record::Policy(record) = &policy.record else {
        unreachable!()
    };
    assert_eq!(policy.arrays[&record.actions].as_u64().unwrap(), [1, 0]);
    assert_eq!(
        policy.arrays[record.masks.as_ref().unwrap()]
            .as_u8()
            .unwrap(),
        [0, 1, 1, 0]
    );
    assert_eq!(
        learner
            .diagnostic_actor()
            .unwrap()
            .act(&observations, ActMode::Sample, &mut rng)
            .unwrap(),
        [Action::Discrete(1), Action::Discrete(0)]
    );
    let bundle = snapshot(&records, "online");
    assert_eq!(bundle.configuration().models.len(), 1);
    assert_eq!(snapshot(&records, "target").configuration().models.len(), 1);
    let online = records
        .iter()
        .find(|c| {
            matches!(c.record, Record::Critic(_)) && c.context.identity.model_role == "online"
        })
        .unwrap();
    assert_eq!(online.arrays["values"].as_f32().unwrap(), [7., 9., 7., 9.]);
    let mut request = InspectionRequest::new("online_q", observations);
    request.critics.push("online_q".into());
    let saved = bundle.inspect(&request).unwrap();
    assert_eq!(saved.policy.arrays["actions"].as_u64().unwrap(), [1, 0]);
    assert_eq!(
        saved.critics["online_q"].arrays["values"].as_f32().unwrap(),
        [7., 9., 7., 9.]
    );
    assert!(value::declaration(online).unwrap().is_some());
    assert!(records.iter().any(
        |c| matches!(c.record, Record::Critic(_)) && c.context.identity.model_role == "target"
    ));
}

#[test]
fn td3_adapter_pairs_both_twins_and_declares_action_conditioning() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let learner = Td3::new(
        Deterministic::new(2, space.clone(), &[4], &mut rng).unwrap(),
        QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
        QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
        Default::default(),
        11,
    )
    .unwrap();
    let set = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "test data",
        vec![Observation::new(vec![0.1, -0.2])],
        Some(vec![Action::Continuous(vec![0.75])]),
        space,
    )
    .unwrap();
    let records = capture(&learner, set);
    let mut indices = Vec::new();
    for chunk in records
        .iter()
        .filter(|c| matches!(c.record, Record::Critic(_)))
    {
        let Record::Critic(record) = &chunk.record else {
            unreachable!()
        };
        let declaration = value::declaration(chunk).unwrap().unwrap();
        let critic = declaration.critics.values().next().unwrap();
        indices.push((
            chunk.context.identity.model_role.as_str(),
            critic.twin.as_ref().unwrap().index,
        ));
        assert_eq!(record.action_condition, ActionCondition::Fixed);
        assert_eq!(
            chunk.arrays[record.actions.as_ref().unwrap()]
                .as_f32()
                .unwrap(),
            [0.75]
        );
    }
    indices.sort_unstable();
    assert_eq!(
        indices,
        [("online", 0), ("online", 1), ("target", 0), ("target", 1)]
    );
    let bundle = snapshot(&records, "online");
    assert_eq!(bundle.configuration().models.len(), 3);
    assert_eq!(snapshot(&records, "target").configuration().models.len(), 3);
    assert!(records.iter().any(|c| matches!(&c.record, Record::Activations(record) if matches!(&record.inputs, ActivationInputs::ObservationsAndActions { source: ActivationActionSource::FixedProbe, .. }))));
}

#[test]
fn d4pg_adapter_preserves_raw_heads_and_nonuniform_grid_in_historical_inference() {
    use rl::nn::critics::quantile::QuantileGrid;

    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut config = D4pgConfig::new(3).unwrap();
    config.grid = QuantileGrid {
        edges: vec![0., 0.25, 0.5, 1.],
    };
    let mut critic = QValue::new(2, 1, 3, &[], &mut rng).unwrap();
    constant_head(&mut critic.network, &[4., -2., 1.]);
    let learner = D4pg::new(
        Deterministic::new(2, space.clone(), &[4], &mut rng).unwrap(),
        critic,
        config,
        23,
    )
    .unwrap();
    let observations = vec![Observation::new(vec![0.1, -0.2])];
    let fixed_actions = vec![Action::Continuous(vec![0.75])];
    let set = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "test data",
        observations.clone(),
        Some(fixed_actions.clone()),
        space,
    )
    .unwrap();
    let records = capture(&learner, set);
    let actual = records
        .iter()
        .find(|c| {
            matches!(c.record, Record::Critic(_)) && c.context.identity.model_role == "online"
        })
        .unwrap();
    let expected = [4., -2., 1.];
    assert_eq!(actual.arrays["values"].as_f32().unwrap(), expected);
    assert_eq!(
        actual.arrays["locations"].as_f32().unwrap(),
        [0.125, 0.375, 0.75]
    );
    assert_eq!(actual.arrays["masses"].as_f32().unwrap(), [0.25, 0.25, 0.5]);
    let mut request = InspectionRequest::new("policy", observations);
    request.critics.push("quantile_q".into());
    request.fixed_actions = Some(fixed_actions);
    let restored = snapshot(&records, "online").inspect(&request).unwrap();
    assert_eq!(
        restored.critics["quantile_q"].arrays["values"]
            .as_f32()
            .unwrap(),
        expected
    );
}

#[test]
fn ppo_adapter_stores_stochastic_policy_and_state_value_for_the_same_revision() {
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut policy = Policy::new(2, space.clone(), &[], &mut rng).unwrap();
    constant_head(&mut policy.network, &[0.5]);
    let mut value_model = Value::new(2, &[], &mut rng).unwrap();
    constant_head(&mut value_model.network, &[1.25]);
    let learner = Ppo::new(
        ActorCritic {
            policy,
            value: value_model,
        },
        Default::default(),
        37,
    )
    .unwrap();
    let observations = vec![Observation::new(vec![0.1, -0.2])];
    let set = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "test data",
        observations.clone(),
        None,
        space,
    )
    .unwrap();
    assert_eq!(
        learner.diagnostic_evaluation_mode(ActMode::Sample),
        ActMode::Sample
    );
    let records = capture(&learner, set);
    assert!(records.iter().any(|c| matches!(&c.record, Record::Policy(p)
        if p.stochastic && matches!(p.parameters, PolicyParameters::Gaussian { .. }))));
    let recorded = records.iter().find(|c| matches!(&c.record, Record::Critic(record) if record.action_condition == ActionCondition::StateValue)).unwrap();
    assert_eq!(
        value::declaration(recorded).unwrap().unwrap().critics["value"]
            .continuation
            .as_ref()
            .unwrap()
            .identity,
        "test@0"
    );
    assert_eq!(recorded.arrays["values"].as_f32().unwrap(), [1.25]);
    let mut request = InspectionRequest::new("policy", observations);
    request.critics.push("value".into());
    let bundle = snapshot(&records, "online");
    let restored = bundle.inspect(&request).unwrap();
    assert_eq!(restored.policy.arrays["mean"].as_f32().unwrap(), [0.5]);
    assert_eq!(
        restored.critics["value"].arrays["values"].as_f32().unwrap(),
        [1.25]
    );
}
#[test]
fn td3_adapter_reports_analytic_critic_derivatives_and_rejects_relu_kinks() {
    use rl::diagnostics::reference::GradientMethod;

    let mut rng = ChaCha8Rng::seed_from_u64(41);
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut q1 = QValue::new(2, 1, 1, &[], &mut rng).unwrap();
    q1.network.layers[0].weight = Param::from_tensor(rl::nn::matrix(vec![2., -3., 4.], 3, 1));
    let mut q2 = QValue::new(2, 1, 1, &[1], &mut rng).unwrap();
    q2.network.layers[0].weight = Param::from_tensor(rl::nn::matrix(vec![0., 0., 1.], 3, 1));
    q2.network.layers[1].weight = Param::from_tensor(rl::nn::matrix(vec![2.], 1, 1));
    let learner = Td3::new(
        Deterministic::new(2, space, &[4], &mut rng).unwrap(),
        q1,
        q2,
        Default::default(),
        43,
    )
    .unwrap();
    let observation = Observation::new(vec![0.1, -0.2]);
    let action = Action::Continuous(vec![0.75]);
    let q1 = learner
        .diagnostic_action_gradient("q1", &observation, &action)
        .unwrap()
        .unwrap();
    assert_eq!(q1.values, [4.]);
    assert!(matches!(q1.method, GradientMethod::Analytic));
    let q2 = learner
        .diagnostic_action_gradient("q2", &observation, &action)
        .unwrap()
        .unwrap();
    assert_eq!(q2.values, [2.]);
    assert!(
        learner
            .diagnostic_action_gradient("q2", &observation, &Action::Continuous(vec![0.]))
            .unwrap()
            .is_none()
    );
}
#[test]
fn td3_conditional_actor_matches_target_smoothing_without_training_exploration() {
    let mut rng = ChaCha8Rng::seed_from_u64(51);
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let mut config = Td3Config::default();
    config.training.target_noise = 100.;
    config.training.noise_clip = 0.2;
    let learner = Td3::new(
        Deterministic::new(2, space, &[4], &mut rng).unwrap(),
        QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
        QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
        config,
        53,
    )
    .unwrap();
    let observations = vec![Observation::new(vec![0., 0.]); 8];
    let deployed = learner
        .diagnostic_actor()
        .unwrap()
        .act(&observations, ActMode::Sample, &mut rng)
        .unwrap();
    assert_eq!(deployed, vec![Action::Continuous(vec![0.]); 8]);
    let (target, continuation) = learner
        .diagnostic_conditional_actor(&RecordContext::new("test", 0))
        .unwrap()
        .unwrap();
    assert!(continuation.stochastic);
    let smoothed = target
        .act(&observations, ActMode::Sample, &mut rng)
        .unwrap();
    assert!(
        smoothed
            .iter()
            .all(|action| matches!(action, Action::Continuous(value) if value[0].abs() <= 0.2))
    );
    assert!(smoothed.contains(&Action::Continuous(vec![0.2])));
    assert!(smoothed.contains(&Action::Continuous(vec![-0.2])));
    assert_eq!(
        target
            .act(&observations, ActMode::Deterministic, &mut rng)
            .unwrap(),
        deployed
    );
}

#[test]
fn md4pg_records_both_frozen_moments_and_preserves_invalid_variance() {
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let space = ActionSpace::Box {
        low: vec![-2., -0.5],
        high: vec![1., 3.],
    };
    let mut first = QValue::new(2, 2, 1, &[], &mut rng).unwrap();
    let mut second = QValue::new(2, 2, 1, &[], &mut rng).unwrap();
    constant_head(&mut first.network, &[2.]);
    constant_head(&mut second.network, &[3.]);
    let learner = Md4pg::new(
        Deterministic::new(2, space.clone(), &[4], &mut rng).unwrap(),
        first,
        second,
        Md4pgConfig {
            risk: rl::objectives::risk::Risk::MeanVariance { penalty: 2. },
            ..Default::default()
        },
        19,
    )
    .unwrap();
    let observations = vec![
        Observation::new(vec![-1., 0.]),
        Observation::new(vec![0., 2.]),
    ];
    let actions = vec![
        Action::Continuous(vec![0., 1.]),
        Action::Continuous(vec![-1., 2.]),
    ];
    let probes = ProbeSet::new(
        "moments",
        ProbePopulation::FixedReference,
        "Two distinct bounded actions",
        observations.clone(),
        Some(actions.clone()),
        space,
    )
    .unwrap();
    let records = capture(&learner, probes);
    for (role, policy) in [("online", "policy"), ("target", "target_policy")] {
        let saved = records
            .iter()
            .find(|chunk| {
                matches!(chunk.record, Record::Critic(_))
                    && chunk.context.identity.model_role == role
            })
            .unwrap();
        let declaration = value::declaration(saved).unwrap().unwrap();
        assert_eq!(
            declaration.critics["moment_q"].utility,
            Some(value::MomentUtility::MeanVariance { penalty: 2. })
        );
        let historical = snapshot(&records, role);
        let mut request = InspectionRequest::new(policy, observations.clone());
        request.fixed_actions = Some(actions.clone());
        request.critics.push("moment_q".into());
        let replayed = historical.inspect(&request).unwrap();
        assert_eq!(
            value::declaration(&replayed.critics["moment_q"])
                .unwrap()
                .as_ref(),
            Some(&declaration),
            "combined inspection retains the frozen target continuation and learner utility"
        );
        for row in 0..observations.len() {
            for chunk in [saved, &replayed.critics["moment_q"]] {
                let prediction = value::ValuePrediction::from_critic(chunk, row, None).unwrap();
                let moments = prediction.moment_summary().unwrap();
                assert_eq!(
                    (moments.mean, moments.raw_second, moments.variance),
                    (2., 3., -1.)
                );
                assert_eq!(moments.standard_deviation, None);
                assert_eq!(
                    prediction
                        .scalar(
                            &value::ValueOutput::Utility,
                            declaration.critics["moment_q"].utility.as_ref()
                        )
                        .unwrap(),
                    2.
                );
            }
        }
    }
}
