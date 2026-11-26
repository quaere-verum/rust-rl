#![cfg(feature = "neural")]

//! Historical policy-evaluation fixtures with neutral critic ownership.
//! Expected values below are literal hand calculations, not a second target helper.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::recurrent_critic::{
        CriticBackup, RecurrentCriticConfig, RecurrentDiscreteCritic, construct_policy_targets,
    },
    core::{
        action::{Action, ActionSpace},
        checkpoint::{Checkpoint, LearnerRecord},
        env::Observation,
        stateful::HistorySpec,
        temporal::BehaviorMetadata,
    },
    data::temporal::{SequenceBatch, SequenceConfig, SequenceOrigin, SequenceSample, TemporalStep},
    nn::{
        network::Mlp,
        temporal::{GruEncoder, RecurrentQ},
    },
};
use std::collections::BTreeMap;

// Burn-pack maps do not have a canonical byte order. Compare every decoded
// parameter ID, tensor bit, optimizer moment, scalar and metadata value.
fn canonical(record: LearnerRecord) -> serde_json::Value {
    fn pack(bytes: Vec<u8>) -> serde_json::Value {
        let reader =
            burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes)).unwrap();
        let scalars = reader
            .scalars()
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect::<BTreeMap<_, _>>();
        let metadata = reader
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect::<BTreeMap<_, _>>();
        let tensors = reader.into_tensors().unwrap().into_iter().map(|t| {
            let (name,dtype,shape,id,bytes) = t.into_parts().unwrap();
            (name,serde_json::json!({"dtype":format!("{dtype:?}"),"shape":shape,"id":id,"bytes":bytes}))
        }).collect::<BTreeMap<_,_>>();
        serde_json::json!({"scalars":scalars,"metadata":metadata,"tensors":tensors})
    }
    let mut v = serde_json::to_value(record).unwrap();
    for m in v["models"].as_object_mut().unwrap().values_mut() {
        m["bytes"] = pack(serde_json::from_value(m["bytes"].take()).unwrap());
    }
    for o in v["optimizers"].as_object_mut().unwrap().values_mut() {
        *o = pack(serde_json::from_value(o.take()).unwrap());
    }
    v
}

fn batch(terminal: bool) -> SequenceBatch {
    let observation = |v, mask| Observation::dense(vec![v], Some(mask));
    SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 2,
            n_step: 1,
        },
        samples: vec![SequenceSample {
            episode_id: 7,
            trajectory_start: 0,
            learn_start: 0,
            burn_start: 0,
            observations: vec![
                observation(1., vec![true, true]),
                observation(2., vec![true, false]),
                observation(
                    3.,
                    if terminal {
                        vec![false, false]
                    } else {
                        vec![true, true]
                    },
                ),
            ],
            steps: vec![
                TemporalStep {
                    action: Action::Discrete(1),
                    reward: 0.25_f32.into(),
                    terminated: false,
                    truncated: false,
                    behavior: BehaviorMetadata::default(),
                },
                TemporalStep {
                    action: Action::Discrete(0),
                    reward: 1.0_f32.into(),
                    terminated: terminal,
                    truncated: !terminal,
                    behavior: BehaviorMetadata::default(),
                },
            ],
            loss_mask: vec![true, true],
            history_start: 0,
            previous: None,
            origin: SequenceOrigin::Reset,
        }],
    }
}
fn learner() -> RecurrentDiscreteCritic {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(832);
    let q = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[], 2, &mut rng).unwrap(),
    )
    .unwrap();
    RecurrentDiscreteCritic::new(
        q,
        spec,
        RecurrentCriticConfig {
            gamma: 0.5,
            n_step: 1,
            ..Default::default()
        },
    )
    .unwrap()
}
fn probabilities(terminal: bool) -> Vec<f32> {
    vec![
        0.2,
        0.8,
        1.,
        0.,
        if terminal { 0. } else { 0.25 },
        if terminal { 0. } else { 0.75 },
    ]
}

#[test]
fn neutral_critic_preserves_ddqn_updates_and_legacy_checkpoint_continuation() {
    use rl::algorithms::recurrent_ddqn::{RecurrentDdqn, RecurrentDdqnConfig};
    let legacy = || {
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(832);
        RecurrentDdqn::new(
            RecurrentQ::new(
                GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
                Mlp::new(4, &[], 2, &mut rng).unwrap(),
            )
            .unwrap(),
            spec,
            RecurrentDdqnConfig {
                gamma: 0.5,
                n_step: 1,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let mut ddqn = legacy();
    let mut critic = learner();
    for _ in 0..3 {
        let expected = ddqn.update(&batch(false)).unwrap();
        let actual = critic
            .fit(&batch(false), CriticBackup::DoubleQOptimality)
            .unwrap();
        assert_eq!(actual.metrics, expected.metrics);
        assert_eq!(
            critic.parameter_values().unwrap(),
            ddqn.parameter_values().unwrap()
        );
        assert_eq!(
            critic.reconstructed_q_values(&batch(false)).unwrap(),
            ddqn.reconstructed_q_values(&batch(false)).unwrap()
        );
    }
    let record = ddqn.checkpoint().unwrap();
    assert_eq!(record.algorithm, "recurrent-ddqn-v1");
    assert_eq!(record.metadata["version"], 1);
    assert_eq!(record.metadata["config"]["ddqn"]["epsilon"], 0.1_f32);
    let mut resumed = legacy();
    resumed.restore(&record).unwrap();
    assert_eq!(
        ddqn.update(&batch(true)).unwrap().metrics,
        resumed.update(&batch(true)).unwrap().metrics
    );
    assert_eq!(
        canonical(ddqn.checkpoint().unwrap()),
        canonical(resumed.checkpoint().unwrap())
    );
    // A neutral critic record cannot be silently accepted as a DDQN record.
    assert!(resumed.restore(&critic.checkpoint().unwrap()).is_err());
}

#[test]
fn expected_backup_uses_legal_probability_average_and_actual_truncation_observation() {
    // t0: .25 + .5 * (1*2 + 0*1000) = 1.25.
    // t1: 1 + .5 * (.25*(-2) + .75*6) = 3.
    let target = [0., 0., 2., 1000., -2., 6.];
    let points =
        construct_policy_targets(&batch(false), &probabilities(false), &target, 2, 0.5).unwrap();
    assert_eq!(
        points.iter().map(|p| p.value).collect::<Vec<_>>(),
        [1.25, 3.]
    );
    assert_eq!(
        points
            .iter()
            .map(|p| (p.row, p.timestep, p.action, p.bootstrap_index))
            .collect::<Vec<_>>(),
        [(0, 0, 1, 1), (0, 1, 0, 2)]
    );
    assert_eq!(points[1].discount, 0.5);
}

#[test]
fn terminal_bootstrap_has_no_distribution_and_gamma_zero_keeps_only_reward() {
    let target = [0., 0., 2., 1000., -2., 6.];
    let points =
        construct_policy_targets(&batch(true), &probabilities(true), &target, 2, 0.5).unwrap();
    assert_eq!(
        points.iter().map(|p| p.value).collect::<Vec<_>>(),
        [1.25, 1.]
    );
    assert_eq!(points[1].discount, 0.);
    let points =
        construct_policy_targets(&batch(false), &probabilities(false), &target, 2, 0.).unwrap();
    assert_eq!(
        points.iter().map(|p| p.value).collect::<Vec<_>>(),
        [0.25, 1.]
    );
}

#[test]
fn malformed_target_distributions_and_uncorrected_multistep_are_rejected() {
    let b = batch(false);
    let q = [0.; 6];
    for invalid in [
        vec![0.; 6],
        vec![0.2, 0.8, 0.5, 0.5, 0.25, 0.75],
        vec![0.2, 0.8, 1., 0., -0.1, 1.1],
        vec![0.2, 0.8, 1., 0., 0.5, 0.25],
        vec![0.2, 0.8, 1., 0., f32::NAN, 0.],
        vec![0.; 5],
    ] {
        assert!(construct_policy_targets(&b, &invalid, &q, 2, 0.5).is_err());
    }
    let mut multi = b;
    multi.config.n_step = 2;
    assert!(construct_policy_targets(&multi, &probabilities(false), &q, 2, 0.5).is_err());
    assert!(construct_policy_targets(&batch(false), &probabilities(false), &q, 2, 1.1).is_err());
}

#[test]
fn one_hot_policy_matches_native_double_q_loss_parameters_targets_and_resume() {
    let data = batch(false);
    let mut native = learner();
    let mut expected = learner();
    for _ in 0..3 {
        let (online, _) = native.reconstructed_q_values(&data).unwrap();
        let mut policy = vec![0.; online.len()];
        for t in 0..3 {
            let mask = data.samples[0].observations[t].dense_mask().unwrap();
            let a = (0..2)
                .filter(|&a| mask[a])
                .max_by(|&a, &b| online[2 * t + a].total_cmp(&online[2 * t + b]))
                .unwrap();
            policy[2 * t + a] = 1.;
        }
        let a = native.fit(&data, CriticBackup::DoubleQOptimality).unwrap();
        let b = expected
            .fit(&data, CriticBackup::PolicyEvaluation(&policy))
            .unwrap();
        assert_eq!(a.metrics, b.metrics);
        assert_eq!(
            native.parameter_values().unwrap(),
            expected.parameter_values().unwrap()
        );
    }
    let record = expected.checkpoint().unwrap();
    let mut restored = learner();
    restored.restore(&record).unwrap();
    let a = expected
        .fit(&data, CriticBackup::PolicyEvaluation(&probabilities(false)))
        .unwrap();
    let b = restored
        .fit(&data, CriticBackup::PolicyEvaluation(&probabilities(false)))
        .unwrap();
    assert_eq!(a.metrics, b.metrics);
    assert!(
        canonical(expected.checkpoint().unwrap()) == canonical(restored.checkpoint().unwrap()),
        "decoded model/target/optimizer state differs after restored expected-policy update"
    );
}

#[test]
fn rejected_policy_update_is_atomic() {
    let mut model = learner();
    let before = canonical(model.checkpoint().unwrap());
    assert!(
        model
            .fit(&batch(false), CriticBackup::PolicyEvaluation(&[0.; 6]))
            .is_err()
    );
    assert!(
        before == canonical(model.checkpoint().unwrap()),
        "rejected target input mutated learner"
    );
}

#[test]
fn heterogeneous_padding_and_burn_in_preserve_target_history_indices() {
    let mut data = batch(false);
    let mut short = batch(true).samples.remove(0);
    short.episode_id = 9;
    short.steps.truncate(1);
    short.steps[0].terminated = true;
    short.observations.truncate(2);
    short.observations[1].as_dense_mut().unwrap().action_mask = Some(vec![false, false]);
    short.loss_mask.truncate(1);
    data.samples.push(short);
    let p = [probabilities(false), vec![0.2, 0.8, 0., 0., 0., 0.]].concat();
    let q = [vec![0., 0., 2., 1000., -2., 6.], vec![999.; 6]].concat();
    let points = construct_policy_targets(&data, &p, &q, 2, 0.5).unwrap();
    assert_eq!(
        points.iter().map(|p| p.value).collect::<Vec<_>>(),
        [1.25, 3., 0.25]
    );
    data.samples.truncate(1);
    data.config.burn_in = 1;
    data.config.learn_len = 1;
    data.samples[0].learn_start = 1;
    data.samples[0].loss_mask[0] = false;
    let points = construct_policy_targets(&data, &probabilities(false), &q[..6], 2, 0.5).unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(
        (
            points[0].timestep,
            points[0].bootstrap_index,
            points[0].value
        ),
        (1, 2, 3.)
    );
}
