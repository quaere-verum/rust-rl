#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{
        ddpg::{Ddpg, DdpgConfig},
        recurrent_critic::{RecurrentCriticConfig, RecurrentDiscreteCritic},
    },
    core::{action::ActionSpace, env::Observation, stateful::HistorySpec},
    nn::{
        critics::{
            CriticInitialization,
            scalar::{ActionValue, DiscreteActionValue, DiscreteQ, QValue},
        },
        matrix,
        network::Mlp,
        policy::Deterministic,
        temporal::{GruEncoder, RecurrentQ},
        to_vec,
    },
};

#[test]
fn a_constant_recurrent_head_remains_trainable_when_the_learner_takes_ownership() {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(35);
    let model = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[], 2, &mut rng).unwrap(),
    )
    .unwrap()
    .with_initialization(CriticInitialization::Constant { value: 0.6 })
    .unwrap();
    assert!(RecurrentDiscreteCritic::new(model, spec, RecurrentCriticConfig::default()).is_ok());
}

#[test]
fn scalar_initialization_is_constant_and_copied_to_the_target() {
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let observations = [Observation::new(vec![1.]), Observation::new(vec![-2.])];
    let actions = matrix(vec![-0.25, 0.9], 2, 1);
    for initial in [0., 0.6, -3.] {
        let actor = Deterministic::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[4], &mut rng)
            .unwrap();
        let q = QValue::new(1, 1, 1, &[4], &mut rng)
            .unwrap()
            .with_initialization(CriticInitialization::Constant { value: initial })
            .unwrap();
        let learner = Ddpg::new(actor, q, DdpgConfig::default(), 32).unwrap();
        for critic in [learner.critic(), learner.target_critic()] {
            let values = critic.values(observations.as_slice().into(), actions.clone());
            assert_eq!(to_vec(values.unwrap()).unwrap(), vec![initial; 2]);
        }
    }
}

#[test]
fn discrete_initialization_covers_every_action_and_existing_preserves_outputs() {
    let mut rng = ChaCha8Rng::seed_from_u64(33);
    let observations = [Observation::dense(vec![2.], Some(vec![true, false, true]))];
    let obs = observations.as_slice().into();
    let q = DiscreteQ::new(1, 3, &[4], &mut rng).unwrap();
    let before = to_vec(q.action_values(obs).unwrap()).unwrap();
    let q = q
        .with_initialization(CriticInitialization::Existing)
        .unwrap();
    assert_eq!(to_vec(q.action_values(obs).unwrap()).unwrap(), before);
    let q = q
        .with_initialization(CriticInitialization::Constant { value: 0.6 })
        .unwrap();
    assert_eq!(to_vec(q.action_values(obs).unwrap()).unwrap(), vec![0.6; 3]);
    assert!(
        q.with_initialization(CriticInitialization::Constant { value: f32::NAN })
            .is_err()
    );
}

#[test]
fn scalar_initialization_rejects_an_undeclared_multichannel_representation() {
    let mut rng = ChaCha8Rng::seed_from_u64(34);
    assert!(
        QValue::new(1, 1, 2, &[4], &mut rng)
            .unwrap()
            .with_initialization(CriticInitialization::Constant { value: 0.6 })
            .is_err()
    );
}
