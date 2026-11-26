#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::reinforce::{CategoricalExploration, Reinforce, ReinforceConfig},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, EpisodeActor, EpisodeBehavior, RolloutBehavior},
        checkpoint::Checkpoint,
        env::Observation,
    },
    nn::{
        critics::{CriticInitialization, scalar::Value},
        distributions::Distribution,
        matrix,
        policy::{ActorCritic, Policy},
        to_vec,
    },
};

#[test]
fn mixed_categorical_distribution_samples_and_scores_the_same_legal_law() {
    let distribution = Distribution::Categorical {
        logits: matrix(vec![3f32.ln(), 0., 1000., 3f32.ln(), 0., 1000.], 2, 3),
        masks: vec![true, true, false, true, true, false],
    }
    .with_uniform_mixture(&[0.1, 0.])
    .unwrap();
    let probabilities = to_vec(distribution.categorical_terms().unwrap().probabilities).unwrap();
    for (actual, expected) in probabilities.iter().zip([0.725, 0.275, 0., 0.75, 0.25, 0.]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let mut sampled = RolloutBehavior::default();
    distribution
        .sample_behavior_into(
            ActMode::Sample,
            &mut ChaCha8Rng::seed_from_u64(19),
            &mut sampled,
        )
        .unwrap();
    let scores = to_vec(
        distribution
            .log_prob(&sampled.actions, &sampled.latents)
            .unwrap(),
    )
    .unwrap();
    for (score, sampled_score) in scores.iter().zip(&sampled.log_probs) {
        assert!((score - sampled_score).abs() < 1e-6);
    }
    assert!(!sampled.actions.contains(&Action::Discrete(2)));
}

#[test]
fn uniform_exploration_has_zero_actor_score_gradient() {
    let logits = matrix(vec![12., -7.], 1, 2).require_grad();
    let distribution = Distribution::Categorical {
        logits: logits.clone(),
        masks: vec![true, true],
    }
    .with_uniform_mixture(&[1.])
    .unwrap();
    let score = distribution.log_prob(&[Action::Discrete(0)], &[]).unwrap();
    assert_eq!(to_vec(score.clone()).unwrap(), vec![0.5f32.ln()]);
    let gradient = logits.grad(&score.backward()).unwrap();
    assert_eq!(to_vec(gradient).unwrap(), vec![0., 0.]);
}

fn learner() -> Reinforce {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    Reinforce::new(
        ActorCritic {
            policy: Policy::new(1, ActionSpace::Discrete { count: 2 }, &[], &mut rng).unwrap(),
            value: Value::new(1, &[], &mut rng)
                .unwrap()
                .with_initialization(CriticInitialization::Constant { value: 0. })
                .unwrap(),
        },
        ReinforceConfig::default(),
        43,
    )
    .unwrap()
}

#[test]
fn reinforce_restores_root_only_exploration_and_replays_collection() {
    let settings = CategoricalExploration {
        epsilon: 0.1,
        only_at: Some(vec![1.]),
    };
    let configured = learner()
        .with_categorical_exploration(settings.clone())
        .unwrap();
    let mut restored = learner();
    restored.restore(&configured.checkpoint().unwrap()).unwrap();
    assert_eq!(restored.categorical_exploration(), Some(&settings));
    let obs = [Observation::new(vec![1.]), Observation::new(vec![-1.])];
    let sample = |learner: &Reinforce| {
        let mut out = EpisodeBehavior::default();
        learner
            .sample_episode_into(
                obs.as_slice().into(),
                &mut ChaCha8Rng::seed_from_u64(91),
                &mut out,
            )
            .unwrap();
        out.actions
    };
    assert_eq!(sample(&configured), sample(&restored));
    let mut first = ChaCha8Rng::seed_from_u64(97);
    let mut second = first.clone();
    assert_eq!(
        configured
            .act(&obs, ActMode::Deterministic, &mut first)
            .unwrap(),
        learner()
            .act(&obs, ActMode::Deterministic, &mut second)
            .unwrap()
    );
}
