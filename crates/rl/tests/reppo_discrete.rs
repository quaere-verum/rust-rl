#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    nn::critics::{categorical::CategoricalQ, scalar::ActionValue},
    nn::{distributions::Distribution, matrix, to_vec},
    objectives::{
        hl_gauss::HlGaussConfig,
        reppo::{categorical_objective, self_predictive_loss},
    },
    prelude::*,
};

#[test]
fn masked_categorical_forward_kl_has_the_required_direction_and_support() {
    let old = Distribution::Categorical {
        logits: matrix(vec![0.8_f32.ln(), 0.2_f32.ln(), 1000.], 1, 3),
        masks: vec![true, true, false],
    };
    let logits = matrix(vec![0.3_f32.ln(), 0.7_f32.ln(), -1000.], 1, 3).require_grad();
    let current = Distribution::Categorical {
        logits: logits.clone(),
        masks: vec![true, true, false],
    };
    assert!(old.kl(&old).unwrap().into_scalar::<f32>().abs() < 1e-7);
    let kl = old.kl(&current).unwrap();
    let expected = 0.8 * (0.8_f32 / 0.3).ln() + 0.2 * (0.2_f32 / 0.7).ln();
    assert!((kl.clone().into_scalar::<f32>() - expected).abs() < 1e-6);
    let grads = kl.sum().backward();
    assert_eq!(to_vec(logits.grad(&grads).unwrap()).unwrap()[2], 0.);
    let empty = Distribution::Categorical {
        logits: matrix(vec![0.; 3], 1, 3),
        masks: vec![false; 3],
    };
    assert!(empty.categorical_terms().is_err());
}

#[test]
fn exact_masked_expected_q_gradient_matches_two_action_bandit() {
    let logits = matrix(vec![4_f32.ln(), 0., 1000.], 1, 3).require_grad();
    let distribution = Distribution::Categorical {
        logits: logits.clone(),
        masks: vec![true, true, false],
    };
    let terms = distribution.categorical_terms().unwrap();
    let q = matrix(vec![1., 3., f32::NAN], 1, 3).require_grad();
    let loss = categorical_objective(terms.probabilities, terms.log_probs, q.clone(), 0.7)
        .unwrap()
        .mean();
    let expected = -1.4 + 0.7 * (0.8 * 0.8_f32.ln() + 0.2 * 0.2_f32.ln());
    assert!((loss.clone().into_scalar::<f32>() - expected).abs() < 1e-6);
    let gradients = loss.backward();
    let gradient = to_vec(logits.grad(&gradients).unwrap()).unwrap();
    let expected_gradient = 0.32 + 0.7 * 0.16 * 4_f32.ln();
    assert!((gradient[0] - expected_gradient).abs() < 1e-6);
    assert!((gradient[1] + expected_gradient).abs() < 1e-6);
    assert_eq!(gradient[2], 0.);
    assert!(q.grad(&gradients).is_none());
}

#[derive(Clone)]
struct Bandit {
    singleton: bool,
    space: ActionSpace,
}
impl Bandit {
    fn observation(&self, terminal: bool) -> Observation {
        Observation::dense(
            vec![1.],
            Some(if terminal {
                vec![false; 3]
            } else if self.singleton {
                vec![false, true, false]
            } else {
                vec![true, true, false]
            }),
        )
    }
}
impl Env for Bandit {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        Ok(EpisodeStart::fresh(self.observation(false)))
    }
    fn step(&mut self, a: &Action) -> Result<Step> {
        self.space.check(a, self.observation(false).dense_mask())?;
        Ok(Step {
            observation: self.observation(true),
            reward: (if *a == Action::Discrete(1) {
                1.0_f32
            } else {
                -1.0_f32
            })
            .into(),
            terminated: true,
            truncated: false,
        })
    }
}
fn build(
    singleton: bool,
    auxiliary: f32,
) -> (
    Reppo<Policy, CategoricalQ>,
    OnPolicyTrainer<SyncVectorEnv<Bandit>>,
) {
    let env = Bandit {
        singleton,
        space: ActionSpace::Discrete { count: 3 },
    };
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let policy = Policy::new(1, env.space.clone(), &[8], &mut rng).unwrap();
    let config = ReppoConfig {
        epochs: 2,
        minibatch_size: 4,
        entropy_target: if singleton { 100. } else { 0.3 },
        auxiliary_coefficient: auxiliary,
        hl_gauss: HlGaussConfig {
            min: -2.,
            max: 2.,
            bins: 11,
            sigma_ratio: 0.75,
        },
        ..Default::default()
    };
    let critic = CategoricalQ::new(1, env.space.clone(), 11, &[8, 8], &mut rng).unwrap();
    let learner = Reppo::new(policy, critic, config, 5).unwrap();
    let collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 6).unwrap();
    let trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 8,
            ..Default::default()
        },
    )
    .unwrap();
    (learner, trainer)
}

#[test]
fn discrete_reppo_updates_with_terminal_empty_masks_and_standard_trainer() {
    use rl::nn::policy::StochasticPolicy;
    let (mut learner, mut trainer) = build(false, 0.);
    let obs = vec![trainer.collector().environment().environments()[0].observation(false)];
    let before = to_vec(
        learner
            .policy()
            .distribution((&obs).into())
            .unwrap()
            .log_prob(&[Action::Discrete(1)], &[None])
            .unwrap(),
    )
    .unwrap();
    let summary = trainer
        .train(&mut learner, Budget::vector_steps(8), None)
        .unwrap();
    let after = to_vec(
        learner
            .policy()
            .distribution((&obs).into())
            .unwrap()
            .log_prob(&[Action::Discrete(1)], &[None])
            .unwrap(),
    )
    .unwrap();
    assert_ne!(before, after);
    assert_eq!(summary.counters.env_steps, 8);
    assert!(
        summary
            .last_update
            .unwrap()
            .metrics
            .values()
            .all(|x| x.is_finite())
    );
}

#[test]
fn singleton_masks_have_finite_losses_and_feasible_entropy_duals() {
    let (mut learner, mut trainer) = build(true, 0.1);
    let alpha = learner.multipliers().0;
    let summary = trainer
        .train(&mut learner, Budget::vector_steps(8), None)
        .unwrap();
    let metrics = summary.last_update.unwrap().metrics;
    assert!(metrics.values().all(|x| x.is_finite()));
    assert_eq!(metrics["entropy"], 0.);
    assert_eq!(
        metrics["auxiliary_loss"], 0.,
        "terminal transitions have no auxiliary continuation"
    );
    assert!((learner.multipliers().0 - alpha).abs() < 1e-7);
}

#[test]
fn normalized_categorical_critic_is_seeded_action_conditioned_and_differentiable() {
    let mut rng = ChaCha8Rng::seed_from_u64(91);
    let mut same = rng.clone();
    let a = CategoricalQ::new(
        1,
        ActionSpace::Unbounded { dimensions: 1 },
        11,
        &[8, 8],
        &mut rng,
    )
    .unwrap();
    let b = CategoricalQ::new(
        1,
        ActionSpace::Unbounded { dimensions: 1 },
        11,
        &[8, 8],
        &mut same,
    )
    .unwrap();
    let obs = vec![Observation::new(vec![0.4])];
    let input = matrix(vec![0.2], 1, 1).require_grad();
    let qa = a.values((&obs).into(), input.clone()).unwrap();
    let qb = b.values((&obs).into(), matrix(vec![0.2], 1, 1)).unwrap();
    assert_eq!(to_vec(qa.clone()).unwrap(), to_vec(qb).unwrap());
    let support = HlGaussConfig {
        min: -2.,
        max: 2.,
        bins: 11,
        sigma_ratio: 0.75,
    };
    let grads = support.decode(qa).unwrap().sum().backward();
    assert!(input.grad(&grads).unwrap().into_scalar::<f32>().abs() > 1e-6);
    assert_eq!(
        a.representation((&obs).into(), matrix(vec![0.2], 1, 1))
            .unwrap()
            .dims(),
        [1, 8]
    );
    assert!(
        CategoricalQ::new(
            1,
            ActionSpace::Unbounded { dimensions: 1 },
            11,
            &[1],
            &mut rng
        )
        .is_err()
    );
}

#[test]
fn auxiliary_target_and_episode_boundary_gradients_are_detached() {
    let prediction = matrix(vec![1., 2., 3., 4.], 2, 2).require_grad();
    let target = matrix(vec![0.; 4], 2, 2).require_grad();
    let valid = rl::nn::vector(&[1., 0.]).require_grad();
    let loss = self_predictive_loss(prediction.clone(), target.clone(), valid.clone()).unwrap();
    assert_eq!(loss.clone().into_scalar::<f32>(), 1.25);
    let grads = loss.backward();
    assert_eq!(
        to_vec(prediction.grad(&grads).unwrap()).unwrap(),
        [0.5, 1., 0., 0.]
    );
    assert!(target.grad(&grads).is_none());
    assert!(valid.grad(&grads).is_none());
}

#[test]
fn auxiliary_learning_runs_on_continuations_and_normalized_checkpoint_continues() {
    use rl::envs::lqg::{Lqg, LqgConfig};
    let env = Lqg::new(
        LqgConfig::scalar(5, 0.9, 1., 1., 0.1, 1., 1., 0.05, 2.).unwrap(),
        1,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let cfg = ReppoConfig {
        epochs: 2,
        minibatch_size: 3,
        auxiliary_coefficient: 0.1,
        hl_gauss: HlGaussConfig {
            min: -50.,
            max: 10.,
            bins: 21,
            sigma_ratio: 0.75,
        },
        ..Default::default()
    };
    let policy = Policy::new(2, env.action_space().clone(), &[8], &mut rng).unwrap();
    let critic = CategoricalQ::new(2, env.action_space().clone(), 21, &[8, 8], &mut rng).unwrap();
    let mut learner = Reppo::new(policy.clone(), critic.clone(), cfg.clone(), 5).unwrap();
    let collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 6).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let summary = trainer
        .train(&mut learner, Budget::vector_steps(3), None)
        .unwrap();
    assert!(summary.last_update.unwrap().metrics["auxiliary_loss"] > 0.);
    let mut restored = Reppo::new(policy, critic, cfg, 99).unwrap();
    restored.restore(&learner.checkpoint().unwrap()).unwrap();
    let mut rollout = RolloutBuffer::default();
    let mut c = trainer.collector().clone();
    let mut d = c.clone();
    c.collect_pathwise_into(&learner, 3, &mut rollout).unwrap();
    let batch = rollout.to_batch(0.99, None, false).unwrap();
    d.collect_pathwise_into(&restored, 3, &mut rollout).unwrap();
    let resumed_batch = rollout.to_batch(0.99, None, false).unwrap();
    assert_eq!(
        serde_json::to_value(&batch).unwrap(),
        serde_json::to_value(&resumed_batch).unwrap()
    );
    let a = learner.update(&batch).unwrap();
    let b = restored.update(&resumed_batch).unwrap();
    assert_eq!(a.metrics, b.metrics);
    assert_eq!(learner.multipliers(), restored.multipliers());
}
