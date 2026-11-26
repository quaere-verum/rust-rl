use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    envs::lqg::{Lqg, LqgConfig, Riccati},
    prelude::*,
    train::trainer::RandomActor,
};
struct Analytic {
    solution: Riccati,
    space: ActionSpace,
    horizon: usize,
}
impl Actor for Analytic {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.resize_with(obs.len(), || Action::Continuous(Vec::new()));
        for (o, slot) in obs.iter().zip(actions) {
            let state: Vec<_> = o.dense_values()[..o.dense_values().len() - 1]
                .iter()
                .map(|x| *x as f64)
                .collect();
            let t = (o.dense_values()[o.dense_values().len() - 1] * self.horizon as f32).round()
                as usize;
            if !matches!(slot, Action::Continuous(_)) {
                *slot = Action::Continuous(Vec::new());
            }
            let Action::Continuous(a) = slot else {
                unreachable!()
            };
            a.clear();
            a.extend(
                self.solution
                    .action(&state, t)?
                    .into_iter()
                    .map(|x| x as f32),
            );
            if let ActionSpace::Box { low, high } = &self.space {
                for (i, a) in a.iter_mut().enumerate() {
                    *a = a.clamp(low[i], high[i]);
                }
            }
        }
        Ok(())
    }
}
fn evaluate(actor: &dyn Actor, config: &LqgConfig, mode: ActMode) -> Result<f64> {
    let mut total = 0.;
    for seed in 100..132 {
        let mut env = Lqg::new(config.clone(), seed)?;
        let mut o = env.reset(Some(seed))?.observation;
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1000);
        for _ in 0..config.horizon() {
            let a = actor.act(&[o], mode, &mut rng)?.remove(0);
            let s = env.step(&a)?;
            total += s.reward.total() as f64;
            o = s.observation;
        }
    }
    Ok(total / 32.)
}
fn main() -> Result<()> {
    let rounds = std::env::args()
        .nth(1)
        .map(|s| s.parse::<u64>().expect("round count"))
        .unwrap_or(80);
    let config = LqgConfig::scalar(16, 0.95, 1., 1., 0.1, 2., 1., 0.05, 2.)?;
    let env = Lqg::new(config.clone(), 0)?;
    let space = env.action_space().clone();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let models = ActorCritic {
        policy: Policy::new(2, space.clone(), &[32, 32], &mut rng)?,
        value: Value::new(2, &[32, 32], &mut rng)?,
    };
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 2,
            minibatch_size: 256,
            policy: OptimizerConfig {
                learning_rate: 1e-3,
                ..Default::default()
            },
            value: OptimizerConfig {
                learning_rate: 1e-3,
                ..Default::default()
            },
            ..Default::default()
        },
        43,
    )?;
    let initial = evaluate(learner.actor(), &config, ActMode::Deterministic)?;
    let collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env; 32])?, 44)?;
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 16,
            ..Default::default()
        },
    )?;
    trainer.train(
        &mut learner,
        Budget::vector_steps(rounds * 16),
        Some(&mut |p| {
            if p.counters.vector_steps.is_multiple_of(160) {
                println!(
                    "transitions={} updates={} collect={:.2}s learn={:.2}s",
                    p.counters.env_steps,
                    p.counters.updates,
                    p.collection_seconds,
                    p.learning_seconds
                );
            }
        }),
    )?;
    let random = RandomActor {
        space: space.clone(),
    };
    let analytic = Analytic {
        solution: config.riccati()?,
        space,
        horizon: 16,
    };
    println!(
        "Mean episode reward (32 fixed evaluation seeds): random={:.3}, initial={initial:.3}, PPO={:.3}, clipped unconstrained Riccati={:.3}",
        evaluate(&random, &config, ActMode::Sample)?,
        evaluate(learner.actor(), &config, ActMode::Deterministic)?,
        evaluate(&analytic, &config, ActMode::Deterministic)?
    );
    Ok(())
}
