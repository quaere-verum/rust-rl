use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::prelude::*;
use serde::{Deserialize, Serialize};
/// Episode length varies across resets. Masks remain separate from observation features.
#[derive(Clone, Serialize, Deserialize)]
struct Choice {
    remaining: usize,
    rng: ChaCha8Rng,
    space: ActionSpace,
}
impl Snapshot for Choice {
    fn validate_snapshot(&self) -> Result<()> {
        if self.space != (ActionSpace::Discrete { count: 2 }) || self.remaining > 5 {
            return Err(Error::State("invalid Choice snapshot".into()));
        }
        Ok(())
    }
}
impl Env for Choice {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.remaining = self.rng.random_range(1..=5);
        Ok(EpisodeStart::fresh(Observation::new(vec![
            self.remaining as f32 / 5.,
        ])))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.remaining == 0 {
            return Err(Error::State("reset required".into()));
        }
        self.space.check(action, None)?;
        self.remaining -= 1;
        Ok(Step {
            observation: Observation::new(vec![self.remaining as f32 / 5.]),
            reward: (if *action == Action::Discrete(1) {
                1.
            } else {
                -1.
            })
            .into(),
            terminated: self.remaining == 0,
            truncated: false,
        })
    }
}
fn main() -> Result<()> {
    let env = Choice {
        remaining: 0,
        rng: ChaCha8Rng::seed_from_u64(1),
        space: ActionSpace::Discrete { count: 2 },
    };
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let models = ActorCritic {
        policy: Policy::new(1, env.space.clone(), &[8], &mut rng)?,
        value: Value::new(1, &[8], &mut rng)?,
    };
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            policy: OptimizerConfig {
                learning_rate: 0.003,
                ..Default::default()
            },
            ..Default::default()
        },
        3,
    )?;
    let collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env; 16])?, 4)?;
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 8,
            ..Default::default()
        },
    )?;
    let report = trainer.train(&mut learner, Budget::vector_steps(128), None)?;
    println!(
        "Variable-length custom environment: {} transitions, {} episodes, {} updates",
        report.counters.env_steps, report.counters.episodes, report.counters.updates
    );
    Ok(())
}
