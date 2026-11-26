//! Reusable bounded proof: ordinary Env, no viewer/provider/branching/frame traits.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        evaluation::{EvaluationConfig, StochasticActor, evaluate},
        observer::DiagnosticObserver,
        probes::{self, ProbeSet},
        schedule::*,
        schema::*,
        storage::*,
    },
    prelude::*,
};
pub const SEED: u64 = 7301;
pub const RUN_ID: &str = "generic-counterbalance-seed7301";
#[path = "categorical_demo.rs"]
mod categorical_demo;

pub fn record_categorical_demo(path: &std::path::Path) -> Result<()> {
    categorical_demo::record(path)
}

struct CounterBalance {
    space: ActionSpace,
    position: f32,
    step: usize,
}
impl CounterBalance {
    fn new() -> Self {
        Self {
            space: ActionSpace::Discrete { count: 2 },
            position: 0.,
            step: 0,
        }
    }
    fn observation(&self) -> Observation {
        Observation::new(vec![self.position, self.step as f32 / 4.])
    }
}
impl Env for CounterBalance {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(2)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        self.position = if seed.unwrap_or(SEED).is_multiple_of(2) {
            -0.5
        } else {
            0.5
        };
        self.step = 0;
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<rl::core::env::Step> {
        if self.step == 4 {
            return Err(Error::State("reset required".into()));
        }
        let Action::Discrete(choice) = action else {
            return Err(Error::Config("discrete action required".into()));
        };
        if *choice > 1 {
            return Err(Error::Config("action out of bounds".into()));
        }
        self.position += if *choice == 0 { -0.25 } else { 0.25 };
        self.step += 1;
        Ok(rl::core::env::Step {
            observation: self.observation(),
            reward: (-self.position.abs()).into(),
            terminated: self.step == 4,
            truncated: false,
        })
    }
}
/// Three real PPO updates, twelve training steps, six four-step evaluations.
pub fn record(path: &std::path::Path) -> Result<()> {
    let space = ActionSpace::Discrete { count: 2 };
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut learner = Ppo::new(
        ActorCritic {
            policy: Policy::new(2, space.clone(), &[4], &mut rng)?,
            value: Value::new(2, &[4], &mut rng)?,
        },
        PpoConfig {
            epochs: 1,
            minibatch_size: 4,
            ..Default::default()
        },
        SEED + 1,
    )?;
    let mut trainer = OnPolicyTrainer::new(
        OnPolicyCollector::new(SyncVectorEnv::new(vec![CounterBalance::new()])?, SEED + 2)?,
        OnPolicyConfig {
            rollout_steps: 4,
            ..Default::default()
        },
    )?;
    let probes = ProbeSet::new(
        "ordinary-inputs",
        ProbePopulation::FixedReference,
        "three fixed CounterBalance observations",
        vec![
            Observation::new(vec![-0.5, 0.]),
            Observation::new(vec![0., 0.5]),
            Observation::new(vec![0.5, 0.]),
        ],
        None,
        space.clone(),
    )?;
    let recorder = Recorder::create(
        path,
        RunManifest::new(RUN_ID, "ordinary custom Env proof", "PPO", 2, space, 0.99)?,
        StorageConfig::default(),
    )?;
    let mut probe_rng = ChaCha8Rng::seed_from_u64(SEED + 3);
    let mut observer = DiagnosticObserver::new(
        recorder,
        RecordingSchedule {
            metrics: Some(Interval { first: 1, every: 1 }),
            probes: Some(Interval { first: 1, every: 1 }),
            monte_carlo: None,
            snapshots: None,
        },
        |learner: &Ppo, context: &RecordContext, _: Due, writer: &mut Recorder| {
            writer.append(&probes.record(context.clone())?)?;
            let policy = learner.policy();
            writer.append(&probes::policy(
                context.clone(),
                &policy,
                &probes,
                ActMode::Sample,
                &mut probe_rng,
            )?)?;
            let result = evaluate(
                |_| Ok(CounterBalance::new()),
                &StochasticActor::new(&policy),
                &EvaluationConfig {
                    seeds: vec![101, 102],
                    policy_seed: SEED + 4,
                    max_steps: 4,
                    trajectories: 2,
                    policy_id: "recorded PPO policy".into(),
                    ..Default::default()
                },
            )?;
            writer.append(&result.empirical_record(context.clone(), None)?)?;
            for i in 0..2 {
                writer.append(&result.trajectory_record(
                    i,
                    context.clone(),
                    &format!(
                        "evaluation-{}-{i}",
                        context.identity.learner_update_revision
                    ),
                    "CounterBalance",
                )?)?;
            }
            println!(
                "revision={} training_budget=12 evaluation_episodes={} evaluation_steps={} seed={SEED}",
                context.identity.learner_update_revision,
                result.returns.len(),
                result.environment_steps
            );
            Ok(())
        },
    );
    trainer.train_observed(&mut learner, Budget::vector_steps(12), None, &mut observer)?;
    observer.finish()?;
    Ok(())
}
