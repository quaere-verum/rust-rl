//! Complete recurrent MPO updates over the standard replay sampler and trainer.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::profiling;
use rl::{
    Error, Result,
    algorithms::{
        recurrent_critic::{RecurrentCriticConfig, RecurrentDiscreteCritic},
        recurrent_mpo::{MpoBackup, RecurrentMpo, RecurrentMpoConfig},
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        stateful::{BehaviorPublisher, HistorySpec},
        temporal::TemporalExecutionMode,
    },
    data::temporal::SequenceConfig,
    envs::delayed_bit::DelayedBit,
    nn::{
        network::Mlp,
        temporal::{GruEncoder, RecurrentQ},
        temporal_categorical_actor::CategoricalMixtureSettings,
        temporal_policy::TemporalPolicy,
    },
    train::{
        behavior::FixedBehavior, recurrent::TemporalTrainer,
        temporal_config::TemporalOffPolicyConfig,
    },
};
use serde::Serialize;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

const USAGE: &str = "recurrent_mpo_allocations HORIZON WINDOW BATCH LANES REPLAY_EPISODES_PER_LANE UPDATES ACTIONS PROJECTION_KL SEED REPEATS evaluation|optimality [--probes] [--optimized]";

// Extra categories are legal only at the decision and alternate between the
// two answers. The task still reveals the cue only at reset.
struct DelayedChoice {
    inner: DelayedBit,
    space: ActionSpace,
    decision: bool,
}

impl DelayedChoice {
    fn expand_mask(&mut self, mut observation: Observation) -> Observation {
        let ActionSpace::Discrete { count } = &self.space else {
            unreachable!()
        };
        let mask = observation
            .as_dense_mut()
            .unwrap()
            .action_mask
            .as_mut()
            .expect("DelayedBit mask");
        self.decision = mask[1];
        mask.resize(*count, self.decision);
        observation
    }
}

impl Env for DelayedChoice {
    fn observation_space(&self) -> ObservationSpace {
        self.inner.observation_space()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        let mut start = self.inner.reset(seed)?;
        start.observation = self.expand_mask(start.observation);
        Ok(start)
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.space.check(action, None)?;
        let Action::Discrete(action) = *action else {
            unreachable!()
        };
        let bit = if self.decision {
            action % 2
        } else {
            // DelayedBit rejects action 1 before the decision.
            usize::from(action != 0)
        };
        let mut step = self.inner.step(&Action::Discrete(bit))?;
        step.observation = self.expand_mask(step.observation);
        Ok(step)
    }
}

#[derive(Serialize)]
struct Workload {
    horizon: usize,
    window: usize,
    batch: usize,
    lanes: usize,
    replay_episodes: usize,
    updates: usize,
    actions: usize,
    projection_kl: f64,
    seed: u64,
    repeats: usize,
    backup: MpoBackup,
    update_mode: TemporalExecutionMode,
    probes: bool,
}

impl Workload {
    fn parse() -> Result<Self> {
        fn read<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>) -> Result<T> {
            args.next()
                .and_then(|arg| arg.parse().ok())
                .ok_or_else(|| Error::Config(USAGE.into()))
        }
        let mut args = std::env::args().skip(1);
        let mut workload = Self {
            horizon: read(&mut args)?,
            window: read(&mut args)?,
            batch: read(&mut args)?,
            lanes: read(&mut args)?,
            replay_episodes: read(&mut args)?,
            updates: read(&mut args)?,
            actions: read(&mut args)?,
            projection_kl: read(&mut args)?,
            seed: read(&mut args)?,
            repeats: read(&mut args)?,
            backup: match args.next().as_deref() {
                Some("evaluation") => MpoBackup::PolicyEvaluation,
                Some("optimality") => MpoBackup::DoubleQOptimality,
                _ => return Err(Error::Config(USAGE.into())),
            },
            update_mode: TemporalExecutionMode::Reference,
            probes: false,
        };
        for flag in args {
            match flag.as_str() {
                "--bench" => {}
                "--probes" if !workload.probes => workload.probes = true,
                "--optimized" if workload.update_mode == TemporalExecutionMode::Reference => {
                    workload.update_mode = TemporalExecutionMode::OptimizedExact;
                }
                _ => return Err(Error::Config(USAGE.into())),
            }
        }
        if !(3..=2048).contains(&workload.horizon)
            || !(1..=workload.horizon.min(64)).contains(&workload.window)
            || !(1..=64).contains(&workload.batch)
            || ![1, 2, 4, 8, 16].contains(&workload.lanes)
            || !(1..=8).contains(&workload.replay_episodes)
            || !(1..=256).contains(&workload.updates)
            || !(2..=32).contains(&workload.actions)
            || !(1..=10).contains(&workload.repeats)
            || !(0. < workload.projection_kl && workload.projection_kl.is_finite())
            || workload.horizon * workload.lanes * workload.replay_episodes > 32_768
            || workload.horizon * workload.batch * workload.updates > 8_388_608
        {
            return Err(Error::Config(
                "MPO benchmark workload exceeds bounds".into(),
            ));
        }
        Ok(workload)
    }

    fn trial(&self, trial: usize) -> Result<serde_json::Value> {
        let space = ActionSpace::Discrete {
            count: self.actions,
        };
        let envs = (0..self.lanes)
            .map(|lane| {
                Ok(DelayedChoice {
                    inner: DelayedBit::new(self.horizon - 1, None, false, self.seed ^ lane as u64)?,
                    space: space.clone(),
                    decision: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let spec = HistorySpec::new(envs[0].observation_space().require_dense_width()?, space)?;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let policy = TemporalPolicy::new(spec.clone(), Some(7), &[], &mut rng)?;
        let critic = RecurrentDiscreteCritic::new(
            RecurrentQ::new(
                GruEncoder::new(spec.input_dim(), 9, &mut rng)?,
                Mlp::new(9, &[], self.actions, &mut rng)?,
            )?,
            spec,
            RecurrentCriticConfig {
                n_step: 1,
                gamma: 1.,
                target_interval: 4,
                ..Default::default()
            },
        )?;
        let mut learner = RecurrentMpo::new(
            policy,
            critic,
            RecurrentMpoConfig {
                backup: self.backup,
                projection_kl: self.projection_kl,
                ..Default::default()
            },
        )?;
        learner.set_update_mode(self.update_mode);
        let warmup_ticks = self.horizon * self.replay_episodes;
        let settings = CategoricalMixtureSettings { epsilon: 0.1 };
        let actor = learner.behavior_actor(&settings, 0)?;
        let mut trainer = TemporalTrainer::new(
            SyncVectorEnv::new(envs)?,
            actor,
            TemporalOffPolicyConfig {
                replay_capacity: (warmup_ticks + self.horizon) * self.lanes,
                batch_size: self.batch,
                warmup_transitions: (warmup_ticks * self.lanes) as u64,
                n_step: 1,
                gamma: 1.,
                updates_per_transition: 1. / self.lanes as f64,
            },
            SequenceConfig {
                burn_in: 2,
                learn_len: self.window,
                n_step: 1,
            },
            FixedBehavior(settings),
            self.seed ^ 0x0043_4f4c_4c45_4354,
        )?;

        let mut warmup_seconds = 0.;
        for tick in 1..=warmup_ticks {
            let start = Instant::now();
            let progress = trainer.step(&mut learner)?;
            warmup_seconds += start.elapsed().as_secs_f64();
            assert!(progress.reports.is_empty());
            if tick.is_multiple_of((warmup_ticks / 4).max(1)) || tick == warmup_ticks {
                eprintln!(
                    "trial={trial}/{} warmup={tick}/{warmup_ticks} elapsed={warmup_seconds:.3}s",
                    self.repeats
                );
            }
        }

        profiling::reset();
        profiling::set_enabled(self.probes);
        let mut wall_seconds = 0.;
        let mut learning_seconds = 0.;
        let mut collection_seconds = 0.;
        let mut metrics = BTreeMap::new();
        let mut projection_attempts = BTreeMap::new();
        let mut episode_groups = 0;
        let mut duplicate_windows = 0;
        let mut min_learn_start_sum = 0;
        for tick in 1..=self.updates {
            let start = Instant::now();
            let progress = trainer.step(&mut learner)?;
            wall_seconds += start.elapsed().as_secs_f64();
            learning_seconds += progress.learning_seconds;
            collection_seconds += progress.collection_seconds;
            assert_eq!(progress.reports.len(), 1);
            let report = &progress.reports[0];
            for key in [
                "learning_positions",
                "actor_optimizer_calls",
                "critic_optimizer_calls",
                "actor_reconstruction_passes",
                "actor_forward_tokens",
                "online_critic_forward_tokens",
                "target_critic_forward_tokens",
                "actor_valid_history_tokens",
                "online_critic_valid_history_tokens",
                "target_critic_valid_history_tokens",
            ] {
                *metrics.entry(key).or_insert(0.) += report.metrics[key];
            }
            let attempts = report.metrics["projection_evaluations"] as u64;
            *projection_attempts.entry(attempts).or_insert(0) += 1;
            let windows = trainer.last_windows();
            let unique: BTreeSet<_> = windows.iter().copied().collect();
            let episodes: BTreeSet<_> = windows.iter().map(|&(id, _)| id).collect();
            episode_groups += episodes.len();
            duplicate_windows += windows.len() - unique.len();
            min_learn_start_sum += windows.iter().map(|&(_, start)| start).min().unwrap();
            if tick.is_multiple_of((self.updates / 8).max(1)) || tick == self.updates {
                eprintln!(
                    "trial={trial}/{} updates={tick}/{} elapsed={wall_seconds:.3}s",
                    self.repeats, self.updates
                );
            }
        }
        profiling::set_enabled(false);
        assert_eq!(trainer.counters().updates, self.updates as u64);
        Ok(json!({
            "trial": trial,
            "warmup_seconds": warmup_seconds,
            "wall_seconds": wall_seconds,
            "collection_seconds": collection_seconds,
            "learning_seconds": learning_seconds,
            "updates": trainer.counters().updates,
            "transitions": trainer.counters().env_steps,
            "updates_per_second": self.updates as f64 / wall_seconds,
            "learning_rows_per_second": metrics["learning_positions"] / wall_seconds,
            "projection_attempts_histogram": projection_attempts,
            "sampled_episode_groups": episode_groups,
            "duplicate_windows": duplicate_windows,
            "min_learn_start_sum": min_learn_start_sum,
            "work": metrics,
            "phases": profiling::snapshot(),
        }))
    }
}

pub(crate) fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--help") {
        println!("{USAGE}");
        return Ok(());
    }
    let workload = Workload::parse()?;
    let trials = (1..=workload.repeats)
        .map(|trial| workload.trial(trial))
        .collect::<Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "workload": workload,
            "rayon_threads": std::env::var("RAYON_NUM_THREADS").ok(),
            "trials": trials,
        }))?
    );
    Ok(())
}
