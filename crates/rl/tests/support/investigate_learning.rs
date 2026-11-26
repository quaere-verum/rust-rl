use crate::investigate_or_commit::{Action as Choice, Context, InvestigateOrCommit};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::{
        OptimizerConfig,
        recurrent_critic::{RecurrentCriticConfig, RecurrentDiscreteCritic},
        recurrent_mpo::{MpoBackup, RecurrentMpo, RecurrentMpoConfig},
        reinforce::{CategoricalExploration, Reinforce, ReinforceConfig},
    },
    core::{
        action::{Action, ActionSpace},
        actor::{EpisodeActor, EpisodeBehavior},
        env::Observation,
        learner::Learner,
        stateful::HistorySpec,
    },
    data::{
        rollout::{OnPolicyBatch, OnPolicyMetadata},
        temporal::{
            BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceOrigin, SequenceSample,
            TemporalStep,
        },
    },
    nn::{
        critics::{CriticInitialization, scalar::Value},
        matrix,
        network::Mlp,
        policy::{ActorCritic, Policy, StochasticPolicy},
        temporal::{GruEncoder, GruQ},
        temporal_policy::TemporalPolicy,
        to_vec,
    },
    planning::pomcp::sample_weighted,
};
use std::time::{Duration, Instant};

pub const EPISODES_PER_UPDATE: usize = 32;
pub const SUCCESS: f64 = 0.8;
const SEQUENCE: SequenceConfig = SequenceConfig {
    burn_in: 0,
    learn_len: 2,
    n_step: 1,
};

pub struct Problem {
    signals: u16,
    observations: Vec<Observation>,
    terminal: Observation,
}

impl Problem {
    pub fn new(signals: u16) -> Self {
        assert!(signals > 0);
        let width = usize::from(signals) + 1;
        let actions = usize::from(signals) + 2;
        let observations = (0..width)
            .map(|context| {
                let mut values = vec![0.; width];
                values[context] = 1.;
                let legal = (0..actions)
                    .map(|action| (context == 0) == (action < 2))
                    .collect();
                Observation::dense(values, Some(legal))
            })
            .collect();
        Self {
            signals,
            observations,
            terminal: Observation::dense(vec![0.; width], Some(vec![false; actions])),
        }
    }

    fn space(&self) -> ActionSpace {
        ActionSpace::Discrete {
            count: self.action_count(),
        }
    }

    fn action_count(&self) -> usize {
        usize::from(self.signals) + 2
    }

    fn root(&self) -> &Observation {
        &self.observations[0]
    }

    fn signal(&self, signal: u16) -> &Observation {
        &self.observations[usize::from(signal) + 1]
    }

    fn law(&self, probabilities: &[f32]) -> PolicyStats {
        let actions = self.action_count();
        assert_eq!(probabilities.len(), self.observations.len() * actions);
        let investigate = f64::from(probabilities[1]);
        let competence = (0..usize::from(self.signals))
            .map(|signal| f64::from(probabilities[(signal + 1) * actions + signal + 2]))
            .sum::<f64>()
            / f64::from(self.signals);
        let value = 0.6 * (1. - investigate) + (0.5 + 0.4 * competence) * investigate;
        PolicyStats {
            investigate,
            competence,
            value,
        }
    }
}

#[test]
fn evaluates_the_masked_root_and_matching_signal_probabilities() {
    let probabilities = [0.2, 0.8, 0., 0., 0., 0., 0.9, 0.1, 0., 0., 0.3, 0.7];
    let result = Problem::new(2).law(&probabilities);
    for (actual, expected) in [
        (result.investigate, 0.8),
        (result.competence, 0.8),
        (result.value, 0.776),
    ] {
        assert!((actual - expected).abs() < 1e-7);
    }
}

#[derive(Clone, Copy, Default)]
pub struct PolicyStats {
    pub investigate: f64,
    pub competence: f64,
    pub value: f64,
}

pub struct Run {
    pub initial_policy: PolicyStats,
    pub final_policy: PolicyStats,
    pub first_success: Option<usize>,
    pub episodes: usize,
    pub transitions: usize,
    pub wall: Duration,
}

impl Run {
    fn new(stats: PolicyStats) -> Self {
        Self {
            initial_policy: stats,
            final_policy: stats,
            first_success: (stats.value >= SUCCESS).then_some(0),
            episodes: 0,
            transitions: 0,
            wall: Duration::ZERO,
        }
    }

    fn record(&mut self, stats: PolicyStats, transitions: usize) {
        self.episodes += EPISODES_PER_UPDATE;
        self.transitions += transitions;
        self.final_policy = stats;
        if self.first_success.is_none() && stats.value >= SUCCESS {
            self.first_success = Some(self.episodes);
        }
    }
}

struct Scenario {
    signal: u16,
    reward_draw: f64,
}

impl Scenario {
    fn draw(problem: &Problem, rng: &mut ChaCha8Rng) -> Self {
        Self {
            signal: rng.random_range(0..problem.signals),
            reward_draw: rng.random(),
        }
    }

    fn reward(&self, guess: Option<usize>) -> f32 {
        let (context, action) = match guess {
            Some(guess) => (
                Context::Signal(self.signal),
                Choice::Guess((guess - 2) as u16),
            ),
            None => (Context::Root, Choice::Commit),
        };
        f32::from(self.reward_draw < InvestigateOrCommit::success_probability(context, action))
    }
}

fn optimizer(learning_rate: f64) -> OptimizerConfig {
    OptimizerConfig {
        learning_rate,
        max_grad_norm: Some(1.),
    }
}

fn mpo_probabilities(learner: &RecurrentMpo, problem: &Problem) -> Result<Vec<f32>> {
    let policy = learner.policy();
    assert!(policy.observation_only());
    let width = problem.observations.len();
    let features = problem
        .observations
        .iter()
        .flat_map(|row| row.dense_values().iter().copied())
        .collect();
    to_vec(
        policy
            .distribution_from_features(
                matrix(features, width, width),
                problem.observations.as_slice().into(),
            )?
            .categorical_terms()?
            .probabilities,
    )
}

fn step(action: usize, reward: f32, terminal: bool, update: usize) -> TemporalStep {
    TemporalStep {
        action: Action::Discrete(action),
        reward: reward.into(),
        terminated: terminal,
        truncated: false,
        behavior: BehaviorMetadata {
            revision: update as u64,
            log_prob: None,
        },
    }
}

// New factual complete episodes on every update. The same seeded exogenous streams
// draw a signal and reward variate for every episode, even on immediate commits.
fn mpo_batch(
    problem: &Problem,
    probabilities: &[f32],
    scenarios: &mut ChaCha8Rng,
    actions: &mut ChaCha8Rng,
    update: usize,
) -> Result<SequenceBatch> {
    let rows: Vec<_> = probabilities.chunks_exact(problem.action_count()).collect();
    let mut samples = Vec::with_capacity(EPISODES_PER_UPDATE);
    for episode in 0..EPISODES_PER_UPDATE {
        let scenario = Scenario::draw(problem, scenarios);
        let root = sample_weighted(rows[0].iter().copied().map(f64::from), actions)?;
        let signal_row = rows[usize::from(scenario.signal) + 1];
        let guess = sample_weighted(signal_row.iter().copied().map(f64::from), actions)?;
        let mut observations = vec![problem.root().clone()];
        let mut steps = Vec::with_capacity(2);
        if root == 1 {
            steps.push(step(root, 0., false, update));
            observations.push(problem.signal(scenario.signal).clone());
            steps.push(step(guess, scenario.reward(Some(guess)), true, update));
        } else {
            steps.push(step(root, scenario.reward(None), true, update));
        }
        observations.push(problem.terminal.clone());
        samples.push(SequenceSample {
            episode_id: (update * EPISODES_PER_UPDATE + episode) as u64,
            trajectory_start: 0,
            learn_start: 0,
            burn_start: 0,
            observations,
            loss_mask: vec![true; steps.len()],
            steps,
            history_start: 0,
            previous: None,
            origin: SequenceOrigin::Reset,
        });
    }
    Ok(SequenceBatch {
        samples,
        config: SEQUENCE,
    })
}

pub fn run_mpo(
    problem: &Problem,
    seed: u64,
    initialization: CriticInitialization,
    backup: MpoBackup,
    updates: usize,
) -> Result<Run> {
    let started = Instant::now();
    let mut actor_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA17C);
    let mut critic_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC117);
    let mut scenarios = ChaCha8Rng::seed_from_u64(seed ^ 0xFAC7);
    let mut actions = ChaCha8Rng::seed_from_u64(seed ^ 0xBEA9);
    let spec = HistorySpec::new(problem.observations.len(), problem.space())?;
    let policy = TemporalPolicy::new(spec.clone(), None, &[], &mut actor_rng)?;
    let critic = RecurrentDiscreteCritic::new(
        GruQ::new(
            GruEncoder::new(spec.input_dim(), 8, &mut critic_rng)?,
            Mlp::new(8, &[], spec.action_width(), &mut critic_rng)?,
        )?
        .with_initialization(initialization)?,
        spec,
        RecurrentCriticConfig {
            optimizer: optimizer(0.02),
            gamma: 1.,
            n_step: 1,
            target_tau: 0.25,
            ..Default::default()
        },
    )?;
    let mut learner = RecurrentMpo::new(
        policy,
        critic,
        RecurrentMpoConfig {
            actor: optimizer(0.03),
            backup,
            ..Default::default()
        },
    )?;
    let mut probabilities = mpo_probabilities(&learner, problem)?;
    let mut run = Run::new(problem.law(&probabilities));
    for update in 0..updates {
        let batch = mpo_batch(
            problem,
            &probabilities,
            &mut scenarios,
            &mut actions,
            update,
        )?;
        learner.update(&batch)?;
        probabilities = mpo_probabilities(&learner, problem)?;
        run.record(
            problem.law(&probabilities),
            batch.samples.iter().map(|sample| sample.steps.len()).sum(),
        );
    }
    run.wall = started.elapsed();
    Ok(run)
}

fn reinforce_probabilities(learner: &Reinforce, problem: &Problem) -> Result<Vec<f32>> {
    to_vec(
        learner
            .policy()
            .distribution(problem.observations.as_slice().into())?
            .categorical_terms()?
            .probabilities,
    )
}

fn discrete(action: &Action) -> usize {
    match action {
        Action::Discrete(action) => *action,
        _ => unreachable!("finite-action study"),
    }
}

fn reinforce_batch(
    learner: &Reinforce,
    problem: &Problem,
    scenarios: &mut ChaCha8Rng,
    actions: &mut ChaCha8Rng,
) -> Result<OnPolicyBatch> {
    let draws: Vec<_> = (0..EPISODES_PER_UPDATE)
        .map(|_| Scenario::draw(problem, scenarios))
        .collect();
    let roots = vec![problem.root().clone(); EPISODES_PER_UPDATE];
    let signals: Vec<_> = draws
        .iter()
        .map(|scenario| problem.signal(scenario.signal).clone())
        .collect();
    let mut root_behavior = EpisodeBehavior::default();
    let mut signal_behavior = EpisodeBehavior::default();
    learner.sample_episode_into(roots.as_slice().into(), actions, &mut root_behavior)?;
    // Draw the equally sized hypothetical action row for every matched seed.
    // Only reached states are included in the factual learning batch.
    learner.sample_episode_into(signals.as_slice().into(), actions, &mut signal_behavior)?;
    let mut batch = OnPolicyBatch {
        episode_count: EPISODES_PER_UPDATE,
        observations: Vec::new(),
        actions: Vec::new(),
        latents: Vec::new(),
        returns: Vec::new(),
        discount_weights: Vec::new(),
        metadata: OnPolicyMetadata::Episodic {
            baseline_values: None,
        },
    };
    let mut baselines = Vec::new();
    for (index, scenario) in draws.iter().enumerate() {
        let investigate = discrete(&root_behavior.actions[index]) == 1;
        let guess = investigate.then(|| discrete(&signal_behavior.actions[index]));
        let reward = scenario.reward(guess);
        for (observation, behavior) in std::iter::once((&roots[index], &root_behavior))
            .chain(investigate.then_some((&signals[index], &signal_behavior)))
        {
            batch.observations.push(observation.clone());
            batch.actions.push(behavior.actions[index].clone());
            batch.latents.push(behavior.latents[index].clone());
            batch.returns.push(reward);
            batch.discount_weights.push(1.);
            baselines.push(
                behavior
                    .baseline_values
                    .as_ref()
                    .expect("configured baseline")[index],
            );
        }
    }
    batch.metadata = OnPolicyMetadata::Episodic {
        baseline_values: Some(baselines),
    };
    Ok(batch)
}

pub fn run_reinforce(
    problem: &Problem,
    seed: u64,
    entropy: f32,
    epsilon: f32,
    updates: usize,
) -> Result<Run> {
    let started = Instant::now();
    let mut actor_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA17C);
    let mut critic_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC117);
    let mut scenarios = ChaCha8Rng::seed_from_u64(seed ^ 0xFAC7);
    let mut actions = ChaCha8Rng::seed_from_u64(seed ^ 0xBEA9);
    let width = problem.observations.len();
    let mut learner = Reinforce::new(
        ActorCritic {
            policy: Policy::new(width, problem.space(), &[], &mut actor_rng)?,
            value: Value::new(width, &[], &mut critic_rng)?,
        },
        ReinforceConfig {
            policy: optimizer(0.06),
            value: optimizer(0.05),
            entropy,
            ..Default::default()
        },
        seed,
    )?;
    if epsilon > 0. {
        learner = learner.with_categorical_exploration(CategoricalExploration {
            epsilon,
            only_at: Some(problem.root().dense_values().to_vec()),
        })?;
    }
    let mut run = Run::new(problem.law(&reinforce_probabilities(&learner, problem)?));
    for _ in 0..updates {
        let batch = reinforce_batch(&learner, problem, &mut scenarios, &mut actions)?;
        learner.update(&batch)?;
        run.record(
            problem.law(&reinforce_probabilities(&learner, problem)?),
            batch.len(),
        );
    }
    run.wall = started.elapsed();
    Ok(run)
}
