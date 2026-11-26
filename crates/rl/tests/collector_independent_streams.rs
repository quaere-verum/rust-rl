use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::stateful::{InferenceMode, StatefulCollector},
    core::{
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        temporal::{TemporalSink, TemporalTransition},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
struct SeedClock {
    rng: ChaCha8Rng,
    seeds: Vec<u64>,
    actions: Vec<Action>,
    time: usize,
    horizon: usize,
}
impl Env for SeedClock {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed ^ 0x454e56);
        }
        let episode = seed.unwrap_or_else(|| self.rng.random());
        self.seeds.push(episode);
        self.time = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        self.time += 1;
        self.actions.push(action.clone());
        Ok(Step {
            observation: Observation::new(vec![0.]),
            reward: 0.0_f32.into(),
            terminated: self.time == self.horizon,
            truncated: false,
        })
    }
}
impl Snapshot for SeedClock {
    fn validate_snapshot(&self) -> Result<()> {
        Ok(())
    }
}
type Vector = SyncVectorEnv<SeedClock>;
fn environment() -> Vector {
    SyncVectorEnv::new(
        [2, 3]
            .into_iter()
            .map(|horizon| SeedClock {
                rng: ChaCha8Rng::seed_from_u64(0),
                seeds: vec![],
                actions: vec![],
                time: 0,
                horizon,
            })
            .collect(),
    )
    .unwrap()
}
#[derive(Clone, Serialize, Deserialize)]
struct RandomActor;
impl Actor for RandomActor {
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        *out = obs
            .iter()
            .map(|_| Action::Discrete(rng.random_range(0..2)))
            .collect();
        Ok(())
    }
}
impl RolloutActor for RandomActor {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> Result<()> {
        self.act_into(obs, ActMode::Sample, rng, &mut out.actions)?;
        out.values = vec![0.; obs.len()];
        out.log_probs = vec![-std::f32::consts::LN_2; obs.len()];
        out.latents = vec![None; obs.len()];
        Ok(())
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> Result<()> {
        *out = vec![0.; obs.len()];
        Ok(())
    }
}
impl StatefulActor for RandomActor {
    type State = u64;
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn revision(&self) -> u64 {
        0
    }
    fn initial_state(&self) -> Result<u64> {
        Ok(0)
    }
    fn validate_state(&self, _: &u64) -> Result<()> {
        Ok(())
    }
    fn assimilate(&self, s: &u64, _: HistoryInput<'_>) -> Result<u64> {
        Ok(s + 1)
    }
    fn decide(
        &self,
        _: &u64,
        _: &Observation,
        _: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        Ok(StatefulDecision {
            action: Action::Discrete(rng.random_range(0..2)),
            log_prob: None,
        })
    }
}
impl Snapshot for RandomActor {
    fn validate_snapshot(&self) -> Result<()> {
        Ok(())
    }
}
#[derive(Default)]
struct Sink;
impl TemporalSink for Sink {
    fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> Result<()> {
        Ok(())
    }
    fn push(&mut self, _: &TemporalTransition) -> Result<()> {
        Ok(())
    }
}
fn seeds(env: &Vector) -> Vec<Vec<u64>> {
    env.environments().iter().map(|e| e.seeds.clone()).collect()
}
fn actions(env: &Vector) -> Vec<Vec<Action>> {
    env.environments()
        .iter()
        .map(|e| e.actions.clone())
        .collect()
}
fn ff_steps(c: &mut OnPolicyCollector<Vector>, steps: usize) {
    c.collect_steps_into(&RandomActor, steps, &mut RolloutBuffer::default())
        .unwrap();
}

#[test]
fn explicit_feedforward_streams_are_independent_and_resume_through_resets() {
    let mut a = OnPolicyCollector::new_with_reset_seeds(environment(), 11, &[101, 202]).unwrap();
    let mut action_changed =
        OnPolicyCollector::new_with_reset_seeds(environment(), 23, &[101, 202]).unwrap();
    let mut reset_changed =
        OnPolicyCollector::new_with_reset_seeds(environment(), 11, &[303, 404]).unwrap();
    for c in [&mut a, &mut action_changed, &mut reset_changed] {
        ff_steps(c, 12);
    }
    assert_eq!(seeds(a.environment()), seeds(action_changed.environment()));
    assert_ne!(
        actions(a.environment()),
        actions(action_changed.environment())
    );
    assert_ne!(seeds(a.environment()), seeds(reset_changed.environment()));
    assert_eq!(
        actions(a.environment()),
        actions(reset_changed.environment())
    );
    let mut reference_rng = ChaCha8Rng::seed_from_u64(11);
    let mut expected = [vec![], vec![]];
    for _ in 0..12 {
        for lane in &mut expected {
            lane.push(Action::Discrete(reference_rng.random_range(0..2)));
        }
    }
    assert_eq!(actions(a.environment()), expected);
    let mut restored: OnPolicyCollector<Vector> =
        serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
    ff_steps(&mut a, 7);
    ff_steps(&mut restored, 7);
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(restored).unwrap()
    );
}

#[test]
fn default_feedforward_preserves_legacy_initial_draws_and_action_sequence() {
    let mut rng = ChaCha8Rng::seed_from_u64(47);
    let initial: Vec<u64> = (0..2).map(|_| rng.random()).collect();
    let mut c = OnPolicyCollector::new(environment(), 47).unwrap();
    assert_eq!(
        seeds(c.environment()),
        initial.iter().map(|s| vec![*s]).collect::<Vec<_>>()
    );
    let mut expected = [vec![], vec![]];
    for _ in 0..12 {
        for lane in &mut expected {
            lane.push(Action::Discrete(rng.random_range(0..2)));
        }
    }
    ff_steps(&mut c, 12);
    assert_eq!(actions(c.environment()), expected);
}

#[test]
fn explicit_stateful_streams_preserve_lane_rngs_and_snapshot_continuation() {
    let build = |action_seed, seeds: &[u64]| {
        StatefulCollector::new_with_reset_seeds(
            environment(),
            RandomActor,
            action_seed,
            InferenceMode::Scalar,
            seeds,
        )
        .unwrap()
    };
    let mut a = build(11, &[101, 202]);
    let mut b = build(23, &[101, 202]);
    let mut c = build(11, &[303, 404]);
    for collector in [&mut a, &mut b, &mut c] {
        collector.collect(12, ActMode::Sample, &mut Sink).unwrap();
    }
    assert_eq!(seeds(a.environment()), seeds(b.environment()));
    assert_ne!(actions(a.environment()), actions(b.environment()));
    assert_ne!(seeds(a.environment()), seeds(c.environment()));
    assert_eq!(actions(a.environment()), actions(c.environment()));
    let mut restored: StatefulCollector<Vector, RandomActor> =
        serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
    for collector in [&mut a, &mut restored] {
        collector.collect(7, ActMode::Sample, &mut Sink).unwrap();
    }
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(restored).unwrap()
    );
    let mut legacy_rng = ChaCha8Rng::seed_from_u64(11);
    let initial: Vec<u64> = (0..2).map(|_| legacy_rng.random()).collect();
    let legacy = StatefulCollector::new(environment(), RandomActor, 11).unwrap();
    let explicit = build(11, &initial);
    let mut expected = serde_json::to_value(legacy).unwrap();
    let actual = serde_json::to_value(explicit).unwrap();
    // Only the driver's unused action RNG differs; stateful lane RNGs are unchanged.
    expected["driver"]["rng"] = actual["driver"]["rng"].clone();
    assert_eq!(expected, actual);
}

#[test]
fn wrong_seed_count_is_rejected() {
    assert!(OnPolicyCollector::new_with_reset_seeds(environment(), 11, &[101]).is_err());
    assert!(
        StatefulCollector::new_with_reset_seeds(
            environment(),
            RandomActor,
            11,
            InferenceMode::Scalar,
            &[101, 202, 303]
        )
        .is_err()
    );
}

#[test]
#[cfg(feature = "neural")]
fn temporal_trainer_forwards_explicit_seeds_and_canonical_mode() {
    use rl::{
        algorithms::{ppo::PpoConfig, recurrent_ppo::RecurrentPpo},
        core::stateful::HistorySpec,
        nn::{temporal_policy::TemporalPolicy, temporal_value::TemporalValue},
        train::temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    };
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let learner = RecurrentPpo::new(
        TemporalPolicy::new(spec.clone(), Some(2), &[], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(2), &[], &mut rng).unwrap(),
        PpoConfig::default(),
        19,
    )
    .unwrap();
    let config = TemporalOnPolicyConfig {
        inference_mode: InferenceMode::CanonicalBatched,
        ..Default::default()
    };
    let mut trainer = TemporalOnPolicyTrainer::new_with_reset_seeds(
        environment(),
        &learner,
        config,
        47,
        &[101, 202],
    )
    .unwrap();
    assert_eq!(
        trainer.collector().inference_mode(),
        InferenceMode::CanonicalBatched
    );
    assert_eq!(
        seeds(trainer.collector().environment()),
        vec![vec![101], vec![202]]
    );
    trainer.collect_tick(&learner).unwrap();
    let mut restored: TemporalOnPolicyTrainer<Vector> =
        serde_json::from_slice(&serde_json::to_vec(&trainer).unwrap()).unwrap();
    for _ in 0..5 {
        trainer.collect_tick(&learner).unwrap();
        restored.collect_tick(&learner).unwrap();
    }
    assert_eq!(
        seeds(trainer.collector().environment()),
        seeds(restored.collector().environment())
    );
    assert_eq!(
        actions(trainer.collector().environment()),
        actions(restored.collector().environment())
    );
}
