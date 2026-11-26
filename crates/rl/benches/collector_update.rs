//! Bounded comparisons against the original universal collector.
//! Copy this unchanged harness into the baseline checkout and compile its benchmark
//! target with `cargo rustc --profile bench --bench collector_update -- --cfg collector_baseline`.
#![allow(unexpected_cfgs)]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::vector_env::VectorEnv,
    data::replay::NStepAccumulator,
    envs::lqg::{Lqg, LqgConfig},
    prelude::*,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

struct Counting;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
fn record(size: usize) {
    if ACTIVE.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}
// SAFETY: All allocation pointer/layout contracts are delegated unchanged to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            record(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            record(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(p, layout, size) };
        if !out.is_null() {
            record(size);
        }
        out
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
const WARMUP: usize = 2;
const SAMPLES: usize = 5;
const STEPS: usize = 16;

/// Preallocated finite observation supply isolates collector wrapper work. Its new
/// step_into path allocates nothing after warmup; the old step API needs an output Vec.
struct PooledVector {
    lanes: usize,
    space: ActionSpace,
    observations: Vec<Observation>,
}
impl PooledVector {
    fn new(lanes: usize) -> Self {
        Self::with_steps(lanes, STEPS)
    }
    fn with_steps(lanes: usize, steps: usize) -> Self {
        Self {
            lanes,
            space: space(),
            observations: (0..(WARMUP + SAMPLES) * steps * lanes)
                .map(|_| Observation::new(vec![1., 0.]))
                .collect(),
        }
    }
    fn fill(&mut self, actions: &[Option<Action>], out: &mut Vec<Option<Step>>) {
        out.resize_with(self.lanes, || None);
        for (slot, action) in out.iter_mut().zip(actions) {
            *slot = action.as_ref().map(|_| Step {
                observation: self.observations.pop().expect("bounded observation supply"),
                reward: 0.1_f32.into(),
                terminated: false,
                truncated: false,
            });
        }
    }
}
impl VectorEnv for PooledVector {
    fn len(&self) -> usize {
        self.lanes
    }
    #[cfg(collector_baseline)]
    fn observation_dim(&self) -> usize {
        2
    }
    #[cfg(not(collector_baseline))]
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(2)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, mask: &[bool], _: Option<&[u64]>) -> Result<Vec<Option<EpisodeStart>>> {
        Ok(mask
            .iter()
            .map(|&reset| reset.then(|| EpisodeStart::fresh(Observation::new(vec![1., 0.]))))
            .collect())
    }
    fn step(&mut self, actions: &[Option<Action>]) -> Result<Vec<Option<Step>>> {
        let mut out = Vec::new();
        self.fill(actions, &mut out);
        Ok(out)
    }
    #[cfg(not(collector_baseline))]
    fn step_into(&mut self, actions: &[Option<Action>], out: &mut Vec<Option<Step>>) -> Result<()> {
        self.fill(actions, out);
        Ok(())
    }
}
fn space() -> ActionSpace {
    ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    }
}
struct CountedActor {
    models: Option<ActorCritic>,
    on_policy: bool,
    space: ActionSpace,
    actions: Cell<u64>,
    values: Cell<u64>,
    likelihoods: Cell<u64>,
}
impl CountedActor {
    fn new(models: Option<ActorCritic>, on_policy: bool) -> Self {
        Self {
            models,
            on_policy,
            space: space(),
            actions: Cell::new(0),
            values: Cell::new(0),
            likelihoods: Cell::new(0),
        }
    }
    fn reset_counts(&self) {
        self.actions.set(0);
        self.values.set(0);
        self.likelihoods.set(0);
    }
}
fn increment(counter: &Cell<u64>, len: usize) {
    counter.set(counter.get() + len as u64);
}
#[cfg(collector_baseline)]
impl Actor for CountedActor {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act(
        &self,
        obs: &[Observation],
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<ActorOutput>> {
        increment(&self.actions, obs.len());
        if self.on_policy {
            increment(&self.values, obs.len());
            increment(&self.likelihoods, obs.len());
        }
        if let Some(models) = &self.models {
            return models.act(obs, mode, rng);
        }
        Ok(obs
            .iter()
            .map(|_| ActorOutput {
                action: Action::Continuous(vec![0.]),
                log_prob: self.on_policy.then_some(0.),
                latent: None,
                value: 0.,
            })
            .collect())
    }
    fn values(&self, obs: &[Observation]) -> Result<Vec<f32>> {
        increment(&self.values, obs.len());
        if let Some(models) = &self.models {
            models.values(obs)
        } else {
            Ok(vec![0.; obs.len()])
        }
    }
}
#[cfg(not(collector_baseline))]
fn fixed_actions(len: usize, out: &mut Vec<Action>) {
    out.resize_with(len, || Action::Continuous(vec![0.]));
    for action in out {
        if let Action::Continuous(values) = action {
            values.fill(0.);
        }
    }
}
#[cfg(not(collector_baseline))]
impl Actor for CountedActor {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        increment(&self.actions, obs.len());
        if let Some(models) = &self.models {
            models.act_into(obs, mode, rng, out)
        } else {
            fixed_actions(obs.len(), out);
            Ok(())
        }
    }
}
#[cfg(not(collector_baseline))]
impl RolloutActor for CountedActor {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> Result<()> {
        assert!(self.on_policy);
        increment(&self.actions, obs.len());
        increment(&self.values, obs.len());
        increment(&self.likelihoods, obs.len());
        if let Some(models) = &self.models {
            models.sample_rollout_into(obs, rng, out)
        } else {
            fixed_actions(obs.len(), &mut out.actions);
            out.values.resize(obs.len(), 0.);
            out.log_probs.resize(obs.len(), 0.);
            out.latents.resize(obs.len(), None);
            Ok(())
        }
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> Result<()> {
        increment(&self.values, obs.len());
        if let Some(models) = &self.models {
            models.values_into(obs, out)
        } else {
            out.resize(obs.len(), 0.);
            Ok(())
        }
    }
}

fn measure(
    label: &str,
    case: &str,
    transitions: usize,
    actor: Option<&CountedActor>,
    mut run: impl FnMut() -> (f64, f64, f64),
) {
    eprintln!("running {label}/{case}");
    for _ in 0..WARMUP {
        black_box(run());
    }
    if let Some(actor) = actor {
        actor.reset_counts();
    }
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    let mut elapsed = [0.; SAMPLES];
    let mut collection = 0.;
    let mut preparation = 0.;
    let mut learning = 0.;
    for value in &mut elapsed {
        ACTIVE.store(true, Ordering::Relaxed);
        let start = Instant::now();
        let (c, p, l) = run();
        *value = start.elapsed().as_secs_f64() * 1e6;
        ACTIVE.store(false, Ordering::Relaxed);
        collection += c;
        preparation += p;
        learning += l;
    }
    elapsed.sort_by(f64::total_cmp);
    let counts = if let Some(actor) = actor {
        format!(
            "{:.1},{:.1},{:.1}",
            actor.actions.get() as f64 / SAMPLES as f64,
            actor.values.get() as f64 / SAMPLES as f64,
            actor.likelihoods.get() as f64 / SAMPLES as f64
        )
    } else {
        "NA,NA,NA".into()
    };
    println!(
        "{label},{case},{transitions},{SAMPLES},{:.3},{:.3},{:.3},{:.1},{:.1},{counts},{:.3},{:.3},{:.3}",
        elapsed[SAMPLES / 2],
        elapsed[0],
        elapsed[SAMPLES - 1],
        CALLS.load(Ordering::Relaxed) as f64 / SAMPLES as f64,
        BYTES.load(Ordering::Relaxed) as f64 / SAMPLES as f64,
        collection * 1e6 / SAMPLES as f64,
        preparation * 1e6 / SAMPLES as f64,
        learning * 1e6 / SAMPLES as f64
    );
}
fn off_policy<V: VectorEnv>(label: &str, case: &str, env: V, n_step: usize) {
    let lanes = env.len();
    let mut replay = ReplayBuffer::new(lanes * 32, 2, space()).unwrap();
    let actor = CountedActor::new(None, false);
    #[cfg(collector_baseline)]
    {
        let mut collector = Collector::new(env, 17).unwrap();
        let mut pending = NStepAccumulator::new(n_step, 0.99, lanes).unwrap();
        measure(label, case, lanes * STEPS, Some(&actor), || {
            let rollout = collector.collect_steps(&actor, STEPS).unwrap();
            for (i, row) in rollout.rows.into_iter().enumerate() {
                if let Some(row) = row {
                    for item in pending.push(i % lanes, row.transition).unwrap() {
                        replay.push(&item).unwrap();
                    }
                }
            }
            black_box(replay.len());
            (0., 0., 0.)
        });
    }
    #[cfg(not(collector_baseline))]
    {
        let mut collector = OffPolicyCollector::new(env, 17).unwrap();
        let mut pending = (n_step > 1).then(|| NStepAccumulator::new(n_step, 0.99, lanes).unwrap());
        measure(label, case, lanes * STEPS, Some(&actor), || {
            let mut writer = ReplayWriter::new(&mut replay, pending.as_mut(), 0.99, lanes).unwrap();
            black_box(collector.collect_into(&actor, STEPS, &mut writer).unwrap());
            (0., 0., 0.)
        });
    }
}
fn models() -> ActorCritic {
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    ActorCritic {
        policy: Policy::new(2, space(), &[32, 32], &mut rng).unwrap(),
        value: Value::new(2, &[32], &mut rng).unwrap(),
    }
}
fn on_policy(label: &str, lanes: usize, neural: bool) {
    let env = PooledVector::new(lanes);
    let actor = CountedActor::new(neural.then(models), true);
    let name = if neural {
        "ppo_vmpo_neural"
    } else {
        "synthetic"
    };
    let case = format!("on_policy/{name}/lanes{lanes}");
    #[cfg(collector_baseline)]
    {
        let mut collector = Collector::new(env, 17).unwrap();
        measure(label, &case, lanes * STEPS, Some(&actor), || {
            black_box(collector.collect_steps(&actor, STEPS).unwrap());
            (0., 0., 0.)
        });
    }
    #[cfg(not(collector_baseline))]
    {
        let mut collector = OnPolicyCollector::new(env, 17).unwrap();
        let mut rollout = RolloutBuffer::default();
        measure(label, &case, lanes * STEPS, Some(&actor), || {
            black_box(
                collector
                    .collect_steps_into(&actor, STEPS, &mut rollout)
                    .unwrap(),
            );
            (0., 0., 0.)
        });
    }
}
/// PPO and VMPO share this collection/GAE preparation path. Include terminal
/// resets separately from the preallocated continuing environment measurements.
fn on_policy_with_returns<V: VectorEnv>(label: &str, case: &str, env: V, steps: usize) {
    let lanes = env.len();
    let actor = CountedActor::new(Some(models()), true);
    #[cfg(collector_baseline)]
    {
        let mut collector = Collector::new(env, 17).unwrap();
        measure(label, case, lanes * steps, Some(&actor), || {
            let start = Instant::now();
            let rollout = collector.collect_steps(&actor, steps).unwrap();
            let collection = start.elapsed().as_secs_f64();
            let start = Instant::now();
            black_box(rollout.to_batch(0.99, Some(0.95), true).unwrap());
            (collection, start.elapsed().as_secs_f64(), 0.)
        });
    }
    #[cfg(not(collector_baseline))]
    {
        let mut collector = OnPolicyCollector::new(env, 17).unwrap();
        let mut rollout = RolloutBuffer::default();
        let mut batch = OnPolicyBatch::default();
        measure(label, case, lanes * steps, Some(&actor), || {
            let start = Instant::now();
            black_box(
                collector
                    .collect_steps_into(&actor, steps, &mut rollout)
                    .unwrap(),
            );
            let collection = start.elapsed().as_secs_f64();
            let start = Instant::now();
            rollout
                .to_batch_into(0.99, Some(0.95), true, &mut batch)
                .unwrap();
            black_box(&batch);
            (collection, start.elapsed().as_secs_f64(), 0.)
        });
    }
}
fn trainers(label: &str) {
    const LANES: usize = 32;
    #[cfg(collector_baseline)]
    type OnCollector = Collector<PooledVector>;
    #[cfg(not(collector_baseline))]
    type OnCollector = OnPolicyCollector<PooledVector>;
    let config = OnPolicyConfig {
        rollout_steps: STEPS,
        ..Default::default()
    };
    let mut trainer = OnPolicyTrainer::new(
        OnCollector::new(PooledVector::new(LANES), 17).unwrap(),
        config.clone(),
    )
    .unwrap();
    let mut learner = Ppo::new(
        models(),
        PpoConfig {
            epochs: 1,
            minibatch_size: LANES * STEPS,
            target_kl: None,
            ..Default::default()
        },
        23,
    )
    .unwrap();
    measure(label, "trainer/ppo/lanes32", LANES * STEPS, None, || {
        let summary = trainer
            .train(&mut learner, Budget::vector_steps(STEPS as u64), None)
            .unwrap();
        (summary.collection_seconds, 0., summary.learning_seconds)
    });
    let mut trainer = OnPolicyTrainer::new(
        OnCollector::new(PooledVector::new(LANES), 17).unwrap(),
        config,
    )
    .unwrap();
    let mut learner = Vmpo::new(
        models(),
        VmpoConfig {
            epochs: 1,
            minibatch_size: LANES * STEPS,
            ..Default::default()
        },
        23,
    )
    .unwrap();
    measure(label, "trainer/vmpo/lanes32", LANES * STEPS, None, || {
        let summary = trainer
            .train(&mut learner, Budget::vector_steps(STEPS as u64), None)
            .unwrap();
        (summary.collection_seconds, 0., summary.learning_seconds)
    });
    #[cfg(collector_baseline)]
    type OffCollector = Collector<PooledVector>;
    #[cfg(not(collector_baseline))]
    type OffCollector = OffPolicyCollector<PooledVector>;
    let config = OffPolicyConfig {
        replay_capacity: 4096,
        batch_size: 32,
        warmup_transitions: 0,
        updates_per_transition: 1. / LANES as f64,
        n_step: 1,
        ..Default::default()
    };
    let mut trainer = OffPolicyTrainer::new(
        OffCollector::new(PooledVector::new(LANES), 17).unwrap(),
        config,
        23,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let policy = Deterministic::new(2, space(), &[32], &mut rng).unwrap();
    let critic = QValue::new(2, 1, 1, &[32, 32], &mut rng).unwrap();
    let mut learner = Ddpg::new(policy, critic, Default::default(), 29).unwrap();
    measure(label, "trainer/ddpg/lanes32", LANES * STEPS, None, || {
        let summary = trainer
            .train(&mut learner, Budget::vector_steps(STEPS as u64), None)
            .unwrap();
        (summary.collection_seconds, 0., summary.learning_seconds)
    });
}
fn main() {
    let label = std::env::var("COLLECTOR_BENCH_LABEL").unwrap_or_else(|_| "current".into());
    println!(
        "label,case,transitions,samples,median_us,min_us,max_us,allocations,requested_bytes,action_obs,value_obs,likelihood_obs,collection_us,preparation_us,learning_us"
    );
    for lanes in [1, 32, 256, 1024] {
        for n in [1, 3] {
            off_policy(
                &label,
                &format!("off_policy/synthetic/n{n}/lanes{lanes}"),
                PooledVector::new(lanes),
                n,
            );
        }
        on_policy(&label, lanes, false);
        if lanes <= 256 {
            on_policy(&label, lanes, true);
        }
        let env = Lqg::new(
            LqgConfig::scalar(32, 0.95, 1., 1., 0.1, 1., 1., 0.1, 1.).unwrap(),
            13,
        )
        .unwrap();
        off_policy(
            &label,
            &format!("off_policy/lqg/n1/lanes{lanes}"),
            SyncVectorEnv::new(vec![env; lanes]).unwrap(),
            1,
        );
    }
    for lanes in [1, 32, 256] {
        for steps in [16, 64] {
            on_policy_with_returns(
                &label,
                &format!("on_policy_with_returns/continuing/lanes{lanes}/steps{steps}"),
                PooledVector::with_steps(lanes, steps),
                steps,
            );
            let env = Lqg::new(
                LqgConfig::scalar(4, 0.95, 1., 1., 0.1, 1., 1., 0.1, 1.).unwrap(),
                13,
            )
            .unwrap();
            on_policy_with_returns(
                &label,
                &format!("on_policy_with_returns/ending4/lanes{lanes}/steps{steps}"),
                SyncVectorEnv::new(vec![env; lanes]).unwrap(),
                steps,
            );
        }
    }
    trainers(&label);
}
