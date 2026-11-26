use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::transition::Transition,
    data::{gae::gae, replay::ReplayItem},
    envs::lqg::{Lqg, LqgConfig},
    nn::{critics::quantile::QuantileGrid, matrix},
    objectives::quantile::{QuantileLoss, quantile_loss},
    prelude::*,
    train::trainer::RandomActor,
};
use std::{hint::black_box, time::Duration};

fn components(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let item = ReplayItem {
        transition: Transition {
            observation: Observation::new(vec![1., 0.]),
            action: Action::Continuous(vec![0.]),
            reward: 1.0_f32.into(),
            next_observation: Observation::new(vec![0., 1.]),
            terminated: true,
            truncated: false,
        },
        discount: 0.,
    };
    let mut replay = ReplayBuffer::new(100_000, 2, space.clone()).unwrap();
    for _ in 0..100_000 {
        replay.push(&item).unwrap();
    }
    c.bench_function("replay/insert", |b| {
        b.iter(|| replay.push(black_box(&item)).unwrap())
    });
    let mut group = c.benchmark_group("replay/sample");
    for batch in [1, 32, 256] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &batch, |b, &n| {
            b.iter(|| black_box(replay.sample(n, &mut rng).unwrap()))
        });
    }
    group.finish();
    let mut group = c.benchmark_group("replay/sample_reused");
    for batch in [1, 32, 256] {
        let mut scratch = ReplayBatch::default();
        replay.sample_into(batch, &mut rng, &mut scratch).unwrap();
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_function(BenchmarkId::from_parameter(batch), |b| {
            b.iter(|| replay.sample_into(batch, &mut rng, &mut scratch).unwrap())
        });
    }
    group.finish();
    let policy = Policy::new(2, space.clone(), &[32, 32], &mut rng).unwrap();
    let actor = ActorCritic {
        policy,
        value: Value::new(2, &[32], &mut rng).unwrap(),
    };
    let mut group = c.benchmark_group("policy/action_only");
    for batch in [1, 32, 256] {
        let obs = vec![Observation::new(vec![1., 0.]); batch];
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(BenchmarkId::from_parameter(batch), &obs, |b, obs| {
            b.iter(|| black_box(actor.act(obs, ActMode::Deterministic, &mut rng).unwrap()))
        });
    }
    group.finish();
    let mut group = c.benchmark_group("policy/behavior");
    for batch in [1, 32, 64, 256] {
        let obs = vec![Observation::new(vec![1., 0.]); batch];
        let mut behavior = RolloutBehavior::default();
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_function(BenchmarkId::from_parameter(batch), |b| {
            b.iter(|| {
                actor
                    .sample_rollout_into((&obs).into(), &mut rng, &mut behavior)
                    .unwrap()
            })
        });
    }
    group.finish();
    let mut group = c.benchmark_group("collection");
    for lanes in [1, 32] {
        let env = Lqg::new(
            LqgConfig::scalar(16, 0.95, 1., 1., 0.1, 1., 1., 0.1, 1.).unwrap(),
            0,
        )
        .unwrap();
        let mut collector =
            OffPolicyCollector::new(SyncVectorEnv::new(vec![env; lanes]).unwrap(), 2).unwrap();
        let mut replay = ReplayBuffer::new(100_000, 2, space.clone()).unwrap();
        let random = RandomActor {
            space: space.clone(),
        };
        group.throughput(Throughput::Elements((lanes * 16) as u64));
        group.bench_function(BenchmarkId::from_parameter(lanes), |b| {
            b.iter(|| {
                let mut writer = ReplayWriter::new(&mut replay, None, 0.99, lanes).unwrap();
                black_box(collector.collect_into(&random, 16, &mut writer).unwrap())
            })
        });
    }
    group.finish();
    let rewards = vec![1.; 1024];
    let mut simulation = Lqg::new(
        LqgConfig::scalar(16, 0.95, 1., 1., 0.1, 1., 1., 0.1, 1.).unwrap(),
        0,
    )
    .unwrap();
    simulation.reset(None).unwrap();
    let control = Action::Continuous(vec![0.]);
    c.bench_function("simulation/lqg_step", |b| {
        b.iter(|| {
            let step = simulation.step(&control).unwrap();
            if step.terminated {
                simulation.reset(None).unwrap();
            }
            black_box(step);
        })
    });
    let pack = vec![Observation::new(vec![1., 0.]); 256];
    c.bench_function("batching/observations256", |b| {
        b.iter(|| black_box(rl::nn::observations((&pack).into(), 2).unwrap()))
    });
    let values = vec![0.; 1024];
    let flags = vec![false; 1024];
    c.bench_function("returns/gae1024", |b| {
        b.iter(|| black_box(gae(&rewards, &values, &values, &flags, &flags, 0.99, 0.95).unwrap()))
    });
    let mut group = c.benchmark_group("quantile/loss_backward");
    for batch in [32, 256] {
        for n in [8, 32, 128] {
            let grid = QuantileGrid::uniform(n).unwrap();
            group.bench_function(BenchmarkId::new(batch.to_string(), n), |b| {
                b.iter(|| {
                    let pred = matrix(vec![0.; batch * n], batch, n).require_grad();
                    let target = matrix(vec![1.; batch * n], batch, n);
                    black_box(
                        quantile_loss(pred, target, &grid, &QuantileLoss::Huber { kappa: 1. })
                            .unwrap()
                            .backward(),
                    );
                })
            });
        }
    }
    group.finish();
    let mut group = c.benchmark_group("learner/ddpg");
    for batch in [16, 64, 256] {
        let p = Deterministic::new(2, space.clone(), &[32], &mut rng).unwrap();
        let q = QValue::new(2, 1, 1, &[32, 32], &mut rng).unwrap();
        let mut learner = Ddpg::new(p, q, Default::default(), 0).unwrap();
        let data = ReplayBatch {
            items: vec![item.clone(); batch],
        };
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_function(BenchmarkId::from_parameter(batch), |b| {
            b.iter(|| black_box(learner.update(&data).unwrap()))
        });
    }
    group.finish();
}
criterion_group! {name=benches;config=Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(200)).measurement_time(Duration::from_millis(500));targets=components}
criterion_main!(benches);
