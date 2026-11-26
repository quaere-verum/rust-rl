//! Bounded paired CPU training measurements; all modes use identical owned seeds.
#[path = "../../../benches/support/allocation.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        compact::*,
        models::SnapshotBuilder,
        observer::DiagnosticObserver,
        probes::{self, ProbeSet},
        schedule::*,
        schema::*,
        storage::*,
    },
    envs::lqg::{Lqg, LqgConfig},
    nn::{
        observations,
        probes::{CriticProbe, ScalarProbe},
    },
    prelude::*,
};
use std::{
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
fn main() -> Result<()> {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "runs/diagnostic-benchmark-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        )
    }));
    let root = if root.is_relative() {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(root)
    } else {
        root
    };
    std::fs::create_dir_all(&root)?;
    println!(
        "mode,repeat,seconds,collection_seconds,learning_seconds,diagnostic_seconds,allocations,requested_bytes,live_delta_bytes,peak_extra_bytes,disk_bytes"
    );
    let mut reference = None;
    for repeat in 0..3 {
        for mode in if repeat % 2 == 0 {
            vec![0, 1, 2, 3, 4, 5]
        } else {
            vec![5, 4, 3, 2, 1, 0]
        } {
            let name = [
                "disabled",
                "metrics",
                "probes",
                "activations",
                "snapshots",
                "summaries",
            ][mode];
            let mut rng = ChaCha8Rng::seed_from_u64(19);
            let env = Lqg::new(
                LqgConfig::scalar(32, 0.95, 1., 1., 0.1, 1., 1., 0.1, 1.)?,
                13,
            )?;
            let space = env.action_space().clone();
            let mut learner = Ddpg::new(
                Deterministic::new(2, space.clone(), &[32], &mut rng)?,
                QValue::new(2, 1, 1, &[32, 32], &mut rng)?,
                Default::default(),
                29,
            )?;
            let mut trainer = OffPolicyTrainer::new(
                OffPolicyCollector::new(SyncVectorEnv::new(vec![env; 32])?, 17)?,
                OffPolicyConfig {
                    replay_capacity: 4096,
                    batch_size: 32,
                    warmup_transitions: 0,
                    updates_per_transition: 1. / 32.,
                    ..Default::default()
                },
                23,
            )?;
            trainer.train(&mut learner, Budget::vector_steps(20), None)?;
            let set = ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "128 fixed numerical inputs, no training RNG",
                (0..128)
                    .map(|i| {
                        Observation::new(vec![(i % 16) as f32 / 8. - 1., (i / 16) as f32 / 8.])
                    })
                    .collect(),
                None,
                space.clone(),
            )?;
            let run = format!("{name}-{repeat}");
            let mut observer = if mode == 0 {
                None
            } else {
                let writer = Recorder::create(
                    root.join(&run),
                    RunManifest::new(&run, "overhead", "DDPG", 2, space.clone(), 0.99)?,
                    StorageConfig::default(),
                )?
                .with_snapshot_retention(2)?;
                let mut writer = if std::env::var("RL_DIAGNOSTIC_BUFFER").as_deref() == Ok("off") {
                    writer
                } else {
                    writer.with_buffering(BufferConfig::default())?
                };
                writer.append(&set.record(RecordContext::new(&run, 20))?)?;
                let schedule = RecordingSchedule {
                    metrics: Some(Interval { first: 1, every: 1 }),
                    probes: (mode >= 2).then_some(Interval {
                        first: 20,
                        every: 20,
                    }),
                    monte_carlo: None,
                    snapshots: (mode == 4).then_some(Interval {
                        first: 200,
                        every: 200,
                    }),
                };
                Some(DiagnosticObserver::new(
                    writer,
                    schedule,
                    |l: &Ddpg, c: &RecordContext, due: Due, w: &mut Recorder| -> Result<()> {
                        if due.probes {
                            let mut c = c.clone();
                            c.probe_set = Some(set.id().into());
                            let p = l.policy();
                            let q = l.critic();
                            let actions = p.actions(set.observations().into())?;
                            let actual = probes::continuous_actions(actions.clone())?;
                            w.append(&probes::deterministic_policy(c.clone(), &p, &set)?)?;
                            w.append(&probes::critic(
                                c.clone(),
                                ScalarProbe::new(&q)?
                                    .predict(set.observations().into(), actions)?,
                                ActionCondition::CurrentPolicy,
                                Some(&actual),
                            )?)?;
                            if mode >= 3 {
                                let storage = if mode == 5 {
                                    ActivationStorage::Summary
                                } else {
                                    ActivationStorage::Full
                                };
                                w.append(&activations_batched(
                                    c,
                                    "policy",
                                    &p.network,
                                    observations(set.observations().into(), 2)?,
                                    &["layers.0.pre", "layers.0.post"],
                                    &ActivationOptions {
                                        storage,
                                        ..Default::default()
                                    },
                                )?)?;
                            }
                        }
                        if due.snapshots {
                            let mut s = SnapshotBuilder::new(
                                &format!("k{}", l.updates()),
                                c.clone(),
                                2,
                                space.clone(),
                            )?;
                            s.deterministic("policy", &l.policy())?;
                            s.scalar("critic", &l.critic())?;
                            w.append(&s.finish()?)?;
                        }
                        Ok(())
                    },
                ))
            };
            let memory = allocation::start();
            let clock = Instant::now();
            let progress = if let Some(observer) = &mut observer {
                trainer.train_observed(&mut learner, Budget::vector_steps(200), None, observer)?
            } else {
                trainer.train(&mut learner, Budget::vector_steps(200), None)?
            };
            if let Some(observer) = &mut observer {
                observer.finish()?;
            }
            let seconds = clock.elapsed().as_secs_f64();
            let memory = allocation::finish(memory);
            let (diagnostics, bytes) = if let Some(o) = &mut observer {
                (
                    o.diagnostic_seconds(),
                    o.recorder().status().committed_bytes - o.recorder().status().reclaimed_bytes,
                )
            } else {
                (0., 0)
            };
            let final_actions =
                rl::nn::to_vec(learner.policy().actions(set.observations().into())?)?;
            if let Some(reference) = &reference {
                assert_eq!(
                    &final_actions, reference,
                    "recording changed training behaviour"
                );
            } else {
                reference = Some(final_actions);
            }
            println!(
                "{name},{repeat},{seconds:.6},{:.6},{:.6},{diagnostics:.6},{},{},{},{},{bytes}",
                progress.collection_seconds,
                progress.learning_seconds,
                memory.calls,
                memory.requested,
                memory.live,
                memory.peak
            );
        }
    }
    Ok(())
}
