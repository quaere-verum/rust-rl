#![allow(dead_code)]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::SnapshotBuilder, probes::ProbeSet, schema::*, storage::*},
    nn::{critics::scalar::QValue, network::Mlp, policy::Deterministic},
};
use std::collections::BTreeMap;

pub fn observations() -> Vec<Vec<f64>> {
    (0..12)
        .map(|i| {
            let t = i as f64 / 11.;
            vec![t, (t * 4.).sin(), (t * 7.).cos()]
        })
        .collect()
}
pub fn snapshot(revision: u64, role: &str, names: &[&str]) -> Chunk {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut rng = ChaCha8Rng::seed_from_u64(revision + 42);
    let mut context = RecordContext::new("network", revision);
    context.identity.model_role = role.into();
    let mut builder =
        SnapshotBuilder::new(&format!("{role}-{revision}"), context, 3, space.clone()).unwrap();
    for name in names {
        match *name {
            "actor" => builder
                .deterministic(
                    name,
                    &Deterministic::from_network(
                        Mlp::new(3, &[4, 3], 1, &mut rng).unwrap(),
                        space.clone(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            "moments" => builder
                .moments(
                    name,
                    &QValue::new(3, 1, 1, &[4], &mut rng).unwrap(),
                    &QValue::new(3, 1, 1, &[3], &mut rng).unwrap(),
                )
                .unwrap(),
            _ => builder
                .scalar(name, &QValue::new(3, 1, 1, &[4], &mut rng).unwrap())
                .unwrap(),
        }
    }
    builder.finish().unwrap()
}
pub fn recorder() -> (tempfile::TempDir, Recorder) {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut writer = Recorder::create(
        dir.path().join("run"),
        RunManifest::new("network", "test", "test", 3, space.clone(), 0.95).unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let mut context = RecordContext::new("network", 0);
    context.probe_set = Some("fixed".into());
    writer
        .append(
            &ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "fixed test population",
                observations()
                    .into_iter()
                    .map(|v| Observation::new(v.into_iter().map(|x| x as f32).collect()))
                    .collect(),
                None,
                space,
            )
            .unwrap()
            .record(context)
            .unwrap(),
        )
        .unwrap();
    (dir, writer)
}
pub fn fixture() -> tempfile::TempDir {
    let (dir, mut writer) = recorder();
    for revision in (0..=250).step_by(10) {
        writer
            .append(&Chunk {
                context: RecordContext::new("network", revision),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("loss".into(), 1.)]),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            })
            .unwrap();
        if [0, 100, 250].contains(&revision) {
            writer
                .append(&snapshot(
                    revision,
                    "online",
                    &["actor", "q1", "q2", "moments"],
                ))
                .unwrap();
        }
    }
    writer.finish().unwrap();
    dir
}
