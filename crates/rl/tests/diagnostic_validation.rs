#![cfg(feature = "diagnostics")]
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use std::collections::BTreeMap;

fn probes() -> Chunk {
    ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "explicit fixture",
        vec![
            Observation::new(vec![1., 2.]),
            Observation::new(vec![3., 4.]),
        ],
        None,
        ActionSpace::Discrete { count: 2 },
    )
    .unwrap()
    .record(RecordContext::new("validation", 0))
    .unwrap()
}
#[test]
fn malformed_probe_shapes_dtypes_masks_and_identities_are_rejected_before_writing() {
    let good = probes();
    let Record::ProbeSet(p) = &good.record else {
        panic!()
    };
    let cases = [
        (&p.observations, Array::f32(vec![2, 0], vec![]).unwrap()),
        (&p.observations, Array::f32(vec![4], vec![0.; 4]).unwrap()),
        (
            &p.observations,
            Array::f32(vec![2, 2], vec![f32::NAN; 4]).unwrap(),
        ),
        (&p.row_ids, Array::u64(vec![2], vec![1, 1]).unwrap()),
        (
            p.action_masks.as_ref().unwrap(),
            Array::u8(vec![2, 2], vec![0; 4]).unwrap(),
        ),
        (
            p.action_masks.as_ref().unwrap(),
            Array::f32(vec![2, 2], vec![1.; 4]).unwrap(),
        ),
    ];
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        RunManifest::new(
            "validation",
            "test",
            "fixture",
            2,
            ActionSpace::Discrete { count: 2 },
            0.99,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    for (key, value) in cases {
        let mut invalid = good.clone();
        invalid.arrays.insert(key.clone(), value);
        assert!(invalid.validate().is_err());
        assert!(writer.append(&invalid).is_err());
    }
    writer.append(&good).unwrap();
    writer.finish().unwrap();
    let reader = Recording::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    assert_eq!(reader.entries().len(), 1);
    reader.load(0).unwrap();
}

#[test]
fn critic_grid_validation_preserves_raw_crossings_and_detects_shape_errors() {
    let mut c = Chunk {
        context: RecordContext::new("validation", 2),
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Quantiles {
                values: "q".into(),
                locations: "tau".into(),
                masses: "mass".into(),
            },
            action_condition: ActionCondition::StateValue,
            actions: None,
            continuation_policy: None,
            continuation_stochastic: None,
        }),
        arrays: BTreeMap::from([
            ("q".into(), Array::f32(vec![1, 2], vec![2., -1.]).unwrap()),
            ("tau".into(), Array::f32(vec![2], vec![0.25, 0.75]).unwrap()),
            ("mass".into(), Array::f32(vec![2], vec![0.5, 0.5]).unwrap()),
        ]),
    };
    c.validate().unwrap(); // Model defects remain inspectable; do not sort away crossings.
    c.arrays.insert(
        "q".into(),
        Array::f32(vec![1, 2], vec![f32::NAN, 1.]).unwrap(),
    );
    c.validate().unwrap(); // Numerical failures are data, not malformed storage.
    for bad in [vec![0.75, 0.25], vec![0.5, 0.5], vec![-0.1, 0.9]] {
        let mut invalid = c.clone();
        invalid
            .arrays
            .insert("tau".into(), Array::f32(vec![2], bad).unwrap());
        assert!(invalid.validate().is_err());
    }
    c.arrays
        .insert("q".into(), Array::f32(vec![2], vec![0., 1.]).unwrap());
    assert!(c.validate().is_err());
}

#[test]
fn trajectory_clocks_and_episode_boundaries_are_checked() {
    use rl::{
        diagnostics::evaluation::*,
        envs::lqg::{Lqg, LqgConfig},
        train::trainer::RandomActor,
    };
    let config = LqgConfig::scalar(2, 1., 1., 1., 1., 1., 1., 0., 2.).unwrap();
    let result = evaluate_with_frames(
        |s| Lqg::new(config.clone(), s),
        &RandomActor {
            space: config.action_space.clone(),
        },
        &EvaluationConfig {
            seeds: vec![1],
            trajectories: 1,
            max_steps: 2,
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    let valid = result
        .trajectory_record(0, RecordContext::new("validation", 1), "episode", "lqg")
        .unwrap();
    valid.validate().unwrap();
    let Record::Trajectory(t) = &valid.record else {
        panic!()
    };
    for (key, data) in [
        (&t.terminated, Array::u8(vec![2], vec![1, 1]).unwrap()),
        (&t.truncated, Array::u8(vec![2], vec![0, 1]).unwrap()),
        (&t.terminated, Array::u8(vec![2], vec![0, 0]).unwrap()),
        (
            t.simulation_times.as_ref().unwrap(),
            Array::f64(vec![3], vec![0., 2., 1.]).unwrap(),
        ),
    ] {
        let mut invalid = valid.clone();
        invalid.arrays.insert(key.clone(), data);
        assert!(invalid.validate().is_err());
    }
}
