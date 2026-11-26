#![allow(dead_code)]
use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*, visualization::*},
};
use std::collections::BTreeMap;

pub fn manifest(space: ActionSpace) -> RunManifest {
    let mut m = RunManifest::new("dynamics", "test", "test", 3, space.clone(), 0.95).unwrap();
    m.environment_descriptor = Some(EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations: (0..3)
            .map(|i| ObservationField {
                id: format!("coord{i}"),
                label: format!("Position {i}"),
                unit: Some("m".into()),
                description: None,
                kind: ObservationType::Numerical,
                range: Some([0., 2.]),
                coordinate: true,
            })
            .collect(),
        actions: match space {
            ActionSpace::Discrete { count } => ActionDescriptor::Discrete {
                choices: (0..count)
                    .map(|i| DiscreteChoice {
                        label: format!("choice {i}"),
                        meaning: None,
                    })
                    .collect(),
            },
            _ => ActionDescriptor::Continuous {
                components: (0..space.dimensions())
                    .map(|i| ActionComponent {
                        label: format!("Motor {i}"),
                        unit: Some("N".into()),
                        meaning: None,
                    })
                    .collect(),
                preferred_component: Some(if space.dimensions() > 1 { 2 } else { 0 }),
            },
        },
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel {
                label: "Revision".into(),
                unit: None,
            },
            decision_step: QuantityLabel {
                label: "Step".into(),
                unit: None,
            },
            simulation: Some(QuantityLabel {
                label: "Time".into(),
                unit: Some("s".into()),
            }),
        },
        reward: RewardDescriptor {
            label: "Reward".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: vec![],
        preferences: VisualizationPreferences {
            coordinates: vec!["coord1".into(), "coord0".into()],
            ..Default::default()
        },
    });
    m
}

pub fn chunk(space: &ActionSpace) -> Chunk {
    let actions = match space {
        ActionSpace::Discrete { .. } => Array::u64(vec![4], vec![0, 2, 1, 0]).unwrap(),
        _ => Array::f64(
            vec![4, space.dimensions()],
            (0..4)
                .flat_map(|t| (0..space.dimensions()).map(move |j| (10 * j + t) as f64))
                .collect(),
        )
        .unwrap(),
    };
    let mut context = RecordContext::new("dynamics", 17);
    context.identity.model_role = "target".into();
    context.trajectory = Some("episode-1".into());
    Chunk {
        context,
        record: Record::Trajectory(TrajectoryRecord {
            render: None,
            id: "episode-1".into(),
            environment: "fixture".into(),
            observations: "obs".into(),
            actions: "actions".into(),
            rewards: "rewards".into(),
            terminated: "terminated".into(),
            truncated: "truncated".into(),
            simulation_times: Some("times".into()),
            action_masks: None,
            frames: BTreeMap::new(),
            evaluation: EvaluationMetadata {
                conditioning: Conditioning::InitialScenario,
                initial_action: ActionCondition::CurrentPolicy,
                continuation_policy: "policy at revision 17".into(),
                stochastic_policy: false,
                discount: 0.95,
                independent_future_streams: true,
                seed_methodology: "fixed seeds".into(),
                truncation_handling: "preserve prefixes".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            (
                "obs".into(),
                Array::f64(
                    vec![5, 3],
                    vec![
                        0., 0., 10., 1., 0., 11., 1., 1., 12., 2., 2., 13., 0., 2., 14.,
                    ],
                )
                .unwrap(),
            ),
            ("actions".into(), actions),
            (
                "rewards".into(),
                Array::f64(vec![4], vec![1., -2., 3., 4.]).unwrap(),
            ),
            (
                "terminated".into(),
                Array::u8(vec![4], vec![0, 0, 0, 1]).unwrap(),
            ),
            ("truncated".into(), Array::u8(vec![4], vec![0; 4]).unwrap()),
            (
                "times".into(),
                Array::f64(vec![5], vec![0., 0.1, 0.3, 0.6, 1.]).unwrap(),
            ),
        ]),
    }
}

/// Exercise the real on-disk codec, index and Recorder validation.
pub fn saved(space: ActionSpace) -> (tempfile::TempDir, RunManifest, Chunk) {
    let dir = tempfile::tempdir().unwrap();
    let m = manifest(space.clone());
    let mut recorder =
        Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
    recorder.append(&chunk(&space)).unwrap();
    recorder.finish().unwrap();
    let recording = Recording::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let loaded = recording.load(0).unwrap();
    (dir, recording.manifest().clone(), loaded)
}

pub fn bounded(dimensions: usize) -> ActionSpace {
    ActionSpace::Box {
        low: vec![-100.; dimensions],
        high: vec![100.; dimensions],
    }
}
