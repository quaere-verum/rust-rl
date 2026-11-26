#![cfg(all(feature = "diagnostics", feature = "neural"))]
//! Numerical fingerprint of the immutable public demonstration recordings.
use std::path::Path;

use rl::{
    core::{action::Action, env::Observation},
    diagnostics::{
        models::{InspectionRequest, ModelBundle, ModelKind, ModelLimits},
        schema::{Chunk, DType, Record, RecordKind},
        storage::{ReadLimits, Recording},
        value::{ValueOutput, ValuePrediction},
    },
};
use sha2::{Digest, Sha256};

const RECORDINGS: [(&str, Option<&str>, &str); 4] = [
    (
        "rocket-ppo-release-candidate",
        None,
        "position-height-slice",
    ),
    (
        "rocket-ppo-history-200",
        Some("rocket-landing"),
        "position-height-slice",
    ),
    (
        "alpha-d4pg-release-candidate",
        Some("alpha-execution"),
        "conditional-alpha",
    ),
    (
        "alpha-cvar-release-candidate",
        Some("alpha-execution-cvar"),
        "conditional-alpha",
    ),
];

fn fingerprint(label: &str, chunk: &Chunk) {
    let mut digest = Sha256::new();
    digest.update(serde_json::to_vec(&chunk.context.identity).unwrap());
    digest.update(serde_json::to_vec(&chunk.record).unwrap());
    for (key, array) in &chunk.arrays {
        digest.update(serde_json::to_vec(&(key, array.dtype(), array.shape())).unwrap());
        match array.dtype() {
            DType::F32 => {
                for value in array.as_f32().unwrap() {
                    digest.update(value.to_le_bytes());
                }
            }
            DType::F64 => {
                for value in array.as_f64().unwrap() {
                    digest.update(value.to_le_bytes());
                }
            }
            DType::U64 => {
                for value in array.as_u64().unwrap() {
                    digest.update(value.to_le_bytes());
                }
            }
            DType::U8 => digest.update(array.as_u8().unwrap()),
        }
    }
    println!("release\t{label}\t{:x}", digest.finalize());
}

#[test]
#[ignore = "requires the complete demonstration recordings"]
fn frozen_release_recordings() {
    let root = std::env::var_os("RL_RELEASE_RECORDINGS").expect("RL_RELEASE_RECORDINGS");
    let root = Path::new(&root);
    let bundled = root.join("rocket-landing").is_dir();
    for (name, public_directory, population) in RECORDINGS {
        let directory = if bundled {
            let Some(directory) = public_directory else {
                continue;
            };
            directory
        } else {
            name
        };
        eprintln!("Checking historical inference for {name}");
        let recording = Recording::open(root.join(directory), ReadLimits::default()).unwrap();
        assert!(recording.complete(), "{name} is incomplete");
        let entries = recording.entries();
        for role in ["online", "target"] {
            let snapshots: Vec<_> = entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.kind == RecordKind::ModelSnapshot && e.model_role == role)
                .collect();
            if snapshots.is_empty() {
                assert!(name.starts_with("rocket") && role == "target");
                continue;
            }
            assert!(snapshots.len() >= 3);
            for offset in [0, snapshots.len() / 2, snapshots.len() - 1] {
                let (index, entry) = snapshots[offset];
                let label = format!("{name}/{role}/{}", entry.revision);
                let snapshot = recording.load(index).unwrap();
                fingerprint(&format!("{label}/snapshot"), &snapshot);
                let model = ModelBundle::load(&snapshot, ModelLimits::default()).unwrap();
                let models = &model.configuration().models;
                let (policy, _) = models
                    .iter()
                    .find(|(_, definition)| {
                        definition.kind.is_stochastic_policy()
                            || matches!(definition.kind, ModelKind::DeterministicPolicy)
                    })
                    .unwrap();
                let critics: Vec<_> = models
                    .iter()
                    .filter_map(|(key, definition)| {
                        matches!(
                            definition.kind,
                            ModelKind::StateValue
                                | ModelKind::ScalarQ
                                | ModelKind::QuantileQ { .. }
                        )
                        .then_some(key.clone())
                    })
                    .collect();
                assert!(!critics.is_empty(), "missing critics for {label}");
                let probe_index = entries
                    .iter()
                    .position(|e| {
                        e.kind == RecordKind::ProbeSet
                            && e.revision == entry.revision
                            && e.model_role == "online"
                            && e.probe_set.as_deref() == Some(population)
                    })
                    .unwrap_or_else(|| panic!("missing probes for {label}"));
                let probe = recording.load(probe_index).unwrap();
                let Record::ProbeSet(record) = &probe.record else {
                    unreachable!()
                };
                let states = &probe.arrays[&record.observations];
                let [count, width] = states.shape() else {
                    panic!("expected dense states")
                };
                let rows = [0, count / 2, count - 1];
                let states = states.as_f32().unwrap();
                let observations = rows
                    .iter()
                    .map(|&row| Observation::new(states[row * width..(row + 1) * width].to_vec()))
                    .collect();
                let mut request = InspectionRequest::new(policy, observations);
                request.critics = critics;
                request.fixed_actions = record.fixed_actions.as_ref().map(|key| {
                    let array = &probe.arrays[key];
                    let [_, width] = array.shape() else {
                        panic!("expected dense actions")
                    };
                    let actions = array.as_f32().unwrap();
                    rows.iter()
                        .map(|&row| {
                            Action::Continuous(actions[row * width..(row + 1) * width].to_vec())
                        })
                        .collect()
                });
                for (key, layout) in model.activation_networks() {
                    if key == *policy || request.critics.contains(&key) {
                        request
                            .layers
                            .insert(key, layout.layers().into_iter().map(|(n, _)| n).collect());
                    }
                }
                let result = model.inspect(&request).unwrap();
                fingerprint(&format!("{label}/policy"), &result.policy);
                for (key, chunk) in result.critics {
                    fingerprint(&format!("{label}/critic/{key}"), &chunk);
                    for (row, _) in rows.iter().enumerate() {
                        let prediction = ValuePrediction::from_critic(&chunk, row, None).unwrap();
                        let mean = prediction
                            .scalar(&ValueOutput::ExpectedReturn, None)
                            .unwrap();
                        let summary = if matches!(prediction, ValuePrediction::Quantiles { .. }) {
                            let forecast = prediction.ordered_quantiles().unwrap();
                            let sd = forecast
                                .values()
                                .iter()
                                .zip(forecast.masses())
                                .map(|(value, mass)| mass * (value - mean).powi(2))
                                .sum::<f64>()
                                .sqrt();
                            let tail = prediction
                                .scalar(&ValueOutput::Cvar { probability: 0.05 }, None)
                                .unwrap();
                            format!("{mean:.17e}\t{sd:.17e}\t{tail:.17e}")
                        } else {
                            format!("{mean:.17e}")
                        };
                        println!("release\t{label}/summary/{key}/{row}\t{summary}");
                    }
                }
                for (key, chunk) in result.activations {
                    fingerprint(&format!("{label}/activations/{key}"), &chunk);
                }
            }
        }
        for kind in [
            RecordKind::Policy,
            RecordKind::Critic,
            RecordKind::Empirical,
        ] {
            if let Some((index, entry)) = entries
                .iter()
                .enumerate()
                .rev()
                .find(|(_, e)| e.kind == kind && e.model_role.starts_with("reference:"))
            {
                fingerprint(
                    &format!("{name}/{}/{}:{kind:?}", entry.model_role, entry.revision),
                    &recording.load(index).unwrap(),
                );
            }
        }
    }
}
