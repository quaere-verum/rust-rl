#![cfg(feature = "diagnostics")]
use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*},
};
use std::collections::BTreeMap;
fn snapshot(k: u64, role: &str) -> Chunk {
    let mut c = RecordContext::new("scale", k);
    c.identity.model_role = role.into();
    c.identity.checkpoint_id = Some(format!("{role}-{k}"));
    Chunk {
        context: c,
        record: Record::ModelSnapshot(ModelSnapshotRecord {
            id: format!("{role}-{k}"),
            architecture: "fixture".into(),
            configuration: serde_json::json!({}),
            models: BTreeMap::from([(
                "weights".into(),
                ModelArtifact {
                    bytes: "weights".into(),
                    signature: "fixture".into(),
                    parameter_shapes: BTreeMap::new(),
                },
            )]),
            capabilities: vec![],
        }),
        arrays: BTreeMap::from([(
            "weights".into(),
            Array::u8(vec![2048], vec![k as u8; 2048]).unwrap(),
        )]),
    }
}
#[test]
fn retention_is_explicit_per_role_and_preserves_independent_readers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "scale",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig {
            max_total_bytes: 24_000,
            compress: false,
            ..Default::default()
        },
    )
    .unwrap()
    .with_snapshot_retention(2)
    .unwrap();
    writer.append(&snapshot(0, "online")).unwrap();
    let mut reader = Recording::open(&path, ReadLimits::default()).unwrap();
    let old = reader.source(0).unwrap();
    for k in 1..20 {
        assert!(matches!(
            writer.append(&snapshot(k, "online")).unwrap(),
            AppendOutcome::Committed { .. }
        ));
    }
    writer.append(&snapshot(20, "target")).unwrap();
    writer.finish().unwrap();
    reader.refresh().unwrap();
    assert!(!reader.is_retained(0));
    assert!(reader.is_retained(18));
    assert!(reader.is_retained(19));
    assert!(reader.is_retained(20));
    assert!(old.load().unwrap_err().to_string().contains("retention"));
    assert_eq!(reader.status().unwrap().pruned_records, 18);
    let files = std::fs::read_dir(path.join("chunks")).unwrap().count();
    assert_eq!(files, 3);
    assert!(writer.status().committed_bytes - writer.status().reclaimed_bytes < 24_000);
    std::fs::remove_file(path.join(&reader.entries()[19].file)).unwrap();
    assert!(reader.is_retained(19));
    assert!(reader.load(19).is_err()); // Missing retained data is corruption, not intentional pruning.
}
#[cfg(feature = "neural")]
#[test]
fn compact_probes_preserve_full_values_and_explicit_neuron_identity() {
    use rand::SeedableRng;
    use rl::{
        diagnostics::{compact::*, probes},
        nn::{matrix, network::Mlp},
    };
    let model = Mlp::new(
        2,
        &[5, 3],
        1,
        &mut rand_chacha::ChaCha8Rng::seed_from_u64(7),
    )
    .unwrap();
    let input = matrix(vec![0.2, -0.3, 0.4, 0.1, -1., 2.], 3, 2);
    let names = ["layers.0.pre", "layers.1.post"];
    let context = RecordContext::new("scale", 0);
    let full =
        probes::activations(context.clone(), "policy", &model, input.clone(), &names).unwrap();
    let batched = activations_batched(
        context.clone(),
        "policy",
        &model,
        input.clone(),
        &names,
        &ActivationOptions {
            batch_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    for name in names {
        assert_eq!(
            full.arrays[name].as_f32().unwrap(),
            batched.arrays[name].as_f32().unwrap()
        );
    }
    let summary = activations_batched(
        context.clone(),
        "policy",
        &model,
        input.clone(),
        &names,
        &ActivationOptions {
            storage: ActivationStorage::Summary,
            batch_size: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(summary.arrays.is_empty());
    let (Record::Activations(a), Record::Activations(b)) = (&full.record, &summary.record) else {
        panic!()
    };
    for name in names {
        assert!((a.summaries[name].mean.unwrap() - b.summaries[name].mean.unwrap()).abs() < 1e-12);
        assert!(
            (a.summaries[name].standard_deviation.unwrap()
                - b.summaries[name].standard_deviation.unwrap())
            .abs()
                < 1e-12
        );
    }
    let selected = activations_batched(
        context.clone(),
        "policy",
        &model,
        input.clone(),
        &names,
        &ActivationOptions {
            storage: ActivationStorage::Neurons(vec![2, 0]),
            batch_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let Record::Activations(a) = &selected.record else {
        panic!()
    };
    for name in names {
        assert_eq!(
            selected.arrays[&a.selected_neurons[name]].as_u64().unwrap(),
            &[2, 0]
        );
        let width = full.arrays[name].shape()[1];
        let expected: Vec<_> = full.arrays[name]
            .as_f32()
            .unwrap()
            .chunks(width)
            .flat_map(|r| [r[2], r[0]])
            .collect();
        assert_eq!(selected.arrays[name].as_f32().unwrap(), expected);
    }
    assert!(
        activations_batched(
            context,
            "policy",
            &model,
            input,
            &names,
            &ActivationOptions {
                max_recorded_elements: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn buffered_pages_are_bounded_atomic_and_interruption_is_visible() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("buffered");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "scale",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap()
    .with_buffering(BufferConfig {
        max_records: 3,
        max_bytes: 8192,
    })
    .unwrap();
    let metric = |k| Chunk {
        context: RecordContext::new("scale", k),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::from([("loss".into(), k as f64)]),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::new(),
    };
    assert!(matches!(
        writer.append(&metric(0)).unwrap(),
        AppendOutcome::Buffered { sequence: 0 }
    ));
    let mut reader = Recording::open(&path, ReadLimits::default()).unwrap();
    assert!(reader.entries().is_empty());
    assert!(reader.pending_window().unwrap().is_some());
    writer.append(&metric(1)).unwrap();
    writer.append(&metric(2)).unwrap();
    assert_eq!(writer.pending_records(), 0);
    reader.refresh().unwrap();
    assert_eq!(reader.entries().len(), 3);
    assert!(reader.pending_window().unwrap().is_none());
    assert_eq!(reader.entries()[0].file, reader.entries()[2].file);
    for i in 0..3 {
        reader.load(i).unwrap();
    }
    writer.append(&metric(3)).unwrap();
    drop(writer);
    reader.refresh().unwrap();
    assert!(!reader.complete());
    assert!(reader.pending_window().unwrap().is_some());
    assert_eq!(reader.entries().len(), 3);
    let mut bytes = std::fs::read(path.join(&reader.entries()[0].file)).unwrap();
    let index = reader.entries()[1].page.as_ref().unwrap().offset as usize;
    bytes[index] ^= 1;
    std::fs::write(path.join(&reader.entries()[0].file), bytes).unwrap();
    assert!(reader.load(1).is_err());
    reader.load(0).unwrap();
    reader.load(2).unwrap();
}
