use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{comparison::Comparison, data::*};
use std::{collections::BTreeMap, sync::Arc};

fn context(revision: u64) -> RecordContext {
    let mut context = RecordContext::new("comparison", revision);
    context.probe_set = Some("fixed".into());
    context
}

fn probes(revision: u64, reversed: bool, methodology: &str, masks: bool) -> Chunk {
    let mut chunk = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        methodology,
        if reversed { [2., 1.] } else { [1., 2.] }
            .into_iter()
            .map(|value| Observation::new(vec![value]))
            .collect(),
        None,
        ActionSpace::Discrete { count: 2 },
    )
    .unwrap()
    .record(context(revision))
    .unwrap();
    let Record::ProbeSet(set) = &mut chunk.record else {
        unreachable!()
    };
    if reversed {
        chunk.arrays.insert(
            set.row_ids.clone(),
            Array::u64(vec![2], vec![1, 0]).unwrap(),
        );
    }
    // ProbeSet::record emits all-valid masks for discrete actions by default.
    // Legacy missing-mask records must explicitly remove that generated array.
    if let Some(key) = set.action_masks.take() {
        chunk.arrays.remove(&key);
    }
    if masks {
        set.action_masks = Some("mask".into());
        chunk
            .arrays
            .insert("mask".into(), Array::u8(vec![2, 2], vec![1; 4]).unwrap());
    }
    chunk
}

fn activations(revision: u64, values: [f32; 2]) -> Chunk {
    Chunk {
        context: context(revision),
        record: Record::Activations(ActivationRecord {
            network: "policy".into(),
            layers: BTreeMap::from([("layers.0.post".into(), "h".into())]),
            ..Default::default()
        }),
        arrays: BTreeMap::from([("h".into(), Array::f32(vec![2, 1], values.to_vec()).unwrap())]),
    }
}

fn fixture(changed_methodology: bool, masks: bool) -> (tempfile::TempDir, RunIndex, Arc<Chunk>) {
    fixture_with_policy(changed_methodology, masks, true)
}

fn fixture_with_policy(
    changed_methodology: bool,
    masks: bool,
    policy: bool,
) -> (tempfile::TempDir, RunIndex, Arc<Chunk>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "comparison",
            "test",
            "ppo",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let source = Arc::new(probes(1, false, "stable grid", false));
    writer.append(&source).unwrap(); // 0
    writer.append(&activations(1, [1., 2.])).unwrap(); // 1
    if policy {
        writer
            .append(&Chunk {
                context: context(11),
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters: PolicyParameters::Deterministic,
                    entropy: None,
                    entropy_method: None,
                    masks: None,
                    stochastic: false,
                }),
                arrays: BTreeMap::from([("a".into(), Array::u64(vec![2], vec![0, 1]).unwrap())]),
            })
            .unwrap();
    } // 2 when a policy exists
    writer
        .append(&probes(
            21,
            true,
            if changed_methodology {
                "different grid"
            } else {
                "stable grid"
            },
            masks,
        ))
        .unwrap(); // 3
    writer.append(&activations(21, [20., 10.])).unwrap(); // 4
    writer
        .append(&Chunk {
            context: context(31),
            record: Record::Critic(CriticRecord {
                representation: CriticRepresentation::Scalar { values: "v".into() },
                action_condition: ActionCondition::StateValue,
                actions: None,
                continuation_policy: None,
                continuation_stochastic: None,
            }),
            arrays: BTreeMap::from([("v".into(), Array::f32(vec![2], vec![200., 100.]).unwrap())]),
        })
        .unwrap(); // 5
    // This later definition must never reinterpret older measurement rows.
    writer
        .append(&probes(35, false, "new unrelated grid", false))
        .unwrap(); // 6
    writer
        .append(&Chunk {
            context: context(41),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        })
        .unwrap();
    writer.finish().unwrap();
    (
        dir,
        RunIndex::open(path, ReadLimits::default()).unwrap(),
        source,
    )
}

#[test]
fn opening_a_main_run_resets_the_reference_without_recreating_comparison() {
    let (_dir, run, _) = fixture(false, false);
    let (_other_dir, other, _) = fixture(false, false);
    let mut comparison = Comparison::new();
    assert!(comparison.reference_run().is_none());
    assert!(comparison.set_reference_revision(11).is_err());
    comparison.reset_reference_to(&run).unwrap();
    assert_eq!(comparison.reference_run().unwrap().root(), run.root());
    assert_eq!(comparison.reference_role(), "online");
    assert_eq!(comparison.reference_revision(), 1);
    comparison.set_reference_revision(37).unwrap();
    comparison.reset_reference_to(&other).unwrap();
    assert_eq!(comparison.reference_run().unwrap().root(), other.root());
    assert_eq!(comparison.reference_revision(), 1);
}

#[test]
fn activation_only_reference_starts_at_its_first_recorded_revision() {
    let (_dir, run, _) = fixture_with_policy(false, false, false);
    let mut comparison = Comparison::new();
    comparison.reset_reference_to(&run).unwrap();
    assert_eq!(comparison.reference_revision(), 1);
    for requested in [11, 21, 31, 37, 11] {
        comparison.set_reference_revision(requested).unwrap();
        assert_eq!(comparison.reference_revision(), requested);
    }
    for invalid in [0, 42] {
        assert!(comparison.set_reference_revision(invalid).is_err());
        assert_eq!(comparison.reference_revision(), 11);
    }
}

#[test]
fn shared_reference_policy_follows_time_without_reanchoring_the_input() {
    use rl_viewer::resolver::{ProbeResolver, ResolveStatus};
    let (_dir, run, _) = fixture(false, false);
    let mut comparison = Comparison::new();
    comparison.reset_reference_to(&run).unwrap();
    let selection = Selection {
        revision: 41,
        input: SelectedInput::ProbeRow {
            set: "fixed".into(),
            row: 0,
            sequence: 0,
        },
        ..Default::default()
    };
    let input = selection.input.clone();
    let mut policy_resolver = ProbeResolver::new(1024 * 1024);
    // Reference time is independent of the learner request. The source anchor
    // stays immutable while Policy selects its own predecessor.
    for requested in [11, 21, 31, 37, 11] {
        comparison.set_reference_revision(requested).unwrap();
        let reference_selection = Selection {
            revision: comparison.reference_revision(),
            input: input.clone(),
            model_role: comparison.reference_role().into(),
        };
        let reference = comparison.reference_run().unwrap();
        let load = |index| Ok(Some(Arc::new(reference.recording().load(index)?)));
        let policy = policy_resolver
            .resolve_policy(reference, &reference_selection, load)
            .unwrap();
        assert_eq!(policy.status, ResolveStatus::Ready);
        assert_eq!(policy.requested_revision, requested);
        let policy = policy.record.unwrap();
        assert_eq!(
            (policy.revision(), policy.row, policy.probe_key.sequence),
            (11, 0, 0)
        );
        assert_eq!(policy.key.root, *run.root());
        assert_eq!(comparison.reference_revision(), requested);
        assert_eq!(selection.input, input);
        assert_eq!(selection.revision, 41);
    }
}

#[test]
fn reference_resolution_distinguishes_missing_source_from_unmatched_input() {
    use rl_viewer::resolver::{ProbeResolver, ResolveRejection, ResolveStatus};
    let (_dir, run, _) = fixture(false, false);
    for (sequence, rejection) in [
        (99, ResolveRejection::MissingSource),
        // Definition #6 belongs to a different construction from the earlier policy.
        (6, ResolveRejection::InputMismatch),
    ] {
        let selection = Selection {
            revision: 11,
            model_role: "online".into(),
            input: SelectedInput::ProbeRow {
                set: "fixed".into(),
                sequence,
                row: 0,
            },
        };
        let view = ProbeResolver::new(1024 * 1024)
            .resolve_policy(&run, &selection, |index| {
                Ok(Some(Arc::new(run.recording().load(index)?)))
            })
            .unwrap();
        assert_eq!(view.status, ResolveStatus::Unavailable);
        assert_eq!(view.rejection, Some(rejection));
        assert!(view.record.is_none());
    }
}
