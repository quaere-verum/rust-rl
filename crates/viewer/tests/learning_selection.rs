use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{
    data::{RunIndex, SelectedInput, Selection},
    pane::SelectionIntent,
};
use std::collections::BTreeMap;

#[test]
fn shared_revision_intent_preserves_input_identity_and_checks_recorded_bounds() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let space = ActionSpace::Discrete { count: 2 };
    let mut writer = Recorder::create(
        &path,
        RunManifest::new("selection", "selection", "fixture", 1, space.clone(), 0.9).unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let mut context = RecordContext::new("selection", 1);
    context.probe_set = Some("fixed".into());
    writer
        .append(
            &ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "stable source",
                vec![Observation::new(vec![0.])],
                None,
                space,
            )
            .unwrap()
            .record(context)
            .unwrap(),
        )
        .unwrap();
    writer
        .append(&Chunk {
            context: RecordContext::new("selection", 7),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        })
        .unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let mut selection = Selection {
        revision: 1,
        model_role: "online".into(),
        input: SelectedInput::ProbeRow {
            set: "fixed".into(),
            row: 0,
            sequence: run.recording().entries()[0].sequence,
        },
    };
    let source = selection.input.clone();
    SelectionIntent::Revision(4)
        .apply(&run, &mut selection)
        .unwrap();
    assert_eq!(selection.revision, 4);
    assert_eq!(selection.input, source);
    assert_eq!(selection.model_role, "online");
    assert!(
        SelectionIntent::Revision(9)
            .apply(&run, &mut selection)
            .is_err()
    );
    assert_eq!(selection.revision, 4);
    assert_eq!(selection.input, source);
}
