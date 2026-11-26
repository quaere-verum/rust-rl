#[path = "support/network.rs"]
mod support;
use rl::diagnostics::{
    models::{ModelBundle, ModelLimits},
    schema::*,
    storage::*,
};
use rl_viewer::{
    data::{Loader, RunIndex},
    interactive::Interactive,
    network::{Request, work::NetworkWorker},
    snapshot_catalog::{SnapshotCatalog, SnapshotEntry, SnapshotPanel},
    timeline::{Direction, RevisionRange, Timeline},
};
use std::time::{Duration, Instant};

fn catalog(run: &RunIndex, role: &str, name: &str) -> SnapshotCatalog {
    let mut interactive = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(4 * 1024 * 1024, || {}).unwrap();
    let mut catalog = SnapshotCatalog::new(
        run,
        role,
        SnapshotPanel::Network {
            name: Some(name.into()),
        },
        &interactive,
    )
    .unwrap();
    settle(&mut catalog, run, &mut loader, &mut interactive);
    catalog
}
fn settle(
    c: &mut SnapshotCatalog,
    run: &RunIndex,
    loader: &mut Loader,
    interactive: &mut Interactive,
) {
    let end = Instant::now() + Duration::from_secs(10);
    while c.pending() {
        loader.poll();
        c.advance(run, loader, interactive).unwrap();
        assert!(Instant::now() < end, "snapshot validation timeout");
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn named_networks_ignore_dense_diagnostics_and_keep_roles_separate() {
    let dir = support::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    for name in ["actor", "q1", "q2", "moments.first", "moments.second"] {
        let c = catalog(&run, "online", name);
        assert_eq!(c.positions(), [0, 100, 250]);
        assert!(c.selected(50).is_none());
        assert!(c.entries().iter().all(|e| e.name == name));
    }
    assert!(catalog(&run, "target", "q1").positions().is_empty());
    assert!(catalog(&run, "online", "missing").positions().is_empty());
}
#[test]
fn missing_incompatible_corrupt_and_duplicate_snapshot_candidates() {
    let (dir, mut writer) = support::recorder();
    writer
        .append(&support::snapshot(0, "online", &["actor", "q1", "q2"]))
        .unwrap();
    writer
        .append(&support::snapshot(100, "online", &["actor", "q1"]))
        .unwrap();
    writer
        .append(&support::snapshot(
            250,
            "online",
            &["actor", "q1", "q2", "moments"],
        ))
        .unwrap();
    let mut corrupt = support::snapshot(250, "online", &["actor", "q1", "q2"]);
    if let Record::ModelSnapshot(meta) = &corrupt.record {
        let bytes = meta.models["actor"].bytes.clone();
        corrupt
            .arrays
            .insert(bytes, Array::u8(vec![3], vec![1, 2, 3]).unwrap());
    }
    writer.append(&corrupt).unwrap();
    let mut wrong = support::snapshot(300, "online", &["actor", "q2"]);
    if let Record::ModelSnapshot(meta) = &mut wrong.record {
        meta.configuration["observation_dim"] = 2.into();
    }
    writer.append(&wrong).unwrap();
    writer
        .append(&support::snapshot(400, "target", &["q2"]))
        .unwrap();
    let mut run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut c = catalog(&run, "online", "q2");
    assert_eq!(c.positions(), [0, 250]);
    let chosen = c.selected(250).unwrap().key.clone();
    assert_eq!(chosen.sequence, run.recording().entries()[3].sequence);
    assert_eq!(catalog(&run, "online", "moments.second").positions(), [250]);
    assert_eq!(catalog(&run, "target", "q2").positions(), [400]);

    // New same-revision weights do not replace an already admitted identity.
    writer
        .append(&support::snapshot(250, "online", &["q2"]))
        .unwrap();
    writer
        .append(&support::snapshot(500, "online", &["q2"]))
        .unwrap();
    run.refresh().unwrap();
    let mut loader = Loader::new(4 * 1024 * 1024, || {}).unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    c.advance(&run, &mut loader, &mut interactive).unwrap();
    settle(&mut c, &run, &mut loader, &mut interactive);
    assert_eq!(c.positions(), [0, 250, 500]);
    assert_eq!(c.selected(250).unwrap().key, chosen);
    let index = run.by_sequence(chosen.sequence).unwrap();
    std::fs::remove_file(run.root().join(&run.recording().entries()[index].file)).unwrap();
    c.recheck_availability(&run);
    assert_eq!(c.positions(), [0, 500]);
    assert!(
        c.selected(250).is_none(),
        "no fallback to a different sibling"
    );
    assert!(run.exact_snapshot_index(&chosen, 250, "online").is_err());
}

#[test]
fn model_selector_discovers_networks_in_separate_same_revision_artifacts() {
    let (dir, mut writer) = support::recorder();
    writer
        .append(&support::snapshot(0, "online", &["actor"]))
        .unwrap();
    writer
        .append(&support::snapshot(0, "online", &["q2"]))
        .unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let c = catalog(&run, "online", "q2");
    assert!(c.networks().iter().any(|c| c.name == "actor"));
    assert_eq!(c.positions(), [0]);
    assert_eq!(catalog(&run, "online", "actor").positions(), [0]);
}

fn submit(w: &mut NetworkWorker, run: &RunIndex, e: &SnapshotEntry) {
    let probes = support::observations();
    let index = run
        .exact_snapshot_index(&e.key, e.revision, &e.role)
        .unwrap();
    w.request(
        format!("{}:{}", e.revision, e.key.sequence),
        "fixed population and actor".into(),
        e.clone(),
        run.recording().source(index).unwrap(),
        Request {
            network: e.name.clone(),
            anchor: probes[3].clone(),
            endpoint: probes[8].clone(),
            probes,
            axes: vec![0, 1, 2],
            axis_scales: vec![1., 2., 2.],
            ranges: [[0., 1.], [-1., 1.]],
            direction_angle: 0.4,
            t: 1.,
            radius: 0.2,
            action_mask: None,
            layer: 1,
            grid: [3, 3],
            previous_basis: vec![],
        },
    );
}
fn wait(w: &mut NetworkWorker) {
    let end = Instant::now() + Duration::from_secs(10);
    while w.pending() {
        w.poll();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(w.ready(), "{:?}", w.error());
}
#[test]
fn geometry_publication_matches_exact_weights_and_keeps_the_reference_basis() {
    let dir = support::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let c = catalog(&run, "online", "actor");
    let mut w = NetworkWorker::new(|| {});
    submit(&mut w, &run, c.selected(0).unwrap());
    wait(&mut w);
    let basis = serde_json::to_string(&w.frame().unwrap().geometry.pca_bases).unwrap();
    let initial_values = w.frame().unwrap().geometry.output_values.clone();
    let mut t = Timeline::new(RevisionRange::new(0, 250));
    t.set_snapshot_positions(&c.positions());
    t.play();
    t.set_speed(100.);
    assert!(t.advance_ready(Duration::from_secs(1), w.ready()));
    assert_eq!(t.revision(), 100);
    submit(&mut w, &run, c.selected(100).unwrap());
    assert!(!w.ready());
    assert_eq!(
        w.frame().unwrap().snapshot.revision,
        0,
        "held identity is never the request"
    );
    assert!(!t.advance_ready(Duration::from_secs(30), w.ready()));
    t.pause();
    wait(&mut w);
    assert!(!t.advance_ready(Duration::from_secs(30), w.ready()));
    assert_eq!(t.revision(), 100, "pause while preparing persists");
    let published = w.frame().unwrap();
    assert_eq!(published.snapshot.key, c.selected(100).unwrap().key);
    assert_eq!(published.snapshot.revision, 100);
    assert_eq!(
        serde_json::to_string(&published.geometry.pca_bases).unwrap(),
        basis
    );
    assert_ne!(published.geometry.output_values, initial_values);
    let index = run.by_sequence(published.snapshot.key.sequence).unwrap();
    let bundle = ModelBundle::load(
        &run.recording().load(index).unwrap(),
        ModelLimits::default(),
    )
    .unwrap();
    let expected = bundle
        .numerical_network("actor")
        .unwrap()
        .forward_layers(&support::observations()[3])
        .unwrap();
    assert_eq!(&published.geometry.output_values, expected.last().unwrap());
    // Superseded work is never published, including a run/network context reset.
    submit(&mut w, &run, c.selected(0).unwrap());
    submit(&mut w, &run, c.selected(250).unwrap());
    wait(&mut w);
    assert_eq!(w.frame().unwrap().snapshot.revision, 250);
    assert_eq!(
        serde_json::to_string(&w.frame().unwrap().geometry.pca_bases).unwrap(),
        basis
    );
    w.reset();
    assert!(w.frame().is_none());
    assert!(!w.ready());
    t.set_snapshot_positions(&[250]);
    t.set_looping(true);
    t.step(Direction::Forward);
    assert_eq!(t.revision(), 250);
}
