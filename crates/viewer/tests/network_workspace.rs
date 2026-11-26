#[path = "support/network.rs"]
mod fixtures;
#[path = "support/policy_ui.rs"]
mod support;

use rl_viewer::ui::View;
use std::time::{Duration, Instant};

fn settle(h: &mut support::Harness) {
    let end = Instant::now() + Duration::from_secs(20);
    loop {
        h.frame(vec![]);
        if h.app.network_ready() && h.app.snapshot_preparation_idle() {
            h.frame(vec![]);
            return;
        }
        assert!(Instant::now() < end, "{:?}", h.text);
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn network_has_three_compact_plots_and_no_statistics_sidebar() {
    let dir = support::fixture(3, 0, true);
    let mut h = support::Harness::new(&dir, eframe::egui::vec2(1920., 1080.));
    h.app.select_view(View::Network);
    settle(&mut h);
    assert!(!h.app.network_frame().unwrap().geometry.paths.is_empty());
    for title in [
        "Local geometry",
        "Propagation by layer",
        "Representation paths",
    ] {
        assert!(
            h.text.iter().any(|(t, _)| t == title),
            "missing {title}: {:?}",
            h.text
        );
    }
    assert!(h.has("Local nonlinearity"));
    for removed in [
        "Layer statistics",
        "Model checkpoint",
        "Reference point A",
        "Use a different B",
        "Historical inference only",
        "Covariance-normalized",
    ] {
        assert!(
            !h.text.iter().any(|(t, _)| t.contains(removed)),
            "visible: {removed}"
        );
    }
    assert!(!h.has("yaw"));
    h.click("Slice");
    h.click("3D");
    settle(&mut h);
    assert!(h.has("yaw") && h.has("pitch"));
    assert!(h.app.network_frame().unwrap().geometry.metric_3d.is_some());
    h.click("2D");
    settle(&mut h);
    assert!(!h.has("yaw") && !h.has("pitch"));
    assert!(h.app.network_frame().unwrap().geometry.metric_3d.is_none());
    h.click("Slice");
    h.click("Details");
    let science = [
        "inverse activation-covariance",
        "0.001",
        "convex",
        "not guaranteed",
        "fixed-reference PCA",
        "Exact finite comparisons",
        "Snapshot source sequence",
        "Raw output",
    ];
    let mut seen = [false; 8];
    let position = h
        .text
        .iter()
        .find(|(text, _)| text.starts_with("Shown k "))
        .unwrap()
        .1
        .center()
        + eframe::egui::vec2(0., 180.);
    for _ in 0..16 {
        for (index, note) in science.iter().enumerate() {
            seen[index] |= h.has(note);
        }
        if seen.iter().all(|seen| *seen) {
            break;
        }
        h.frame(vec![
            eframe::egui::Event::PointerMoved(position),
            eframe::egui::Event::MouseWheel {
                unit: eframe::egui::MouseWheelUnit::Point,
                delta: eframe::egui::vec2(0., -180.),
                phase: eframe::egui::TouchPhase::Move,
                source: eframe::egui::MouseWheelSource::Unknown,
                modifiers: eframe::egui::Modifiers::NONE,
            },
        ]);
    }
    for (note, seen) in science.iter().zip(seen) {
        assert!(seen, "Details missing {note}: {:?}", h.text);
    }
}

#[test]
fn network_transport_skips_metrics_preserves_inputs_and_publishes_the_exact_checkpoint() {
    let dir = fixtures::fixture();
    let mut h = support::Harness::new(&dir, eframe::egui::vec2(1920., 1080.));
    h.app.select_view(View::Network);
    settle(&mut h);
    let input = h.app.selection().input.clone();
    let basis = serde_json::to_string(&h.app.network_frame().unwrap().geometry.pca_bases).unwrap();
    for label in ["‹", "‹"] {
        h.click(label);
        settle(&mut h);
    }
    assert_eq!(h.app.selection().revision, 0);
    h.click("Slice");
    h.click("3D");
    settle(&mut h);
    h.click("Slice");
    for revision in [100, 250] {
        h.click("›");
        settle(&mut h);
        assert_eq!(h.app.selection().revision, revision);
        let frame = h.app.network_frame().unwrap();
        assert_eq!(frame.snapshot.revision, revision);
        assert_eq!(frame.snapshot.name, "actor");
        assert!(frame.geometry.metric_3d.is_some());
        assert_eq!(
            serde_json::to_string(&frame.geometry.pca_bases).unwrap(),
            basis
        );
        assert_eq!(h.app.selection().input, input);
    }
    h.click("Rate");
    h.click("Loop");
    h.click("Rate");
    h.click("›");
    settle(&mut h);
    assert_eq!(h.app.selection().revision, 0);
    h.frame(vec![eframe::egui::Event::Key {
        key: eframe::egui::Key::ArrowRight,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: eframe::egui::Modifiers::NONE,
    }]);
    settle(&mut h);
    assert_eq!(h.app.selection().revision, 100);
    assert_eq!(h.app.selection().input, input);
}

#[test]
fn pruning_a_displayed_checkpoint_requires_explicit_navigation() {
    let dir = fixtures::fixture();
    let mut h = support::Harness::new(&dir, eframe::egui::vec2(1920., 1080.));
    h.app.select_view(View::Network);
    settle(&mut h);
    let selected = h.app.network_frame().unwrap().snapshot.key.clone();
    let run = rl_viewer::data::RunIndex::open(
        dir.path().join("run"),
        rl::diagnostics::storage::ReadLimits::default(),
    )
    .unwrap();
    let index = run.by_sequence(selected.sequence).unwrap();
    std::fs::remove_file(run.root().join(&run.recording().entries()[index].file)).unwrap();
    for _ in 0..5 {
        h.frame(vec![]);
    }
    assert_eq!(
        h.app.selection().revision,
        250,
        "never silently snap to earlier weights"
    );
    assert!(h.app.network_frame().is_none());
    h.click("Unavailable");
    assert!(h.has("Checkpoint unavailable; choose another snapshot"));
    h.click("Unavailable");
    h.click("‹");
    settle(&mut h);
    assert_eq!(
        h.app.selection().revision,
        100,
        "manual Back chooses the nearest earlier eligible snapshot"
    );
}

#[test]
fn pruning_then_appending_a_same_revision_sibling_does_not_replace_the_request() {
    let (dir, mut writer) = fixtures::recorder();
    writer
        .append(&fixtures::snapshot(0, "online", &["actor"]))
        .unwrap();
    writer
        .append(&fixtures::snapshot(100, "online", &["actor"]))
        .unwrap();
    let mut h = support::Harness::new(&dir, eframe::egui::vec2(1920., 1080.));
    h.app.select_view(View::Network);
    settle(&mut h);
    let run = rl_viewer::data::RunIndex::open(
        dir.path().join("run"),
        rl::diagnostics::storage::ReadLimits::default(),
    )
    .unwrap();
    let identity = h.app.network_frame().unwrap().snapshot.key.clone();
    let index = run.by_sequence(identity.sequence).unwrap();
    std::fs::remove_file(run.root().join(&run.recording().entries()[index].file)).unwrap();
    h.frame(vec![]);
    writer
        .append(&fixtures::snapshot(100, "online", &["actor"]))
        .unwrap();
    // RunIndex refresh is deliberately wall-clock based and bounded to 1 Hz.
    let until = Instant::now() + Duration::from_millis(1300);
    while Instant::now() < until {
        h.frame(vec![]);
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        h.app.network_frame().is_none(),
        "must not substitute a new same-revision identity"
    );
    assert_eq!(h.app.selection().revision, 100);
    h.click("Unavailable");
    assert!(h.has("Checkpoint unavailable; choose another snapshot"));
    h.click("Unavailable");
    h.click("‹");
    settle(&mut h);
    h.click("›");
    settle(&mut h);
    assert_eq!(h.app.network_frame().unwrap().snapshot.revision, 100);
    assert_ne!(h.app.network_frame().unwrap().snapshot.key, identity);
}

#[test]
fn selected_models_change_the_playback_domain_without_replacing_inputs() {
    let (dir, mut writer) = fixtures::recorder();
    writer
        .append(&fixtures::snapshot(0, "online", &["actor", "q2"]))
        .unwrap();
    writer
        .append(&fixtures::snapshot(100, "online", &["actor", "moments"]))
        .unwrap();
    writer
        .append(&fixtures::snapshot(250, "online", &["actor", "q2"]))
        .unwrap();
    writer
        .append(&fixtures::snapshot(100, "target", &["q2"]))
        .unwrap();
    writer.finish().unwrap();
    let mut h = support::Harness::new(&dir, eframe::egui::vec2(1920., 1080.));
    h.app.select_view(View::Network);
    settle(&mut h);
    let input = h.app.selection().input.clone();
    h.click("actor · Actor");
    h.click("q2 · Critic");
    settle(&mut h);
    h.click("‹");
    settle(&mut h);
    assert_eq!(h.app.selection().revision, 0);
    assert_eq!(h.app.network_frame().unwrap().snapshot.name, "q2");
    h.click("q2 · Critic");
    h.click("moments · Second-moment critic");
    settle(&mut h);
    assert_eq!(h.app.selection().revision, 100);
    assert_eq!(
        h.app.network_frame().unwrap().snapshot.name,
        "moments.second"
    );
    h.click("›");
    settle(&mut h);
    assert_eq!(
        h.app.selection().revision,
        100,
        "singleton snapshot remains selected"
    );
    assert_eq!(h.app.selection().input, input);
    h.click("online");
    h.click("target");
    settle(&mut h);
    assert_eq!(h.app.network_frame().unwrap().snapshot.role, "target");
    assert_eq!(h.app.network_frame().unwrap().snapshot.name, "q2");
    assert_eq!(h.app.selection().input, input);
}
