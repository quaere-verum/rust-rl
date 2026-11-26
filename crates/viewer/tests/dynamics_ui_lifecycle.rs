#[path = "support/dynamics.rs"]
mod fixture;
#[path = "support/policy_ui.rs"]
mod support;

use eframe::egui;
use rl::diagnostics::{schema::*, storage::*};
use rl_viewer::{
    data::{RunIndex, SelectedInput},
    ui::View,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

fn recording(trajectories: bool) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let space = fixture::bounded(3);
    let mut writer = Recorder::create(
        directory.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    if trajectories {
        for revision in [17, 63] {
            let mut chunk = fixture::chunk(&space);
            chunk.context.identity.model_role = "online".into();
            chunk.context.identity.learner_update_revision = revision;
            chunk.context.counters.updates = revision;
            writer.append(&chunk).unwrap();
        }
    }
    writer
        .append(&Chunk {
            context: RecordContext::new("dynamics", 80),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::from([("loss".into(), 0.1)]),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        })
        .unwrap();
    writer.finish().unwrap();
    directory
}

fn open(directory: &tempfile::TempDir) -> support::Harness {
    let mut harness = support::Harness::new(directory, egui::vec2(1440., 900.));
    harness.app.select_view(View::Dynamics);
    harness
}

fn wait_for(
    harness: &mut support::Harness,
    description: &str,
    condition: impl Fn(&support::Harness) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        harness.frame(vec![]);
        if condition(harness) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{description}: {:?}",
            harness
                .text
                .iter()
                .map(|(text, _)| text)
                .collect::<Vec<_>>()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn settle(harness: &mut support::Harness) {
    wait_for(harness, "Dynamics did not publish a complete frame", |h| {
        h.app.dynamics_ready()
    });
    harness.frame(vec![]);
}

#[test]
fn exact_plot_click_is_queued_before_cursor_changes() {
    let directory = recording(true);
    let mut h = open(&directory);
    settle(&mut h);
    h.click("Final state");
    let outer = h.app.dynamics_plot_bounds()[0];
    let chart = egui::Rect::from_min_max(
        outer.min + egui::vec2(54., 8.),
        outer.max - egui::vec2(134., 44.),
    );
    // Default X/Y axes are coordinates 1/0; first observation is [0,0].
    let pos = chart.left_bottom();
    let button = |pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    h.frame(vec![egui::Event::PointerMoved(pos), button(true)]);
    h.frame(vec![button(false)]);
    assert!(
        matches!(
            h.app.selection().input,
            SelectedInput::TrajectoryStep { step: 4, .. }
        ),
        "exact nearest scan must not change selection inside the click paint"
    );
    wait_for(
        &mut h,
        "queued click did not select the exact first state",
        |h| {
            matches!(
                h.app.selection().input,
                SelectedInput::TrajectoryStep { step: 0, .. }
            )
        },
    );
    // The accepted async selection is painted after controls in that frame.
    // Refresh their geometry before targeting the next button.
    h.frame(vec![]);
    h.click("Final state");
    assert!(matches!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep { step: 4, .. }
    ));
    h.frame(vec![egui::Event::PointerMoved(pos), button(true)]);
    h.frame(vec![button(false)]);
    assert!(
        matches!(
            h.app.selection().input,
            SelectedInput::TrajectoryStep { step: 4, .. }
        ),
        "second click applied inside its paint"
    );
    h.size -= egui::vec2(100., 50.);
    h.frame(vec![]);
    wait_for(&mut h, "resized click did not retire", |h| {
        !h.app.dynamics_pending()
    });
    assert!(
        matches!(
            h.app.selection().input,
            SelectedInput::TrajectoryStep { step: 4, .. }
        ),
        "a queued click must not select a point after the chart transform changes"
    );
}

fn wait_for_release<T>(lease: &Weak<T>) {
    // The visible frame retires synchronously; the bounded worker clears its
    // private caches on its next wake. Do not hold an external strong lease.
    let deadline = Instant::now() + Duration::from_secs(5);
    while lease.strong_count() != 0 {
        assert!(
            Instant::now() < deadline,
            "retired Dynamics data still has {} owners",
            lease.strong_count()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn retiring_a_populated_view_releases_its_frame_and_reentry_rebuilds_it() {
    let directory = recording(true);
    let mut h = open(&directory);
    settle(&mut h);

    for destination in [View::PolicyMap, View::Value, View::Network] {
        let frame = h.app.dynamics_frame().unwrap();
        assert_eq!(frame.state.revision, 63);
        let state = Arc::downgrade(&frame.state);
        let selected = Arc::downgrade(&frame.selected);

        h.app.select_view(destination);
        assert!(h.app.dynamics_frame().is_none(), "{destination:?}");
        assert!(!h.app.dynamics_ready(), "{destination:?}");
        wait_for_release(&state);
        wait_for_release(&selected);

        h.app.select_view(View::Dynamics);
        assert!(h.app.dynamics_frame().is_none());
        settle(&mut h);
        assert_eq!(h.app.dynamics_frame().unwrap().state.revision, 63);
        h.click("Spatial settings");
        assert!(h.has("Evaluation k=63"));
        h.click("Spatial settings");
    }
}

#[test]
fn clearing_and_reopening_the_recording_discards_the_previous_episode_selection() {
    let directory = recording(true);
    let mut h = open(&directory);
    settle(&mut h);
    h.app.select_revision(17).unwrap();
    settle(&mut h);
    h.click("Final state");
    assert!(matches!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 0,
            step: 4
        }
    ));
    let state = Arc::downgrade(&h.app.dynamics_frame().unwrap().state);
    let selected = Arc::downgrade(&h.app.dynamics_frame().unwrap().selected);

    h.app.clear_recording();
    assert!(h.app.dynamics_frame().is_none());
    assert!(!h.app.dynamics_ready());
    wait_for_release(&state);
    wait_for_release(&selected);
    h.frame(vec![]);
    assert!(h.has("Open recording"));
    assert!(h.has("Recent recordings"));

    // Reopen through the ordinary recent-recording control.
    h.click("Run");
    wait_for(&mut h, "recording did not reopen", |h| {
        !h.has("Open recording")
    });
    h.app.select_view(View::Dynamics);
    settle(&mut h);
    assert_eq!(h.app.dynamics_frame().unwrap().state.revision, 63);
    assert!(matches!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 1,
            step: 0
        }
    ));
    assert_eq!(h.app.dynamics_frame().unwrap().selected.key.sequence, 1);
}

#[test]
fn metrics_without_trajectories_show_missing_evidence_and_allow_retry() {
    let directory = recording(false);
    let mut h = open(&directory);
    wait_for(&mut h, "missing evidence did not settle", |h| {
        !h.app.dynamics_pending() && h.has("No compatible unconditional evaluation trajectories")
    });
    assert!(h.app.dynamics_frame().is_none());
    assert!(!h.app.dynamics_ready());
    h.click("Unavailable");
    assert!(h.has("No compatible evaluation trajectories"));
    assert!(!h.has("Evaluation revision 80"));

    h.click("Retry");
    wait_for(&mut h, "retry did not report missing evidence", |h| {
        !h.app.dynamics_pending() && h.has("No compatible unconditional evaluation trajectories")
    });
    assert!(h.app.dynamics_frame().is_none());
    assert!(!h.app.dynamics_ready());
}

#[test]
fn live_source_removal_retires_the_frame_until_the_user_retries() {
    let directory = recording(true);
    let mut h = open(&directory);
    settle(&mut h);
    let identity = h.app.dynamics_frame().unwrap().selected.key.clone();
    let selection = h.app.selection().clone();
    let state = Arc::downgrade(&h.app.dynamics_frame().unwrap().state);
    let run = RunIndex::open(directory.path().join("run"), ReadLimits::default()).unwrap();
    let index = run.by_sequence(identity.sequence).unwrap();
    let file = run.root().join(&run.recording().entries()[index].file);
    let saved = std::fs::read(&file).unwrap();
    std::fs::remove_file(&file).unwrap();

    // The app's normal one-second refresh detects the removed immutable file.
    wait_for(
        &mut h,
        "live refresh did not retire the missing source",
        |h| h.app.dynamics_frame().is_none() && h.has("source file is missing"),
    );
    assert!(!h.app.dynamics_ready());
    assert_eq!(h.app.selection().revision, selection.revision);
    assert_eq!(h.app.selection().input, selection.input);
    wait_for_release(&state);

    std::fs::write(&file, saved).unwrap();
    h.frame(vec![]);
    assert!(h.app.dynamics_frame().is_none());
    assert!(!h.app.dynamics_ready());
    h.click("Unavailable");
    assert!(h.has("source file is missing"));
    h.click("Retry");
    settle(&mut h);
    assert_eq!(h.app.dynamics_frame().unwrap().selected.key, identity);
    assert_eq!(h.app.dynamics_frame().unwrap().state.revision, 63);
    assert_eq!(h.app.selection().input, selection.input);
}
#[test]
fn coverage_controls_reuse_preparation_and_domain_recompute_replaces_the_grid() {
    let directory = recording(true);
    let mut h = open(&directory);
    settle(&mut h);
    let state = h.app.dynamics_frame().unwrap().state.clone();
    let counters = h.app.dynamics_counters();
    h.click("Current");
    assert!(Arc::ptr_eq(&state, &h.app.dynamics_frame().unwrap().state));
    h.click("New");
    assert!(Arc::ptr_eq(&state, &h.app.dynamics_frame().unwrap().state));
    h.click("Cumulative");
    assert_eq!(counters, h.app.dynamics_counters());
    h.click("Spatial settings");
    h.click("Plot bounds");
    h.click("Apply");
    h.click("Spatial settings");
    settle(&mut h);
    assert!(
        h.app
            .dynamics_frame()
            .unwrap()
            .state
            .key
            .domain()
            .manual_bounds()
            .is_some()
    );
    let episode_key = h.app.dynamics_frame().unwrap().selected.key.clone();
    let automatic_domain = rl_viewer::dynamics::history::DomainSpec::automatic(1);
    h.click("Spatial settings");
    h.frame(vec![]);
    h.click("Recompute");
    assert!(
        !h.app.dynamics_ready()
            || h.app
                .dynamics_frame()
                .is_some_and(|frame| frame.state.key.domain() == automatic_domain
                    && frame.selected.key == episode_key
                    && frame.state.revision == state.revision
                    && frame.state.key.axes() == state.key.axes()),
        "A ready frame after recompute must already use the new automatic domain"
    );
    settle(&mut h);
    let current = h.app.dynamics_frame().unwrap();
    assert_ne!(current.state.key.domain(), state.key.domain());
    assert_eq!(current.state.key.domain(), automatic_domain);
    assert_eq!(current.selected.key, episode_key);
    assert!(current.state.key.domain().manual_bounds().is_none());
}
#[test]
fn invalid_manual_bounds_do_not_submit_and_valid_bounds_replace_the_visible_domain() {
    let directory = recording(true);
    let mut h = open(&directory);
    settle(&mut h);
    h.click("Spatial settings");
    h.click("Plot bounds");
    let state = h.app.dynamics_frame().unwrap().state.clone();
    let counters = h.app.dynamics_counters();
    let x_row = h.text.iter().rev().find(|(text, _)| text == "X").unwrap().1;
    let lower_x = h
        .text
        .iter()
        .find(|(text, rect)| text == "0" && (rect.center().y - x_row.center().y).abs() < 10.)
        .unwrap()
        .1;
    h.click_at(lower_x.center());
    let select_all = || egui::Event::Key {
        key: egui::Key::A,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::COMMAND,
    };
    h.frame(vec![select_all(), egui::Event::Text("NaN".into())]);
    h.click("Apply");
    assert!(h.has("finite increasing spans"));
    assert_eq!(counters, h.app.dynamics_counters());
    assert!(Arc::ptr_eq(&state, &h.app.dynamics_frame().unwrap().state));
    h.click("NaN");
    h.frame(vec![select_all(), egui::Event::Text("-1.5".into())]);
    h.click("Apply");
    // An obsolete automatic-domain answer may not become the visible frame.
    if let Some(frame) = h.app.dynamics_frame() {
        assert_eq!(
            frame.state.key.domain().manual_bounds(),
            Some([[-1.5, 2.], [0., 2.]])
        );
    }
    settle(&mut h);
    assert_eq!(
        h.app
            .dynamics_frame()
            .unwrap()
            .state
            .reference
            .grid
            .grid
            .bounds(),
        [[-1.5, 2.], [0., 2.]]
    );
}
