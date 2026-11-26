#[path = "../../rl/examples/support/generic_run.rs"]
mod generic_run;
use rl::diagnostics::{schema::*, storage::ReadLimits, visualization::EnvironmentSemantics};
use rl_viewer::{data::*, pane::PolicySource, resolver::*};
use std::sync::Arc;
#[test]
fn categorical_demo_records_actual_irregular_snapshots_and_unmasked_trajectory_anchor() {
    let dir = tempfile::tempdir().unwrap();
    generic_run::record_categorical_demo(&dir.path().join("demo")).unwrap();
    let run = RunIndex::open(dir.path().join("demo"), ReadLimits::default()).unwrap();
    assert_eq!(run.snapshot_revisions("online"), &[0, 4, 12]);
    assert!(run.recording().manifest().environment_descriptor.is_some());
    let mut episodes = 0;
    for (i, entry) in run.recording().entries().iter().enumerate() {
        if entry.kind == RecordKind::Trajectory {
            let chunk = run.recording().load(i).unwrap();
            let Record::Trajectory(t) = &chunk.record else {
                unreachable!()
            };
            assert!(t.action_masks.is_none());
            let values = chunk.arrays[&t.observations].to_f64();
            assert_eq!(
                values[3], 0.25,
                "original step 1 is absent from fixed probes"
            );
            episodes += 1;
        }
    }
    assert_eq!(episodes, 6);
}
#[test]
fn genuine_ordinary_environment_trains_records_and_opens_without_an_adapter() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("counterbalance");
    generic_run::record(&path).unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(run.recording().manifest().run_id, generic_run::RUN_ID);
    assert!(run.recording().manifest().environment_descriptor.is_none());
    let environment = EnvironmentSemantics::new(run.recording().manifest());
    assert_eq!(environment.observation_dim(), 2);
    assert_eq!(environment.observation(1).unwrap().label, "obs[1]");
    let mut selected = Selection {
        revision: 3,
        ..Default::default()
    };
    selected.select_probe(&run, "ordinary-inputs", 0);
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let policy = resolver
        .resolve_policy(&run, &selected, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    let checked = PolicySource::check_frame(
        policy.record.as_ref().expect("recorded ordinary policy"),
        environment.action_space(),
        32 * 1024,
    )
    .unwrap();
    assert_eq!(checked.batch.rows(), 3);
    assert_eq!(checked.batch.actions().shape(), &[3]);
    assert_eq!(checked.frame.revision(), 3);
    assert_eq!(policy.status, ResolveStatus::Ready);
    assert!(
        run.candidates(RecordKind::Critic, "online", None, 3)
            .next()
            .is_none(),
        "the run contains no state-value critic"
    );
    assert!(
        run.candidates(RecordKind::Empirical, "online", Some("ordinary-inputs"), 3)
            .next()
            .is_none(),
        "episode samples cannot establish conditional evidence"
    );
    let ctx = eframe::egui::Context::default();
    let mut laboratory =
        rl_viewer::ui::Laboratory::from_context(&ctx, path.to_string_lossy().into_owned()).unwrap();
    let mut output = ctx.run_ui(Default::default(), |ui| laboratory.show(ui));
    output.textures_delta.clear();
    laboratory.select_view(rl_viewer::ui::View::PolicyMap);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !laboratory.policy_pane().ready() {
        let mut output = ctx.run_ui(
            eframe::egui::RawInput {
                screen_rect: Some(eframe::egui::Rect::from_min_size(
                    eframe::egui::Pos2::ZERO,
                    eframe::egui::vec2(1440., 900.),
                )),
                ..Default::default()
            },
            |ui| laboratory.show(ui),
        );
        output.textures_delta.clear();
        assert!(laboratory.notice().is_none(), "{:?}", laboratory.notice());
        assert!(
            std::time::Instant::now() < deadline,
            "{:?}",
            laboratory.policy_pane().status()
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(!laboratory.policy_frame().unwrap().grid.regular());
    assert!(
        laboratory
            .policy_pane()
            .status()
            .unwrap()
            .contains("surface unavailable")
    );
    println!(
        "run={} seed={} records={} revisions={:?} descriptor=false frames=false branching=false",
        generic_run::RUN_ID,
        generic_run::SEED,
        run.recording().entries().len(),
        run.revisions()
    );
}
