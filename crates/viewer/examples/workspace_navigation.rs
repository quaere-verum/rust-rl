//! Measure real Policy-to-Value publication and snapshot-catalogue readiness.
//! Arguments: recording, revision, probe set, source sequence, row.
use eframe::egui;
use rl_viewer::{
    data::SelectedInput,
    ui::{Laboratory, View},
};
use std::time::{Duration, Instant};

fn paint(app: &mut Laboratory, ctx: &egui::Context) {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280., 800.),
            )),
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    let shapes = std::mem::take(&mut output.shapes);
    std::hint::black_box(ctx.tessellate(shapes, output.pixels_per_point));
    output.textures_delta.clear();
}

fn main() -> rl::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(rl::Error::Config(
            "expected recording revision probe sequence row".into(),
        ));
    }
    let parse = |value: &str| {
        value
            .parse::<u64>()
            .map_err(|_| rl::Error::Config("invalid numeric argument".into()))
    };
    let revision = parse(&args[1])?;
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, args[0].clone())?;
    app.select_revision(revision)?;
    app.select_input(SelectedInput::ProbeRow {
        set: args[2].clone(),
        sequence: parse(&args[3])?,
        row: usize::try_from(parse(&args[4])?)
            .map_err(|_| rl::Error::Config("row is too large".into()))?,
    })?;
    for view in [View::PolicyMap, View::Value] {
        app.select_view(view);
        let start = Instant::now();
        let mut first_frame = None;
        let mut next_report = Duration::from_secs(1);
        loop {
            paint(&mut app, &ctx);
            let frame = match view {
                View::PolicyMap => app
                    .policy_frame()
                    .is_some_and(|f| f.requested_revision == revision),
                View::Value => app
                    .value_frame()
                    .is_some_and(|f| f.requested_revision == revision),
                _ => unreachable!(),
            };
            if frame && first_frame.is_none() {
                first_frame = Some(start.elapsed());
            }
            if frame && app.snapshot_preparation_idle() {
                println!(
                    "{view:?},first_frame_ms={:.3},catalog_ready_ms={:.3},policy={:?},value={:?}",
                    first_frame.unwrap().as_secs_f64() * 1000.,
                    start.elapsed().as_secs_f64() * 1000.,
                    app.policy_metrics(),
                    app.value_metrics()
                );
                break;
            }
            if start.elapsed() >= next_report {
                eprintln!(
                    "{view:?}: {:.1}s elapsed; frame={}, catalogue idle={}, value pending={}, notice={:?}, value view reserved={}, owned={}, grid={:?}, value metrics={:?}",
                    start.elapsed().as_secs_f64(),
                    frame,
                    app.snapshot_preparation_idle(),
                    app.value_pending(),
                    app.notice(),
                    app.value_view().state_reservation_bytes(),
                    app.value_view().owned_state_bytes(),
                    app.value_frame().map(|f| (
                        f.grid.slice.resolution,
                        f.grid.values.len(),
                        f.grid.empirical.conditions.len()
                    )),
                    app.value_metrics()
                );
                next_report += Duration::from_secs(1);
            }
            if start.elapsed() >= Duration::from_secs(30) {
                return Err(rl::Error::State(format!(
                    "{view:?} did not settle within 30s"
                )));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    Ok(())
}
