//! Measure CPU-side publication when switching between Policy and Value.
//! Arguments: recording, revision, round trips.
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
    if args.len() != 3 {
        return Err(rl::Error::Config(
            "expected recording revision round-trips".into(),
        ));
    }
    let revision = args[1]
        .parse::<u64>()
        .map_err(|_| rl::Error::Config("invalid revision".into()))?;
    let repetitions = args[2]
        .parse::<usize>()
        .map_err(|_| rl::Error::Config("invalid round-trip count".into()))?;
    if !(1..=1000).contains(&repetitions) {
        return Err(rl::Error::Config(
            "round trips must be between 1 and 1000".into(),
        ));
    }
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, args[0].clone())?;
    app.select_revision(revision)?;
    let settle = |app: &mut Laboratory, view: View, start: Instant| -> rl::Result<(f64, f64)> {
        let mut first = None;
        loop {
            paint(app, &ctx);
            let ready = match view {
                View::PolicyMap => {
                    app.policy_pane().ready()
                        && app.policy_frame().is_some_and(|f| {
                            f.requested_revision == revision && f.grid.current.revision == revision
                        })
                }
                View::Value => {
                    app.value_view().ready()
                        && app.value_frame().is_some_and(|f| {
                            f.requested_revision == revision
                                && f.grid.current().identity().revision == revision
                        })
                }
                _ => unreachable!(),
            };
            if ready && first.is_none() {
                first = Some(start.elapsed().as_secs_f64() * 1000.);
            }
            if ready && app.snapshot_preparation_idle() {
                return Ok((first.unwrap(), start.elapsed().as_secs_f64() * 1000.));
            }
            if start.elapsed() >= Duration::from_secs(60) {
                return Err(rl::Error::State(format!(
                    "{view:?} did not settle within 60s: {:?}",
                    app.notice()
                )));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    };
    app.select_view(View::PolicyMap);
    settle(&mut app, View::PolicyMap, Instant::now())?;
    let selected = app.selection().input.clone();
    if matches!(selected, SelectedInput::None) {
        return Err(rl::Error::State(
            "no recorded input selected for the comparison".into(),
        ));
    }
    eprintln!("selected input: {selected:?}");
    if std::env::var("RL_VIEWER_NAV_AUDIT").as_deref() == Ok("true") {
        for view in [View::PolicyMap, View::Value] {
            if view == View::Value {
                app.select_view(view);
                settle(&mut app, view, Instant::now())?;
            }
            let visible = app.policy_metrics().prepared_bytes;
            if view == View::PolicyMap {
                let frame = app.policy_frame().expect("settled Policy frame");
                let present = (0..frame.grid.values.len())
                    .filter(|&i| frame.grid.reference_at(i).is_some())
                    .count();
                eprintln!(
                    "Policy grid {} values, {present} matched reference values, {} projection bytes",
                    frame.grid.values.len(),
                    frame.grid.projection_bytes()
                );
            }
            if view == View::Value {
                let frame = app.value_frame().expect("settled Value frame");
                eprintln!(
                    "Value grid {:?}; empirical conditions {}",
                    frame.grid.slice.resolution,
                    frame.grid.empirical.conditions.len()
                );
            }
            app.select_view(View::Training);
            paint(&mut app, &ctx);
            let suspended = app.policy_metrics().prepared_bytes;
            eprintln!("{view:?}: visible {visible} bytes, suspended {suspended} bytes");
        }
        return Ok(());
    }
    println!(
        "round,workspace,publication_ms,catalog_idle_ms,restores,snapshot_verifications,grid_batches,point_batches,grid_x,grid_y,empirical_conditions,selected_futures,cdf_points,prepared_bytes,prepared_peak_bytes"
    );
    for round in 1..=repetitions {
        for view in [View::Value, View::PolicyMap] {
            let before = app.policy_metrics();
            let start = Instant::now();
            app.select_view(view);
            let (publication, idle) = settle(&mut app, view, start)?;
            if app.selection().input != selected {
                return Err(rl::Error::State(
                    "workspace switch changed the recorded input".into(),
                ));
            }
            let after = app.policy_metrics();
            let (resolution, empirical, selected_futures, cdf_points) = match view {
                View::PolicyMap => (app.policy_frame().unwrap().grid.config.resolution, 0, 0, 0),
                View::Value => {
                    let frame = app.value_frame().unwrap();
                    let selected = frame.selected.as_ref().and_then(|p| p.empirical.as_ref());
                    (
                        frame.grid.slice.resolution,
                        frame.grid.empirical.conditions.len(),
                        selected.map_or(0, |e| e.condition.summary.count),
                        selected.map_or(0, |e| e.cdf.len()),
                    )
                }
                _ => unreachable!(),
            };
            println!(
                "{round},{view:?},{publication:.3},{idle:.3},{},{},{},{},{},{},{},{},{},{},{}",
                after.restores - before.restores,
                after.snapshot_verifications - before.snapshot_verifications,
                after.grid_batches - before.grid_batches,
                after.point_batches - before.point_batches,
                resolution[0],
                resolution[1],
                empirical,
                selected_futures,
                cdf_points,
                after.prepared_bytes,
                after.prepared_peak
            );
            eprintln!(
                "completed {round}/{repetitions} {view:?}: {publication:.3} ms, catalogue idle {idle:.3} ms"
            );
        }
    }
    Ok(())
}
