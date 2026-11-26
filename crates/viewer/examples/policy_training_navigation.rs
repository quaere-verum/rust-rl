//! Measure CPU-side Policy or Value publication after round trips through Training.
//! Arguments: recording, revision, repetitions. Set RL_VIEWER_NAV_VIEW=value for Value.
use eframe::egui;
use rl_viewer::ui::{Laboratory, View};
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
            "expected recording revision repetitions".into(),
        ));
    }
    let revision = args[1]
        .parse::<u64>()
        .map_err(|_| rl::Error::Config("invalid revision".into()))?;
    let repetitions = args[2]
        .parse::<usize>()
        .map_err(|_| rl::Error::Config("invalid repetition count".into()))?;
    if !(1..=1000).contains(&repetitions) {
        return Err(rl::Error::Config(
            "repetitions must be between 1 and 1000".into(),
        ));
    }
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, args[0].clone())?;
    app.select_revision(revision)?;
    let value = std::env::var("RL_VIEWER_NAV_VIEW").as_deref() == Ok("value");
    let view = if value { View::Value } else { View::PolicyMap };
    app.select_view(view);
    let settle = |app: &mut Laboratory| -> rl::Result<()> {
        let start = Instant::now();
        while !((if value {
            app.value_view().ready()
                && app
                    .value_frame()
                    .is_some_and(|f| f.requested_revision == revision)
        } else {
            app.policy_pane().ready()
                && app
                    .policy_frame()
                    .is_some_and(|f| f.requested_revision == revision)
        }) && app.snapshot_preparation_idle())
        {
            paint(app, &ctx);
            if start.elapsed() > Duration::from_secs(60) {
                return Err(rl::Error::State(format!(
                    "{view:?} did not settle within 60s"
                )));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    };
    settle(&mut app)?;
    println!("iteration,publication_ms,restores,grid_batches,point_batches,prepared_peak_bytes");
    for i in 1..=repetitions {
        app.select_view(View::Training);
        paint(&mut app, &ctx);
        let before = app.policy_metrics();
        let start = Instant::now();
        app.select_view(view);
        settle(&mut app)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let after = app.policy_metrics();
        println!(
            "{i},{elapsed:.3},{},{},{},{}",
            after.restores - before.restores,
            after.grid_batches - before.grid_batches,
            after.point_batches - before.point_batches,
            after.prepared_peak
        );
        eprintln!("completed {i}/{repetitions}: {elapsed:.3} ms");
    }
    Ok(())
}
