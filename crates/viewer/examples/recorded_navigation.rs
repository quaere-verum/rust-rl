//! Measure publication of real retained-model checkpoints, including revisits.
//! Usage: cargo run -p rl-viewer --example recorded_navigation -- <recording> <revision>...
use eframe::egui;
use rl_viewer::ui::{Laboratory, View};
use std::time::{Duration, Instant};

fn paint(app: &mut Laboratory, ctx: &egui::Context) {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440., 960.),
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
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| rl::Error::Config("recording path required".into()))?;
    let revisions: Vec<u64> = args
        .map(|arg| {
            arg.parse()
                .map_err(|_| rl::Error::Config("revision must be an integer".into()))
        })
        .collect::<rl::Result<_>>()?;
    if revisions.is_empty() {
        return Err(rl::Error::Config(
            "at least one revision is required".into(),
        ));
    }
    let ctx = egui::Context::default();
    let dwell = std::env::var("RL_VIEWER_NAV_DWELL_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_default();
    let open = Instant::now();
    let mut app = Laboratory::from_context(&ctx, path)?;
    eprintln!("recording opened in {:.3}s", open.elapsed().as_secs_f64());
    let value_view = std::env::var("RL_VIEWER_NAV_VIEW").as_deref() == Ok("value");
    app.select_view(if value_view {
        View::Value
    } else {
        View::PolicyMap
    });
    println!(
        "position,revision,publication_ms,restore_delta,grid_batches_delta,restore_ms,inference_ms,preparation_ms,ui_passes,ui_total_ms,ui_max_ms,prepared_peak_bytes,scratch_peak_bytes"
    );
    for (position, revision) in revisions.into_iter().enumerate() {
        eprintln!(
            "preparing position {} at revision {}",
            position + 1,
            revision
        );
        let before = if value_view {
            app.value_metrics()
        } else {
            app.policy_metrics()
        };
        let start = Instant::now();
        app.select_revision(revision)?;
        let mut passes = 0_u64;
        let mut ui_total = Duration::ZERO;
        let mut ui_max = Duration::ZERO;
        loop {
            let paint_start = Instant::now();
            paint(&mut app, &ctx);
            let elapsed = paint_start.elapsed();
            passes += 1;
            ui_total += elapsed;
            ui_max = ui_max.max(elapsed);
            let exact = if value_view {
                app.value_frame().is_some_and(|frame| {
                    frame.requested_revision == revision
                        && frame.grid.current().identity().revision == revision
                })
            } else {
                app.policy_frame().is_some_and(|frame| {
                    frame.requested_revision == revision && frame.grid.current.revision == revision
                })
            };
            let pending = if value_view {
                app.value_pending()
            } else {
                app.policy_pending()
            };
            if exact && !pending {
                break;
            }
            if start.elapsed() >= Duration::from_secs(30) {
                return Err(rl::Error::State(format!(
                    "checkpoint {revision} did not publish within 30s; selection={:?}, notice={:?}, pending={}, current={:?}, policy_mode={:?}, policy_status={:?}",
                    app.selection(),
                    app.notice(),
                    pending,
                    if value_view {
                        app.value_frame()
                            .map(|f| (f.requested_revision, f.grid.current().identity().revision))
                    } else {
                        app.policy_frame()
                            .map(|f| (f.requested_revision, f.grid.current.revision))
                    },
                    app.policy_mode(),
                    app.policy_pane().status()
                )));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        let after = if value_view {
            app.value_metrics()
        } else {
            app.policy_metrics()
        };
        println!(
            "{position},{revision},{:.3},{},{},{:.3},{:.3},{:.3},{},{:.3},{:.3},{},{}",
            start.elapsed().as_secs_f64() * 1000.,
            after.restores.saturating_sub(before.restores),
            after.grid_batches.saturating_sub(before.grid_batches),
            after.restore_micros.saturating_sub(before.restore_micros) as f64 / 1000.,
            after
                .inference_micros
                .saturating_sub(before.inference_micros) as f64
                / 1000.,
            after
                .preparation_micros
                .saturating_sub(before.preparation_micros) as f64
                / 1000.,
            passes,
            ui_total.as_secs_f64() * 1000.,
            ui_max.as_secs_f64() * 1000.,
            after.prepared_peak,
            after.scratch_peak
        );
        if !dwell.is_zero() {
            let idle_start = Instant::now();
            while idle_start.elapsed() < dwell {
                paint(&mut app, &ctx);
                std::thread::sleep(Duration::from_millis(2));
            }
            let idle = if value_view {
                app.value_metrics()
            } else {
                app.policy_metrics()
            };
            eprintln!(
                "dwell after {revision}: {:.3}ms, total_restores_including_catalog_validation={}, grid_batches={}",
                idle_start.elapsed().as_secs_f64() * 1000.,
                idle.restores.saturating_sub(after.restores),
                idle.grid_batches.saturating_sub(after.grid_batches)
            );
        }
    }
    Ok(())
}
