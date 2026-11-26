//! Measure superseded historical Policy requests against a real recording.
//! Usage: rapid_navigation <recording> <rounds> <revision> <revision>...
use eframe::egui;
use rl_viewer::ui::{Laboratory, View};
use std::time::{Duration, Instant};

fn paint(app: &mut Laboratory, ctx: &egui::Context) -> rl::Result<()> {
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
    if let Some(frame) = app.policy_frame()
        && frame.requested_revision != frame.grid.current.revision
    {
        return Err(rl::Error::State(
            "a held Policy frame was relabelled as a different checkpoint".into(),
        ));
    }
    Ok(())
}

fn wait_for(
    app: &mut Laboratory,
    ctx: &egui::Context,
    revision: u64,
    catalogue: bool,
) -> rl::Result<()> {
    let started = Instant::now();
    loop {
        paint(app, ctx)?;
        if app.policy_frame().is_some_and(|f| {
            f.requested_revision == revision && f.grid.current.revision == revision
        }) && !app.policy_pending()
            && (!catalogue || app.snapshot_preparation_idle())
        {
            return Ok(());
        }
        if started.elapsed() > Duration::from_secs(30) {
            return Err(rl::Error::State(format!(
                "revision {revision} did not settle; notice={:?}",
                app.notice()
            )));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn main() -> rl::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or_else(|| rl::Error::Config("recording path required".into()))?;
    let parse = |text: String| {
        text.parse::<u64>()
            .map_err(|_| rl::Error::Config("expected a nonnegative integer".into()))
    };
    let rounds = args
        .next()
        .ok_or_else(|| rl::Error::Config("round count required".into()))
        .and_then(parse)?;
    let revisions = args.map(parse).collect::<rl::Result<Vec<_>>>()?;
    if rounds == 0 || revisions.len() < 2 {
        return Err(rl::Error::Config(
            "use at least one round and two checkpoint revisions".into(),
        ));
    }
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path)?;
    app.select_view(View::PolicyMap);
    app.select_revision(revisions[0])?;
    wait_for(&mut app, &ctx, revisions[0], true)?;
    println!(
        "round,last_revision,requests,request_median_ms,request_max_ms,last_publication_ms,restore_delta,grid_batches_delta,held_paints,prepared_peak_bytes"
    );
    for round in 0..rounds {
        eprintln!("Scrubbing round {}/{}", round + 1, rounds);
        let before = app.policy_metrics();
        let mut timings = Vec::with_capacity(revisions.len());
        let mut held = 0usize;
        let mut last_start = Instant::now();
        let mut last = revisions[0];
        for index in 0..revisions.len() {
            let offset = usize::try_from(round % revisions.len() as u64).unwrap();
            let revision = revisions[(index + offset) % revisions.len()];
            last_start = Instant::now();
            app.select_revision(revision)?;
            paint(&mut app, &ctx)?;
            timings.push(last_start.elapsed());
            held += usize::from(
                app.policy_frame()
                    .is_some_and(|f| f.requested_revision != revision),
            );
            last = revision;
        }
        wait_for(&mut app, &ctx, last, false)?;
        let publication = last_start.elapsed();
        let after = app.policy_metrics();
        timings.sort_unstable();
        let median = (timings[(timings.len() - 1) / 2].as_secs_f64()
            + timings[timings.len() / 2].as_secs_f64())
            * 500.;
        let max = timings.last().unwrap().as_secs_f64() * 1000.;
        println!(
            "{round},{last},{},{median:.3},{max:.3},{:.3},{},{},{held},{}",
            timings.len(),
            publication.as_secs_f64() * 1000.,
            after.restores.saturating_sub(before.restores),
            after.grid_batches.saturating_sub(before.grid_batches),
            after.prepared_peak,
        );
    }
    Ok(())
}
