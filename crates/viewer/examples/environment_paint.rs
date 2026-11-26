//! Release measurement of cached Environment painting, including tessellation.
//! Usage: environment_paint <recording> <revision> <episode-sequence> <step> <samples>
use eframe::egui;
use rl_viewer::{
    data::SelectedInput,
    ui::{Laboratory, View},
};
use std::time::{Duration, Instant};

fn paint(app: &mut Laboratory, ctx: &egui::Context) -> egui::FullOutput {
    ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280., 800.),
            )),
            ..Default::default()
        },
        |ui| app.show(ui),
    )
}

fn has_text(shape: &egui::Shape, expected: &str) -> bool {
    match shape {
        egui::Shape::Text(text) => text.galley.text() == expected,
        egui::Shape::Vec(shapes) => shapes.iter().any(|shape| has_text(shape, expected)),
        _ => false,
    }
}

fn main() -> rl::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(rl::Error::Config(
            "expected recording revision sequence step samples".into(),
        ));
    }
    let parse = |value: &str| {
        value
            .parse::<u64>()
            .map_err(|_| rl::Error::Config("invalid numeric argument".into()))
    };
    let revision = parse(&args[1])?;
    let sequence = parse(&args[2])?;
    let step = usize::try_from(parse(&args[3])?)
        .map_err(|_| rl::Error::Config("step too large".into()))?;
    let samples = usize::try_from(parse(&args[4])?)
        .map_err(|_| rl::Error::Config("sample count too large".into()))?;
    if !(1..=10_000).contains(&samples) {
        return Err(rl::Error::Config("sample count must be 1..=10000".into()));
    }
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, args[0].clone())?;
    app.select_view(View::Environment);
    app.select_revision(revision)?;
    let wait = |app: &mut Laboratory, selected: Option<SelectedInput>| -> rl::Result<()> {
        let start = Instant::now();
        let mut next_report = Duration::from_secs(1);
        loop {
            let output = paint(app, &ctx);
            std::hint::black_box(ctx.tessellate(output.shapes, output.pixels_per_point));
            if app.environment_ready()
                && selected
                    .as_ref()
                    .is_none_or(|s| &app.selection().input == s)
            {
                return Ok(());
            }
            if start.elapsed() >= next_report {
                eprintln!(
                    "Environment preparation: {:.1}s, status {:?}",
                    start.elapsed().as_secs_f64(),
                    app.environment_status()
                );
                next_report += Duration::from_secs(1);
            }
            if start.elapsed() > Duration::from_secs(30) {
                return Err(rl::Error::State(
                    "environment did not settle in 30 seconds".into(),
                ));
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    };
    wait(&mut app, None)?;
    app.select_environment_step(sequence, step)?;
    wait(
        &mut app,
        Some(SelectedInput::TrajectoryStep { sequence, step }),
    )?;
    let ready = paint(&mut app, &ctx);
    if !ready
        .shapes
        .iter()
        .any(|shape| has_text(&shape.shape, "Recorded trajectory"))
    {
        return Err(rl::Error::State(
            "selected recording did not display a frame trajectory".into(),
        ));
    }
    let mut csv = String::from("sample,paint_and_tessellation_ms\n");
    for index in 0..samples {
        let start = Instant::now();
        let output = paint(&mut app, &ctx);
        std::hint::black_box(ctx.tessellate(output.shapes, output.pixels_per_point));
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        csv.push_str(&format!("{index},{elapsed:.3}\n"));
        if (index + 1) % 10 == 0 {
            eprintln!("Painted {}/{} frames", index + 1, samples);
        }
        if !app.environment_ready() {
            return Err(rl::Error::State(
                "scene changed during cached paint measurement".into(),
            ));
        }
    }
    if let Some(path) = std::env::var_os("RL_ENVIRONMENT_BENCH_OUTPUT") {
        std::fs::write(path, &csv)?;
    }
    print!("{csv}");
    Ok(())
}
