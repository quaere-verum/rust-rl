//! Bounded CPU measurements for current Network paints and observation packing.
//! Run alone; allocator counts include every thread in this process, not RSS.
#[path = "../../../benches/support/allocation.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;

use eframe::egui;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::SnapshotBuilder, probes::ProbeSet, schema::*, storage::*},
    nn::{network::Mlp, policy::Deterministic},
};
use rl_viewer::ui::{Laboratory, View};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn measure(case: &str, repetitions: usize, mut operation: impl FnMut()) {
    let calibration = Instant::now();
    for _ in 0..repetitions {
        operation();
    }
    // Keep even sub-microsecond packing cases above timer/scheduling noise.
    // The cap also bounds work if a future case is unexpectedly inexpensive.
    let repetitions = ((repetitions as f64 * 0.1 / calibration.elapsed().as_secs_f64()).ceil()
        as usize)
        .clamp(repetitions, 1_000_000);
    eprintln!("{case}: calibrated to {repetitions} operations per sample (target 100ms)");
    for sample in 0..7 {
        let before = allocation::start();
        let start = Instant::now();
        for _ in 0..repetitions {
            operation();
        }
        let seconds = start.elapsed().as_secs_f64();
        let usage = allocation::finish(before);
        println!(
            "{case},{sample},{repetitions},{seconds:.9},{},{},{},{}",
            usage.calls, usage.requested, usage.live, usage.peak
        );
    }
}

fn paint(app: &mut Laboratory, ctx: &egui::Context) {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1920., 1080.),
            )),
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    let shapes = std::mem::take(&mut output.shapes);
    black_box(ctx.tessellate(shapes, output.pixels_per_point));
    output.textures_delta.clear();
}

fn network_recording(count: usize) -> Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut writer = Recorder::create(
        dir.path().join("run"),
        RunManifest::new(
            "hot-paths",
            "measurement",
            "fixed synthetic inputs",
            3,
            space.clone(),
            0.95,
        )?,
        StorageConfig::default(),
    )?;
    let inputs = (0..count)
        .map(|i| {
            let t = i as f32 / (count - 1) as f32;
            Observation::new(vec![t, (4. * t).sin(), (7. * t).cos()])
        })
        .collect();
    writer.append(
        &ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "Deterministic synthetic inputs; no learning claim",
            inputs,
            None,
            space.clone(),
        )?
        .record(RecordContext::new("hot-paths", 0))?,
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut snapshot = SnapshotBuilder::new(
        "actor-0",
        RecordContext::new("hot-paths", 0),
        3,
        space.clone(),
    )?;
    snapshot.deterministic(
        "actor",
        &Deterministic::from_network(Mlp::new(3, &[16, 12], 1, &mut rng)?, space)?,
    )?;
    writer.append(&snapshot.finish()?)?;
    writer.finish()?;
    Ok(dir)
}

fn main() -> Result<()> {
    println!(
        "case,sample,repetitions,seconds,allocations,requested_bytes,live_delta_bytes,peak_extra_bytes"
    );
    for width in [2, 64] {
        for batch in [1, 32, 256] {
            eprintln!("packing width={width} batch={batch}");
            let observations = vec![Observation::new(vec![0.25; width]); batch];
            for _ in 0..20 {
                black_box(rl::nn::observations((&observations).into(), width)?);
            }
            assert_eq!(
                rl::nn::to_vec(rl::nn::observations((&observations).into(), width)?)?,
                vec![0.25; width * batch]
            );
            measure(&format!("pack_{batch}x{width}"), 500, || {
                black_box(rl::nn::observations((&observations).into(), width).unwrap());
            });
        }
    }
    for count in [12, rl_viewer::network::MAX_PROBES] {
        geometry(count)?;
        eprintln!("Network fixture: {count} probes; waiting at most 30s for exact checkpoint");
        let dir = network_recording(count)?;
        let ctx = egui::Context::default();
        let mut app = Laboratory::from_context(&ctx, dir.path().join("run").display().to_string())?;
        app.select_view(View::Network);
        let deadline = Instant::now() + Duration::from_secs(30);
        while !app.network_ready() {
            paint(&mut app, &ctx);
            assert!(
                Instant::now() < deadline,
                "Network preparation did not settle"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        for _ in 0..40 {
            paint(&mut app, &ctx);
        }
        let identity = app.network_frame().unwrap().snapshot.key.clone();
        let values = app.network_frame().unwrap().geometry.output_values.clone();
        assert_eq!(app.network_frame().unwrap().geometry.reference_count, count);
        eprintln!("Network warmed: {count} probes; measuring seven blocks of 100 paints");
        measure(&format!("network_{count}_cached_paints"), 100, || {
            paint(&mut app, &ctx)
        });
        assert!(app.network_ready());
        assert_eq!(app.network_frame().unwrap().snapshot.key, identity);
        assert_eq!(app.network_frame().unwrap().geometry.output_values, values);
    }
    wide_geometry()?;
    Ok(())
}

fn geometry(count: usize) -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = rl::diagnostics::network_geometry::NumericalMlp::from_model(&Mlp::new(
        3,
        &[16, 12],
        1,
        &mut rng,
    )?)?;
    let probes: Vec<_> = (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64;
            vec![t, (4. * t).sin(), (7. * t).cos()]
        })
        .collect();
    let mut request = rl_viewer::network::Request {
        network: "actor".into(),
        anchor: probes[0].clone(),
        endpoint: probes[count / 3].clone(),
        probes,
        axes: vec![0, 1],
        axis_scales: vec![1., 2.],
        ranges: [[0., 1.], [-1., 1.]],
        direction_angle: 0.3,
        t: 1.,
        radius: 0.2,
        action_mask: None,
        layer: 2,
        grid: [12, 8],
        previous_basis: Vec::new(),
    };
    let initial = rl_viewer::network::prepare(&net, &request)?;
    eprintln!("Network geometry {count} probes: {initial:?}");
    request.previous_basis = initial.pca_bases;
    let reused = rl_viewer::network::prepare(&net, &request)?;
    eprintln!("Network geometry {count} probes reused basis: {reused:?}");
    for _ in 0..5 {
        black_box(rl_viewer::network::prepare(&net, &request)?);
    }
    measure(&format!("network_{count}_prepare"), 20, || {
        black_box(rl_viewer::network::prepare(&net, &request).unwrap());
    });
    let mut preparation = rl_viewer::network::Preparation::new(net);
    let cached = preparation.prepare(&request)?;
    assert_eq!(cached.output_values, reused.output_values);
    eprintln!(
        "network_{count}_cached_prepare retained_bytes={}",
        preparation.retained_bytes()
    );
    measure(&format!("network_{count}_cached_prepare"), 20, || {
        black_box(preparation.prepare(&request).unwrap());
    });
    Ok(())
}

fn wide_geometry() -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = rl::diagnostics::network_geometry::NumericalMlp::from_model(&Mlp::new(
        3,
        &[4096],
        1,
        &mut rng,
    )?)?;
    let probes: Vec<_> = (0..12)
        .map(|i| {
            let t = i as f64 / 11.;
            vec![t, (4. * t).sin(), (7. * t).cos()]
        })
        .collect();
    let mut request = rl_viewer::network::Request {
        network: "wide".into(),
        anchor: probes[0].clone(),
        endpoint: probes[4].clone(),
        probes,
        axes: vec![0, 1],
        axis_scales: vec![1., 2.],
        ranges: [[0., 1.], [-1., 1.]],
        direction_angle: 0.3,
        t: 1.,
        radius: 0.2,
        action_mask: None,
        layer: 1,
        grid: [3, 3],
        previous_basis: vec![],
    };
    let mut cached = rl_viewer::network::Preparation::new(net.clone());
    let first = cached.prepare(&request)?;
    assert!(first.grid.iter().all(|g| g.strength.is_finite()));
    request.previous_basis = first.pca_bases;
    eprintln!(
        "network_4096_cached_prepare retained_bytes={}",
        cached.retained_bytes()
    );
    let mut uncached = rl_viewer::network::Preparation::with_cache_budget(net, 0);
    measure("network_4096_low_rank_uncached", 2, || {
        black_box(uncached.prepare(&request).unwrap());
    });
    measure("network_4096_low_rank_cached", 2, || {
        black_box(cached.prepare(&request).unwrap());
    });
    Ok(())
}
