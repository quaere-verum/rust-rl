#![cfg(feature = "diagnostics")]

use rl as recording_rl;

#[path = "support/recording_env.rs"]
mod recording_env;
use recording_env::{NAME, RecordingEnv};
use rl::{
    core::{actor::ActMode, env::Env},
    diagnostics::{evaluation::*, render::*, schema::*, storage::*},
    train::trainer::RandomActor,
};

#[test]
fn combined_frames_and_render_respect_the_explicit_finite_horizon() -> rl::Result<()> {
    let factory = |seed| Ok(RecordingEnv::new(5, seed));
    let actor = RandomActor {
        space: factory(0)?.action_space().clone(),
    };
    let config = EvaluationConfig {
        seeds: vec![1],
        trajectories: 1,
        max_steps: 5,
        discount: 1.,
        ..Default::default()
    };
    let captured = evaluate_with_frames_and_render_and_horizon(
        factory,
        &actor,
        &config,
        EvaluationHorizon::FiniteDecisions(2),
        &RenderCaptureConfig::default(),
        |_| {},
    )?;
    assert_eq!(captured.horizon, EvaluationHorizon::FiniteDecisions(2));
    assert_eq!(captured.environment_steps, 2);
    assert_eq!(captured.trajectories[0].frames.len(), 3);
    assert!(captured.trajectories[0].render.is_some());
    let chunk = captured.trajectory_record(0, RecordContext::new("frames", 0), "episode", NAME)?;
    assert_eq!(
        evaluation_horizon(&chunk)?,
        EvaluationHorizon::FiniteDecisions(2)
    );
    assert!(scene_at(&chunk, 2, &RenderLimits::default())?.is_some());
    Ok(())
}

#[test]
fn portable_recording_preserves_frames_privacy_clocks_and_captured_rollout() -> rl::Result<()> {
    const STEPS: usize = 3;
    let factory = |seed| Ok(RecordingEnv::new(STEPS, seed));
    let space = factory(0)?.action_space().clone();
    let actor = RandomActor {
        space: space.clone(),
    };
    let settings = EvaluationConfig {
        seeds: vec![1, 2],
        mode: ActMode::Sample,
        trajectories: 2,
        discount: 1.,
        max_steps: STEPS,
        ..Default::default()
    };
    let plain = evaluate(factory, &actor, &settings)?;
    let rendered = evaluate_with_frames_and_render(
        factory,
        &actor,
        &settings,
        &RenderCaptureConfig {
            trajectories: 1,
            limits: RenderLimits::default(),
        },
        |_| {},
    )?;
    assert_eq!(rendered.returns, plain.returns);
    assert_eq!(rendered.environment_steps, plain.environment_steps);
    for (captured, ordinary) in rendered.trajectories.iter().zip(&plain.trajectories) {
        assert_eq!(captured.observations, ordinary.observations);
        assert_eq!(captured.actions, ordinary.actions);
        assert_eq!(captured.rewards, ordinary.rewards);
        assert_eq!(captured.terminated, vec![false, false, true]);
        assert_eq!(captured.terminated, ordinary.terminated);
        assert_eq!(captured.truncated, ordinary.truncated);
    }
    assert!(rendered.trajectories[1].render.is_none());
    let chunk = rendered.trajectory_record(0, RecordContext::new("frames", 0), "seed-1", NAME)?;

    let root = tempfile::tempdir()?;
    let path = root.path().join("portable");
    let mut manifest = RunManifest::new("frames", "portable", "random", 3, space, 1.)?;
    manifest.environment = NAME.into();
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default())?;
    writer.append(&chunk)?;
    writer.finish()?;
    let run = Recording::open(path, ReadLimits::default())?;
    assert!(run.complete());
    let loaded = run.load(0)?;
    let Record::Trajectory(trajectory) = &loaded.record else {
        panic!("expected recorded trajectory");
    };
    let observations = &loaded.arrays[&trajectory.observations];
    assert_eq!(observations.shape(), &[STEPS + 1, 3]);
    assert_eq!(&observations.as_f32()?[..3], &[2., 0., 1.]);
    assert_eq!(
        loaded.arrays["frames/diagnostic_signal"].shape(),
        &[STEPS + 1, 1]
    );
    let private_signal = loaded.arrays["frames/diagnostic_signal"].as_f64()?[0];
    assert!(private_signal.abs() > 0. && private_signal.abs() < 0.5);
    assert_eq!(
        loaded.arrays["simulation_times"].as_f64()?,
        &[0., 1., 2., 3.]
    );
    let limits = RenderLimits::default();
    for step in 0..=STEPS {
        let scene = scene_at(&loaded, step, &limits)?.unwrap();
        assert!(matches!(
            &scene.items[0].primitive,
            RenderPrimitive::Text { text, .. } if text == &step.to_string()
        ));
    }
    assert!(scene_at(&loaded, STEPS + 1, &limits).is_err());
    Ok(())
}
