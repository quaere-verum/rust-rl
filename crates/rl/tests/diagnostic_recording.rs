#![cfg(feature = "diagnostics")]

use rl::{
    Result,
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*},
};
use std::collections::BTreeMap;

#[cfg(feature = "neural")]
use rl as recording_rl;
#[cfg(feature = "neural")]
#[path = "support/recording_env.rs"]
mod recording_env;

#[cfg(feature = "neural")]
mod neural {
    use super::recording_env::{NAME, RecordingEnv};
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{
        diagnostics::{
            evaluation::{
                EvaluationConfig, StochasticActor, evaluate_with_frames,
                evaluate_with_frames_and_render,
            },
            models::{InspectionRequest, ModelBundle, ModelLimits, SnapshotBuilder},
            render::*,
            value::{self, CriticDeclaration, ValueCriticKind, ValueDeclaration},
        },
        prelude::*,
    };
    use std::sync::Arc;

    const RUN: &str = "policy-recording";
    const HORIZON: usize = 4;

    fn record_checkpoint(
        writer: &mut Recorder,
        learner: &Vmpo,
        mut context: RecordContext,
    ) -> Result<()> {
        let policy = learner.policy();
        let value = learner.value();
        let checkpoint = format!("vmpo-{}", learner.updates());
        context.identity.checkpoint_id = Some(checkpoint.clone());
        let mut snapshot = SnapshotBuilder::new(
            &checkpoint,
            context.clone(),
            3,
            policy.action_space().clone(),
        )?;
        snapshot.policy("policy", &policy)?;
        snapshot.value("value", &value)?;
        let mut snapshot = snapshot.finish()?;
        value::attach_declaration(
            &mut snapshot,
            &ValueDeclaration {
                version: 1,
                run_id: RUN.into(),
                revision: learner.updates(),
                model_role: context.identity.model_role.clone(),
                conventions: None,
                critics: BTreeMap::from([(
                    "value".into(),
                    CriticDeclaration {
                        kind: ValueCriticKind::StateValue,
                        learned_baseline: Some(true),
                        actor: None,
                        continuation: None,
                        twin: None,
                        utility: None,
                    },
                )]),
            },
        )?;
        writer.append(&snapshot)?;

        let actor = StochasticActor::new(&policy);
        let evaluation = EvaluationConfig {
            seeds: vec![101, 102],
            policy_seed: 103,
            mode: ActMode::Sample,
            discount: 1.,
            max_steps: HORIZON,
            trajectories: 2,
            ..Default::default()
        };
        let plain = evaluate_with_frames(
            |seed| Ok(RecordingEnv::new(HORIZON, seed)),
            &actor,
            &evaluation,
            |_| {},
        )?;
        let rendered = evaluate_with_frames_and_render(
            |seed| Ok(RecordingEnv::new(HORIZON, seed)),
            &actor,
            &evaluation,
            &RenderCaptureConfig {
                trajectories: 1,
                limits: RenderLimits::default(),
            },
            |_| {},
        )?;
        assert_eq!(plain.returns, rendered.returns);
        for (without, with) in plain.trajectories.iter().zip(&rendered.trajectories) {
            assert_eq!(without.observations, with.observations);
            assert_eq!(without.actions, with.actions);
            assert_eq!(without.rewards, with.rewards);
        }
        assert!(rendered.trajectories[1].render.is_none());
        writer.append(&rendered.trajectory_record(
            0,
            context,
            &format!("{checkpoint}-rollout"),
            NAME,
        )?)?;
        Ok(())
    }

    #[test]
    fn trained_policy_and_rendered_trajectory_share_reopened_checkpoints() -> Result<()> {
        let space = RecordingEnv::new(HORIZON, 1).action_space().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut learner = Vmpo::new(
            ActorCritic {
                policy: Policy::new(3, space.clone(), &[8], &mut rng)?,
                value: Value::new(3, &[8], &mut rng)?,
            },
            VmpoConfig {
                epochs: 1,
                minibatch_size: 8,
                ..Default::default()
            },
            3,
        )?;
        let lanes = [4, 5]
            .into_iter()
            .map(|seed| RecordingEnv::new(HORIZON, seed))
            .collect::<Vec<_>>();
        let mut trainer = OnPolicyTrainer::new(
            OnPolicyCollector::new(SyncVectorEnv::new(lanes)?, 6)?,
            OnPolicyConfig {
                rollout_steps: 4,
                gamma: 1.,
                ..Default::default()
            },
        )?;
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("recording");
        let mut manifest = RunManifest::new(RUN, "recording fixture", "V-MPO", 3, space, 1.)?;
        manifest.environment = NAME.into();
        let mut writer = Recorder::create(&path, manifest.clone(), StorageConfig::default())?
            .with_buffering(BufferConfig::default())?;

        record_checkpoint(&mut writer, &learner, RecordContext::new(RUN, 0))?;
        trainer.train(&mut learner, Budget::vector_steps(4), None)?;
        assert_eq!(trainer.counters().env_steps, 8);
        let revision = learner.updates();
        assert!(revision > 0);
        let mut final_context = RecordContext::new(RUN, revision);
        final_context.counters = *trainer.counters();
        record_checkpoint(&mut writer, &learner, final_context)?;
        writer.finish()?;

        assert!(Recorder::create(&path, manifest, StorageConfig::default()).is_err());
        let recording = Recording::open(path, ReadLimits::default())?;
        assert!(recording.complete());
        assert!(recording.issues().is_empty());
        assert_eq!(recording.status()?.skipped_records, 0);
        assert_eq!(recording.entries().len(), 4);
        for (pair, revision) in [0, revision].into_iter().enumerate() {
            let checkpoint = recording.load(2 * pair)?;
            let trajectory = Arc::new(recording.load(2 * pair + 1)?);
            assert!(matches!(checkpoint.record, Record::ModelSnapshot(_)));
            assert_eq!(checkpoint.context.identity, trajectory.context.identity);
            assert_eq!(
                checkpoint.context.identity.learner_update_revision,
                revision
            );

            value::declaration(&checkpoint)?
                .unwrap()
                .validate_manifest(recording.manifest())?;
            let model = ModelBundle::load(&checkpoint, ModelLimits::default())?;
            let Record::Trajectory(track) = &trajectory.record else {
                panic!("expected recorded trajectory")
            };
            let observations = &trajectory.arrays[&track.observations];
            assert_eq!(observations.shape(), &[HORIZON + 1, 3]);
            assert_eq!(trajectory.arrays[&track.actions].shape(), &[HORIZON, 1]);
            let mut request = InspectionRequest::new(
                "policy",
                vec![Observation::new(observations.as_f32()?[..3].to_vec())],
            );
            request.critics = vec!["value".into()];
            let restored = model.inspect(&request)?;
            assert_eq!(restored.actions.len(), 1);
            assert!(restored.critics.contains_key("value"));

            let limits = RenderLimits::default();
            let playback = ValidatedRenderTrack::new(&trajectory, limits)?;
            let terminal = playback.scene_at(&trajectory, HORIZON)?.unwrap();
            assert_eq!(
                scene_at(&trajectory, HORIZON, &limits)?,
                Some(terminal.clone())
            );
            assert!(matches!(
                &terminal.items[0].primitive,
                RenderPrimitive::Text { text, .. } if text == &HORIZON.to_string()
            ));
        }
        Ok(())
    }
}

#[test]
fn rejected_record_does_not_discard_buffered_history() -> Result<()> {
    let root = tempfile::tempdir()?;
    let path = root.path().join("recording");
    let manifest = RunManifest::new(
        "budget",
        "recording fixture",
        "V-MPO",
        1,
        ActionSpace::Discrete { count: 2 },
        1.,
    )?;
    let mut writer = Recorder::create(
        &path,
        manifest,
        StorageConfig {
            max_chunk_bytes: 1024,
            compress: false,
            ..Default::default()
        },
    )?
    .with_buffering(BufferConfig::default())?;
    let mut chunk = Chunk {
        context: RecordContext::new("budget", 0),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::from([("accepted".into(), 1.)]),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::new(),
    };
    assert_eq!(
        writer.append(&chunk)?,
        AppendOutcome::Buffered { sequence: 0 }
    );
    let mut malformed = chunk.clone();
    let Record::Metrics(metrics) = &mut malformed.record else {
        unreachable!();
    };
    metrics.values.insert("invalid".into(), f64::NAN);
    assert!(writer.append(&malformed).is_err());
    assert_eq!(writer.status().committed_records, 0);
    assert_eq!(writer.status().skipped_records, 0);

    chunk
        .arrays
        .insert("oversized".into(), Array::u8(vec![2048], vec![0; 2048])?);
    assert!(matches!(
        writer.append(&chunk)?,
        AppendOutcome::Skipped { .. }
    ));

    writer.finish()?;
    let recording = Recording::open(path, ReadLimits::default())?;
    assert_eq!(recording.entries().len(), 1);
    let accepted = recording.load(0)?;
    let Record::Metrics(metrics) = accepted.record else {
        panic!("expected buffered metrics")
    };
    assert_eq!(metrics.values["accepted"], 1.);
    assert_eq!(recording.status()?.skipped_records, 1);
    Ok(())
}
