#![cfg(feature = "diagnostics")]
use rl as recording_rl;

#[path = "support/recording_env.rs"]
mod recording_env;
use rl::{
    Result,
    core::{
        action::{Action, ActionSpace},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
    },
    diagnostics::{
        evaluation::*,
        render::*,
        schema::{Record, RecordContext},
    },
    train::trainer::RandomActor,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Custom {
    state: usize,
    space: ActionSpace,
    calls: Arc<AtomicUsize>,
}
impl Env for Custom {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.state = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        self.state += 1;
        Ok(Step {
            observation: Observation::new(vec![self.state as f32]),
            reward: (if *action == Action::Discrete(0) {
                1.
            } else {
                -1.
            })
            .into(),
            terminated: self.state == 2,
            truncated: false,
        })
    }
}
fn scene(step: usize) -> RenderScene {
    RenderScene {
        version: RENDER_VERSION,
        bounds: RenderBounds {
            min: [-1., -1.],
            max: [3., 3.],
        },
        background: [0, 0, 0, 255],
        items: vec![RenderItem {
            layer: 0,
            primitive: RenderPrimitive::Text {
                position: [0., 0.],
                text: step.to_string(),
                color: [255; 4],
                size: 12.,
            },
        }],
    }
}
impl RenderSource for Custom {
    fn render_scene(&self) -> Result<RenderScene> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(scene(self.state))
    }
}
fn config() -> EvaluationConfig {
    EvaluationConfig {
        seeds: vec![11, 12, 13],
        trajectories: 2,
        max_steps: 3,
        ..Default::default()
    }
}
#[test]
fn selected_render_capture_is_independent_and_aligned() {
    let calls = Arc::new(AtomicUsize::new(0));
    let factory = |_| {
        Ok(Custom {
            state: 0,
            space: ActionSpace::Discrete { count: 2 },
            calls: calls.clone(),
        })
    };
    let actor = RandomActor {
        space: ActionSpace::Discrete { count: 2 },
    };
    let plain = evaluate(factory, &actor, &config()).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let rendered = evaluate_with_render(
        factory,
        &actor,
        &config(),
        &RenderCaptureConfig {
            trajectories: 1,
            limits: RenderLimits::default(),
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert_eq!(plain.returns, rendered.returns);
    for (a, b) in plain.trajectories.iter().zip(&rendered.trajectories) {
        assert_eq!(a.observations, b.observations);
        assert_eq!(a.actions, b.actions);
        assert_eq!(a.rewards, b.rewards);
    }
    for i in 0..2 {
        let chunk = rendered
            .trajectory_record(i, RecordContext::new("render", 0), "episode", "arbitrary")
            .unwrap();
        chunk.validate().unwrap();
        for step in 0..3 {
            assert_eq!(
                scene_at(&chunk, step, &RenderLimits::default()).unwrap(),
                (i == 0).then(|| scene(step))
            );
        }
        if i == 0 {
            assert!(scene_at(&chunk, 3, &RenderLimits::default()).is_err());
        }
        assert!(matches!(&chunk.record, Record::Trajectory(t) if t.render.is_some() == (i == 0)));
    }
    let empty = evaluate_with_render(
        factory,
        &actor,
        &config(),
        &RenderCaptureConfig {
            trajectories: 0,
            limits: RenderLimits::default(),
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 3);
    assert!(empty.trajectories.iter().all(|t| t.render.is_none()));
}
#[test]
fn limits_and_invalid_geometry_are_rejected() {
    let limits = RenderLimits::default();
    scene(0).validate(&limits).unwrap();
    assert!(
        scene(0)
            .validate(&RenderLimits {
                max_primitives: 0,
                ..limits
            })
            .is_err()
    );
    assert!(
        RenderLimits {
            max_scene_bytes: 65537,
            ..limits
        }
        .validate()
        .is_err()
    );
    let mut bad = scene(0);
    bad.bounds.max[0] = bad.bounds.min[0];
    assert!(bad.validate(&limits).is_err());
    let mut bad = scene(0);
    bad.items[0].primitive = RenderPrimitive::Polygon {
        points: vec![[0., 0.], [2., 0.], [1., 0.5], [2., 2.], [0., 2.]],
        fill: Some([255; 4]),
        stroke: None,
    };
    assert!(bad.validate(&limits).is_err());
}

impl FrameSource for Custom {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        Ok(EnvironmentFrame {
            environment: "arbitrary".into(),
            simulation_time: Some(self.state as f64),
            channels: std::collections::BTreeMap::from([("state".into(), vec![self.state as f64])]),
        })
    }
}
fn captured() -> rl::diagnostics::schema::Chunk {
    let actor = RandomActor {
        space: ActionSpace::Discrete { count: 2 },
    };
    evaluate_with_frames_and_render(
        |_| {
            Ok(Custom {
                state: 0,
                space: actor.space.clone(),
                calls: Arc::new(AtomicUsize::new(0)),
            })
        },
        &actor,
        &config(),
        &RenderCaptureConfig {
            trajectories: 1,
            limits: RenderLimits::default(),
        },
        |_| {},
    )
    .unwrap()
    .trajectory_record(0, RecordContext::new("render", 0), "episode", "arbitrary")
    .unwrap()
}
fn replace_scenes(chunk: &mut rl::diagnostics::schema::Chunk, bytes: &[u8]) {
    use rl::diagnostics::schema::Array;
    let Record::Trajectory(t) = &chunk.record else {
        panic!()
    };
    let r = t.render.as_ref().unwrap();
    chunk.arrays.insert(
        r.data.clone(),
        Array::u8(vec![bytes.len() * 3], bytes.repeat(3)).unwrap(),
    );
    chunk.arrays.insert(
        r.offsets.clone(),
        Array::u64(vec![4], (0..4).map(|i| (i * bytes.len()) as u64).collect()).unwrap(),
    );
}
#[test]
fn validated_playback_matches_every_scene_and_preserves_errors() {
    use rl::diagnostics::schema::{Chunk, MetricRecord};
    let chunk = Arc::new(captured());
    let limits = RenderLimits::default();
    let proof = ValidatedRenderTrack::new(&chunk, limits).unwrap();
    for step in 0..=3 {
        assert_eq!(
            proof.scene_at(&chunk, step).map_err(|e| e.to_string()),
            scene_at(&chunk, step, &limits).map_err(|e| e.to_string())
        );
    }
    let mut absent = captured();
    if let Record::Trajectory(t) = &mut absent.record {
        t.render = None;
    }
    let absent = Arc::new(absent);
    let no_render = ValidatedRenderTrack::new(&absent, limits).unwrap();
    assert_eq!(no_render.scene_at(&absent, usize::MAX).unwrap(), None);
    let metric = Arc::new(Chunk {
        context: RecordContext::new("render", 0),
        record: Record::Metrics(MetricRecord {
            values: Default::default(),
            unavailable: Default::default(),
        }),
        arrays: Default::default(),
    });
    assert_eq!(
        ValidatedRenderTrack::new(&metric, limits)
            .err()
            .unwrap()
            .to_string(),
        scene_at(&metric, 0, &limits).unwrap_err().to_string()
    );

    // Track validation admits structure only; each selected scene still validates.
    let mut invalid_scene = captured();
    replace_scenes(&mut invalid_scene, b"{}");
    let invalid_scene = Arc::new(invalid_scene);
    let invalid_proof = ValidatedRenderTrack::new(&invalid_scene, limits).unwrap();
    assert_eq!(
        invalid_proof
            .scene_at(&invalid_scene, 0)
            .unwrap_err()
            .to_string(),
        scene_at(&invalid_scene, 0, &limits)
            .unwrap_err()
            .to_string()
    );
    assert!(invalid_scene.validate().is_err());

    let mut later = (*chunk).clone();
    let Record::Trajectory(record) = &later.record else {
        panic!()
    };
    let track = record.render.as_ref().unwrap().clone();
    let start = later.arrays[&track.offsets].as_u64().unwrap()[2] as usize;
    let mut bytes = later.arrays[&track.data].as_u8().unwrap().to_vec();
    bytes[start] = b'!';
    later.arrays.insert(
        track.data,
        rl::diagnostics::schema::Array::u8(vec![bytes.len()], bytes).unwrap(),
    );
    let later = Arc::new(later);
    assert!(!proof.matches(&later));
    assert!(proof.scene_at(&later, 0).is_err());
    let replacement_proof = ValidatedRenderTrack::new(&later, limits).unwrap();
    assert_eq!(
        replacement_proof.scene_at(&later, 0).unwrap(),
        Some(scene(0))
    );
    assert_eq!(
        replacement_proof
            .scene_at(&later, 2)
            .unwrap_err()
            .to_string(),
        scene_at(&later, 2, &limits).unwrap_err().to_string()
    );
    assert!(
        later.validate().is_err(),
        "loading still validates nonselected scenes"
    );
}

#[test]
fn validated_playback_binds_exact_owner_and_does_not_pin_payload() {
    let mut chunk = Arc::new(captured());
    let proof = ValidatedRenderTrack::new(&chunk, RenderLimits::default()).unwrap();
    assert!(proof.matches(&chunk));
    assert!(Arc::get_mut(&mut chunk).is_none());
    let replacement = Arc::new((*chunk).clone());
    assert!(!proof.matches(&replacement));
    assert!(proof.scene_at(&replacement, 0).is_err());
    let weak = Arc::downgrade(&chunk);
    // With one strong owner, make_mut dissociates Weak references.
    Arc::make_mut(&mut chunk).context.identity.run_id = "new-owner".into();
    assert!(!proof.matches(&chunk));
    assert!(proof.scene_at(&chunk, 0).is_err());
    assert!(weak.upgrade().is_none());
    let proof = ValidatedRenderTrack::new(&chunk, RenderLimits::default()).unwrap();
    let another_strong = chunk.clone();
    Arc::make_mut(&mut chunk).context.identity.run_id = "cloned-owner".into();
    assert!(!proof.matches(&chunk));
    assert!(proof.matches(&another_strong));
    let weak = Arc::downgrade(&another_strong);
    drop(another_strong);
    assert!(weak.upgrade().is_none());
    drop(proof);
}

#[test]
fn validated_playback_keeps_all_tighter_limits() {
    let mut fixture = captured();
    let mut selected = scene(0);
    selected.items.push(RenderItem {
        layer: 0,
        primitive: RenderPrimitive::Polyline {
            points: vec![[0., 0.], [1., 1.]],
            stroke: RenderStroke {
                color: [255; 4],
                width: 1.,
            },
        },
    });
    let encoded = serde_json::to_vec(&selected).unwrap();
    replace_scenes(&mut fixture, &encoded);
    let chunk = Arc::new(fixture);
    let charge = encoded.len() * 3 + 4 * 8;
    let exact = RenderLimits {
        max_scene_bytes: encoded.len(),
        max_primitives: 2,
        max_points: 2,
        max_text_bytes: 1,
        max_frames: 3,
        max_trajectory_bytes: charge,
        max_evaluation_bytes: 0,
    };
    let proof = ValidatedRenderTrack::new(&chunk, exact).unwrap();
    assert_eq!(proof.scene_at(&chunk, 2).unwrap(), Some(selected));
    let default_proof = ValidatedRenderTrack::new(&chunk, RenderLimits::default()).unwrap();
    for tight in [
        RenderLimits {
            max_scene_bytes: exact.max_scene_bytes - 1,
            ..exact
        },
        RenderLimits {
            max_primitives: 1,
            ..exact
        },
        RenderLimits {
            max_points: 1,
            ..exact
        },
        RenderLimits {
            max_text_bytes: 0,
            ..exact
        },
        RenderLimits {
            max_frames: 2,
            ..exact
        },
        RenderLimits {
            max_trajectory_bytes: charge - 1,
            ..exact
        },
    ] {
        let expected = scene_at(&chunk, 0, &tight).unwrap_err().to_string();
        let actual = match ValidatedRenderTrack::new(&chunk, tight) {
            Err(e) => e.to_string(),
            Ok(proof) => proof.scene_at(&chunk, 0).unwrap_err().to_string(),
        };
        assert_eq!(actual, expected);
        assert!(default_proof.scene_at(&chunk, 0).is_ok());
    }
    // A large later scene rejects the track even when frame zero is small.
    let mut later = captured();
    let Record::Trajectory(record) = &later.record else {
        panic!()
    };
    let track = record.render.as_ref().unwrap().clone();
    let offsets = later.arrays[&track.offsets].as_u64().unwrap().to_vec();
    let mut bytes = later.arrays[&track.data].as_u8().unwrap().to_vec();
    bytes.extend_from_slice(b" ");
    let mut offsets = offsets;
    *offsets.last_mut().unwrap() += 1;
    later.arrays.insert(
        track.data,
        rl::diagnostics::schema::Array::u8(vec![bytes.len()], bytes).unwrap(),
    );
    later.arrays.insert(
        track.offsets,
        rl::diagnostics::schema::Array::u64(vec![4], offsets).unwrap(),
    );
    let limit = RenderLimits {
        max_scene_bytes: serde_json::to_vec(&scene(0)).unwrap().len(),
        ..Default::default()
    };
    assert!(ValidatedRenderTrack::new(&Arc::new(later), limit).is_err());
}

/// Retains the public validating adapter as the original-algorithm reference in
/// the same release binary. No timing threshold is part of the test contract.
#[test]
#[ignore = "serialized release measurement; run only in a coordinated timing slot"]
fn measure_validated_playback_against_original_adapter() {
    use rl::diagnostics::schema::Array;
    use std::{hint::black_box, time::Instant};
    const FRAMES: usize = 4096;
    let mut chunk = captured();
    let Record::Trajectory(record) = &chunk.record else {
        panic!()
    };
    let record = record.clone();
    chunk.arrays.insert(
        record.observations.clone(),
        Array::f32(vec![FRAMES, 1], (0..FRAMES).map(|s| s as f32).collect()).unwrap(),
    );
    chunk.arrays.insert(
        record.actions.clone(),
        Array::u64(vec![FRAMES - 1], vec![0; FRAMES - 1]).unwrap(),
    );
    chunk.arrays.insert(
        record.rewards.clone(),
        Array::f64(vec![FRAMES - 1], vec![1.; FRAMES - 1]).unwrap(),
    );
    let mut terminated = vec![0; FRAMES - 1];
    *terminated.last_mut().unwrap() = 1;
    chunk.arrays.insert(
        record.terminated.clone(),
        Array::u8(vec![FRAMES - 1], terminated).unwrap(),
    );
    chunk.arrays.insert(
        record.truncated.clone(),
        Array::u8(vec![FRAMES - 1], vec![0; FRAMES - 1]).unwrap(),
    );
    if let Some(times) = &record.simulation_times {
        chunk.arrays.insert(
            times.clone(),
            Array::f64(vec![FRAMES], (0..FRAMES).map(|s| s as f64).collect()).unwrap(),
        );
    }
    for key in record.frames.values() {
        chunk.arrays.insert(
            key.clone(),
            Array::f64(vec![FRAMES, 1], (0..FRAMES).map(|s| s as f64).collect()).unwrap(),
        );
    }
    let mut bytes = Vec::new();
    let mut offsets = vec![0];
    for step in 0..FRAMES {
        bytes.extend_from_slice(&serde_json::to_vec(&scene(step)).unwrap());
        offsets.push(bytes.len() as u64);
    }
    let track = record.render.unwrap();
    chunk
        .arrays
        .insert(track.data, Array::u8(vec![bytes.len()], bytes).unwrap());
    chunk.arrays.insert(
        track.offsets,
        Array::u64(vec![FRAMES + 1], offsets).unwrap(),
    );
    // Whole-chunk validation still decodes every recorded scene on load.
    chunk.validate().unwrap();
    let chunk = Arc::new(chunk);
    let limits = RenderLimits::default();
    let proof = ValidatedRenderTrack::new(&chunk, limits).unwrap();
    let mut fingerprint = 0u64;
    for step in 0..FRAMES {
        let indexed = proof.scene_at(&chunk, step).unwrap().unwrap();
        let original = scene_at(&chunk, step, &limits).unwrap().unwrap();
        assert_eq!(indexed, original);
        let hash = |scene: &RenderScene| crc32fast::hash(&serde_json::to_vec(scene).unwrap());
        assert_eq!(hash(&indexed), hash(&original));
        fingerprint = fingerprint.rotate_left(1) ^ u64::from(hash(&indexed));
    }
    let measure = |indexed: bool| {
        let start = Instant::now();
        for step in 0..FRAMES {
            let decoded = if indexed {
                proof.scene_at(black_box(&chunk), step)
            } else {
                scene_at(black_box(&chunk), step, &limits)
            }
            .unwrap()
            .unwrap();
            // Consume and drop the complete selected scene inside both timings.
            // Full equality/serialized-scene fingerprints are checked above.
            drop(black_box(decoded));
        }
        start.elapsed()
    };
    for trial in 0..5 {
        let (original, indexed) = if trial % 2 == 0 {
            (measure(false), measure(true))
        } else {
            let indexed = measure(true);
            (measure(false), indexed)
        };
        eprintln!(
            "playback trial={trial} frames={FRAMES} original_us={} proof_us={} fingerprint={fingerprint:016x}",
            original.as_micros(),
            indexed.as_micros()
        );
    }
}
#[test]
fn malformed_tracks_and_json_fail_without_panicking() {
    use rl::diagnostics::schema::Array;
    let original = captured();
    for offsets in [
        vec![0, 0, 1, 2],
        vec![1, 2, 3, 4],
        vec![0, 1, 2, u64::MAX],
        vec![0, 2, 1, 3],
        vec![0, 1, 2],
    ] {
        let mut chunk = original.clone();
        let Record::Trajectory(t) = &chunk.record else {
            panic!()
        };
        let key = t.render.as_ref().unwrap().offsets.clone();
        chunk
            .arrays
            .insert(key, Array::u64(vec![offsets.len()], offsets).unwrap());
        assert!(chunk.validate().is_err());
        assert!(scene_at(&chunk, 0, &RenderLimits::default()).is_err());
        assert!(ValidatedRenderTrack::new(&Arc::new(chunk), RenderLimits::default()).is_err());
    }
    for bytes in [b"{".to_vec(), vec![255], b"{} trailing".to_vec(), serde_json::to_vec(&serde_json::json!({"version":999,"bounds":{"min":[0,0],"max":[1,1]},"background":[0,0,0,255],"items":[]})).unwrap()] {
        let mut chunk = original.clone(); replace_scenes(&mut chunk, &bytes); assert!(chunk.validate().is_err());
    }
    let json = serde_json::to_string(&scene(0)).unwrap();
    for invalid in [
        json.replace("\"version\":1", "\"version\":1,\"version\":1"),
        json.replace("\"version\":1", "\"version\":1,\"unexpected\":0"),
        json.replace("\"kind\":\"text\"", "\"kind\":\"alien\""),
        json.replace("\"size\":12.0", "\"size\":0"),
        json.replace(
            "\"text\":\"0\"",
            &format!("\"text\":\"{}\"", "x".repeat(1025)),
        ),
    ] {
        let mut chunk = original.clone();
        replace_scenes(&mut chunk, invalid.as_bytes());
        assert!(chunk.validate().is_err(), "{invalid}");
    }
    let mut chunk = original.clone();
    if let Record::Trajectory(t) = &mut chunk.record {
        t.render.as_mut().unwrap().version = 999;
    }
    assert!(chunk.validate().is_err());
    let mut chunk = original;
    if let Record::Trajectory(t) = &mut chunk.record {
        t.render.as_mut().unwrap().data = "missing".into();
    }
    assert!(scene_at(&chunk, 0, &RenderLimits::default()).is_err());
}
#[test]
fn captured_scene_roundtrips_through_storage_with_frames() {
    use rl::diagnostics::{
        schema::RunManifest,
        storage::{BufferConfig, ReadLimits, Recorder, Recording, StorageConfig},
    };
    for compress in [false, true] {
        for buffered in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let mut writer = Recorder::create(
                dir.path().join("run"),
                RunManifest::new(
                    "render",
                    "test",
                    "test",
                    1,
                    ActionSpace::Discrete { count: 2 },
                    0.99,
                )
                .unwrap(),
                StorageConfig {
                    compress,
                    ..Default::default()
                },
            )
            .unwrap();
            if buffered {
                writer = writer.with_buffering(BufferConfig::default()).unwrap();
            }
            writer.append(&captured()).unwrap();
            writer.finish().unwrap();
            let run = Recording::open(dir.path().join("run"), ReadLimits::default()).unwrap();
            let chunk = run.load(0).unwrap();
            assert_eq!(
                scene_at(&chunk, 2, &RenderLimits::default()).unwrap(),
                Some(scene(2))
            );
            assert_eq!(
                chunk.arrays["frames/state"].as_f64().unwrap(),
                &[0., 1., 2.]
            );
        }
    }
}
#[test]
fn independent_capture_count_and_budgets_are_enforced() {
    let actor = RandomActor {
        space: ActionSpace::Discrete { count: 2 },
    };
    let factory = |_| {
        Ok(Custom {
            state: 0,
            space: actor.space.clone(),
            calls: Arc::new(AtomicUsize::new(0)),
        })
    };
    for capture in [
        RenderCaptureConfig {
            trajectories: 3,
            limits: RenderLimits::default(),
        },
        RenderCaptureConfig {
            trajectories: 1,
            limits: RenderLimits {
                max_frames: 2,
                ..Default::default()
            },
        },
        RenderCaptureConfig {
            trajectories: 1,
            limits: RenderLimits {
                max_scene_bytes: 1,
                ..Default::default()
            },
        },
        RenderCaptureConfig {
            trajectories: 1,
            limits: RenderLimits {
                max_trajectory_bytes: 16,
                ..Default::default()
            },
        },
        RenderCaptureConfig {
            trajectories: 1,
            limits: RenderLimits {
                max_evaluation_bytes: 16,
                ..Default::default()
            },
        },
    ] {
        assert!(evaluate_with_render(factory, &actor, &config(), &capture, |_| {}).is_err());
    }
}

#[test]
fn scene_count_point_text_and_byte_boundaries_are_exact() {
    let limits = RenderLimits::default();
    let mut chunk = captured();
    let mut s = scene(0);
    s.items = vec![s.items[0].clone(); limits.max_primitives];
    for valid in [true, false] {
        if !valid {
            s.items.push(s.items[0].clone());
        }
        replace_scenes(&mut chunk, &serde_json::to_vec(&s).unwrap());
        assert_eq!(scene_at(&chunk, 0, &limits).is_ok(), valid);
    }
    s = scene(0);
    s.items[0].primitive = RenderPrimitive::Polyline {
        points: vec![[0., 0.]; limits.max_points],
        stroke: RenderStroke {
            color: [255; 4],
            width: 1.,
        },
    };
    for valid in [true, false] {
        if !valid && let RenderPrimitive::Polyline { points, .. } = &mut s.items[0].primitive {
            points.push([1., 1.]);
        }
        replace_scenes(&mut chunk, &serde_json::to_vec(&s).unwrap());
        assert_eq!(scene_at(&chunk, 0, &limits).is_ok(), valid);
    }
    s = scene(0);
    if let RenderPrimitive::Text { text, .. } = &mut s.items[0].primitive {
        *text = "é".repeat(512);
    }
    s.items = vec![s.items[0].clone(); 4];
    replace_scenes(&mut chunk, &serde_json::to_vec(&s).unwrap());
    scene_at(&chunk, 0, &limits).unwrap();
    s.items.push(scene(0).items.remove(0));
    replace_scenes(&mut chunk, &serde_json::to_vec(&s).unwrap());
    assert!(
        scene_at(&chunk, 0, &limits)
            .unwrap_err()
            .to_string()
            .contains("text budget")
    );
    let encoded = serde_json::to_vec(&scene(0)).unwrap();
    replace_scenes(&mut chunk, &encoded);
    scene_at(
        &chunk,
        0,
        &RenderLimits {
            max_scene_bytes: encoded.len(),
            ..limits
        },
    )
    .unwrap();
    assert!(
        scene_at(
            &chunk,
            0,
            &RenderLimits {
                max_scene_bytes: encoded.len() - 1,
                ..limits
            }
        )
        .is_err()
    );
    let Record::Trajectory(t) = &chunk.record else {
        panic!()
    };
    let track = t.render.as_ref().unwrap();
    let charge = chunk.arrays[&track.data].byte_len() + chunk.arrays[&track.offsets].byte_len();
    scene_at(
        &chunk,
        0,
        &RenderLimits {
            max_trajectory_bytes: charge,
            ..limits
        },
    )
    .unwrap();
    assert!(
        scene_at(
            &chunk,
            0,
            &RenderLimits {
                max_trajectory_bytes: charge - 1,
                ..limits
            }
        )
        .is_err()
    );
}
#[test]
fn all_primitives_and_polygon_order_roundtrip() {
    let stroke = RenderStroke {
        color: [30, 60, 90, 128],
        width: 2.,
    };
    let mut s = scene(0);
    s.items.extend(
        [
            RenderPrimitive::Line {
                from: [0., 0.],
                to: [1., 1.],
                stroke: stroke.clone(),
            },
            RenderPrimitive::Polyline {
                points: vec![[0., 0.], [1., 0.], [1., 1.]],
                stroke: stroke.clone(),
            },
            RenderPrimitive::Polygon {
                points: vec![[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
                fill: Some([1, 2, 3, 128]),
                stroke: Some(stroke.clone()),
            },
            RenderPrimitive::Circle {
                center: [0., 0.],
                radius: 0.5,
                fill: None,
                stroke: Some(stroke.clone()),
            },
            RenderPrimitive::Rectangle {
                min: [-1., -1.],
                max: [0., 0.],
                fill: Some([0; 4]),
                stroke: None,
            },
        ]
        .into_iter()
        .enumerate()
        .map(|(i, primitive)| RenderItem {
            layer: i as i16 - 3,
            primitive,
        }),
    );
    let mut chunk = captured();
    replace_scenes(&mut chunk, &serde_json::to_vec(&s).unwrap());
    assert_eq!(
        scene_at(&chunk, 0, &RenderLimits::default()).unwrap(),
        Some(s.clone())
    );
    if let RenderPrimitive::Polygon { points, .. } = &mut s.items[3].primitive {
        points.reverse();
    }
    s.validate(&RenderLimits::default()).unwrap();
    for points in [
        vec![[0., 0.], [1., 1.], [1., 0.], [0., 1.]],
        vec![[0., 0.], [1., 0.], [2., 0.]],
        vec![[0., 0.], [1., 0.], [1., 1.], [0., 0.]],
    ] {
        s.items[3].primitive = RenderPrimitive::Polygon {
            points,
            fill: Some([255; 4]),
            stroke: None,
        };
        assert!(s.validate(&RenderLimits::default()).is_err());
    }
    let circle: Vec<_> = (0..2048)
        .map(|i| {
            let angle = i as f64 * std::f64::consts::TAU / 2048.;
            [angle.cos(), angle.sin()]
        })
        .collect();
    s.items = vec![RenderItem {
        layer: 0,
        primitive: RenderPrimitive::Polygon {
            points: circle,
            fill: Some([255; 4]),
            stroke: None,
        },
    }];
    s.validate(&RenderLimits::default()).unwrap();
}
#[test]
fn structural_budget_admission_precedes_decoding_next_element() {
    let mut chunk = captured();
    let prefix =
        r#"{"version":1,"bounds":{"min":[0,0],"max":[1,1]},"background":[0,0,0,0],"items":["#;
    replace_scenes(
        &mut chunk,
        format!("{prefix}{{\"primitive\": INVALID}}]}}").as_bytes(),
    );
    let error = scene_at(
        &chunk,
        0,
        &RenderLimits {
            max_primitives: 0,
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("primitive budget"), "{error}");
    replace_scenes(&mut chunk,format!("{prefix}{{\"layer\":0,\"primitive\":{{\"kind\":\"polyline\",\"points\":[INVALID]}}}}]}}").as_bytes());
    let error = scene_at(
        &chunk,
        0,
        &RenderLimits {
            max_points: 0,
            ..Default::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("point budget"), "{error}");
}
#[test]
fn ordinary_env_never_requires_render_or_frame_traits() {
    struct Plain(Custom);
    impl Env for Plain {
        fn observation_space(&self) -> ObservationSpace {
            self.0.observation_space()
        }
        fn action_space(&self) -> &ActionSpace {
            self.0.action_space()
        }
        fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
            self.0.reset(seed)
        }
        fn step(&mut self, action: &Action) -> Result<Step> {
            self.0.step(action)
        }
    }
    let actor = RandomActor {
        space: ActionSpace::Discrete { count: 2 },
    };
    let result = evaluate(
        |_| {
            Ok(Plain(Custom {
                state: 0,
                space: actor.space.clone(),
                calls: Arc::new(AtomicUsize::new(0)),
            }))
        },
        &actor,
        &config(),
    )
    .unwrap();
    let chunk = result
        .trajectory_record(0, RecordContext::new("render", 0), "plain", "unknown")
        .unwrap();
    assert!(
        scene_at(&chunk, 0, &RenderLimits::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn rendering_preserves_stochastic_environment_and_policy_streams() {
    use recording_env::RecordingEnv;
    let actor = RandomActor {
        space: RecordingEnv::new(8, 0).action_space().clone(),
    };
    let settings = EvaluationConfig {
        seeds: vec![0, 1, 999],
        trajectories: 3,
        max_steps: 9,
        ..Default::default()
    };
    let factory = |seed| Ok(RecordingEnv::new(8, seed));
    let plain = evaluate(factory, &actor, &settings).unwrap();
    let rendered = evaluate_with_render(
        factory,
        &actor,
        &settings,
        &RenderCaptureConfig {
            trajectories: 2,
            limits: RenderLimits::default(),
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(plain.returns, rendered.returns);
    assert_eq!(plain.environment_steps, rendered.environment_steps);
    for (a, b) in plain.trajectories.iter().zip(rendered.trajectories) {
        assert_eq!(a.observations, b.observations);
        assert_eq!(a.actions, b.actions);
        assert_eq!(a.rewards, b.rewards);
        assert_eq!(a.terminated, b.terminated);
        assert_eq!(a.truncated, b.truncated);
    }
}
#[test]
fn every_hard_ceiling_and_nonfinite_style_is_enforced() {
    let hard = RenderLimits::default();
    for too_large in [
        RenderLimits {
            max_scene_bytes: hard.max_scene_bytes + 1,
            ..hard
        },
        RenderLimits {
            max_primitives: hard.max_primitives + 1,
            ..hard
        },
        RenderLimits {
            max_points: hard.max_points + 1,
            ..hard
        },
        RenderLimits {
            max_text_bytes: hard.max_text_bytes + 1,
            ..hard
        },
        RenderLimits {
            max_frames: hard.max_frames + 1,
            ..hard
        },
        RenderLimits {
            max_trajectory_bytes: hard.max_trajectory_bytes + 1,
            ..hard
        },
        RenderLimits {
            max_evaluation_bytes: hard.max_evaluation_bytes + 1,
            ..hard
        },
    ] {
        assert!(too_large.validate().is_err());
        assert!(ValidatedRenderTrack::new(&Arc::new(captured()), too_large).is_err());
    }
    for primitive in [
        RenderPrimitive::Line {
            from: [f64::NAN, 0.],
            to: [0., 0.],
            stroke: RenderStroke {
                color: [0; 4],
                width: 1.,
            },
        },
        RenderPrimitive::Circle {
            center: [0., 0.],
            radius: f64::INFINITY,
            fill: Some([0; 4]),
            stroke: None,
        },
        RenderPrimitive::Circle {
            center: [0., 0.],
            radius: 1.,
            fill: None,
            stroke: None,
        },
        RenderPrimitive::Rectangle {
            min: [0., 0.],
            max: [0., 1.],
            fill: Some([0; 4]),
            stroke: None,
        },
        RenderPrimitive::Polyline {
            points: vec![[0., 0.]],
            stroke: RenderStroke {
                color: [0; 4],
                width: 1.,
            },
        },
        RenderPrimitive::Line {
            from: [0., 0.],
            to: [1., 1.],
            stroke: RenderStroke {
                color: [0; 4],
                width: f32::NAN,
            },
        },
        RenderPrimitive::Text {
            position: [0., 0.],
            text: "x".into(),
            color: [0; 4],
            size: 1025.,
        },
    ] {
        let mut s = scene(0);
        s.items[0].primitive = primitive;
        assert!(s.validate(&hard).is_err());
    }
    let mut s = scene(0);
    s.bounds.min[0] = -f64::MAX;
    s.bounds.max[0] = f64::MAX;
    assert!(s.validate(&hard).is_err());
}

#[test]
fn recorder_rejects_corrupt_scenes_without_committing_a_record() {
    use rl::diagnostics::{
        schema::RunManifest,
        storage::{ReadLimits, Recorder, Recording, StorageConfig},
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = Recorder::create(
        &path,
        RunManifest::new(
            "render",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 2 },
            0.99,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let mut invalid = captured();
    replace_scenes(&mut invalid, b"{}");
    assert!(recorder.append(&invalid).is_err());
    recorder.finish().unwrap();
    assert!(
        Recording::open(&path, ReadLimits::default())
            .unwrap()
            .entries()
            .is_empty()
    );
}

#[test]
fn cartpole_scene_projects_the_live_cart_and_pole_without_consuming_randomness() -> Result<()> {
    use rl::envs::cartpole::CartPole;

    let theta = 0.125_f64;
    let environment = CartPole::from_state(9, [0.6, 0., theta, 0.])?;
    let before = serde_json::to_value(&environment)?;
    let scene = environment.render_scene()?;
    scene.validate(&RenderLimits::default())?;
    assert_eq!(before, serde_json::to_value(&environment)?);
    assert_eq!(scene, environment.render_scene()?);
    assert!(scene.items.iter().any(|item| {
        let RenderPrimitive::Line { from, to, .. } = item.primitive else {
            return false;
        };
        (from[0] - 0.6).abs() < 1e-12
            && (to[0] - from[0] - theta.sin()).abs() < 1e-12
            && (to[1] - from[1] - theta.cos()).abs() < 1e-12
    }));
    Ok(())
}

#[test]
fn pendulum_scene_projects_a_horizontal_arm_without_consuming_randomness() -> Result<()> {
    use rl::envs::pendulum::Pendulum;

    let environment = Pendulum::from_state(9, [std::f64::consts::FRAC_PI_2, 0.])?;
    let before = serde_json::to_value(&environment)?;
    let scene = environment.render_scene()?;
    scene.validate(&RenderLimits::default())?;
    assert_eq!(before, serde_json::to_value(&environment)?);
    assert_eq!(scene, environment.render_scene()?);
    assert!(scene.items.iter().any(|item| {
        matches!(item.primitive, RenderPrimitive::Line { from: [0., 0.], to, .. }
            if (to[0] - 1.).abs() < 1e-12 && to[1].abs() < 1e-12)
    }));
    Ok(())
}

#[test]
fn finite_horizon_pendulum_render_round_trips_every_state_and_matches_plain_evaluation()
-> Result<()> {
    use rl::envs::pendulum::Pendulum;

    let factory = |seed| Ok(Pendulum::new(seed));
    let actor = RandomActor {
        space: Pendulum::new(0).action_space().clone(),
    };
    let config = EvaluationConfig {
        seeds: vec![11, 12],
        trajectories: 1,
        max_steps: 8,
        ..Default::default()
    };
    let horizon = EvaluationHorizon::FiniteDecisions(8);
    let plain = evaluate_with_horizon(factory, &actor, &config, horizon)?;
    let rendered = evaluate_with_render_and_horizon(
        factory,
        &actor,
        &config,
        horizon,
        &RenderCaptureConfig::default(),
        |_| {},
    )?;
    assert_eq!(plain.returns, rendered.returns);
    assert!(rendered.metadata.complete_returns);
    let recorded = |result: &EvaluationResult| {
        result.trajectory_record(0, RecordContext::new("pendulum", 0), "episode", "pendulum")
    };
    let plain = recorded(&plain)?;
    let rendered = recorded(&rendered)?;
    assert_eq!(evaluation_horizon(&rendered)?, horizon);
    assert_eq!(trajectory_scenario_seed(&plain)?, Some(11));
    assert_eq!(trajectory_scenario_seed(&rendered)?, Some(11));
    for key in [
        "observations",
        "actions",
        "rewards",
        "terminated",
        "truncated",
    ] {
        assert_eq!(plain.arrays[key].to_f64(), rendered.arrays[key].to_f64());
    }
    let limits = RenderLimits::default();
    for step in 0..=8 {
        assert!(scene_at(&rendered, step, &limits)?.is_some());
    }
    assert!(scene_at(&rendered, 9, &limits).is_err());
    assert_ne!(
        scene_at(&rendered, 0, &limits)?,
        scene_at(&rendered, 8, &limits)?
    );
    let mut legacy = rendered;
    legacy.arrays.remove("trajectory/scenario_seed.v1");
    legacy.validate()?;
    assert_eq!(trajectory_scenario_seed(&legacy)?, None);
    legacy.arrays.insert(
        "trajectory/scenario_seed.v1".into(),
        rl::diagnostics::schema::Array::u64(vec![2], vec![11, 12])?,
    );
    assert!(legacy.validate().is_err());
    Ok(())
}
