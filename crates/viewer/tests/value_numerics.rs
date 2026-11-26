use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{schema::RunManifest, visualization::EnvironmentSemantics},
};
use rl_viewer::{
    policy::slice::SliceMode,
    value::slice::{ActionConditioning, ValueCoordinate as C, ValueSlice},
};

fn manifest(space: ActionSpace) -> RunManifest {
    RunManifest::new("value", "test", "test", 4, space, 0.97).unwrap()
}
#[test]
fn arbitrary_and_reversed_joint_axes_preserve_complete_fixed_inputs() {
    let m = manifest(ActionSpace::Box {
        low: vec![-10.; 3],
        high: vec![10.; 3],
    });
    let env = EnvironmentSemantics::new(&m);
    let mut s = ValueSlice::new(env, &Observation::new(vec![1., 2., 3., 4.]), &[]).unwrap();
    s.x = C::Action(2);
    s.y = Some(C::Observation(3));
    s.mode = SliceMode::TwoDimensional;
    s.resolution = [3, 2];
    s.action_ranges[2] = [-2., 2.];
    s.state.ranges[3] = [4., 8.];
    s.validate(env).unwrap();
    assert_eq!(
        s.observation_at(4).unwrap().dense_values(),
        vec![1., 2., 3., 8.]
    );
    assert_eq!(
        s.action_at(4, Some(&Action::Continuous(vec![5., 6., 7.])))
            .unwrap(),
        Some(Action::Continuous(vec![5., 6., 0.]))
    );
    s.x = C::Observation(3);
    s.y = Some(C::Action(2));
    s.resolution = [2, 3];
    assert_eq!(
        s.observation_at(3).unwrap().dense_values(),
        vec![1., 2., 3., 8.]
    );
    assert_eq!(
        s.action_at(3, Some(&Action::Continuous(vec![5., 6., 7.])))
            .unwrap(),
        Some(Action::Continuous(vec![5., 6., 0.]))
    );
}
#[test]
fn action_action_and_state_only_conditioning_never_replace_unplotted_components() {
    let m = manifest(ActionSpace::Unbounded { dimensions: 3 });
    let env = EnvironmentSemantics::new(&m);
    let mut s = ValueSlice::new(env, &Observation::new(vec![1., 2., 3., 4.]), &[]).unwrap();
    s.x = C::Action(1);
    s.y = Some(C::Action(0));
    s.resolution = [2, 2];
    s.action_ranges = vec![[0., 2.], [-1., 1.], [-4., 4.]];
    assert_eq!(
        s.action_at(2, Some(&Action::Continuous(vec![8., 9., 10.])))
            .unwrap(),
        Some(Action::Continuous(vec![2., -1., 10.]))
    );
    s.conditioning = ActionConditioning::Fixed(Action::Continuous(vec![4., 5., 6.]));
    assert_eq!(
        s.action_at(3, None).unwrap(),
        Some(Action::Continuous(vec![2., 1., 6.]))
    );
    s.x = C::Observation(2);
    s.y = None;
    s.mode = SliceMode::OneDimensional;
    s.conditioning = ActionConditioning::Policy;
    assert_eq!(
        s.action_at(0, Some(&Action::Continuous(vec![7., 8., 9.])))
            .unwrap(),
        Some(Action::Continuous(vec![7., 8., 9.]))
    );
    assert_eq!(s.action_at(0, None).unwrap(), None);
}
#[test]
fn masked_categories_enumerate_only_legal_actions_and_never_synthesize_states() {
    let m = manifest(ActionSpace::Discrete { count: 4 });
    let env = EnvironmentSemantics::new(&m);
    let mut anchor = Observation::new(vec![1., 2., 3., 4.]);
    anchor
        .as_dense_mut()
        .expect("dense diagnostic input")
        .action_mask = Some(vec![false, true, false, true]);
    let mut s = ValueSlice::new(env, &anchor, &[]).unwrap();
    assert!(s.observation_at(0).is_err());
    s.x = C::DiscreteAction;
    s.y = None;
    s.mode = SliceMode::OneDimensional;
    s.validate(env).unwrap();
    assert_eq!(s.len().unwrap(), 2);
    assert_eq!(s.observation_at(1).unwrap(), anchor);
    assert_eq!(s.action_at(0, None).unwrap(), Some(Action::Discrete(1)));
    assert_eq!(s.action_at(1, None).unwrap(), Some(Action::Discrete(3)));
    assert!(s.action_at(2, None).is_err());
    s.conditioning = ActionConditioning::Fixed(Action::Discrete(0));
    assert!(s.validate(env).is_err());
}
#[test]
fn invalid_ranges_coordinates_history_and_terminal_masks_are_explicit_errors() {
    let m = manifest(ActionSpace::Box {
        low: vec![-1.; 2],
        high: vec![1.; 2],
    });
    let env = EnvironmentSemantics::new(&m);
    let mut s = ValueSlice::new(env, &Observation::new(vec![0.; 4]), &[]).unwrap();
    s.x = C::Action(0);
    s.y = None;
    s.mode = SliceMode::OneDimensional;
    s.action_ranges[0] = [-2., 2.];
    assert!(s.validate(env).is_err());
    s.action_ranges[0] = [-1., 1.];
    s.state = s.state.with_history();
    assert!(s.observation_at(0).is_err());
    let m = manifest(ActionSpace::Discrete { count: 2 });
    let mut o = Observation::new(vec![0.; 4]);
    o.as_dense_mut()
        .expect("dense diagnostic input")
        .action_mask = Some(vec![false, false]);
    assert!(ValueSlice::new(EnvironmentSemantics::new(&m), &o, &[]).is_err());
}

mod preparation {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{
        diagnostics::{models::*, schema::*, storage::*, value::*},
        nn::{
            critics::scalar::{QValue, Value},
            matrix,
            network::Mlp,
            policy::Deterministic,
            vector,
        },
    };
    use rl_viewer::{data::RunIndex, interactive::Interactive, value::prepared::*};
    use std::{
        collections::BTreeMap,
        sync::Arc,
        time::{Duration, Instant},
    };
    fn fixture() -> (tempfile::TempDir, RunIndex) {
        let dir = tempfile::tempdir().unwrap();
        let space = ActionSpace::Unbounded { dimensions: 3 };
        let m = manifest(space.clone());
        let mut writer =
            Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
        for rev in [3, 7] {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let mut net = Mlp::new(4, &[], 3, &mut rng).unwrap();
            net.layers[0].weight = net.layers[0]
                .weight
                .clone()
                .map(|_| matrix(vec![0.; 12], 4, 3));
            net.layers[0].bias = net.layers[0]
                .bias
                .take()
                .map(|p| p.map(|_| vector(&[rev as f32, 2., 3.])));
            let actor = Deterministic::from_network(net, space.clone()).unwrap();
            let mut b =
                SnapshotBuilder::new("value", RecordContext::new("value", rev), 4, space.clone())
                    .unwrap();
            b.deterministic("actor", &actor).unwrap();
            for (name, bias) in [("q1", 0.), ("q2", 2.)] {
                let mut q = QValue::new(4, 3, 1, &[], &mut rng).unwrap();
                q.network.layers[0].weight = q.network.layers[0]
                    .weight
                    .clone()
                    .map(|_| matrix(vec![1., 0., 0., 0., 1., 10., 100.], 7, 1));
                q.network.layers[0].bias = q.network.layers[0]
                    .bias
                    .take()
                    .map(|p| p.map(|_| vector(&[bias])));
                b.scalar(name, &q).unwrap();
            }
            b.value("v", &Value::new(4, &[], &mut rng).unwrap())
                .unwrap();
            let mut chunk = b.finish().unwrap();
            let critics = [("q1", 0), ("q2", 1)]
                .into_iter()
                .map(|(name, index)| {
                    (
                        name.into(),
                        CriticDeclaration {
                            kind: ValueCriticKind::ScalarQ,
                            learned_baseline: None,
                            actor: Some(ContinuationIdentity {
                                identity: "actor".into(),
                                stochastic: false,
                            }),
                            continuation: Some(ContinuationIdentity {
                                identity: format!("actor-{rev}"),
                                stochastic: false,
                            }),
                            twin: Some(TwinMembership {
                                group: "twin".into(),
                                index,
                            }),
                            utility: None,
                        },
                    )
                })
                .collect();
            attach_declaration(
                &mut chunk,
                &ValueDeclaration {
                    version: 1,
                    run_id: "value".into(),
                    revision: rev,
                    model_role: "online".into(),
                    conventions: Some(ValueConventions {
                        dynamics: "test".into(),
                        observation: "obs".into(),
                        action: "action".into(),
                        reward_units: "reward".into(),
                        reward_convention: m.reward_convention.clone(),
                        discount: 0.97,
                        horizon: "episodic".into(),
                        terminal_treatment: "reward-once".into(),
                        action_constraints: "unbounded".into(),
                    }),
                    critics,
                },
            )
            .unwrap();
            writer.append(&chunk).unwrap();
        }
        for id in ["episode-a", "episode-b"] {
            let mut context = RecordContext::new("value", 3);
            context.trajectory = Some(id.into());
            let chunk = Chunk {
                context,
                record: Record::Trajectory(TrajectoryRecord {
                    render: None,
                    id: id.into(),
                    environment: "test".into(),
                    observations: "o".into(),
                    actions: "a".into(),
                    rewards: "r".into(),
                    terminated: "t".into(),
                    truncated: "x".into(),
                    simulation_times: None,
                    action_masks: None,
                    frames: BTreeMap::new(),
                    evaluation: EvaluationMetadata {
                        conditioning: Conditioning::InitialScenario,
                        initial_action: ActionCondition::CurrentPolicy,
                        continuation_policy: "actor-3".into(),
                        stochastic_policy: false,
                        discount: 0.5,
                        independent_future_streams: true,
                        seed_methodology: "fixed".into(),
                        truncation_handling: "no bootstrap".into(),
                        complete_returns: true,
                    },
                }),
                arrays: BTreeMap::from([
                    (
                        "o".into(),
                        Array::f64(
                            vec![4, 4],
                            vec![
                                0.25, 0., 0., 0., 0.5, 0., 0., 0., 0.75, 0., 0., 0., 1., 0., 0., 0.,
                            ],
                        )
                        .unwrap(),
                    ),
                    (
                        "a".into(),
                        Array::f64(vec![3, 3], vec![3., 2., 3., 3., 2., 3., 3., 2., 3.]).unwrap(),
                    ),
                    ("r".into(), Array::f64(vec![3], vec![1., 2., 4.]).unwrap()),
                    ("t".into(), Array::u8(vec![3], vec![0, 0, 1]).unwrap()),
                    ("x".into(), Array::u8(vec![3], vec![0, 0, 0]).unwrap()),
                ]),
            };
            writer.append(&chunk).unwrap();
        }
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        (dir, run)
    }
    fn source(run: &RunIndex, index: usize, name: &str) -> ValueSource {
        ValueSource::model(run, index, name, &run.recording().load(index).unwrap()).unwrap()
    }
    fn request(run: &RunIndex, index: usize) -> ValueRequest {
        let slice = ValueSlice::new(
            EnvironmentSemantics::new(run.recording().manifest()),
            &Observation::new(vec![0.; 4]),
            &[],
        )
        .unwrap();
        let mut r = ValueRequest::new(
            source(run, index, "q1"),
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::State(Observation::new(vec![0.25, 0., 0., 0.])),
        )
        .unwrap();
        Arc::make_mut(&mut r.slice).resolution = [2, 2];
        r
    }
    fn finish(i: &mut Interactive) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while i.value_pending() {
            i.poll_value().unwrap();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(i.value_error().is_none(), "{:?}", i.value_error());
    }
    #[test]
    fn adaptive_color_limits_follow_the_slice_and_keep_reference_colours_shared() {
        let (_dir, run) = fixture();
        let mut i = Interactive::new(|| {}).unwrap();
        let mut r = request(&run, 0);
        {
            let slice = Arc::make_mut(&mut r.slice);
            slice.state.ranges.fill([-1., 1.]);
            slice.action_ranges.fill([-2., 2.]);
        }
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        // q = s[0] + a[0] + 10*a[1] + 100*a[2]; the actor chooses [3,2,3].
        // State plots must scale q(s, pi(s)), not arbitrary actions in [-2,2].
        let grid = &i.value_frame().unwrap().grid;
        assert_eq!(grid.color_domain, Some([321.8, 324.2]));
        assert!(grid.values.iter().all(|v| (322. ..=324.).contains(v)));
        {
            let slice = Arc::make_mut(&mut r.slice);
            slice.x = rl_viewer::value::slice::ValueCoordinate::Observation(2);
            slice.state.fixed[0] = 0.5;
            slice.resolution = [5, 3];
        }
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let grid = &i.value_frame().unwrap().grid;
        let constant_range = grid.color_domain.unwrap();
        assert!(constant_range[0] < 323.5 && constant_range[1] > 323.5);
        assert!(constant_range[1] - constant_range[0] < 0.001);
        assert!(grid.values.iter().all(|v| *v == 323.5));

        r.reference_value = Some(source(&run, 1, "q1"));
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        assert_eq!(
            i.value_frame().unwrap().grid.reference_color_domain,
            Some(constant_range)
        );
        r.condition = ComparisonCondition::ControllerValues;
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        assert_eq!(
            i.value_frame().unwrap().grid.reference_color_domain,
            Some(constant_range)
        );

        r.reference_value = None;
        r.condition = ComparisonCondition::SameInput;
        Arc::make_mut(&mut r.slice).conditioning =
            ActionConditioning::Fixed(Action::Continuous(vec![0.; 3]));
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let fixed_range = i.value_frame().unwrap().grid.color_domain.unwrap();
        assert!(fixed_range[0] < 0.5 && fixed_range[1] > 0.5);
        assert!(fixed_range[1] - fixed_range[0] < 0.001);
        Arc::make_mut(&mut r.slice).x = C::Action(2);
        i.submit_value(r).unwrap();
        finish(&mut i);
        // Only the plotted action component varies; unplotted components stay fixed.
        assert_eq!(
            i.value_frame().unwrap().grid.color_domain,
            Some([-239.5, 240.5])
        );
    }
    #[test]
    fn actual_actions_twins_references_pins_and_warm_shared_worker_are_coherent() {
        let (_dir, run) = fixture();
        let mut i = Interactive::new(|| {}).unwrap();
        let mut r = request(&run, 0);
        r.reference_value = Some(source(&run, 1, "q1"));
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let frame = i.value_frame().unwrap();
        let p = frame.selected.as_ref().unwrap();
        assert_eq!(p.value, 323.25);
        assert_eq!(p.reference_value, Some(323.25));
        assert_eq!(p.action, Some(Action::Continuous(vec![3., 2., 3.])));
        let counts = i.value_metrics();
        let grid = frame.grid.clone();
        r.pin = ValuePin::State(Observation::new(vec![0.625, 0., 0., 0.]));
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        assert!(Arc::ptr_eq(&grid, &i.value_frame().unwrap().grid));
        assert_eq!(i.value_metrics().grid_batches, counts.grid_batches);
        assert_eq!(i.value_metrics().restores, 2);
        let idle = i.value_metrics();
        for _ in 0..4 {
            i.submit_value(r.clone()).unwrap();
            i.poll_value().unwrap();
        }
        assert_eq!(i.value_metrics().point_batches, idle.point_batches);
        r.condition = ComparisonCondition::ControllerValues;
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        assert_eq!(
            i.value_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .reference_value,
            Some(327.625)
        );
        r.current = source(&run, 0, "q1").with_twin("twin").unwrap();
        r.output = ValueOutput::Disagreement;
        r.reference_value = None;
        r.condition = ComparisonCondition::SameInput;
        i.submit_value(r).unwrap();
        finish(&mut i);
        assert_eq!(
            i.value_frame().unwrap().selected.as_ref().unwrap().value,
            2.
        );
        i.retire_session();
        assert!(i.value_frame().is_none());
    }
    #[test]
    fn source_validation_uses_real_manifest_and_encoding_separately() {
        let (_dir, run) = fixture();
        let mut c = run.recording().load(0).unwrap();
        let mut d = declaration(&c).unwrap().unwrap();
        c.arrays.remove(VALUE_METADATA_KEY);
        d.conventions.as_mut().unwrap().discount = 0.9;
        attach_declaration(&mut c, &d).unwrap();
        assert!(ValueSource::model(&run, 0, "q1", &c).is_err());
        let mut c = run.recording().load(0).unwrap();
        if let Record::ModelSnapshot(s) = &mut c.record {
            s.configuration["observation_dim"] = serde_json::json!(3);
        }
        assert!(ValueSource::model(&run, 0, "q1", &c).is_err());
        assert_eq!(source(&run, 0, "q1").manifest().discount, 0.97);
        assert!(
            ValueSource::model(&run, 0, "v", &run.recording().load(0).unwrap()).is_ok(),
            "an undeclared actual baseline remains inspectable with unknown semantics"
        );
    }
    fn visit(run: &RunIndex, index: usize) -> rl_viewer::policy::prepared::VisitSource {
        let key = run.key(index).unwrap();
        let mut resolver = rl_viewer::resolver::ProbeResolver::new(1024 * 1024);
        let chunk = resolver
            .lease_visitation(run, &key, Arc::new(run.recording().load(index).unwrap()))
            .unwrap();
        rl_viewer::policy::prepared::VisitSource { key, chunk }
    }
    #[test]
    fn state_value_accepts_a_terminal_trajectory_observation_without_an_action() {
        let (_dir, run) = fixture();
        let mut request = request(&run, 0);
        request.current = source(&run, 0, "v");
        request.pin = ValuePin::Recorded {
            source: visit(&run, 2),
            row: 3,
        };
        let mut interactive = Interactive::new(|| {}).unwrap();
        interactive.submit_value(request).unwrap();
        finish(&mut interactive);
        let point = interactive
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap();
        assert_eq!(point.input.dense_values(), vec![1., 0., 0., 0.]);
        assert_eq!(point.recorded_row, Some(3));
        assert!(point.final_observation);
        assert!(point.action.is_none());
        assert!(point.observed_return.is_none());
    }
    #[test]
    fn visits_recenter_exact_inputs_and_keep_individual_episode_identity_across_checkpoints() {
        let (_dir, run) = fixture();
        let mut r = request(&run, 0);
        r.visits = vec![visit(&run, 2), visit(&run, 3)];
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let grid = &i.value_frame().unwrap().grid;
        let visits = grid.visits.as_ref().unwrap();
        assert_eq!(visits.filtered_count, 6);
        assert_eq!(visits.markers.len(), 6);
        assert_ne!(visits.markers[0].source.key, visits.markers[3].source.key);
        let (pin, slice, intent) = grid.select_visit(0).unwrap();
        assert!(matches!(
            intent,
            rl_viewer::pane::SelectionIntent::Input(
                rl_viewer::data::SelectedInput::TrajectoryStep { step: 0, .. }
            )
        ));
        assert_eq!(slice.state.fixed, vec![0.25, 0., 0., 0.]);
        assert_eq!(
            slice.conditioning,
            ActionConditioning::Fixed(Action::Continuous(vec![3., 2., 3.]))
        );
        r.pin = pin;
        r.slice = Arc::new(slice);
        r.current = source(&run, 1, "q1");
        r.requested_revision = 7;
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let p = i.value_frame().unwrap().selected.as_ref().unwrap();
        assert_eq!(p.observed_return, Some(3.));
        assert_eq!(p.recorded_revision, Some(3));
        assert_eq!(p.action, Some(Action::Continuous(vec![3., 2., 3.])));
        let evidence = visit(&run, 2);
        assert_eq!(
            rl_viewer::value::visitation::observed_return(
                &evidence,
                0,
                &p.input,
                Some(&Action::Continuous(vec![7., 2., 3.])),
                true,
                &run.recording().manifest().action_space
            )
            .unwrap(),
            None
        );
        r.pin = ValuePin::State(p.input.clone());
        Arc::make_mut(&mut r.slice).conditioning = ActionConditioning::Policy;
        i.submit_value(r).unwrap();
        finish(&mut i);
        let p = i.value_frame().unwrap().selected.as_ref().unwrap();
        assert_eq!(p.observed_return, None);
        assert_eq!(p.action, Some(Action::Continuous(vec![7., 2., 3.])));
        assert_eq!(
            i.value_frame()
                .unwrap()
                .grid
                .visits
                .as_ref()
                .unwrap()
                .filtered_count,
            0,
            "unplotted policy actions must match"
        );
    }
    #[test]
    fn full_action_pin_keeps_unplotted_components_through_actor_changes() {
        let (_dir, run) = fixture();
        let mut r = request(&run, 0);
        let s = Arc::make_mut(&mut r.slice);
        s.x = C::Observation(0);
        s.y = Some(C::Action(1));
        s.action_ranges[1] = [1., 3.];
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let (pin, slice) = i.value_frame().unwrap().grid.pin_at(3).unwrap();
        assert_eq!(
            slice.conditioning,
            ActionConditioning::Fixed(Action::Continuous(vec![3., 3., 3.]))
        );
        r.pin = pin;
        r.slice = Arc::new(slice);
        r.current = source(&run, 1, "q1");
        r.requested_revision = 7;
        i.submit_value(r).unwrap();
        finish(&mut i);
        assert_eq!(
            i.value_frame().unwrap().selected.as_ref().unwrap().action,
            Some(Action::Continuous(vec![3., 3., 3.]))
        );
    }
    #[test]
    fn observed_return_edges_use_decisions_exact_inputs_and_actual_evaluation_metadata() {
        use rl_viewer::value::visitation::{exact_input, observed_return};
        let (_base_dir, base) = fixture();
        let dir = tempfile::tempdir().unwrap();
        let m = base.recording().manifest().clone();
        let mut writer =
            Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
        for variant in 0..4 {
            let mut chunk = base.recording().load(2).unwrap();
            chunk.arrays.insert(
                "o".into(),
                Array::f64(
                    vec![2, 4],
                    vec![
                        if variant == 3 { 0.1 } else { 0.25 },
                        0.,
                        0.,
                        0.,
                        1.,
                        0.,
                        0.,
                        0.,
                    ],
                )
                .unwrap(),
            );
            chunk.arrays.insert(
                "a".into(),
                Array::f64(vec![1, 3], vec![3., 2., 3.]).unwrap(),
            );
            chunk
                .arrays
                .insert("r".into(), Array::f64(vec![1], vec![7.]).unwrap());
            chunk.arrays.insert(
                "t".into(),
                Array::u8(vec![1], vec![u8::from(variant != 1)]).unwrap(),
            );
            chunk.arrays.insert(
                "x".into(),
                Array::u8(vec![1], vec![u8::from(variant == 1)]).unwrap(),
            );
            if let Record::Trajectory(t) = &mut chunk.record {
                t.id = format!("edge-{variant}");
                chunk.context.trajectory = Some(t.id.clone());
                if variant == 1 {
                    t.evaluation.complete_returns = false;
                }
                if variant == 2 {
                    t.evaluation.continuation_policy.clear();
                }
            }
            if variant == 2 {
                assert!(
                    writer.append(&chunk).is_err(),
                    "missing policy identity is rejected by the recording schema"
                );
            } else {
                writer.append(&chunk).unwrap();
            }
        }
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let o = Observation::new(vec![0.25, 0., 0., 0.]);
        let a = Action::Continuous(vec![3., 2., 3.]);
        assert_eq!(
            observed_return(&visit(&run, 0), 0, &o, Some(&a), true, &m.action_space).unwrap(),
            Some(7.)
        );
        assert!(
            exact_input(&visit(&run, 0), 1, &m.action_space).is_err(),
            "T+1 terminal observation is not a decision"
        );
        assert_eq!(
            observed_return(&visit(&run, 1), 0, &o, Some(&a), true, &m.action_space).unwrap(),
            None
        );
        assert!(
            exact_input(&visit(&run, 2), 0, &m.action_space).is_err(),
            "nonexact f64 input cannot acquire a synthetic f32 identity"
        );
        let near = Observation::new(vec![0.250001, 0., 0., 0.]);
        assert_eq!(
            observed_return(&visit(&run, 0), 0, &near, Some(&a), true, &m.action_space).unwrap(),
            None
        );
    }
    fn search_finish(
        run: &RunIndex,
        search: &mut rl_viewer::policy::prepared::ModelSearch,
        loader: &mut rl_viewer::data::Loader,
        resolver: &mut rl_viewer::resolver::ProbeResolver,
    ) -> Option<ValueSource> {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            loader.begin_pass();
            loader.poll();
            let result = search.advance_value(run, loader, resolver).unwrap();
            loader.finish_pass().unwrap();
            if result.is_some() || !search.pending() {
                return result;
            }
            assert!(Instant::now() < deadline, "{:?}", search.last_rejection);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn search_last_candidate_and_negative_append_run_revision_invalidation() {
        use rl_viewer::{data::Loader, policy::prepared::ModelSearch, resolver::ProbeResolver};
        let (_base_dir, base) = fixture();
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Recorder::create(
            dir.path().join("run"),
            base.recording().manifest().clone(),
            StorageConfig::default(),
        )
        .unwrap();
        let mut run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut loader = Loader::new(8 * 1024 * 1024, || {}).unwrap();
        let mut resolver = ProbeResolver::new(8 * 1024 * 1024);
        let mut search = ModelSearch::new("online", 258, Some("q1")).unwrap();
        assert!(search_finish(&run, &mut search, &mut loader, &mut resolver).is_none());
        let idle = loader.completed_loads;
        for _ in 0..5 {
            assert!(
                search
                    .advance_value(&run, &mut loader, &mut resolver)
                    .unwrap()
                    .is_none()
            );
        }
        assert_eq!(loader.completed_loads, idle);
        writer.append(&base.recording().load(0).unwrap()).unwrap();
        run.refresh().unwrap();
        assert!(search_finish(&run, &mut search, &mut loader, &mut resolver).is_some());
        for rev in 4..259 {
            let mut b = SnapshotBuilder::new(
                "unsupported",
                RecordContext::new("value", rev),
                4,
                run.recording().manifest().action_space.clone(),
            )
            .unwrap();
            let net = Mlp::new(4, &[], 1, &mut ChaCha8Rng::seed_from_u64(2)).unwrap();
            b.opaque("opaque", &net, "unsupported").unwrap();
            writer.append(&b.finish().unwrap()).unwrap();
        }
        run.refresh().unwrap();
        search.retarget("online", 258, Some("q1")).unwrap();
        let s = search_finish(&run, &mut search, &mut loader, &mut resolver).unwrap();
        assert_eq!(s.identity().revision, 3);
        assert_eq!(search.checked_candidates(), 256);
        search.retarget("online", 2, Some("q1")).unwrap();
        assert!(search_finish(&run, &mut search, &mut loader, &mut resolver).is_none());
        search.retarget("online", 7, Some("q1")).unwrap();
        assert_eq!(
            search_finish(&base, &mut search, &mut loader, &mut resolver)
                .unwrap()
                .identity()
                .revision,
            7
        );
        writer.finish().unwrap();
    }
    #[test]
    fn immutable_probe_only_critics_stay_sparse_and_keep_genuine_pins() {
        let (_base_dir, base) = fixture();
        let model =
            ModelBundle::load(&base.recording().load(0).unwrap(), ModelLimits::default()).unwrap();
        let result = model
            .inspect_critics(&CriticInspectionRequest {
                critics: vec!["q1".into()],
                observations: vec![
                    Observation::new(vec![-0.5, 0., 0., 0.]),
                    Observation::new(vec![0.75, 0., 0., 0.]),
                ],
                fixed_actions: Some(vec![Action::Continuous(vec![3., 2., 3.]); 2]),
            })
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Recorder::create(
            dir.path().join("run"),
            base.recording().manifest().clone(),
            StorageConfig::default(),
        )
        .unwrap();
        writer.append(&result.probes).unwrap();
        writer.append(&result.critics["q1"]).unwrap();
        let policy = model
            .inspect(&InspectionRequest::new(
                "actor",
                vec![
                    Observation::new(vec![-0.5, 0., 0., 0.]),
                    Observation::new(vec![0.75, 0., 0., 0.]),
                ],
            ))
            .unwrap();
        writer.append(&policy.probes).unwrap();
        writer.append(&policy.policy).unwrap();
        let mut later = result.probes.clone();
        if let Record::ProbeSet(p) = &later.record {
            let key = p.observations.clone();
            later
                .arrays
                .insert(key, Array::f32(vec![2, 4], vec![99.; 8]).unwrap());
        }
        writer.append(&later).unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut loader = rl_viewer::data::Loader::new(1024 * 1024, || {}).unwrap();
        let mut resolver = rl_viewer::resolver::ProbeResolver::new(1024 * 1024);
        let mut search =
            rl_viewer::policy::prepared::ModelSearch::new("online", 3, Some("q1")).unwrap();
        let source = search_finish(&run, &mut search, &mut loader, &mut resolver).unwrap();
        assert!(source.identity().recorded);
        let slice =
            ValueSlice::new(source.semantics(), &Observation::new(vec![0.; 4]), &[]).unwrap();
        let mut r = ValueRequest::new(
            source.clone(),
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::Input {
                observation: Observation::new(vec![-0.5, 0., 0., 0.]),
                action: Some(Action::Continuous(vec![3., 2., 3.])),
            },
        )
        .unwrap();
        r.reference_value = Some(source);
        let selection = rl_viewer::data::Selection {
            revision: 3,
            input: rl_viewer::data::SelectedInput::ProbeRow {
                set: "interactive".into(),
                row: 0,
                sequence: 2,
            },
            ..Default::default()
        };
        let policy_frame = resolver
            .resolve_policy(&run, &selection, |index| {
                Ok(Some(Arc::new(run.recording().load(index)?)))
            })
            .unwrap()
            .record
            .unwrap();
        r.reference_policy = Some(
            rl_viewer::policy::prepared::PolicySource::recorded_controller(
                &run,
                &policy_frame,
                "recorded actor",
            )
            .unwrap(),
        );
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let g = &i.value_frame().unwrap().grid;
        assert!(!g.regular());
        assert_eq!(g.positions, vec![[-0.5, 0.], [0.75, 0.]]);
        assert_eq!(g.values, vec![322.5, 323.75]);
        assert_eq!(
            g.reference_series.as_ref().unwrap().positions,
            vec![[-0.5, 0.], [0.75, 0.]]
        );
        assert_eq!(
            g.reference_policy_series.as_ref().unwrap().actions,
            vec![Action::Continuous(vec![3., 2., 3.]); 2]
        );
        assert_eq!(g.reference_values, vec![Some(322.5), Some(323.75)]);
        let (pin, slice) = g.pin_at(1).unwrap();
        assert!(matches!(pin, ValuePin::Probe { row: 1, .. }));
        r.pin = pin;
        r.slice = Arc::new(slice);
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let p = i.value_frame().unwrap().selected.as_ref().unwrap();
        assert_eq!(p.value, 323.75);
        assert_eq!(p.recorded_row, Some(1));
        assert_eq!(p.observed_return, None);
        assert_eq!(i.value_metrics().restores, 0);
        let held = i.value_frame().unwrap().selected.as_ref().unwrap().clone();
        let mut rejected = Vec::new();
        for altered_action in [Some(Action::Continuous(vec![7., 2., 3.])), None] {
            let mut altered = r.clone();
            let ValuePin::Probe { action, .. } = &mut altered.pin else {
                panic!("genuine pin expected")
            };
            *action = altered_action;
            i.submit_value(altered).unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            while i.value_pending() {
                i.poll_value().unwrap();
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(1));
            }
            rejected.push(
                i.value_error().is_some()
                    && i.value_frame()
                        .and_then(|f| f.selected.as_ref())
                        .is_some_and(|p| {
                            std::sync::Arc::ptr_eq(p, &held)
                                && p.action == held.action
                                && p.input == held.input
                        }),
            );
        }
        assert_eq!(
            rejected,
            vec![true, true],
            "a genuine recorded Q pin cannot change or omit its first action"
        );
    }

    #[test]
    fn value_only_full_quantile_heads_and_batched_moment_utility_use_real_models() {
        use rl::nn::critics::quantile::QuantileGrid;
        let dir = tempfile::tempdir().unwrap();
        let m = manifest(ActionSpace::Unbounded { dimensions: 3 });
        let mut writer =
            Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        for (index, name) in [(0, "v"), (1, "z"), (2, "m")] {
            let mut b = SnapshotBuilder::new(
                name,
                RecordContext::new("value", index),
                4,
                m.action_space.clone(),
            )
            .unwrap();
            let constant = |heads: usize, values: Vec<f32>, rng: &mut ChaCha8Rng| {
                let mut q = QValue::new(4, 3, heads, &[], rng).unwrap();
                q.network.layers[0].weight = q.network.layers[0]
                    .weight
                    .clone()
                    .map(|_| matrix(vec![0.; 7 * heads], 7, heads));
                q.network.layers[0].bias = q.network.layers[0]
                    .bias
                    .take()
                    .map(|p| p.map(|_| vector(&values)));
                q
            };
            if name == "v" {
                b.value(name, &Value::new(4, &[], &mut rng).unwrap())
                    .unwrap();
            } else if name == "z" {
                b.quantiles(
                    name,
                    &constant(4096, (0..4096).rev().map(|i| i as f32).collect(), &mut rng),
                    &QuantileGrid::uniform(4096).unwrap(),
                )
                .unwrap();
            } else {
                b.moments(
                    name,
                    &constant(1, vec![2.], &mut rng),
                    &constant(1, vec![3.], &mut rng),
                )
                .unwrap();
            }
            let mut chunk = b.finish().unwrap();
            if name == "m" {
                attach_declaration(
                    &mut chunk,
                    &ValueDeclaration {
                        version: 1,
                        run_id: "value".into(),
                        revision: index,
                        model_role: "online".into(),
                        conventions: None,
                        critics: BTreeMap::from([(
                            "m".into(),
                            CriticDeclaration {
                                kind: ValueCriticKind::MomentQ,
                                learned_baseline: None,
                                actor: None,
                                continuation: None,
                                twin: None,
                                utility: Some(MomentUtility::MeanVariance { penalty: 2. }),
                            },
                        )]),
                    },
                )
                .unwrap();
            }
            writer.append(&chunk).unwrap();
        }
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut i = Interactive::new(|| {}).unwrap();
        for (index, name, output) in [
            (0, "v", ValueOutput::ExpectedReturn),
            (1, "z", ValueOutput::ExpectedReturn),
            (2, "m", ValueOutput::Utility),
        ] {
            let source = source(&run, index, name);
            let mut slice =
                ValueSlice::new(source.semantics(), &Observation::new(vec![0.; 4]), &[]).unwrap();
            slice.resolution = [2, 2];
            if index != 0 {
                slice.conditioning = ActionConditioning::Fixed(Action::Continuous(vec![0.; 3]));
            }
            let pin = ValuePin::Input {
                observation: Observation::new(vec![0.; 4]),
                action: if index == 0 {
                    None
                } else {
                    Some(Action::Continuous(vec![0.; 3]))
                },
            };
            i.submit_value(ValueRequest::new(source, output, slice, pin).unwrap())
                .unwrap();
            finish(&mut i);
            let frame = i.value_frame().unwrap();
            let p = frame.selected.as_ref().unwrap();
            if index == 1 {
                assert_eq!(p.value, 2047.5);
                let ValuePrediction::Quantiles {
                    raw,
                    locations,
                    masses,
                } = &p.prediction
                else {
                    panic!()
                };
                assert_eq!(
                    (raw.len(), locations.len(), masses.len()),
                    (4096, 4096, 4096)
                );
                assert_eq!((raw[0], raw[4095]), (4095., 0.));
            }
            if index == 2 {
                assert_eq!(frame.grid.values, vec![2.; 4]);
                assert_eq!(p.value, 2.);
                assert_eq!(p.prediction.moment_summary().unwrap().variance, -1.);
            }
        }
        assert!(i.value_metrics().prepared_peak <= 4 * 1024 * 1024);
        assert!(i.value_metrics().scratch_peak <= 4 * 1024 * 1024);
        // Persist real moment inspection, then exercise current and reference
        // sparse utility preparation without restoring model weights.
        let model =
            ModelBundle::load(&run.recording().load(2).unwrap(), ModelLimits::default()).unwrap();
        let mut inspected = model
            .inspect_critics(&CriticInspectionRequest {
                critics: vec!["m".into()],
                observations: vec![Observation::new(vec![0.; 4]); 2],
                fixed_actions: Some(vec![Action::Continuous(vec![0.; 3]); 2]),
            })
            .unwrap();
        let c = inspected.critics.get_mut("m").unwrap();
        let mut d = declaration(c).unwrap().unwrap();
        d.conventions = Some(ValueConventions {
            dynamics: "test".into(),
            observation: "obs".into(),
            action: "action".into(),
            reward_units: "reward".into(),
            reward_convention: m.reward_convention.clone(),
            discount: m.discount,
            horizon: "episodic".into(),
            terminal_treatment: "reward-once".into(),
            action_constraints: "unbounded".into(),
        });
        d.critics.get_mut("m").unwrap().continuation = Some(ContinuationIdentity {
            identity: "recorded-moment-policy".into(),
            stochastic: false,
        });
        c.arrays.remove(VALUE_METADATA_KEY);
        attach_declaration(c, &d).unwrap();
        let mut writer = Recorder::create(
            dir.path().join("sparse"),
            m.clone(),
            StorageConfig::default(),
        )
        .unwrap();
        writer.append(&inspected.probes).unwrap();
        writer.append(c).unwrap();
        writer.finish().unwrap();
        let sparse = RunIndex::open(dir.path().join("sparse"), ReadLimits::default()).unwrap();
        let mut loader = rl_viewer::data::Loader::new(1024 * 1024, || {}).unwrap();
        let mut resolver = rl_viewer::resolver::ProbeResolver::new(1024 * 1024);
        let mut search =
            rl_viewer::policy::prepared::ModelSearch::new("online", 2, Some("m")).unwrap();
        let source = search_finish(&sparse, &mut search, &mut loader, &mut resolver).unwrap();
        let mut slice =
            ValueSlice::new(source.semantics(), &Observation::new(vec![0.; 4]), &[]).unwrap();
        slice.resolution = [2, 2];
        slice.conditioning = ActionConditioning::Fixed(Action::Continuous(vec![0.; 3]));
        let mut r = ValueRequest::new(
            source.clone(),
            ValueOutput::Utility,
            slice,
            ValuePin::Input {
                observation: Observation::new(vec![0.; 4]),
                action: Some(Action::Continuous(vec![0.; 3])),
            },
        )
        .unwrap();
        r.reference_value = Some(source);
        i.submit_value(r).unwrap();
        finish(&mut i);
        let frame = i.value_frame().unwrap();
        assert_eq!(frame.grid.values, vec![2., 2.]);
        assert_eq!(frame.grid.reference_error, None);
        assert_eq!(frame.grid.reference_values, vec![Some(2.), Some(2.)]);
        assert_eq!(
            frame.grid.reference_series.as_ref().unwrap().values,
            vec![2., 2.]
        );
        assert_eq!(frame.selected.as_ref().unwrap().reference_value, Some(2.));
    }

    #[test]
    fn discrete_greedy_and_masked_categories_use_actual_critic_values() {
        use rl::nn::critics::scalar::DiscreteQ;
        let dir = tempfile::tempdir().unwrap();
        let m = manifest(ActionSpace::Discrete { count: 3 });
        let mut writer =
            Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
        let mut q = DiscreteQ::new(4, 3, &[], &mut ChaCha8Rng::seed_from_u64(1)).unwrap();
        q.network.layers[0].weight = q.network.layers[0]
            .weight
            .clone()
            .map(|_| matrix(vec![0.; 12], 4, 3));
        q.network.layers[0].bias = q.network.layers[0]
            .bias
            .take()
            .map(|p| p.map(|_| vector(&[1., 99., 3.])));
        let mut b = SnapshotBuilder::new(
            "q",
            RecordContext::new("value", 0),
            4,
            m.action_space.clone(),
        )
        .unwrap();
        b.discrete("q", &q).unwrap();
        writer.append(&b.finish().unwrap()).unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut o = Observation::new(vec![0.; 4]);
        o.as_dense_mut()
            .expect("dense diagnostic input")
            .action_mask = Some(vec![true, false, true]);
        let mut slice = ValueSlice::new(EnvironmentSemantics::new(&m), &o, &[]).unwrap();
        slice.x = C::DiscreteAction;
        slice.y = None;
        slice.mode = SliceMode::OneDimensional;
        let request = ValueRequest::new(
            source(&run, 0, "q"),
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::State(o),
        )
        .unwrap();
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(request).unwrap();
        finish(&mut i);
        let frame = i.value_frame().unwrap();
        assert_eq!(frame.grid.values, vec![1., 3.]);
        assert_eq!(frame.grid.categories, vec![0, 2]);
        let point = frame.selected.as_ref().unwrap();
        assert_eq!(point.value, 3.);
        assert_eq!(point.action, Some(Action::Discrete(2)));
    }

    #[test]
    fn reppo_action_axis_evaluates_each_legal_action_under_default_conditioning() {
        use rl::{
            nn::{
                critics::categorical::CategoricalQ, critics::scalar::ActionValue, policy::Policy,
            },
            objectives::hl_gauss::HlGaussConfig,
        };

        let space = ActionSpace::Discrete { count: 4 };
        let m = manifest(space.clone());
        let mut rng = ChaCha8Rng::seed_from_u64(51);
        let actor = Policy::new(4, space.clone(), &[5], &mut rng).unwrap();
        let support = HlGaussConfig {
            bins: 11,
            min: -3.,
            max: 7.,
            ..Default::default()
        };
        let critic = CategoricalQ::new(4, space.clone(), support.bins, &[6, 4], &mut rng).unwrap();
        let anchor = Observation::dense(
            vec![0.3, -0.7, 0.2, 1.],
            Some(vec![false, true, false, true]),
        );
        let observations = [anchor.clone(), anchor.clone()];
        let encoded_actions = matrix(vec![0., 1., 0., 0., 0., 0., 0., 1.], 2, 4);
        let logits = critic
            .values(observations.as_slice().into(), encoded_actions)
            .unwrap();
        let expected = rl::nn::to_vec(support.decode(logits).unwrap()).unwrap();

        let mut snapshot =
            SnapshotBuilder::new("reppo", RecordContext::new("value", 7), 4, space).unwrap();
        snapshot.policy("policy", &actor).unwrap();
        snapshot
            .categorical("soft_q", &critic, &support, false)
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run");
        let mut writer = Recorder::create(&path, m.clone(), StorageConfig::default()).unwrap();
        writer.append(&snapshot.finish().unwrap()).unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
        let mut slice = ValueSlice::new(EnvironmentSemantics::new(&m), &anchor, &[]).unwrap();
        assert_eq!(slice.conditioning, ActionConditioning::Policy);
        slice.x = C::DiscreteAction;
        slice.y = None;
        slice.mode = SliceMode::OneDimensional;
        let request = ValueRequest::new(
            source(&run, 0, "soft_q"),
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::Input {
                observation: anchor,
                action: Some(Action::Discrete(3)),
            },
        )
        .unwrap();
        let mut worker = Interactive::new(|| {}).unwrap();
        worker.submit_value(request).unwrap();
        finish(&mut worker);
        let frame = worker.value_frame().unwrap();
        assert_eq!(frame.grid.categories, [1, 3]);
        assert_eq!(
            frame.grid.actions,
            [Some(Action::Discrete(1)), Some(Action::Discrete(3))]
        );
        assert_eq!(frame.grid.values.len(), expected.len());
        for (actual, expected) in frame.grid.values.iter().zip(expected) {
            assert!((actual - f64::from(expected)).abs() < 3e-6);
        }
    }

    #[test]
    fn stochastic_q_requires_full_action_and_unknown_values_do_not_disable_policy_overlay() {
        let dir = tempfile::tempdir().unwrap();
        let m = manifest(ActionSpace::Unbounded { dimensions: 3 });
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let actor = rl::nn::policy::Policy::new(4, m.action_space.clone(), &[], &mut rng).unwrap();
        let q = QValue::new(4, 3, 1, &[], &mut rng).unwrap();
        let mut b = SnapshotBuilder::new(
            "stochastic",
            RecordContext::new("value", 0),
            4,
            m.action_space.clone(),
        )
        .unwrap();
        b.policy("actor", &actor).unwrap();
        b.scalar("q", &q).unwrap();
        let mut writer =
            Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
        writer.append(&b.finish().unwrap()).unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let o = Observation::new(vec![0.; 4]);
        let mut slice = ValueSlice::new(EnvironmentSemantics::new(&m), &o, &[]).unwrap();
        slice.resolution = [2, 2];
        let mut r = ValueRequest::new(
            source(&run, 0, "q"),
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::State(o.clone()),
        )
        .unwrap();
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while i.value_pending() {
            i.poll_value().unwrap();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            i.value_error()
                .unwrap()
                .contains("stochastic Q expectation")
        );
        let action = Action::Continuous(vec![1., 2., 3.]);
        Arc::make_mut(&mut r.slice).conditioning = ActionConditioning::Fixed(action.clone());
        r.pin = ValuePin::Input {
            observation: o,
            action: Some(action),
        };
        r.reference_value = Some(source(&run, 0, "q"));
        r.reference_policy =
            Some(rl_viewer::policy::prepared::PolicySource::model(&run, 0, Some("actor")).unwrap());
        i.submit_value(r).unwrap();
        finish(&mut i);
        let frame = i.value_frame().unwrap();
        assert!(
            frame
                .grid
                .reference_error
                .as_ref()
                .unwrap()
                .contains("conventions unknown")
        );
        assert!(frame.grid.reference_policy_error.is_none());
        assert_eq!(frame.grid.reference_policy_overlay.len(), 4);
        assert!(frame.selected.as_ref().unwrap().value.is_finite());
    }

    fn sparse_discrete_request() -> (tempfile::TempDir, ValueRequest) {
        sparse_discrete_request_with_mask(None)
    }
    fn sparse_discrete_request_with_mask(
        mask: Option<Vec<bool>>,
    ) -> (tempfile::TempDir, ValueRequest) {
        use rl::nn::critics::scalar::DiscreteQ;
        let dir = tempfile::tempdir().unwrap();
        let m = manifest(ActionSpace::Discrete { count: 3 });
        let mut writer =
            Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for (revision, values, states) in [
            (0, vec![1., 2., 3.], vec![0.]),
            (1, vec![9., 0., 4.], vec![0., 0.5]),
        ] {
            let mut q = DiscreteQ::new(4, 3, &[], &mut rng).unwrap();
            q.network.layers[0].weight = q.network.layers[0]
                .weight
                .clone()
                .map(|_| matrix(vec![0.; 12], 4, 3));
            q.network.layers[0].bias = q.network.layers[0]
                .bias
                .take()
                .map(|p| p.map(|_| vector(&values)));
            let mut b = SnapshotBuilder::new(
                "q",
                RecordContext::new("value", revision),
                4,
                m.action_space.clone(),
            )
            .unwrap();
            b.discrete("q", &q).unwrap();
            let mut snapshot = b.finish().unwrap();
            attach_declaration(
                &mut snapshot,
                &ValueDeclaration {
                    version: 1,
                    run_id: "value".into(),
                    revision,
                    model_role: "online".into(),
                    conventions: Some(ValueConventions {
                        dynamics: "test".into(),
                        observation: "obs".into(),
                        action: "action".into(),
                        reward_units: "reward".into(),
                        reward_convention: m.reward_convention.clone(),
                        discount: m.discount,
                        horizon: "episodic".into(),
                        terminal_treatment: "reward-once".into(),
                        action_constraints: "legal".into(),
                    }),
                    critics: BTreeMap::from([(
                        "q".into(),
                        CriticDeclaration {
                            kind: ValueCriticKind::DiscreteQ,
                            learned_baseline: None,
                            actor: None,
                            continuation: Some(ContinuationIdentity {
                                identity: format!("greedy-{revision}"),
                                stochastic: false,
                            }),
                            twin: None,
                            utility: None,
                        },
                    )]),
                },
            )
            .unwrap();
            let model = ModelBundle::load(&snapshot, ModelLimits::default()).unwrap();
            let recorded = model
                .inspect_critics(&CriticInspectionRequest {
                    critics: vec!["q".into()],
                    observations: states
                        .into_iter()
                        .map(|x| Observation::dense(vec![x, 0., 0., 0.], mask.clone()))
                        .collect(),
                    fixed_actions: None,
                })
                .unwrap();
            writer.append(&recorded.probes).unwrap();
            writer.append(&recorded.critics["q"]).unwrap();
        }
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut loader = rl_viewer::data::Loader::new(1024 * 1024, || {}).unwrap();
        let mut resolver = rl_viewer::resolver::ProbeResolver::new(1024 * 1024);
        let mut search =
            rl_viewer::policy::prepared::ModelSearch::new("online", 0, Some("q")).unwrap();
        let current = search_finish(&run, &mut search, &mut loader, &mut resolver).unwrap();
        search.retarget("online", 1, Some("q")).unwrap();
        let reference = search_finish(&run, &mut search, &mut loader, &mut resolver).unwrap();
        let observation = current.recorded_input(0).unwrap().0;
        let mut slice = ValueSlice::new(current.semantics(), &observation, &[]).unwrap();
        slice.resolution = [2, 2];
        let mut r = ValueRequest::new(
            current,
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::State(observation),
        )
        .unwrap();
        r.reference_value = Some(reference);
        (dir, r)
    }
    #[test]
    fn sparse_discrete_fixed_action_is_the_stored_grid_action() {
        let (_dir, mut r) = sparse_discrete_request();
        Arc::make_mut(&mut r.slice).conditioning = ActionConditioning::Fixed(Action::Discrete(0));
        let observation = r.current.recorded_input(0).unwrap().0;
        r.pin = ValuePin::Input {
            observation,
            action: Some(Action::Discrete(0)),
        };
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r).unwrap();
        finish(&mut i);
        let frame = i.value_frame().unwrap();
        assert_eq!(frame.grid.values, vec![1.]);
        assert_eq!(frame.grid.actions, vec![Some(Action::Discrete(0))]);
        assert_eq!(frame.grid.reference_values, vec![Some(9.)]);
        assert_eq!(frame.selected.as_ref().unwrap().value, 1.);
    }
    #[test]
    fn sparse_discrete_same_input_uses_current_greedy_action_for_reference() {
        let (_dir, r) = sparse_discrete_request();
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r).unwrap();
        finish(&mut i);
        let frame = i.value_frame().unwrap();
        assert_eq!(frame.grid.values, vec![3.]);
        assert_eq!(frame.grid.actions, vec![Some(Action::Discrete(2))]);
        assert_eq!(frame.grid.reference_values, vec![Some(4.)]);
        assert_eq!(frame.selected.as_ref().unwrap().reference_value, Some(4.));
    }
    #[test]
    fn sparse_discrete_comparison_conditions_retain_actual_legal_actions() {
        for masked in [false, true] {
            for condition in [
                ComparisonCondition::SameInput,
                ComparisonCondition::ControllerValues,
            ] {
                let (_dir, mut r) =
                    sparse_discrete_request_with_mask(masked.then(|| vec![true, true, false]));
                r.condition = condition;
                let mut i = Interactive::new(|| {}).unwrap();
                i.submit_value(r).unwrap();
                finish(&mut i);
                let f = i.value_frame().unwrap();
                let current_action = if masked { 1 } else { 2 };
                let current = if masked { 2. } else { 3. };
                let same = condition == ComparisonCondition::SameInput;
                let reference_action = if same { current_action } else { 0 };
                let reference = if same {
                    if masked { 0. } else { 4. }
                } else {
                    9.
                };
                assert_eq!(f.grid.values, vec![current]);
                assert_eq!(f.grid.reference_values, vec![Some(reference)]);
                assert_eq!(f.grid.actions, vec![Some(Action::Discrete(current_action))]);
                assert_eq!(
                    f.grid.reference_actions,
                    vec![Some(Action::Discrete(reference_action))]
                );
                let independent = f.grid.reference_series.as_ref().unwrap();
                assert_eq!(independent.values, vec![9., 9.]);
                assert_eq!(independent.actions, vec![Some(Action::Discrete(0)); 2]);
                assert_eq!(independent.rows, vec![0, 1]);
                let p = f.selected.as_ref().unwrap();
                assert_eq!(p.value, current);
                assert_eq!(p.reference_value, Some(reference));
                assert_eq!(p.action, Some(Action::Discrete(current_action)));
                assert_eq!(p.reference_action, Some(Action::Discrete(reference_action)));
                assert_eq!(
                    f.grid.values[0] - f.grid.reference_values[0].unwrap(),
                    p.value - p.reference_value.unwrap()
                );
            }
        }
    }
    #[test]
    fn recorded_color_domain_follows_greedy_and_action_axis_views() {
        let (_dir, mut r) = sparse_discrete_request_with_mask(Some(vec![true, true, false]));
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let g = &i.value_frame().unwrap().grid;
        assert_eq!(g.values, vec![2.]);
        let range = g.color_domain.unwrap();
        assert!(range[0] < 2. && range[1] > 2. && range[1] - range[0] < 0.001);
        assert_eq!(g.reference_color_domain, g.color_domain);
        Arc::make_mut(&mut r.slice).y = Some(C::DiscreteAction);
        i.submit_value(r).unwrap();
        finish(&mut i);
        let range = i.value_frame().unwrap().grid.color_domain.unwrap();
        assert!((range[0] - 0.924).abs() < 1e-9 && (range[1] - 2.076).abs() < 1e-9);
    }
    #[test]
    fn sparse_discrete_reference_series_expands_legal_categories_at_its_own_states() {
        let (_dir, mut r) = sparse_discrete_request();
        Arc::make_mut(&mut r.slice).y = Some(C::DiscreteAction);
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r).unwrap();
        finish(&mut i);
        let g = &i.value_frame().unwrap().grid;
        assert_eq!(g.values, vec![1., 2., 3.]);
        assert_eq!(g.reference_values, vec![Some(9.), Some(0.), Some(4.)]);
        let series = g.reference_series.as_ref().unwrap();
        assert_eq!(
            series.positions,
            vec![
                [0., 0.],
                [0., 1.],
                [0., 2.],
                [0.5, 0.],
                [0.5, 1.],
                [0.5, 2.]
            ]
        );
        assert_eq!(series.values, vec![9., 0., 4., 9., 0., 4.]);
        assert_eq!(series.rows, vec![0, 0, 0, 1, 1, 1]);
    }
    #[test]
    fn recorded_twin_replacement_keys_every_constituent_immutable_record() {
        let (_base_dir, base) = fixture();
        let model =
            ModelBundle::load(&base.recording().load(0).unwrap(), ModelLimits::default()).unwrap();
        let recorded = model
            .inspect_critics(&CriticInspectionRequest {
                critics: vec!["q1".into(), "q2".into()],
                observations: vec![Observation::new(vec![0.; 4])],
                fixed_actions: Some(vec![Action::Continuous(vec![3., 2., 3.])]),
            })
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Recorder::create(
            dir.path().join("run"),
            base.recording().manifest().clone(),
            StorageConfig::default(),
        )
        .unwrap();
        writer.append(&recorded.probes).unwrap();
        writer.append(&recorded.critics["q1"]).unwrap();
        writer.append(&recorded.critics["q2"]).unwrap();
        let mut run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut loader = rl_viewer::data::Loader::new(1024 * 1024, || {}).unwrap();
        let mut resolver = rl_viewer::resolver::ProbeResolver::new(1024 * 1024);
        let mut search =
            rl_viewer::policy::prepared::ModelSearch::new("online", 3, Some("q1")).unwrap();
        let q1 = search_finish(&run, &mut search, &mut loader, &mut resolver).unwrap();
        search.retarget("online", 3, Some("q2")).unwrap();
        let q2 = search_finish(&run, &mut search, &mut loader, &mut resolver).unwrap();
        let current = q1.clone().recorded_twin(q2, "twin").unwrap();
        let mut slice =
            ValueSlice::new(current.semantics(), &Observation::new(vec![0.; 4]), &[]).unwrap();
        slice.resolution = [2, 2];
        slice.conditioning = ActionConditioning::Fixed(Action::Continuous(vec![3., 2., 3.]));
        let mut r = ValueRequest::new(
            current,
            ValueOutput::Minimum,
            slice,
            ValuePin::Input {
                observation: Observation::new(vec![0.; 4]),
                action: Some(Action::Continuous(vec![3., 2., 3.])),
            },
        )
        .unwrap();
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let first = i.value_frame().unwrap().grid.clone();
        assert_eq!(first.values, vec![323.]);
        let mut changed = recorded.critics["q2"].clone();
        let Record::Critic(CriticRecord {
            representation: CriticRepresentation::Scalar { values },
            ..
        }) = &changed.record
        else {
            panic!()
        };
        let values = values.clone();
        changed
            .arrays
            .insert(values, Array::f32(vec![1], vec![300.]).unwrap());
        writer.append(&changed).unwrap();
        run.refresh().unwrap();
        search.retarget("online", 3, Some("q2")).unwrap();
        let q2 = search_finish(&run, &mut search, &mut loader, &mut resolver).unwrap();
        r.current = q1.recorded_twin(q2, "twin").unwrap();
        i.submit_value(r).unwrap();
        finish(&mut i);
        assert!(
            !Arc::ptr_eq(&first, &i.value_frame().unwrap().grid),
            "replacing Q2 must invalidate the grid despite identical context/declaration"
        );
        assert_eq!(i.value_frame().unwrap().grid.values, vec![300.]);
        assert_eq!(
            i.value_frame().unwrap().selected.as_ref().unwrap().value,
            300.
        );
        writer.finish().unwrap();
    }

    #[test]
    fn twin_historical_reference_keeps_explicit_q1_q2_and_pair_statistics() {
        let (_dir, run) = fixture();
        let mut r = request(&run, 0);
        r.current = r.current.with_twin("twin").unwrap();
        r.reference_value = Some(source(&run, 1, "q1").with_twin("twin").unwrap());
        let mut i = Interactive::new(|| {}).unwrap();
        for (output, expected) in [
            (ValueOutput::Q1, 323.25),
            (ValueOutput::Q2, 325.25),
            (ValueOutput::Minimum, 323.25),
            (ValueOutput::Disagreement, 2.),
        ] {
            r.output = output;
            i.submit_value(r.clone()).unwrap();
            finish(&mut i);
            let p = i.value_frame().unwrap().selected.as_ref().unwrap();
            assert_eq!(p.reference_error, None);
            assert_eq!(p.reference_value, Some(expected));
            assert!(
                i.value_frame()
                    .unwrap()
                    .grid
                    .reference_values
                    .iter()
                    .all(Option::is_some)
            );
        }
        r.output = ValueOutput::Q2;
        r.condition = ComparisonCondition::ControllerValues;
        i.submit_value(r).unwrap();
        finish(&mut i);
        assert_eq!(
            i.value_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .reference_value,
            Some(329.25)
        );
    }

    #[test]
    fn many_small_visit_sources_use_global_marker_capacity_and_retire() {
        let (_dir, run) = fixture();
        let mut r = request(&run, 0);
        Arc::make_mut(&mut r.slice).resolution = [64, 64];
        r.visits = (0..16).map(|n| visit(&run, 2 + n % 2)).collect();
        r.reference_value = Some(source(&run, 1, "q1"));
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let held = i.value_frame().unwrap().grid.clone();
        assert_eq!(held.visits.as_ref().unwrap().markers.len(), 48);
        assert!(i.value_metrics().prepared_peak <= 4 * 1024 * 1024);
        i.retire_session();
        assert!(
            i.value_metrics().prepared_bytes > 0,
            "held grid remains charged"
        );
        drop(held);
        drop(r);
        let deadline = Instant::now() + Duration::from_secs(5);
        while i.value_metrics().prepared_bytes != 0 {
            i.housekeeping();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn multiple_sources_above_marker_cap_keep_all_counts_and_retire_held_grid() {
        let (_dir, run) = fixture();
        let n = 600;
        let mut c = run.recording().load(2).unwrap();
        c.arrays.insert(
            "o".into(),
            Array::f32(vec![n + 1, 4], vec![0.; (n + 1) * 4]).unwrap(),
        );
        c.arrays
            .insert("a".into(), Array::f32(vec![n, 3], vec![0.; n * 3]).unwrap());
        c.arrays
            .insert("r".into(), Array::f32(vec![n], vec![0.; n]).unwrap());
        let mut terminal = vec![0; n];
        terminal[n - 1] = 1;
        c.arrays
            .insert("t".into(), Array::u8(vec![n], terminal).unwrap());
        c.arrays
            .insert("x".into(), Array::u8(vec![n], vec![0; n]).unwrap());
        c.validate().unwrap();
        let mut resolver = rl_viewer::resolver::ProbeResolver::new(1024 * 1024);
        let mut r = request(&run, 0);
        Arc::make_mut(&mut r.slice).conditioning =
            ActionConditioning::Fixed(Action::Continuous(vec![0.; 3]));
        for index in [2, 3] {
            let identity = run.recording().load(index).unwrap();
            c.context = identity.context;
            c.record = identity.record;
            let key = run.key(index).unwrap();
            let chunk = resolver
                .lease_visitation(&run, &key, Arc::new(c.clone()))
                .unwrap();
            r.visits
                .push(rl_viewer::policy::prepared::VisitSource { key, chunk });
        }
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let held = i.value_frame().unwrap().grid.clone();
        let visits = held.visits.as_ref().unwrap();
        assert_eq!(visits.markers.len(), 1024);
        assert_eq!(visits.filtered_count, 1200);
        assert_eq!(visits.counts.iter().sum::<u32>(), 1200);
        assert!(i.value_metrics().prepared_peak <= 4 * 1024 * 1024);
        i.retire_session();
        assert!(
            i.value_metrics().prepared_bytes > 0,
            "held capped grid remains charged"
        );
        drop(held);
        drop(r);
        let deadline = Instant::now() + Duration::from_secs(5);
        while i.value_metrics().prepared_bytes != 0 {
            i.housekeeping();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn visit_population_above_global_row_admission_is_rejected_before_preparation() {
        let (_dir, run) = fixture();
        let mut c = run.recording().load(2).unwrap();
        let n = 4097;
        c.arrays.insert(
            "o".into(),
            Array::f32(vec![n + 1, 4], vec![0.; (n + 1) * 4]).unwrap(),
        );
        c.arrays
            .insert("a".into(), Array::f32(vec![n, 3], vec![0.; n * 3]).unwrap());
        c.arrays
            .insert("r".into(), Array::f32(vec![n], vec![0.; n]).unwrap());
        let mut terminal = vec![0; n];
        terminal[n - 1] = 1;
        c.arrays
            .insert("t".into(), Array::u8(vec![n], terminal).unwrap());
        c.arrays
            .insert("x".into(), Array::u8(vec![n], vec![0; n]).unwrap());
        c.validate().unwrap();
        let mut resolver = rl_viewer::resolver::ProbeResolver::new(1024 * 1024);
        let mut r = request(&run, 0);
        for index in [2, 3] {
            let identity = run.recording().load(index).unwrap();
            c.context = identity.context;
            c.record = identity.record;
            let key = run.key(index).unwrap();
            let chunk = resolver
                .lease_visitation(&run, &key, Arc::new(c.clone()))
                .unwrap();
            r.visits
                .push(rl_viewer::policy::prepared::VisitSource { key, chunk });
        }
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while i.value_pending() {
            i.poll_value().unwrap();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(i.value_error().is_some_and(|s| s.contains("8192")));
        assert!(i.value_frame().is_none());
        assert_eq!(i.value_metrics().grid_batches, 0);
    }

    #[test]
    fn twin_q1_compares_to_independent_scalar_reference_and_invalid_admission_drops_owners() {
        let (_dir, run) = fixture();
        let mut r = request(&run, 0);
        r.current = r.current.with_twin("twin").unwrap();
        r.output = ValueOutput::Q1;
        r.reference_value = Some(source(&run, 1, "q1"));
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let point = i.value_frame().unwrap().selected.as_ref().unwrap();
        assert_eq!(point.reference_value, Some(323.25));
        assert_eq!(point.reference_error, None);
        r.condition = ComparisonCondition::ControllerValues;
        i.submit_value(r.clone()).unwrap();
        finish(&mut i);
        let point = i.value_frame().unwrap().selected.as_ref().unwrap();
        assert_eq!(point.reference_value, Some(327.25));
        assert_eq!(
            point.reference_action,
            Some(Action::Continuous(vec![7., 2., 3.]))
        );
        assert_eq!(point.reference_error, None);
        Arc::make_mut(&mut r.slice).resolution = [256, 256];
        i.submit_value(r).unwrap();
        while i.value_pending() {
            i.poll_value().unwrap();
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(i.value_error().is_some());
        i.retire_session();
        let deadline = Instant::now() + Duration::from_secs(5);
        while i.retirement_status().0 != i.retirement_status().1 {
            i.housekeeping();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(i.value_metrics().prepared_bytes, 0);
    }
}
