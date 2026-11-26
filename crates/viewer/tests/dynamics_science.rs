#[path = "support/dynamics.rs"]
mod support;
use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, visualization::*},
};
use rl_viewer::dynamics::{
    ActionAxis, ActionValue, Axes, EpisodeStatus, Grid, Limits, occupancy, prepare_episode,
};

#[test]
fn histogram_counts_decision_states_and_normalizes_only_in_grid_visits() {
    let (_dir, m, c) = support::saved(support::bounded(1));
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    let grid = Grid::new([[0., 2.], [0., 2.]], [2, 2], &Limits::default()).unwrap();
    let o = occupancy(&[e], Axes { x: 0, y: 1 }, &grid).unwrap();
    assert_eq!(o.counts, [1, 1, 0, 2]);
    assert_eq!(o.mass, [0.25, 0.25, 0., 0.5]);
    assert_eq!(o.included, 4);
    assert_eq!(o.excluded, 0);
    assert_eq!(o.log_mass(2), None);
    assert_eq!(o.log_mass(3), Some(0.5_f64.ln()));
}

#[test]
fn fixed_domain_outliers_are_excluded_without_clamping_or_changing_grid() {
    let (_dir, m, c) = support::saved(support::bounded(1));
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    let grid = Grid::new([[0., 1.], [0., 1.]], [2, 2], &Limits::default()).unwrap();
    let o = occupancy(&[e], Axes { x: 0, y: 1 }, &grid).unwrap();
    assert_eq!(o.counts, [1, 1, 0, 1]);
    assert_eq!(o.excluded, 1);
    assert_eq!(o.mass, [1. / 3., 1. / 3., 0., 1. / 3.]);
    assert_eq!(grid.bin([0.5, 0.5]), Some(3));
    assert_eq!(grid.bin([1., 1.]), Some(3));
    assert_eq!(grid.bin([f64::NAN, 0.]), None);
}

#[test]
fn exact_time_cursor_has_t_plus_one_states_and_no_terminal_action_or_reward() {
    let (_dir, m, c) = support::saved(support::bounded(1));
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    assert_eq!(e.transitions(), 4);
    assert_eq!(e.project(0, Axes { x: 0, y: 1 }).unwrap(), [0., 0.]);
    assert_eq!(e.project(4, Axes { x: 0, y: 1 }).unwrap(), [0., 2.]);
    let step = e.step(2).unwrap();
    assert_eq!(step.observation, [1., 1., 12.]);
    assert_eq!(step.reward, Some(3.));
    assert_eq!(step.simulation_time, Some(0.3));
    assert_eq!(
        e.action_value(2, ActionAxis::Continuous(0)).unwrap(),
        Some(ActionValue::Continuous(2.))
    );
    assert_eq!(e.step(4).unwrap().reward, None);
    assert_eq!(e.action_value(4, ActionAxis::Continuous(0)).unwrap(), None);
    assert_eq!(
        e.project_state_action(4, Axes { x: 0, y: 1 }, ActionAxis::Continuous(0))
            .unwrap(),
        None
    );
    assert!(e.step(5).is_err());
    assert_eq!(e.realized_return(), 6.);
    assert_eq!(e.status(), EpisodeStatus::Terminated);
    assert_eq!(e.context().identity.learner_update_revision, 17);
    assert_eq!(e.context().identity.model_role, "target");
    assert_eq!(
        e.record().evaluation.continuation_policy,
        "policy at revision 17"
    );
}

#[test]
fn every_continuous_component_drives_the_same_action_and_state_action_projection() {
    for space in [
        support::bounded(3),
        ActionSpace::Unbounded { dimensions: 3 },
    ] {
        let (_dir, m, c) = support::saved(space);
        let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
        assert_eq!(
            e.action_axes(),
            [
                ActionAxis::Continuous(0),
                ActionAxis::Continuous(1),
                ActionAxis::Continuous(2)
            ]
        );
        assert_eq!(e.default_action_axis(), ActionAxis::Continuous(2));
        assert_eq!(
            e.action_label(ActionAxis::Continuous(1)).unwrap(),
            "Motor 1"
        );
        assert_eq!(e.action_unit(ActionAxis::Continuous(1)).unwrap(), Some("N"));
        for (j, want) in [1., 11., 21.].into_iter().enumerate() {
            assert_eq!(
                e.action_value(1, ActionAxis::Continuous(j)).unwrap(),
                Some(ActionValue::Continuous(want))
            );
            assert_eq!(
                e.project_state_action(1, Axes { x: 0, y: 1 }, ActionAxis::Continuous(j))
                    .unwrap(),
                Some([1., 0., want])
            );
        }
        assert_eq!(e.step(1).unwrap().reward, Some(-2.));
        assert_eq!(e.project_state3(4, [0, 1, 2]).unwrap(), [0., 2., 14.]);
        assert!(e.project_state3(0, [0, 1, 1]).is_err());
        assert!(e.action_value(0, ActionAxis::Continuous(3)).is_err());
    }
}

#[test]
fn discrete_ids_are_categorical_and_cannot_be_used_as_a_continuous_3d_coordinate() {
    let (_dir, m, c) = support::saved(ActionSpace::Discrete { count: 3 });
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    assert_eq!(e.action_axes(), [ActionAxis::Discrete]);
    assert_eq!(e.action_label(ActionAxis::Discrete).unwrap(), "Action ID");
    assert_eq!(
        e.action_value(1, ActionAxis::Discrete).unwrap(),
        Some(ActionValue::Discrete(2))
    );
    assert_eq!(e.discrete_label(2), Some("choice 2"));
    assert!(e.action_value(1, ActionAxis::Continuous(0)).is_err());
    assert!(
        e.project_state_action(1, Axes { x: 0, y: 1 }, ActionAxis::Discrete)
            .is_err()
    );
}

#[test]
fn semantic_numeric_coordinates_and_preferences_control_projection() {
    let (_dir, mut m, c) = support::saved(support::bounded(1));
    m.environment_descriptor.as_mut().unwrap().observations[2].kind =
        ObservationType::Categorical {
            categories: vec![Category {
                code: 10.,
                label: "phase".into(),
            }],
        };
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    assert_eq!(e.numeric_coordinates(), [0, 1]);
    assert_eq!(e.default_axes(), Some(Axes { x: 1, y: 0 }));
    assert_eq!(e.coordinates()[0].label, "Position 0");
    assert_eq!(e.coordinates()[0].unit.as_deref(), Some("m"));
    assert!(e.project(0, Axes { x: 0, y: 2 }).is_err());
    assert!(e.project(0, Axes { x: 0, y: 0 }).is_err());
    assert!(e.project(0, Axes { x: 0, y: 3 }).is_err());
    m.environment_descriptor = None;
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    assert_eq!(e.coordinates()[1].label, "s[1]");
    assert_eq!(e.default_action_axis(), ActionAxis::Continuous(0));
}

#[test]
fn status_comes_from_flags_and_does_not_confuse_prefix_or_batch_return_metadata() {
    let (_dir, m, mut c) = support::saved(support::bounded(1));
    if let Record::Trajectory(r) = &mut c.record {
        r.evaluation.complete_returns = false;
    }
    assert_eq!(
        prepare_episode(&c, &m, &Limits::default())
            .unwrap()
            .status(),
        EpisodeStatus::Terminated
    );
    c.arrays
        .insert("terminated".into(), Array::u8(vec![4], vec![0; 4]).unwrap());
    c.arrays.insert(
        "truncated".into(),
        Array::u8(vec![4], vec![0, 0, 0, 1]).unwrap(),
    );
    assert_eq!(
        prepare_episode(&c, &m, &Limits::default())
            .unwrap()
            .status(),
        EpisodeStatus::Truncated
    );
    c.arrays
        .insert("truncated".into(), Array::u8(vec![4], vec![0; 4]).unwrap());
    assert_eq!(
        prepare_episode(&c, &m, &Limits::default())
            .unwrap()
            .status(),
        EpisodeStatus::Prefix
    );
}

#[test]
fn complete_finite_objective_keeps_a_coincident_environment_truncation_as_a_raw_flag() {
    use rl::{
        core::env::TimeLimit,
        diagnostics::evaluation::{EvaluationConfig, EvaluationHorizon, evaluate_with_horizon},
        envs::lqg::{Lqg, LqgConfig},
        train::trainer::RandomActor,
    };
    let config = LqgConfig::scalar(3, 1., 0., 1., 1., 1., 1., 0., 2.).unwrap();
    let actor = RandomActor {
        space: config.action_space.clone(),
    };
    let evaluation = evaluate_with_horizon(
        |seed| TimeLimit::new(Lqg::new(config.clone(), seed)?, 2),
        &actor,
        &EvaluationConfig {
            seeds: vec![1],
            discount: 0.5,
            max_steps: 2,
            trajectories: 1,
            ..Default::default()
        },
        EvaluationHorizon::FiniteDecisions(2),
    )
    .unwrap();
    let chunk = evaluation
        .trajectory_record(0, RecordContext::new("eval", 0), "episode", "lqg")
        .unwrap();
    let manifest =
        RunManifest::new("eval", "test", "fixture", 2, config.action_space, 0.5).unwrap();

    let episode = prepare_episode(&chunk, &manifest, &Limits::default()).unwrap();
    assert_eq!(episode.status(), EpisodeStatus::FiniteHorizon);
    assert_eq!(episode.step(0).unwrap().return_to_go, Some(-1.5));
    let last = episode.step(1).unwrap();
    assert_eq!(last.terminated, Some(false));
    assert_eq!(last.truncated, Some(true));
    assert_eq!(last.return_to_go, Some(-1.));
}

#[test]
fn preparation_rejects_shape_space_identity_and_nonfinite_mismatches() {
    let (_dir, m, c) = support::saved(support::bounded(3));
    for (key, array) in [
        ("obs", Array::f64(vec![5, 2], vec![0.; 10]).unwrap()),
        ("actions", Array::f64(vec![4, 1], vec![0.; 4]).unwrap()),
        ("actions", Array::f64(vec![4, 3], vec![101.; 12]).unwrap()),
        ("rewards", Array::f64(vec![3], vec![0.; 3]).unwrap()),
        (
            "times",
            Array::f64(vec![5], vec![0., 1., 0., 3., 4.]).unwrap(),
        ),
        ("terminated", Array::u8(vec![4], vec![1, 0, 0, 1]).unwrap()),
        ("obs", Array::f64(vec![5, 3], vec![f64::NAN; 15]).unwrap()),
    ] {
        let mut bad = c.clone();
        bad.arrays.insert(key.into(), array);
        assert!(
            prepare_episode(&bad, &m, &Limits::default()).is_err(),
            "{key}"
        );
    }
    let mut wrong = c.clone();
    wrong.context.identity.run_id = "another run".into();
    assert!(prepare_episode(&wrong, &m, &Limits::default()).is_err());
    let (_dir, m, mut c) = support::saved(ActionSpace::Discrete { count: 3 });
    c.arrays.insert(
        "actions".into(),
        Array::u64(vec![4], vec![0, 3, 1, 0]).unwrap(),
    );
    assert!(prepare_episode(&c, &m, &Limits::default()).is_err());
}

#[test]
fn one_transition_is_valid_empty_trajectory_is_rejected_without_panicking() {
    let (_dir, m, mut c) = support::saved(support::bounded(1));
    c.arrays.insert(
        "obs".into(),
        Array::f64(vec![2, 3], vec![0., 0., 0., 2., 2., 2.]).unwrap(),
    );
    c.arrays
        .insert("actions".into(), Array::f64(vec![1, 1], vec![7.]).unwrap());
    c.arrays
        .insert("rewards".into(), Array::f64(vec![1], vec![8.]).unwrap());
    c.arrays
        .insert("terminated".into(), Array::u8(vec![1], vec![1]).unwrap());
    c.arrays
        .insert("truncated".into(), Array::u8(vec![1], vec![0]).unwrap());
    c.arrays
        .insert("times".into(), Array::f64(vec![2], vec![0., 1.]).unwrap());
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    assert_eq!(e.transitions(), 1);
    assert_eq!(e.step(1).unwrap().observation, [2., 2., 2.]);
    c.arrays
        .insert("rewards".into(), Array::f64(vec![0], vec![]).unwrap());
    assert!(prepare_episode(&c, &m, &Limits::default()).is_err());
}

#[test]
fn byte_budget_rejects_full_episode_rather_than_sampling_exact_cursor_data() {
    let (_dir, m, c) = support::saved(support::bounded(3));
    let e = prepare_episode(&c, &m, &Limits::default()).unwrap();
    let limits = Limits {
        max_episode_bytes: e.retained_bytes() - 1,
        ..Limits::default()
    };
    assert!(prepare_episode(&c, &m, &limits).is_err());
    let limits = Limits {
        max_episode_bytes: e.retained_bytes(),
        ..Limits::default()
    };
    let admitted = prepare_episode(&c, &m, &limits).unwrap();
    assert_eq!(admitted.step(3).unwrap().observation, [2., 2., 13.]);
}

#[test]
fn unrelated_manifest_metadata_does_not_change_retained_charge_or_admission() {
    let (_dir, mut m, c) = support::saved(support::bounded(3));
    let retained_bytes = prepare_episode(&c, &m, &Limits::default())
        .unwrap()
        .retained_bytes();
    m.metadata
        .insert("unrelated run notes".into(), "x".repeat(700 * 1024));
    let limits = Limits {
        max_episode_bytes: retained_bytes,
        ..Limits::default()
    };
    let admitted = prepare_episode(&c, &m, &limits).unwrap();
    assert_eq!(admitted.retained_bytes(), retained_bytes);
    assert_eq!(admitted.step(3).unwrap().observation, [2., 2., 13.]);
}

#[test]
fn near_limit_exact_episode_is_admitted_with_large_unretained_run_metadata() {
    let (_dir, mut m, mut c) = support::saved(support::bounded(1));
    let transitions = 750_000;
    if let Record::Trajectory(record) = &mut c.record {
        record.simulation_times = None;
    }
    c.arrays.remove("times");
    c.arrays.insert(
        "obs".into(),
        Array::f64(vec![transitions + 1, 3], vec![1.; (transitions + 1) * 3]).unwrap(),
    );
    c.arrays.insert(
        "actions".into(),
        Array::f64(vec![transitions, 1], vec![0.; transitions]).unwrap(),
    );
    c.arrays.insert(
        "rewards".into(),
        Array::f64(vec![transitions], vec![0.; transitions]).unwrap(),
    );
    let mut terminated = vec![0; transitions];
    terminated[transitions - 1] = 1;
    c.arrays.insert(
        "terminated".into(),
        Array::u8(vec![transitions], terminated).unwrap(),
    );
    c.arrays.insert(
        "truncated".into(),
        Array::u8(vec![transitions], vec![0; transitions]).unwrap(),
    );
    let retained_bytes = prepare_episode(&c, &m, &Limits::default())
        .unwrap()
        .retained_bytes();
    m.metadata
        .insert("unrelated run notes".into(), "x".repeat(700 * 1024));
    let admitted = prepare_episode(&c, &m, &Limits::default()).unwrap();
    assert_eq!(admitted.retained_bytes(), retained_bytes);
    assert_eq!(admitted.transitions(), 750_000);
    assert_eq!(admitted.step(750_000).unwrap().observation, [1., 1., 1.]);
    assert_eq!(
        admitted
            .action_value(750_000, ActionAxis::Continuous(0))
            .unwrap(),
        None
    );
}

#[test]
fn individual_blank_legacy_labels_use_state_fallback_and_keep_meaningful_labels() {
    let (_dir, mut m, c) = support::saved(support::bounded(1));
    m.environment_descriptor = None;
    m.observation_labels = vec!["".into(), "Known position".into(), " \t ".into()];
    let episode = prepare_episode(&c, &m, &Limits::default()).unwrap();
    let labels: Vec<_> = episode
        .coordinates()
        .iter()
        .map(|c| c.label.as_str())
        .collect();
    assert_eq!(labels, ["s[0]", "Known position", "s[2]"]);
}

#[test]
fn an_exceeded_base_budget_is_rejected_before_declared_label_count_arithmetic() {
    let (_dir, mut m, c) = support::saved(ActionSpace::Discrete { count: 3 });
    m.environment_descriptor = None;
    m.action_space = ActionSpace::Discrete { count: usize::MAX };
    let limits = Limits {
        max_episode_bytes: 1,
        ..Limits::default()
    };
    let error = prepare_episode(&c, &m, &limits).unwrap_err().to_string();
    assert!(error.contains("exceeds byte budget"), "{error}");
}

#[test]
fn a_billion_declared_discrete_choices_are_rejected_with_only_tiny_recorded_arrays() {
    let (_dir, mut m, c) = support::saved(ActionSpace::Discrete { count: 3 });
    m.environment_descriptor = None;
    m.action_space = ActionSpace::Discrete {
        count: 1_000_000_000,
    };
    let error = prepare_episode(&c, &m, &Limits::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds byte budget"), "{error}");
}

#[test]
fn grid_rejects_invalid_bounds_zero_cells_overflow_and_excessive_allocation() {
    let l = Limits::default();
    for (bounds, cells) in [
        ([[0., 0.], [0., 1.]], [2, 2]),
        ([[0., 1.], [0., 1.]], [0, 2]),
        ([[0., f64::INFINITY], [0., 1.]], [2, 2]),
        ([[0., 1.], [0., 1.]], [usize::MAX, 2]),
        ([[0., 1.], [0., 1.]], [100_000, 100_000]),
    ] {
        assert!(Grid::new(bounds, cells, &l).is_err());
    }
    let grid = Grid::new([[0., 1.], [0., 1.]], [2, 2], &l).unwrap();
    let o = occupancy(&[], Axes { x: 0, y: 1 }, &grid).unwrap();
    assert_eq!(o.mass, [0.; 4]);
    assert_eq!(o.included, 0);
}
