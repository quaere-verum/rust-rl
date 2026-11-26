use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{schema::RunManifest, visualization::*},
};
use rl_viewer::policy::{slice::*, visitation::*};
fn manifest() -> RunManifest {
    RunManifest::new(
        "r",
        "e",
        "p",
        3,
        ActionSpace::Unbounded { dimensions: 2 },
        0.9,
    )
    .unwrap()
}
fn annotated() -> RunManifest {
    let mut m = manifest();
    m.environment_descriptor = Some(EnvironmentDescriptor {
        version: 1,
        observations: (0..3)
            .map(|i| ObservationField {
                id: format!("s{i}"),
                label: format!("S{i}"),
                unit: None,
                description: None,
                kind: if i == 1 {
                    ObservationType::Categorical {
                        categories: vec![
                            Category {
                                code: 4.,
                                label: "a".into(),
                            },
                            Category {
                                code: 8.,
                                label: "b".into(),
                            },
                        ],
                    }
                } else {
                    ObservationType::Numerical
                },
                range: Some([0., 10.]),
                coordinate: i != 1,
            })
            .collect(),
        actions: ActionDescriptor::Continuous {
            components: (0..2)
                .map(|_| ActionComponent {
                    label: "a".into(),
                    unit: None,
                    meaning: None,
                })
                .collect(),
            preferred_component: Some(1),
        },
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel {
                label: "rev".into(),
                unit: None,
            },
            decision_step: QuantityLabel {
                label: "step".into(),
                unit: None,
            },
            simulation: None,
        },
        reward: RewardDescriptor {
            label: "r".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: vec![],
        preferences: VisualizationPreferences {
            coordinates: vec!["s2".into(), "s0".into()],
            ..Default::default()
        },
    });
    m.validate().unwrap();
    m
}

#[test]
fn exported_slice_reconstructs_categories_from_the_recording_encoding() {
    let m = annotated();
    let e = EnvironmentSemantics::new(&m);
    let config = SliceConfig::new(e, &Observation::new(vec![1., 4., 3.]), &[]).unwrap();
    let mut spec = PolicySliceSpec::from_config("online", "actor", &config);
    assert_eq!(spec.to_config(e).unwrap(), config);
    spec.fixed[1] = 6.;
    assert!(
        spec.to_config(e).is_err(),
        "authored specs cannot invent categorical codes"
    );
    spec.fixed[1] = 4.;
    spec.x = 1;
    assert!(
        spec.to_config(e).is_err(),
        "authored specs cannot turn categories into numerical axes"
    );
}
#[test]
fn arbitrary_axes_use_exact_fixed_values_and_grid_coordinates_without_checkpoint_resets() {
    let m = manifest();
    let e = EnvironmentSemantics::new(&m);
    let anchor = Observation::new(vec![1., 2., 3.]);
    let mut c = SliceConfig::new(e, &anchor, &[]).unwrap();
    c.set_axes(e, 2, Some(0)).unwrap();
    c.ranges[2] = [-2., 2.];
    c.ranges[0] = [10., 20.];
    c.resolution = [3, 2];
    c.fixed[1] = 4.25;
    c.action = 1;
    assert_eq!(c.len().unwrap(), 6);
    assert_eq!(
        c.observation_at(0).unwrap().dense_values(),
        vec![10., 4.25, -2.]
    );
    assert_eq!(
        c.observation_at(5).unwrap().dense_values(),
        vec![20., 4.25, 2.]
    );
    assert!(c.observation_at(6).is_err());
    assert!(c.set_axes(e, 0, Some(0)).is_err());
    assert_eq!(c.x, 2);
    c.mode = SliceMode::OneDimensional;
    c.resolution[0] = 5;
    assert_eq!(c.len().unwrap(), 5);
    assert_eq!(
        c.observation_at(2).unwrap().dense_values(),
        vec![1., 4.25, 0.]
    );
    let stable = c.clone();
    c.validate(e).unwrap();
    assert_eq!(c, stable);
}
#[test]
fn metadata_preferences_ranges_and_categories_and_fallback_population_are_honest() {
    let m = annotated();
    let e = EnvironmentSemantics::new(&m);
    let mut c = SliceConfig::new(e, &Observation::new(vec![1., 4., 3.]), &[]).unwrap();
    assert_eq!((c.x, c.y, c.action), (2, Some(0), 1));
    assert_eq!(c.ranges[0], [0., 10.]);
    c.fixed[1] = 8.;
    c.validate(e).unwrap();
    c.fixed[1] = 6.;
    assert!(c.validate(e).is_err());
    assert!(c.set_axes(e, 1, None).is_err());
    let m = manifest();
    let e = EnvironmentSemantics::new(&m);
    let c = SliceConfig::new(
        e,
        &Observation::new(vec![1., 2., 3.]),
        &[
            Observation::new(vec![-3., 7., 9.]),
            Observation::new(vec![5., 8., 10.]),
        ],
    )
    .unwrap();
    assert_eq!(c.ranges[0], [-3., 5.]);
    let m = RunManifest::new(
        "r",
        "e",
        "p",
        1,
        ActionSpace::Unbounded { dimensions: 1 },
        0.9,
    )
    .unwrap();
    let c = SliceConfig::new(
        EnvironmentSemantics::new(&m),
        &Observation::new(vec![2.]),
        &[],
    )
    .unwrap();
    assert_eq!(c.mode, SliceMode::OneDimensional);
    assert_eq!(c.len().unwrap(), 256);
}
#[test]
fn invalid_sizes_and_nonfinite_values_fail_before_grid_allocation_and_conditioning_is_preserved() {
    let m = manifest();
    let e = EnvironmentSemantics::new(&m);
    let a = Observation::new(vec![1., 2., 3.]);
    let mut c = SliceConfig::new(e, &a, &[]).unwrap();
    c.resolution = [usize::MAX, usize::MAX];
    assert!(c.len().is_err());
    c.resolution = [64, 64];
    c.fixed[0] = f32::NAN;
    assert!(c.validate(e).is_err());
    c.fixed[0] = 1.;
    c.ranges[0] = [1., 1.];
    assert!(c.validate(e).is_err());
    c.ranges[0] = [0., 1.];
    c.action = 2;
    assert!(c.validate(e).is_err());
    let c = SliceConfig::new(e, &a, &[]).unwrap().with_history();
    c.validate(e).unwrap();
    assert!(c.observation_at(0).is_err());
    let mut m = manifest();
    m.action_space = ActionSpace::Discrete { count: 2 };
    let e = EnvironmentSemantics::new(&m);
    let a = Observation::dense(vec![1., 2., 3.], Some(vec![true, false]));
    let c = SliceConfig::new(e, &a, &[]).unwrap();
    c.validate(e).unwrap();
    assert!(c.observation_at(0).is_err());
    m.observation_dim = usize::MAX;
    assert!(SliceConfig::new(EnvironmentSemantics::new(&m), &a, &[]).is_err());
}
#[test]
fn unmasked_discrete_actions_allow_numeric_grids_without_relaxing_input_legality() {
    let mut m = manifest();
    m.action_space = ActionSpace::Discrete { count: 2 };
    let e = EnvironmentSemantics::new(&m);
    let anchor = Observation::new(vec![1., 2., 3.]);
    let mut config = SliceConfig::new(e, &anchor, &[]).unwrap();
    config.resolution = [2, 2];
    config.action = 1;
    config.validate(e).unwrap();
    let input = config.observation_at(0).unwrap();
    input.validate(3, &m.action_space, true).unwrap();
    assert_eq!(input.dense_mask(), None);
    assert_eq!(input.dense_values(), vec![0., 1., 3.]);
    config.action = 2;
    assert!(config.validate(e).is_err());
    assert!(config.observation_at(0).is_err());
    assert!(
        SliceConfig::new(e, &anchor, &[])
            .unwrap()
            .with_history()
            .observation_at(0)
            .is_err()
    );
    for mask in [vec![true, false], vec![true, true]] {
        let masked = Observation::dense(anchor.dense_values().to_vec(), Some(mask));
        let config = SliceConfig::new(e, &masked, &[]).unwrap();
        assert_eq!(config.grid_eligibility(), GridEligibility::Masked);
        assert!(config.observation_at(0).is_err());
    }
    for mask in [vec![false, false], vec![true]] {
        assert!(
            SliceConfig::new(
                e,
                &Observation::dense(anchor.dense_values().to_vec(), Some(mask)),
                &[]
            )
            .is_err()
        );
    }
    let mut categorical = annotated();
    for field in &mut categorical
        .environment_descriptor
        .as_mut()
        .unwrap()
        .observations
    {
        field.kind = ObservationType::Categorical {
            categories: vec![Category {
                code: 4.,
                label: "only".into(),
            }],
        };
    }
    let samples = SliceConfig::new(
        EnvironmentSemantics::new(&categorical),
        &Observation::new(vec![4.; 3]),
        &[],
    )
    .unwrap();
    assert_eq!(samples.mode, SliceMode::SamplesOnly);
    assert_eq!(samples.len().unwrap(), 0);
    assert!(samples.observation_at(0).is_err());
    assert_eq!(
        samples.visit_bin(&Observation::new(vec![4.; 3])).unwrap(),
        None
    );
}
#[test]
fn visits_share_exact_categorical_filter_and_local_bins_with_bounded_deterministic_markers() {
    let m = annotated();
    let e = EnvironmentSemantics::new(&m);
    let mut c = SliceConfig::new(e, &Observation::new(vec![1., 4., 3.]), &[]).unwrap();
    c.resolution = [2, 2];
    c.tolerance = 0.25;
    let rows = vec![
        Observation::new(vec![1., 4., 3.]),
        Observation::new(vec![1., 8., 3.]),
        Observation::new(vec![9., 4., 9.]),
        Observation::new(vec![1., 4., 3.]),
    ];
    let evidence = VisitationEvidence {
        source: VisitationSource::EvaluationTrajectory,
        revision: 7,
        observations: &rows,
    };
    let v = prepare_visitation(e, &c, Some(evidence), 1)
        .unwrap()
        .unwrap();
    assert_eq!(v.filtered_count, 3);
    assert_eq!(v.counts, vec![2, 0, 0, 1]);
    assert_eq!(v.markers.len(), 1);
    assert_eq!(v.revision, 7);
    assert_eq!(
        v,
        prepare_visitation(e, &c, Some(evidence), 1)
            .unwrap()
            .unwrap()
    );
    assert!(prepare_visitation(e, &c, None, 10).unwrap().is_none());
    let m = manifest();
    let e = EnvironmentSemantics::new(&m);
    let mut c = SliceConfig::new(e, &Observation::new(vec![1., 2., 3.]), &[]).unwrap();
    c.set_axes(e, 0, Some(2)).unwrap();
    c.tolerance = 0.25;
    let rows = [
        Observation::new(vec![1., 2.25, 3.]),
        Observation::new(vec![1., 2.2501, 3.]),
    ];
    assert_eq!(
        prepare_visitation(
            e,
            &c,
            Some(VisitationEvidence {
                source: VisitationSource::PolicyVisited,
                revision: 9,
                observations: &rows
            }),
            10
        )
        .unwrap()
        .unwrap()
        .filtered_count,
        1
    );
}

#[test]
fn visit_bins_and_selected_counts_share_edges_and_empty_evidence_has_no_card() {
    let m = manifest();
    let e = EnvironmentSemantics::new(&m);
    let mut c = SliceConfig::new(e, &Observation::new(vec![1., 2., 3.]), &[]).unwrap();
    c.ranges[0] = [0., 10.];
    c.ranges[1] = [0., 10.];
    c.resolution = [2, 2];
    for (v, bin) in [
        (vec![0., 0., 3.], Some(0)),
        (vec![5., 5., 3.], Some(3)),
        (vec![10., 10., 3.], Some(3)),
        (vec![10.01, 0., 3.], None),
        (vec![1., 1., 4.], None),
    ] {
        assert_eq!(c.visit_bin(&Observation::new(v)).unwrap(), bin);
    }
    assert!(
        prepare_visitation(
            e,
            &c,
            Some(VisitationEvidence {
                source: VisitationSource::EvaluationTrajectory,
                revision: 2,
                observations: &[]
            }),
            10
        )
        .unwrap()
        .is_none()
    );
}
#[test]
fn high_dimensions_are_admitted_by_bytes_and_populations_are_bounded_before_scanning() {
    let m = RunManifest::new(
        "r",
        "e",
        "p",
        16_000,
        ActionSpace::Unbounded { dimensions: 1 },
        0.9,
    )
    .unwrap();
    let c = SliceConfig::new(
        EnvironmentSemantics::new(&m),
        &Observation::new(vec![0.; 16_000]),
        &[],
    )
    .unwrap();
    assert!(c.owned_bytes() <= MAX_SLICE_STATE_BYTES);
    assert_eq!(c.observation_at(0).unwrap().dense_values().len(), 16_000);
    let m = manifest();
    let e = EnvironmentSemantics::new(&m);
    let a = Observation::new(vec![1., 2., 3.]);
    let rows = vec![a.clone(); MAX_POPULATION_ROWS + 1];
    assert!(SliceConfig::new(e, &a, &rows).is_err());
}

#[test]
fn entering_two_dimensions_normalizes_both_inactive_y_resolution_boundaries() {
    let m = manifest();
    let e = EnvironmentSemantics::new(&m);
    for inactive_y in [1, usize::MAX] {
        let mut config = SliceConfig::new(e, &Observation::new(vec![1., 2., 3.]), &[]).unwrap();
        config.set_axes(e, 0, None).unwrap();
        config.resolution = [256, inactive_y];
        config.validate(e).unwrap();
        config.set_axes(e, 0, Some(1)).unwrap();
        config.validate(e).unwrap();
        assert!(config.len().unwrap() <= 256 * 256);
        assert!(config.observation_at(0).is_ok());
        let before = config.clone();
        assert!(config.set_axes(e, 1, Some(1)).is_err());
        assert_eq!(config, before);
    }
}
