use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{schema::*, visualization::*},
};
use rl_viewer::policy::numerics::SelectedDistribution;
use rl_viewer::policy::prepared::*;
#[test]
fn wide_observation_and_action_widths_reduce_batches_before_allocation() {
    assert_eq!(batch_capacity(2, 1, false).unwrap(), 1024);
    assert!(batch_capacity(4096, 1, false).unwrap() < 1024);
    assert!(batch_capacity(2, 4096, true).unwrap() < 1024);
    assert!(batch_capacity(usize::MAX, 1, false).is_err());
}
#[test]
fn cross_run_unknown_encoding_is_not_certified_by_equal_width() {
    let a = RunManifest::new(
        "a",
        "env",
        "alg",
        1,
        ActionSpace::Unbounded { dimensions: 1 },
        0.9,
    )
    .unwrap();
    let b = RunManifest::new(
        "b",
        "env",
        "alg",
        1,
        ActionSpace::Unbounded { dimensions: 1 },
        0.9,
    )
    .unwrap();
    assert!(!compatible_encoding(
        false,
        EnvironmentSemantics::new(&a),
        EnvironmentSemantics::new(&b)
    ));
    assert!(compatible_encoding(
        true,
        EnvironmentSemantics::new(&a),
        EnvironmentSemantics::new(&b)
    ));
}
#[test]
fn very_narrow_gaussian_uses_a_cdf_on_the_stable_explicit_domain() {
    let distribution = SelectedDistribution::Gaussian(rl::diagnostics::policy::GaussianMarginal {
        mean: 0.12345,
        log_std: -30.,
        bounds: None,
    });
    let plot = distribution_plot(&distribution, Some([-1., 1.]))
        .unwrap()
        .unwrap();
    assert!(plot.is_cdf);
    assert_eq!(plot.domain, [-1., 1.]);
    assert_eq!(plot.samples.first().unwrap()[1], 0.);
    assert_eq!(plot.samples.last().unwrap()[1], 1.);
    let _ = Observation::new(vec![0.]);
}
#[test]
fn current_and_reference_charts_always_use_the_same_quantity() {
    let broad = SelectedDistribution::Gaussian(rl::diagnostics::policy::GaussianMarginal {
        mean: 0.,
        log_std: 0.,
        bounds: None,
    });
    let narrow = SelectedDistribution::Gaussian(rl::diagnostics::policy::GaussianMarginal {
        mean: 0.12345,
        log_std: -30.,
        bounds: None,
    });
    let (a, b) = distribution_plots(&broad, Some(&narrow), Some([-4., 4.])).unwrap();
    assert!(a.as_ref().unwrap().is_cdf);
    assert!(b.as_ref().unwrap().is_cdf);
    assert_eq!(a.unwrap().domain, b.unwrap().domain);
}

use rl::diagnostics::{probes::ProbeSet, storage::*};
use rl_viewer::{
    data::{RunIndex, SelectedInput, Selection},
    interactive::Interactive,
    policy::slice::SliceConfig,
    resolver::{ProbeResolver, ResolvedRecord},
};
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};
fn descriptor(m: &mut RunManifest) {
    m.environment_descriptor = Some(EnvironmentDescriptor {
        version: 1,
        observations: vec![ObservationField {
            id: "position".into(),
            label: "Position".into(),
            unit: Some("m".into()),
            description: None,
            kind: ObservationType::Numerical,
            range: Some([-1., 1.]),
            coordinate: true,
        }],
        actions: match &m.action_space {
            ActionSpace::Discrete { count } => ActionDescriptor::Discrete {
                choices: (0..*count)
                    .map(|i| DiscreteChoice {
                        label: format!("a{i}"),
                        meaning: None,
                    })
                    .collect(),
            },
            _ => ActionDescriptor::Continuous {
                components: vec![ActionComponent {
                    label: "effort".into(),
                    unit: None,
                    meaning: None,
                }],
                preferred_component: None,
            },
        },
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel {
                label: "revision".into(),
                unit: None,
            },
            decision_step: QuantityLabel {
                label: "step".into(),
                unit: None,
            },
            simulation: None,
        },
        reward: RewardDescriptor {
            label: "reward".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: vec![],
        preferences: Default::default(),
    });
}
fn recorded(
    id: &str,
    count: usize,
    history: Option<f32>,
    descriptor_enabled: bool,
) -> (tempfile::TempDir, RunIndex, ProbeResolver, ResolvedRecord) {
    recorded_modified(id, count, history, descriptor_enabled, |_| {})
}
fn recorded_modified(
    id: &str,
    count: usize,
    history: Option<f32>,
    descriptor_enabled: bool,
    modify: fn(&mut Chunk),
) -> (tempfile::TempDir, RunIndex, ProbeResolver, ResolvedRecord) {
    let dir = tempfile::tempdir().unwrap();
    let space = if count == 0 {
        ActionSpace::Unbounded { dimensions: 1 }
    } else {
        ActionSpace::Discrete { count }
    };
    let mut m = RunManifest::new(id, "ordinary", "unknown", 1, space.clone(), 0.9).unwrap();
    if descriptor_enabled {
        descriptor(&mut m);
    }
    let mut writer = Recorder::create(dir.path().join("run"), m, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new(id, 5);
    context.probe_set = Some("real-population".into());
    let inputs = vec![
        Observation::dense(vec![-0.5], (count > 0).then(|| vec![true; count])),
        Observation::dense(vec![0.5], (count > 0).then(|| vec![true; count])),
    ];
    let mut probes = ProbeSet::new(
        "real-population",
        ProbePopulation::FixedReference,
        "real controlled inputs",
        inputs,
        None,
        space,
    )
    .unwrap();
    if let Some(v) = history {
        probes = probes
            .with_history("state", Array::f32(vec![2, 1], vec![v, v]).unwrap())
            .unwrap();
    }
    let mut probes = probes.record(context.clone()).unwrap();
    modify(&mut probes);
    let Record::ProbeSet(probe_record) = &probes.record else {
        panic!()
    };
    let probe_id = probe_record.id.clone();
    writer.append(&probes).unwrap();
    let (parameters, arrays) = if count == 0 {
        (
            PolicyParameters::Deterministic,
            BTreeMap::from([(
                "actions".into(),
                Array::f32(vec![2, 1], vec![-0.25, 0.75]).unwrap(),
            )]),
        )
    } else {
        (
            PolicyParameters::Categorical {
                probabilities: "p".into(),
            },
            BTreeMap::from([
                ("actions".into(), Array::u64(vec![2], vec![0, 0]).unwrap()),
                (
                    "p".into(),
                    Array::f64(vec![2, count], vec![1. / count as f64; 2 * count]).unwrap(),
                ),
            ]),
        )
    };
    let mut policy = Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters,
            stochastic: false,
            masks: None,
            entropy: None,
            entropy_method: None,
        }),
        arrays,
    };
    modify(&mut policy);
    writer.append(&policy).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let selection = Selection {
        revision: 5,
        input: SelectedInput::ProbeRow {
            set: probe_id,
            row: 0,
            sequence: 0,
        },
        ..Default::default()
    };
    let mut resolver = ProbeResolver::new(8 * 1024 * 1024);
    let frame = resolver
        .resolve_policy(&run, &selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap()
        .record
        .unwrap();
    (dir, run, resolver, frame)
}
fn recorded_request(
    run: &RunIndex,
    frame: &ResolvedRecord,
    reference: Option<PolicySource>,
    history: bool,
) -> PolicyRequest {
    let checked = rl_viewer::policy_change::admission::checked_batch(
        &frame.chunk,
        &frame.probes,
        &run.recording().manifest().action_space,
        4 * 1024 * 1024,
    )
    .unwrap();
    let input = recorded_input(&checked, 0).unwrap();
    let mut slice = SliceConfig::new(
        EnvironmentSemantics::new(run.recording().manifest()),
        &input,
        &[],
    )
    .unwrap();
    if history {
        slice = slice.with_history();
    }
    PolicyRequest::new(
        PolicySource::recorded(run, frame, "recorded controller").unwrap(),
        reference,
        9,
        slice,
        PolicyPin::Recorded {
            key: frame.key.clone(),
            probe_key: frame.probe_key.clone(),
            row: 0,
        },
        Some([-1., 1.]),
    )
    .unwrap()
}
fn finish(worker: &mut Interactive) {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        worker.poll_policy().unwrap();
        if !worker.policy_pending() {
            break;
        }
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn policy_only_mask(chunk: &mut Chunk, mask: Option<[u8; 2]>, probabilities: [f64; 2]) {
    match &mut chunk.record {
        Record::ProbeSet(p) => {
            if let Some(key) = p.action_masks.take() {
                chunk.arrays.remove(&key);
            }
        }
        Record::Policy(p) => {
            p.masks = mask.map(|_| "policy-mask".into());
            if let Some(mask) = mask {
                chunk.arrays.insert(
                    "policy-mask".into(),
                    Array::u8(vec![2, 2], mask.repeat(2)).unwrap(),
                );
            }
            chunk.arrays.insert(
                "p".into(),
                Array::f64(vec![2, 2], probabilities.repeat(2)).unwrap(),
            );
        }
        _ => {}
    }
}
fn all_policy_mask(c: &mut Chunk) {
    policy_only_mask(c, Some([1, 1]), [0.5, 0.5]);
}
fn restricted_policy_mask(c: &mut Chunk) {
    policy_only_mask(c, Some([1, 0]), [1., 0.]);
}
fn all_mask_point_mass(c: &mut Chunk) {
    policy_only_mask(c, Some([1, 1]), [1., 0.]);
}
fn absent_policy_mask(c: &mut Chunk) {
    policy_only_mask(c, None, [0.5, 0.5]);
}

#[test]
fn policy_only_masks_certify_grid_and_exact_comparisons() {
    for (current, reference, expected, kl) in [
        (
            all_policy_mask as fn(&mut Chunk),
            restricted_policy_mask as fn(&mut Chunk),
            None,
            None,
        ),
        (all_policy_mask, absent_policy_mask, None, None),
        (absent_policy_mask, all_policy_mask, None, None),
        (
            restricted_policy_mask,
            restricted_policy_mask,
            Some(1.),
            Some(0.),
        ),
        (
            all_policy_mask,
            all_mask_point_mass,
            Some(1.),
            Some(f64::INFINITY),
        ),
    ] {
        let (_a, run, _, frame) = recorded_modified("current", 2, None, true, current);
        let (_b, other_run, _, other) = recorded_modified("reference", 2, None, true, reference);
        let mut worker = Interactive::new(|| {}).unwrap();
        worker
            .submit_policy(recorded_request(
                &run,
                &frame,
                Some(PolicySource::recorded_controller(&other_run, &other, "reference").unwrap()),
                false,
            ))
            .unwrap();
        finish(&mut worker);
        assert!(
            worker.policy_error().is_none(),
            "{:?}",
            worker.policy_error()
        );
        let f = worker.policy_frame().unwrap();
        assert_eq!(
            f.grid.reference_at(0),
            expected,
            "grid must certify actual optional mask conditioning"
        );
        assert_eq!(f.grid.difference_at(0).is_some(), expected.is_some());
        let selected = f.selected.as_ref().unwrap();
        assert_eq!(selected.reference_value, expected);
        assert_eq!(
            selected.reference_distribution.is_some(),
            expected.is_some()
        );
        assert_eq!(
            selected.reference_recorded_input.is_some(),
            expected.is_some()
        );
        assert_eq!(selected.kl, kl);
        let expected_mask = frame.chunk.arrays.get("policy-mask").map(|a| {
            a.as_u8().unwrap()[..2]
                .iter()
                .map(|&v| v != 0)
                .collect::<Vec<_>>()
        });
        assert_eq!(
            f.grid.input(0).unwrap().dense_mask(),
            expected_mask.as_deref()
        );
        assert_eq!(selected.input.dense_mask(), expected_mask.as_deref());
        assert!(worker.policy_metrics().scratch_peak <= 4 * 1024 * 1024);
    }
}

#[test]
fn policy_only_mask_matching_finds_later_controller_row_without_weakening_ordinary_identity() {
    let (_a, run, _, frame) = recorded_modified("current", 2, None, true, all_policy_mask);
    let (_b, other_run, _, other) = recorded_modified("reference", 2, None, true, |c| {
        policy_only_mask(c, Some([1, 1]), [0.5, 0.5]);
        match &c.record {
            Record::ProbeSet(p) => {
                c.arrays.insert(
                    p.observations.clone(),
                    Array::f32(vec![2, 1], vec![-0.5, -0.5]).unwrap(),
                );
            }
            Record::Policy(_) => {
                c.arrays.insert(
                    "policy-mask".into(),
                    Array::u8(vec![2, 2], vec![1, 0, 1, 1]).unwrap(),
                );
                c.arrays.insert(
                    "p".into(),
                    Array::f64(vec![2, 2], vec![1., 0., 0.5, 0.5]).unwrap(),
                );
            }
            _ => {}
        }
    });
    for explicit in [false, true] {
        let source = if explicit {
            PolicySource::recorded_controller(&other_run, &other, "controller")
        } else {
            PolicySource::recorded(&other_run, &other, "ordinary reference")
        }
        .unwrap();
        let mut worker = Interactive::new(|| {}).unwrap();
        worker
            .submit_policy(recorded_request(&run, &frame, Some(source), false))
            .unwrap();
        finish(&mut worker);
        assert!(
            worker.policy_error().is_none(),
            "{:?}",
            worker.policy_error()
        );
        let f = worker.policy_frame().unwrap();
        let p = f.selected.as_ref().unwrap();
        assert_eq!(f.grid.reference_at(0), explicit.then_some(0.5));
        assert_eq!(p.reference_value, explicit.then_some(0.5));
        assert_eq!(
            p.reference_recorded_input,
            explicit.then_some(SelectedInput::ProbeRow {
                set: "real-population".into(),
                row: 1,
                sequence: 0
            })
        );
    }
}
#[test]
fn masked_recorded_inputs_preserve_every_category_real_identity_and_leases() {
    let (_dir, run, mut resolver, frame) = recorded("masked", 70, None, false);
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(&run, &frame, None, false))
        .unwrap();
    drop(frame);
    resolver.clear();
    assert!(resolver.resident_bytes() > 0);
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let f = worker.policy_frame().unwrap();
    assert!(!f.grid.regular());
    assert_eq!(f.grid.rows, vec![0, 1]);
    let SelectedDistribution::Categorical {
        probabilities,
        valid,
    } = &f.selected.as_ref().unwrap().distribution
    else {
        panic!()
    };
    assert_eq!(probabilities.len(), 70);
    assert_eq!(valid.len(), 70);
    assert!(f.grid.selection_intent(1).is_some());
    worker.retire_session();
    let end = Instant::now() + Duration::from_secs(5);
    while worker.pending() {
        worker.housekeeping();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(resolver.resident_bytes(), 0);
}
#[test]
fn certified_cross_run_recorded_controller_compares_only_exact_complete_samples() {
    let (_a, run, _, frame) = recorded("current", 0, None, true);
    let (_b, reference, _, other) = recorded("analytical-controller", 0, None, true);
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut request = recorded_request(
        &run,
        &frame,
        Some(PolicySource::recorded(&reference, &other, "analytical controller samples").unwrap()),
        false,
    );
    worker.submit_policy(request.clone()).unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let f = worker.policy_frame().unwrap();
    assert_eq!(
        f.grid.reference.as_ref().unwrap().name,
        "analytical controller samples"
    );
    assert_eq!(f.selected.as_ref().unwrap().difference, Some(0.));
    assert!(f.selected.as_ref().unwrap().kl.is_none());
    let held_grid = f.grid.clone();
    let held_point = f.selected.as_ref().unwrap().clone();
    request.pin = PolicyPin::synthetic(Observation::new(vec![0.123]));
    worker.submit_policy(request).unwrap();
    finish(&mut worker);
    assert!(worker.policy_error().unwrap().contains("not recorded"));
    let held = worker.policy_frame().unwrap();
    assert!(std::sync::Arc::ptr_eq(&held.grid, &held_grid));
    let point = held.selected.as_ref().unwrap();
    assert_eq!(point.input, held_point.input);
    assert_ne!(point.input.dense_values(), vec![0.123]);
    assert_eq!(point.value, held_point.value);
    assert_eq!(point.reference_value, held_point.reference_value);
}
#[test]
fn recorded_history_is_kept_and_different_history_never_becomes_a_comparison() {
    let (_a, run, _, frame) = recorded("current", 70, Some(1.), true);
    let (_b, reference, _, other) = recorded("reference", 70, Some(2.), true);
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(
            &run,
            &frame,
            Some(
                PolicySource::recorded_controller(&reference, &other, "history reference").unwrap(),
            ),
            true,
        ))
        .unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!(point.reference_value.is_none());
    assert!(point.kl.is_none());
}
#[test]
fn changed_units_and_missing_descriptors_reject_cross_run_request() {
    let (_a, run, _, frame) = recorded("current", 0, None, true);
    let (_b, reference, _, other) = recorded("unknown", 0, None, false);
    let source = PolicySource::recorded(&reference, &other, "controller").unwrap();
    let base = recorded_request(&run, &frame, None, false);
    assert!(
        PolicyRequest::new(
            base.current,
            Some(source),
            9,
            (*base.slice).clone(),
            base.pin,
            base.distribution_domain
        )
        .is_err()
    );
    let mut a = run.recording().manifest().clone();
    let mut b = a.clone();
    b.environment_descriptor.as_mut().unwrap().observations[0].unit = Some("seconds".into());
    assert!(!compatible_encoding(
        false,
        EnvironmentSemantics::new(&a),
        EnvironmentSemantics::new(&b)
    ));
    a.environment_descriptor = None;
    assert!(!compatible_encoding(
        false,
        EnvironmentSemantics::new(&a),
        EnvironmentSemantics::new(&b)
    ));
}
#[test]
fn full_current_reference_kl_retains_true_support_infinity_at_identical_inputs() {
    let (_a, run, _, frame) = recorded("current", 70, None, true);
    let (_b, reference, _, other) = recorded_modified("reference", 70, None, true, |chunk| {
        if let Record::Policy(_) = &chunk.record {
            let mut p = vec![0.; 140];
            p[0] = 1.;
            p[70] = 1.;
            chunk
                .arrays
                .insert("p".into(), Array::f64(vec![2, 70], p).unwrap());
        }
    });
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(
            &run,
            &frame,
            Some(PolicySource::recorded(&reference, &other, "reference").unwrap()),
            false,
        ))
        .unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.kl, Some(f64::INFINITY));
    assert_eq!(point.reference_value, Some(1.));
}
#[test]
fn changed_masks_are_different_conditioning_even_when_numeric_observations_match() {
    let (_a, run, _, frame) = recorded("current", 70, None, true);
    let (_b, reference, _, other) =
        recorded_modified("reference", 70, None, true, |chunk| match &chunk.record {
            Record::ProbeSet(p) => {
                let mut masks = vec![1; 140];
                masks[69] = 0;
                masks[139] = 0;
                chunk.arrays.insert(
                    p.action_masks.clone().unwrap(),
                    Array::u8(vec![2, 70], masks).unwrap(),
                );
            }
            Record::Policy(_) => {
                let mut p = vec![1. / 69.; 140];
                p[69] = 0.;
                p[139] = 0.;
                chunk
                    .arrays
                    .insert("p".into(), Array::f64(vec![2, 70], p).unwrap());
            }
            _ => {}
        });
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(
            &run,
            &frame,
            Some(PolicySource::recorded_controller(&reference, &other, "reference").unwrap()),
            false,
        ))
        .unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!(point.reference_value.is_none());
    assert!(point.kl.is_none());
}
#[test]
fn different_continuous_families_compare_physical_output_without_fabricated_kl() {
    let (_a, run, _, frame) = recorded("current", 0, None, true);
    let (_b, reference, _, other) = recorded_modified("reference", 0, None, true, |chunk| {
        if let Record::Policy(p) = &mut chunk.record {
            p.parameters = PolicyParameters::Gaussian {
                mean: "mean".into(),
                log_std: "std".into(),
                tanh_bounded: false,
            };
            chunk.arrays.insert(
                "mean".into(),
                Array::f64(vec![2, 1], vec![-0.25, 0.75]).unwrap(),
            );
            chunk
                .arrays
                .insert("std".into(), Array::f64(vec![1], vec![0.]).unwrap());
        }
    });
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(
            &run,
            &frame,
            Some(PolicySource::recorded(&reference, &other, "reference").unwrap()),
            false,
        ))
        .unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.difference, Some(0.));
    assert!(point.kl.is_none());
    assert!(point.reference_plot.is_some());
}
#[test]
fn visitation_uses_real_policy_visited_rows_and_the_displayed_local_bin() {
    let (_a, run, _, frame) = recorded_modified("visits", 0, None, true, |chunk| {
        if let Record::ProbeSet(p) = &mut chunk.record {
            p.population = ProbePopulation::PolicyVisited;
        }
    });
    let mut r = recorded_request(&run, &frame, None, false);
    r.visits = Some(VisitSource {
        key: frame.probe_key.clone(),
        chunk: frame.probes.clone(),
    });
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(r).unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let f = worker.policy_frame().unwrap();
    let visits = f.grid.visits.as_ref().unwrap();
    assert_eq!(visits.revision, 5);
    assert_eq!(visits.filtered_count, 2);
    assert_eq!(visits.counts.iter().sum::<u32>(), 2);
    assert_eq!(visits.markers, vec![[-0.5, 0.], [0.5, 0.]]);
    assert_eq!(f.selected.as_ref().unwrap().visits, Some(1));
}
#[test]
fn imported_input_that_cannot_be_represented_exactly_is_not_relabelled_as_an_f32_pin() {
    let (_a, run, _, frame) = recorded_modified("precision", 0, None, true, |chunk| {
        if let Record::ProbeSet(p) = &chunk.record {
            chunk.arrays.insert(
                p.observations.clone(),
                Array::f64(vec![2, 1], vec![-0.5000000001, 0.5]).unwrap(),
            );
        }
    });
    let checked = rl_viewer::policy_change::admission::checked_batch(
        &frame.chunk,
        &frame.probes,
        &run.recording().manifest().action_space,
        4 * 1024 * 1024,
    )
    .unwrap();
    assert!(
        recorded_input(&checked, 0).is_err(),
        "keyed initialization must also reject rounded inputs"
    );
    // A manually supplied valid slice still cannot relabel the imported pin.
    let slice = SliceConfig::new(
        EnvironmentSemantics::new(run.recording().manifest()),
        &Observation::new(vec![-0.5]),
        &[],
    )
    .unwrap();
    let request = PolicyRequest::new(
        PolicySource::recorded(&run, &frame, "precision").unwrap(),
        None,
        9,
        slice,
        PolicyPin::Recorded {
            key: frame.key.clone(),
            probe_key: frame.probe_key.clone(),
            row: 0,
        },
        None,
    )
    .unwrap();
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(request).unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_some(),
        "must not silently round imported physical inputs"
    );
    assert!(worker.policy_frame().is_none());
}
#[test]
fn explicit_controller_can_have_independent_population_and_row_names() {
    let (_a, run, _, frame) = recorded("current", 0, None, true);
    let (_b, reference, _, other) = recorded_modified("controller", 0, None, true, |chunk| {
        chunk.context.probe_set = Some("controller-scenarios".into());
        if let Record::ProbeSet(p) = &mut chunk.record {
            p.id = "controller-scenarios".into();
            p.methodology = "independently evaluated controller scenarios".into();
            chunk.arrays.insert(
                p.row_ids.clone(),
                Array::u64(vec![2], vec![33, 44]).unwrap(),
            );
        }
    });
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(
            &run,
            &frame,
            Some(
                PolicySource::recorded_controller(&reference, &other, "analytical controller")
                    .unwrap(),
            ),
            false,
        ))
        .unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let f = worker.policy_frame().unwrap();
    let point = f.selected.as_ref().unwrap();
    assert_eq!(point.difference, Some(0.));
    assert_eq!(
        point.reference_recorded_input,
        Some(SelectedInput::ProbeRow {
            set: "controller-scenarios".into(),
            row: 0,
            sequence: 0
        })
    );
    assert_ne!(
        f.grid.current.key.root,
        f.grid.reference.as_ref().unwrap().key.root
    );
}
#[test]
fn oversized_borrowed_input_population_is_rejected_before_preparation() {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let m = RunManifest::new(
        "wide-population",
        "ordinary",
        "unknown",
        257,
        space.clone(),
        0.9,
    )
    .unwrap();
    let mut writer = Recorder::create(dir.path().join("run"), m, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("wide-population", 5);
    context.probe_set = Some("bulk".into());
    let probes = ProbeSet::new(
        "bulk",
        ProbePopulation::FixedReference,
        "bounded fixture",
        vec![Observation::new(vec![0.; 257]); 4096],
        None,
        space,
    )
    .unwrap();
    writer
        .append(&probes.record(context.clone()).unwrap())
        .unwrap();
    writer
        .append(&Chunk {
            context,
            record: Record::Policy(PolicyRecord {
                actions: "actions".into(),
                parameters: PolicyParameters::Deterministic,
                stochastic: false,
                masks: None,
                entropy: None,
                entropy_method: None,
            }),
            arrays: BTreeMap::from([(
                "actions".into(),
                Array::f32(vec![4096, 1], vec![0.; 4096]).unwrap(),
            )]),
        })
        .unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let selection = Selection {
        revision: 5,
        input: SelectedInput::ProbeRow {
            set: "bulk".into(),
            row: 0,
            sequence: 0,
        },
        ..Default::default()
    };
    let mut resolver = ProbeResolver::new(28 * 1024 * 1024);
    let frame = resolver
        .resolve_policy(&run, &selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap()
        .record
        .unwrap();
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(&run, &frame, None, false))
        .unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_some(),
        "the 4 MiB borrowed input population limit must be admitted before preparation"
    );
    assert!(worker.policy_frame().is_none());
}
#[test]
fn explicit_controller_changed_observations_have_no_comparison() {
    let (_a, run, _, frame) = recorded("current", 0, None, true);
    let (_b, reference, _, other) = recorded_modified("controller", 0, None, true, |chunk| {
        if let Record::ProbeSet(p) = &chunk.record {
            chunk.arrays.insert(
                p.observations.clone(),
                Array::f32(vec![2, 1], vec![-0.25, 0.25]).unwrap(),
            );
        }
    });
    let mut worker = Interactive::new(|| {}).unwrap();
    worker
        .submit_policy(recorded_request(
            &run,
            &frame,
            Some(PolicySource::recorded_controller(&reference, &other, "controller").unwrap()),
            false,
        ))
        .unwrap();
    finish(&mut worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!(point.reference_value.is_none());
    assert!(point.reference_recorded_input.is_none());
    assert!(point.kl.is_none());
}

#[test]
fn trajectory_visits_lease_real_decision_inputs_and_exclude_terminal_observations() {
    for (steps, truncated) in [(1usize, false), (2, true)] {
        let (_original_dir, original, _, original_frame) = recorded("visits", 2, None, true);
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Recorder::create(
            dir.path().join("run"),
            original.recording().manifest().clone(),
            StorageConfig::default(),
        )
        .unwrap();
        writer.append(&original_frame.probes).unwrap();
        writer.append(&original_frame.chunk).unwrap();
        let mut context = RecordContext::new("visits", 5);
        context.trajectory = Some("episode".into());
        let mut masks = vec![1; steps * 2];
        masks.extend([0, 0]);
        let mut observations = vec![-0.5];
        if steps == 2 {
            observations.push(0.5);
        }
        observations.push(0.);
        let mut terminated = vec![0; steps];
        let mut truncation = vec![0; steps];
        if truncated {
            truncation[steps - 1] = 1;
        } else {
            terminated[steps - 1] = 1;
        }
        writer
            .append(&Chunk {
                context,
                record: Record::Trajectory(TrajectoryRecord {
                    render: None,
                    id: "episode".into(),
                    environment: "ordinary".into(),
                    observations: "o".into(),
                    actions: "a".into(),
                    rewards: "r".into(),
                    terminated: "t".into(),
                    truncated: "u".into(),
                    simulation_times: None,
                    action_masks: Some("m".into()),
                    frames: BTreeMap::new(),
                    evaluation: EvaluationMetadata {
                        conditioning: Conditioning::InitialScenario,
                        initial_action: ActionCondition::CurrentPolicy,
                        continuation_policy: "recorded".into(),
                        stochastic_policy: true,
                        discount: 0.9,
                        independent_future_streams: false,
                        seed_methodology: "fixture".into(),
                        truncation_handling: "recorded".into(),
                        complete_returns: !truncated,
                    },
                }),
                arrays: BTreeMap::from([
                    (
                        "o".into(),
                        Array::f32(vec![steps + 1, 1], observations).unwrap(),
                    ),
                    ("a".into(), Array::u64(vec![steps], vec![0; steps]).unwrap()),
                    (
                        "r".into(),
                        Array::f32(vec![steps], vec![0.; steps]).unwrap(),
                    ),
                    ("t".into(), Array::u8(vec![steps], terminated).unwrap()),
                    ("u".into(), Array::u8(vec![steps], truncation).unwrap()),
                    ("m".into(), Array::u8(vec![steps + 1, 2], masks).unwrap()),
                ]),
            })
            .unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let selection = Selection {
            revision: 5,
            input: SelectedInput::ProbeRow {
                set: "real-population".into(),
                row: 0,
                sequence: 0,
            },
            ..Default::default()
        };
        let mut resolver = ProbeResolver::new(8 * 1024 * 1024);
        let frame = resolver
            .resolve_policy(&run, &selection, |i| {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            })
            .unwrap()
            .record
            .unwrap();
        let key = run.key(2).unwrap();
        let raw = Arc::new(run.recording().load(2).unwrap());
        let mut visit_resolver = ProbeResolver::new(1024 * 1024);
        assert!(
            visit_resolver
                .lease_visitation(&run, &run.key(0).unwrap(), raw.clone())
                .is_err()
        );
        assert!(
            visit_resolver
                .lease_visitation(
                    &run,
                    &run.key(0).unwrap(),
                    Arc::new(run.recording().load(0).unwrap())
                )
                .is_err()
        );
        assert!(
            ProbeResolver::new(1)
                .lease_visitation(&run, &key, raw.clone())
                .is_err()
        );
        let leased = visit_resolver.lease_visitation(&run, &key, raw).unwrap();
        let charged = visit_resolver.resident_bytes();
        assert!(charged > 0);
        let alias = leased.clone();
        assert_eq!(visit_resolver.resident_bytes(), charged);
        drop(alias);
        visit_resolver.clear();
        assert_eq!(visit_resolver.resident_bytes(), charged);
        let mut request = recorded_request(&run, &frame, None, false);
        request.visits = Some(VisitSource { key, chunk: leased });
        let mut worker = Interactive::new(|| {}).unwrap();
        worker.submit_policy(request).unwrap();
        finish(&mut worker);
        assert!(
            worker.policy_error().is_none(),
            "{:?}",
            worker.policy_error()
        );
        let visits = worker.policy_frame().unwrap().grid.visits.as_ref().unwrap();
        assert_eq!(
            visits.source,
            rl_viewer::policy::visitation::VisitationSource::EvaluationTrajectory
        );
        assert_eq!(visits.filtered_count, steps);
        assert_eq!(visits.counts.iter().sum::<u32>() as usize, steps);
        assert!(!visits.markers.contains(&[0., 0.]));
        worker.retire_session();
        let deadline = Instant::now() + Duration::from_secs(10);
        while visit_resolver.resident_bytes() != 0 {
            worker.housekeeping();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(worker.policy_metrics().prepared_bytes, 0);
    }
}

#[test]
fn automatic_gaussian_extent_is_common_and_explicit_overrides_are_preserved() {
    let gaussian = |mean, log_std| {
        SelectedDistribution::Gaussian(rl::diagnostics::policy::GaussianMarginal {
            mean,
            log_std,
            bounds: None,
        })
    };
    let a = gaussian(-2., 0.);
    let b = gaussian(3., 2f64.ln());
    let (current, reference) = distribution_plots(&a, Some(&b), None).unwrap();
    assert_eq!(current.as_ref().unwrap().domain, [-7., 13.]);
    assert_eq!(reference.unwrap().domain, [-7., 13.]);
    let retained = current.unwrap().domain;
    let (next, _) = distribution_plots(&gaussian(10., 0.), None, Some(retained)).unwrap();
    assert_eq!(next.unwrap().domain, retained);
    let (explicit, _) = distribution_plots(&a, Some(&b), Some([-1., 1.])).unwrap();
    assert_eq!(explicit.unwrap().domain, [-1., 1.]);
    assert!(distribution_plots(&gaussian(0., 1000.), None, None).is_err());
    assert!(distribution_plots(&gaussian(f64::MAX, -1.), None, None).is_err());
}
