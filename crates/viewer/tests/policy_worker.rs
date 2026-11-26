use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::*, schema::*, storage::*, visualization::EnvironmentSemantics},
    nn::{network::Mlp, policy::Deterministic},
};
use rl_viewer::{
    data::RunIndex,
    interactive::Interactive,
    policy::{prepared::*, slice::SliceConfig},
};
use std::time::{Duration, Instant};

fn fixture() -> (tempfile::TempDir, RunIndex) {
    fixture_models(&["actor-real-name"], false, &[3, 7])
}

#[test]
fn final_admitted_uncached_model_finishes_before_candidate_limit() {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest = RunManifest::new("limit", "ordinary", "unknown", 2, space.clone(), 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for revision in 0..=256 {
        let mut builder = SnapshotBuilder::new(
            "candidate",
            RecordContext::new("limit", revision),
            2,
            space.clone(),
        )
        .unwrap();
        let network = Mlp::new(2, &[], 1, &mut ChaCha8Rng::seed_from_u64(1)).unwrap();
        if revision == 1 {
            let policy = Deterministic::from_network(network, space.clone()).unwrap();
            builder.deterministic("actor", &policy).unwrap();
        } else {
            builder.opaque("opaque", &network, "unsupported").unwrap();
        }
        writer.append(&builder.finish().unwrap()).unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut loader = rl_viewer::data::Loader::new(1024 * 1024, || {}).unwrap();
    let mut search = ModelSearch::new("online", 256, None).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let source = loop {
        loader.begin_pass();
        loader.poll();
        let source = search.advance(&run, &mut loader).unwrap();
        loader.finish_pass().unwrap();
        if let Some(source) = source {
            break source;
        }
        assert!(search.pending(), "{:?}", search.last_rejection);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(source.identity().revision, 1);
    assert_eq!(search.checked_candidates(), 256);
    assert_eq!(loader.completed_loads, 256);
}

#[test]
fn recorded_policy_only_mask_is_forwarded_to_model_reference() {
    use rl::{
        diagnostics::probes::ProbeSet,
        nn::{critics::scalar::DiscreteQ, matrix, vector},
    };
    use rl_viewer::{
        data::{SelectedInput, Selection},
        resolver::ProbeResolver,
    };
    use std::{collections::BTreeMap, sync::Arc};
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 2 };
    let manifest =
        RunManifest::new("masked-model", "ordinary", "unknown", 1, space.clone(), 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("masked-model", 3);
    context.probe_set = Some("inputs".into());
    let mut probes = ProbeSet::new(
        "inputs",
        ProbePopulation::FixedReference,
        "fixture",
        vec![Observation::new(vec![0.])],
        None,
        space.clone(),
    )
    .unwrap()
    .record(context.clone())
    .unwrap();
    if let Record::ProbeSet(p) = &mut probes.record
        && let Some(key) = p.action_masks.take()
    {
        probes.arrays.remove(&key);
    }
    writer.append(&probes).unwrap();
    writer
        .append(&Chunk {
            context,
            record: Record::Policy(PolicyRecord {
                actions: "a".into(),
                parameters: PolicyParameters::Categorical {
                    probabilities: "p".into(),
                },
                stochastic: false,
                masks: Some("m".into()),
                entropy: None,
                entropy_method: None,
            }),
            arrays: BTreeMap::from([
                ("a".into(), Array::u64(vec![1], vec![0]).unwrap()),
                ("p".into(), Array::f64(vec![1, 2], vec![1., 0.]).unwrap()),
                ("m".into(), Array::u8(vec![1, 2], vec![1, 0]).unwrap()),
            ]),
        })
        .unwrap();
    let mut q = DiscreteQ::new(1, 2, &[], &mut ChaCha8Rng::seed_from_u64(9)).unwrap();
    q.network.layers[0].weight = q.network.layers[0]
        .weight
        .clone()
        .map(|_| matrix(vec![0., 0.], 1, 2));
    q.network.layers[0].bias = q.network.layers[0]
        .bias
        .take()
        .map(|p| p.map(|_| vector(&[0., 10.])));
    let mut builder =
        SnapshotBuilder::new("q", RecordContext::new("masked-model", 7), 1, space).unwrap();
    builder.discrete("q", &q).unwrap();
    writer.append(&builder.finish().unwrap()).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let selection = Selection {
        revision: 3,
        input: SelectedInput::ProbeRow {
            set: "inputs".into(),
            row: 0,
            sequence: 0,
        },
        ..Default::default()
    };
    let frame = resolver
        .resolve_policy(&run, &selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap()
        .record
        .unwrap();
    let checked = rl_viewer::policy_change::admission::checked_batch(
        &frame.chunk,
        &frame.probes,
        &run.recording().manifest().action_space,
        4 * 1024 * 1024,
    )
    .unwrap();
    let input = recorded_input(&checked, 0).unwrap();
    let slice = SliceConfig::new(
        EnvironmentSemantics::new(run.recording().manifest()),
        &input,
        &[],
    )
    .unwrap();
    let request = PolicyRequest::new(
        PolicySource::recorded(&run, &frame, "current").unwrap(),
        Some(PolicySource::model(&run, 2, None).unwrap()),
        7,
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
    worker.submit_policy(request.clone()).unwrap();
    wait(&mut worker);
    let f = worker.policy_frame().unwrap();
    assert_eq!(
        f.grid.reference_at(0),
        Some(1.),
        "the model must receive the policy-only mask, excluding its preferred action1"
    );
    let point = f.selected.as_ref().unwrap();
    assert_eq!(point.reference_value, Some(1.));
    assert_eq!(point.input.dense_mask(), Some(&[true, false][..]));
    assert_eq!(worker.policy_metrics().restores, 1);
    let mut model_request = request;
    model_request.input_population = Some(model_request.current.clone());
    model_request.current = model_request.reference.take().unwrap();
    worker.submit_policy(model_request.clone()).unwrap();
    wait(&mut worker);
    let frame = worker.policy_frame().unwrap();
    assert!(!frame.grid.current.recorded);
    assert!(!frame.grid.regular());
    assert_eq!(frame.grid.population_count, 1);
    assert_eq!(frame.grid.values, [1.]);
    assert_eq!(
        frame.selected.as_ref().unwrap().input.dense_mask(),
        Some(&[true, false][..])
    );
    assert!(matches!(
        frame.selected.as_ref().unwrap().recorded_input,
        Some(SelectedInput::ProbeRow {
            row: 0,
            sequence: 0,
            ..
        })
    ));
    let held = frame.grid.clone();
    for invalid in 0..3 {
        let mut bad = model_request.clone();
        let PolicyPin::Recorded {
            key,
            probe_key,
            row,
        } = &mut bad.pin
        else {
            unreachable!()
        };
        match invalid {
            0 => key.sequence += 1,
            1 => probe_key.sequence += 1,
            _ => *row = 1,
        }
        assert!(worker.submit_policy(bad).is_err());
        assert!(Arc::ptr_eq(&held, &worker.policy_frame().unwrap().grid));
    }
    let mut oversized = model_request.clone();
    Arc::make_mut(&mut oversized.slice)
        .fixed
        .reserve_exact(70_000);
    assert!(worker.submit_policy(oversized).is_err());
    assert!(Arc::ptr_eq(&held, &worker.policy_frame().unwrap().grid));
    assert!(worker.policy_metrics().prepared_bytes <= rl_viewer::budgets::POLICY_FRAME);
    worker.retire_session();
    let end = Instant::now() + Duration::from_secs(5);
    while worker.pending() {
        worker.housekeeping();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(worker.policy_metrics().prepared_bytes > 0);
    drop(held);
    assert_eq!(worker.policy_metrics().prepared_bytes, 0);
}
fn fixture_models(
    names: &[&str],
    opaque_last: bool,
    revisions: &[u64],
) -> (tempfile::TempDir, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest = RunManifest::new(
        "worker",
        "ordinary",
        "not-a-policy-name",
        2,
        space.clone(),
        0.9,
    )
    .unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for (position, &revision) in revisions.iter().enumerate() {
        let network = Mlp::new(2, &[4], 1, &mut ChaCha8Rng::seed_from_u64(revision)).unwrap();
        let policy = Deterministic::from_network(network, space.clone()).unwrap();
        let mut builder = SnapshotBuilder::new(
            &format!("k{revision}"),
            RecordContext::new("worker", revision),
            2,
            space.clone(),
        )
        .unwrap();
        if opaque_last && position + 1 == revisions.len() {
            builder
                .opaque(
                    "external-controller",
                    &policy.network,
                    "no inference adapter",
                )
                .unwrap();
        } else {
            for name in names {
                builder.deterministic(name, &policy).unwrap();
            }
        }
        writer.append(&builder.finish().unwrap()).unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    (dir, run)
}
fn request(run: &RunIndex, one_dimensional: bool) -> PolicyRequest {
    let environment = EnvironmentSemantics::new(run.recording().manifest());
    let mut slice = SliceConfig::new(environment, &Observation::new(vec![0., 0.]), &[]).unwrap();
    if one_dimensional {
        slice.set_axes(environment, 0, None).unwrap();
        slice.resolution = [256, 1];
    }
    PolicyRequest::new(
        PolicySource::model(run, 1, None).unwrap(),
        None,
        9,
        slice,
        PolicyPin::synthetic(Observation::new(vec![0.12345, -0.23456])),
        Some([-5., 5.]),
    )
    .unwrap()
}
fn finish(worker: &mut Interactive) {
    let end = Instant::now() + Duration::from_secs(30);
    loop {
        worker.poll_policy().unwrap();
        if !worker.policy_pending() {
            break;
        }
        assert!(Instant::now() < end, "policy worker timed out");
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn wait(worker: &mut Interactive) {
    finish(worker);
    assert!(
        worker.policy_error().is_none(),
        "{:?}",
        worker.policy_error()
    );
}
#[test]
fn real_grid_batches_reuse_snapshot_and_exact_pin_is_not_a_cell() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(request(&run, false)).unwrap();
    wait(&mut worker);
    let frame = worker.policy_frame().unwrap();
    assert_eq!(frame.grid.values.len(), 4096);
    assert_eq!(frame.grid.current.revision, 7);
    assert_eq!(frame.requested_revision, 9);
    let point = frame.selected.as_ref().unwrap();
    assert_eq!(point.input.dense_values(), vec![0.12345, -0.23456]);
    let model =
        ModelBundle::load(&run.recording().load(1).unwrap(), ModelLimits::default()).unwrap();
    let expected = model
        .inspect(&InspectionRequest::new(
            "actor-real-name",
            vec![point.input.clone()],
        ))
        .unwrap();
    assert_eq!(
        point.value,
        expected.policy.arrays["actions"].as_f32().unwrap()[0] as f64
    );
    let stats = worker.policy_metrics();
    println!(
        "real_grid_metrics={}",
        serde_json::to_string(&stats).unwrap()
    );
    assert_eq!(stats.grid_batches, 4);
    // One exact selected point and one exact selected-input cross-section.
    assert_eq!(stats.point_batches, 2);
    assert_eq!(stats.restores, 1);
    assert!(stats.scratch_peak <= 4 * 1024 * 1024);
    assert!(stats.prepared_peak <= 4 * 1024 * 1024);
}

#[test]
fn selected_policy_cross_section_uses_exact_fixed_input_and_reuses_compatible_pins() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut requested = request(&run, false);
    worker.submit_policy(requested.clone()).unwrap();
    wait(&mut worker);
    let frame = worker.policy_frame().unwrap();
    let point = frame.selected.as_ref().unwrap();
    let cross = point
        .cross_section()
        .expect("deterministic model cross-section");
    assert_eq!(
        cross.fixed_value.to_bits(),
        point.input.dense_values()[0].to_bits()
    );
    assert_eq!(cross.varying_axis, 1);
    let observations = cross
        .points
        .iter()
        .map(|p| Observation::new(vec![cross.fixed_value, p[0] as f32]))
        .collect::<Vec<_>>();
    let model =
        ModelBundle::load(&run.recording().load(1).unwrap(), ModelLimits::default()).unwrap();
    let expected = model
        .inspect(&InspectionRequest::new("actor-real-name", observations))
        .unwrap();
    for (point, expected) in cross
        .points
        .iter()
        .zip(expected.policy.arrays["actions"].as_f32().unwrap())
    {
        assert_eq!(point[1].to_bits(), f64::from(*expected).to_bits());
    }
    let previous_cross = cross as *const _;
    let before = worker.policy_metrics();
    requested.pin = PolicyPin::synthetic(Observation::new(vec![0.12345, 0.3]));
    worker.submit_policy(requested.clone()).unwrap();
    wait(&mut worker);
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.cross_section().unwrap() as *const _, previous_cross);
    assert_eq!(worker.policy_metrics().grid_batches, before.grid_batches);
    assert_eq!(
        worker.policy_metrics().point_batches,
        before.point_batches + 1
    );
    let before = worker.policy_metrics();
    requested.pin = PolicyPin::synthetic(Observation::new(vec![0.3, 0.3]));
    worker.submit_policy(requested).unwrap();
    wait(&mut worker);
    assert_eq!(worker.policy_metrics().grid_batches, before.grid_batches);
    assert_eq!(
        worker.policy_metrics().point_batches,
        before.point_batches + 2
    );
    assert_eq!(
        worker
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .cross_section()
            .unwrap()
            .fixed_value,
        0.3
    );
}

#[test]
fn policy_cross_section_keeps_only_reference_rows_with_the_exact_other_inputs() {
    use rl::diagnostics::probes::ProbeSet;
    use rl_viewer::{
        data::{SelectedInput, Selection},
        resolver::ProbeResolver,
    };
    use std::{collections::BTreeMap, sync::Arc};
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest = RunManifest::new(
        "cross-reference",
        "ordinary",
        "fixture",
        3,
        space.clone(),
        0.9,
    )
    .unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut builder = SnapshotBuilder::new(
        "actor",
        RecordContext::new("cross-reference", 7),
        3,
        space.clone(),
    )
    .unwrap();
    let network = Mlp::new(3, &[4], 1, &mut ChaCha8Rng::seed_from_u64(7)).unwrap();
    builder
        .deterministic(
            "actor",
            &Deterministic::from_network(network, space.clone()).unwrap(),
        )
        .unwrap();
    writer.append(&builder.finish().unwrap()).unwrap();
    let mut context = RecordContext::new("cross-reference", 3);
    context.probe_set = Some("reference".into());
    let observations = vec![
        Observation::new(vec![0.25, -0.5, 0.4]),
        Observation::new(vec![0.25, 0., 0.4]),
        Observation::new(vec![0.25, 0.5, 0.4]),
        Observation::new(vec![0.26, 0., 0.4]),
        Observation::new(vec![0.25, 0., 0.5]),
    ];
    writer
        .append(
            &ProbeSet::new(
                "reference",
                ProbePopulation::FixedReference,
                "real reference rows",
                observations,
                None,
                space,
            )
            .unwrap()
            .record(context.clone())
            .unwrap(),
        )
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
                Array::f64(vec![5, 1], vec![0.11, 0.22, 0.33, 0.44, 0.55]).unwrap(),
            )]),
        })
        .unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let selection = Selection {
        revision: 3,
        model_role: "online".into(),
        input: SelectedInput::ProbeRow {
            set: "reference".into(),
            sequence: 1,
            row: 0,
        },
    };
    let mut resolver = ProbeResolver::new(8 * 1024 * 1024);
    let reference = resolver
        .resolve_policy(&run, &selection, |index| {
            Ok(Some(Arc::new(run.recording().load(index)?)))
        })
        .unwrap()
        .record
        .unwrap();
    let current = PolicySource::model(&run, 0, Some("actor")).unwrap();
    let input = Observation::new(vec![0.25, 0., 0.4]);
    let mut slice = SliceConfig::new(current.semantics(), &input, &[]).unwrap();
    slice.set_axes(current.semantics(), 0, Some(1)).unwrap();
    slice.resolution = [32, 32];
    let mut request = PolicyRequest::new(
        current,
        Some(PolicySource::recorded_controller(&run, &reference, "benchmark").unwrap()),
        7,
        slice,
        PolicyPin::synthetic(input),
        None,
    )
    .unwrap();
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(request.clone()).unwrap();
    wait(&mut worker);
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.reference_value, Some(0.22));
    assert_eq!(
        point.cross_section().unwrap().reference_samples,
        vec![[-0.5, 0.11], [0., 0.22], [0.5, 0.33]]
    );
    request.pin = PolicyPin::synthetic(Observation::new(vec![0.25, 0., 0.5]));
    worker.submit_policy(request).unwrap();
    wait(&mut worker);
    let point = worker.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.reference_value, Some(0.55));
    assert_eq!(
        point.cross_section().unwrap().reference_samples,
        vec![[0., 0.55]]
    );
}
#[test]
fn pin_churn_does_not_restart_an_unfinished_grid_and_reuses_committed_storage() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut requested = request(&run, false);
    worker.submit_policy(requested.clone()).unwrap();
    for i in 0..80 {
        requested.pin = PolicyPin::synthetic(Observation::new(vec![i as f32 / 100., -0.2]));
        worker.submit_policy(requested.clone()).unwrap();
    }
    wait(&mut worker);
    assert_eq!(worker.policy_metrics().grid_batches, 4);
    assert_eq!(
        worker
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values()[0],
        0.79
    );
    let grid = std::sync::Arc::as_ptr(&worker.policy_frame().unwrap().grid);
    requested.pin = PolicyPin::synthetic(Observation::new(vec![0.314, -0.2]));
    worker.submit_policy(requested).unwrap();
    assert_eq!(
        worker
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values()[0],
        0.79
    );
    wait(&mut worker);
    assert_eq!(
        grid,
        std::sync::Arc::as_ptr(&worker.policy_frame().unwrap().grid)
    );
    assert_eq!(worker.policy_metrics().grid_batches, 4);
}
#[test]
fn line_is_one_batch_and_retirement_releases_all_policy_owners() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(request(&run, true)).unwrap();
    wait(&mut worker);
    assert_eq!(worker.policy_frame().unwrap().grid.values.len(), 256);
    assert_eq!(worker.policy_metrics().grid_batches, 1);
    worker.retire_session();
    let end = Instant::now() + Duration::from_secs(10);
    while worker.pending() {
        worker.housekeeping();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(worker.policy_frame().is_none());
    assert_eq!(worker.policy_metrics().prepared_bytes, 0);
    worker.submit_policy(request(&run, true)).unwrap();
    wait(&mut worker);
    assert_eq!(worker.policy_metrics().restores, 2);
}
#[test]
fn cached_model_cannot_survive_removal_of_its_immutable_source() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, true);
    worker.submit_policy(r.clone()).unwrap();
    wait(&mut worker);
    let held_point = worker
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .clone();
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    r.pin = PolicyPin::synthetic(Observation::new(vec![0.3, 0.4]));
    worker.submit_policy(r).unwrap();
    finish(&mut worker);
    assert!(worker.policy_error().is_some());
    assert!(std::sync::Arc::ptr_eq(
        worker.policy_frame().unwrap().selected.as_ref().unwrap(),
        &held_point
    ));
    assert_eq!(
        worker.policy_frame().unwrap().grid.current.key,
        run.key(1).unwrap()
    );
    assert_eq!(worker.policy_metrics().restores, 1);
}
#[test]
fn current_and_reference_keep_independent_actual_identity() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, true);
    r.reference = Some(PolicySource::model(&run, 0, Some("actor-real-name")).unwrap());
    worker.submit_policy(r).unwrap();
    wait(&mut worker);
    let f = worker.policy_frame().unwrap();
    assert_eq!(f.grid.current.key, run.key(1).unwrap());
    assert_eq!(f.grid.reference.as_ref().unwrap().key, run.key(0).unwrap());
    assert_eq!(f.grid.reference.as_ref().unwrap().revision, 3);
    assert_eq!(f.grid.current.name, "actor-real-name");
    assert!(f.selected.as_ref().unwrap().reference_value.is_some());
    assert_eq!(worker.policy_metrics().restores, 2);
}

#[test]
fn bounded_gaussian_reference_projection_matches_the_same_current_model() {
    use rl::nn::gaussian_policy::GaussianPolicy;
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let manifest =
        RunManifest::new("gaussian", "ordinary", "fixture", 2, space.clone(), 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let policy =
        GaussianPolicy::new(2, space.clone(), &[4], &mut ChaCha8Rng::seed_from_u64(19)).unwrap();
    let mut snapshot =
        SnapshotBuilder::new("gaussian", RecordContext::new("gaussian", 3), 2, space).unwrap();
    snapshot.state_gaussian("actor", &policy).unwrap();
    writer.append(&snapshot.finish().unwrap()).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let source = PolicySource::model(&run, 0, Some("actor")).unwrap();
    let input = Observation::new(vec![0., 0.]);
    let mut slice = SliceConfig::new(source.semantics(), &input, &[]).unwrap();
    slice.resolution = [16, 16];
    let request = PolicyRequest::new(
        source.clone(),
        Some(source),
        3,
        slice,
        PolicyPin::synthetic(input),
        None,
    )
    .unwrap();
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(request).unwrap();
    wait(&mut worker);
    let grid = &worker.policy_frame().unwrap().grid;
    assert_eq!(grid.values.len(), 256);
    for (row, value) in grid.values.iter().enumerate() {
        assert_eq!(value.to_bits(), grid.reference_at(row).unwrap().to_bits());
    }
}
#[test]
fn latest_axes_source_reference_and_revision_supersede_all_previous_results() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, false);
    worker.submit_policy(r.clone()).unwrap();
    r = request(&run, true);
    r.current = PolicySource::model(&run, 0, None).unwrap();
    r.requested_revision = 4;
    worker.submit_policy(r.clone()).unwrap();
    r.reference = Some(PolicySource::model(&run, 1, None).unwrap());
    r.requested_revision = 8;
    worker.submit_policy(r).unwrap();
    wait(&mut worker);
    let frame = worker.policy_frame().unwrap();
    assert_eq!(frame.grid.values.len(), 256);
    assert_eq!(frame.requested_revision, 8);
    assert_eq!(frame.grid.current.key, run.key(0).unwrap());
    assert_eq!(
        frame.grid.reference.as_ref().unwrap().key,
        run.key(1).unwrap()
    );
}
#[test]
fn held_source_search_uses_actual_earlier_revision_without_replacing_selected_identity() {
    let (_dir, run) = fixture();
    let mut search = ModelSearch::new("online", 6, None).unwrap();
    let mut loader = rl_viewer::data::Loader::new(8 * 1024 * 1024, || {}).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    let source = loop {
        loader.poll();
        if let Some(source) = search.advance(&run, &mut loader).unwrap() {
            break source;
        }
        assert!(search.pending());
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(source.identity().revision, 3);
    assert_eq!(source.identity().name, "actor-real-name");
    std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
    assert!(search.advance(&run, &mut loader).is_err());
}
#[test]
fn pin_edits_during_an_observed_partial_grid_finish_once_with_the_latest_point() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, false);
    std::sync::Arc::make_mut(&mut r.slice).resolution = [256, 256];
    worker.submit_policy(r.clone()).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while worker.policy_metrics().grid_batches == 0 {
        assert!(Instant::now() < end);
        std::thread::yield_now();
    }
    assert!(
        worker.policy_metrics().grid_batches < 64,
        "fixture must catch a partially completed grid"
    );
    for i in 0..150 {
        r.pin = PolicyPin::synthetic(Observation::new(vec![i as f32 / 200., 0.13]));
        worker.submit_policy(r.clone()).unwrap();
        worker.poll_policy().unwrap();
        if let Some(f) = worker.policy_frame() {
            assert!(
                f.selected.is_none()
                    || f.selected.as_ref().unwrap().input.dense_values()[0] == i as f32 / 200.
            );
        }
    }
    wait(&mut worker);
    assert_eq!(worker.policy_metrics().grid_batches, 64);
    assert_eq!(
        worker
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values(),
        vec![0.745, 0.13]
    );
}
#[test]
fn external_grid_aliases_and_unpolled_completions_remain_accounted_until_dropped() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(request(&run, false)).unwrap();
    wait(&mut worker);
    let before = worker.policy_metrics().prepared_bytes;
    let held = worker.policy_frame().unwrap().grid.clone();
    let held_again = held.clone();
    assert_eq!(
        worker.policy_metrics().prepared_bytes,
        before,
        "Arc aliases charge once"
    );
    worker.retire_session();
    let end = Instant::now() + Duration::from_secs(5);
    while worker.pending() {
        worker.housekeeping();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        worker.policy_metrics().prepared_bytes > 0,
        "externally retained grid must remain counted"
    );
    drop(held);
    assert!(worker.policy_metrics().prepared_bytes > 0);
    drop(held_again);
    assert_eq!(worker.policy_metrics().prepared_bytes, 0);
    worker.submit_policy(request(&run, false)).unwrap();
    let end = Instant::now() + Duration::from_secs(5);
    while worker.policy_metrics().grid_batches < 8 {
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    worker.retire_session();
    while worker.pending() {
        worker.housekeeping();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        worker.policy_metrics().prepared_bytes,
        0,
        "late completion dropped before retirement acknowledgment"
    );
}
#[test]
fn ambiguous_model_names_require_a_choice_and_opaque_models_do_not_fabricate_grids() {
    let (_dir, run) = fixture_models(&["actor-real-name", "alternate"], false, &[3, 7]);
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, true);
    worker.submit_policy(r.clone()).unwrap();
    finish(&mut worker);
    assert!(worker.policy_error().unwrap().contains("Multiple"));
    assert!(worker.policy_frame().is_none());
    r.current = PolicySource::model(&run, 1, Some("alternate")).unwrap();
    worker.submit_policy(r).unwrap();
    wait(&mut worker);
    assert_eq!(
        worker.policy_frame().unwrap().grid.current.name,
        "alternate"
    );
    let (_opaque, run) = fixture_models(&["actor-real-name"], true, &[3, 7]);
    worker.retire_session();
    worker.submit_policy(request(&run, true)).unwrap();
    finish(&mut worker);
    assert!(worker.policy_error().unwrap().contains("adapter"));
    assert!(worker.policy_frame().is_none());
    let mut search = ModelSearch::new("online", 9, None).unwrap();
    let mut loader = rl_viewer::data::Loader::new(8 * 1024 * 1024, || {}).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    let selected = loop {
        loader.poll();
        if let Some(source) = search.advance(&run, &mut loader).unwrap() {
            break source;
        }
        assert!(search.pending());
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(selected.identity().revision, 3);
    assert_eq!(search.checked_candidates(), 2);
}
#[test]
fn policy_same_revision_sources_do_not_substitute_a_sibling() {
    let (_dir, run) = fixture_models(&["actor-real-name"], false, &[3, 7, 7]);
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, true);
    worker.submit_policy(r.clone()).unwrap();
    wait(&mut worker);
    assert_eq!(
        worker.policy_frame().unwrap().grid.current.key,
        run.key(1).unwrap()
    );
    let held_point = worker
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .clone();
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    r.pin = PolicyPin::synthetic(Observation::new(vec![0.25, 0.5]));
    worker.submit_policy(r.clone()).unwrap();
    finish(&mut worker);
    assert!(std::sync::Arc::ptr_eq(
        worker.policy_frame().unwrap().selected.as_ref().unwrap(),
        &held_point
    ));
    assert_eq!(
        worker.policy_frame().unwrap().grid.current.key,
        run.key(1).unwrap()
    );
    assert!(worker.policy_error().is_some());
    r.current = PolicySource::model(&run, 2, None).unwrap();
    worker.submit_policy(r).unwrap();
    wait(&mut worker);
    assert_eq!(
        worker.policy_frame().unwrap().grid.current.key,
        run.key(2).unwrap()
    );
}
#[test]
fn rejected_submission_holds_old_exact_point_without_relabelling_it() {
    let (_dir, run) = fixture();
    let mut worker = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, true);
    worker.submit_policy(r.clone()).unwrap();
    wait(&mut worker);
    let held_point = worker
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .clone();
    r.pin = PolicyPin::synthetic(Observation::new(vec![f32::NAN, 0.]));
    assert!(worker.submit_policy(r).is_err());
    assert!(std::sync::Arc::ptr_eq(
        worker.policy_frame().unwrap().selected.as_ref().unwrap(),
        &held_point
    ));
    assert!(worker.policy_error().is_some());
}

#[test]
fn same_grid_pin_owners_fit_the_committed_frame_allowance() {
    use rl::diagnostics::visualization::*;
    let (_dir, original) = fixture();
    let mut manifest = original.recording().manifest().clone();
    manifest.environment_descriptor = Some(EnvironmentDescriptor {
        version: 1,
        observations: (0..2)
            .map(|i| ObservationField {
                id: format!("x{i}"),
                label: format!("Input{i}"),
                unit: None,
                description: None,
                kind: ObservationType::Numerical,
                range: Some([-1., 1.]),
                coordinate: true,
            })
            .collect(),
        actions: ActionDescriptor::Continuous {
            components: vec![ActionComponent {
                label: "action".into(),
                unit: None,
                meaning: None,
            }],
            preferred_component: None,
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
        frames: (0..24)
            .map(|i| FrameChannel {
                name: format!("channel{i}"),
                label: "f".repeat(900),
                width: 1,
                unit: None,
                visibility: Default::default(),
                representation: None,
            })
            .collect(),
        preferences: Default::default(),
    });
    manifest.validate().unwrap();
    std::fs::write(
        original.root().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let run = RunIndex::open(original.root().as_ref(), ReadLimits::default()).unwrap();
    let mut initial = request(&run, false);
    let slice = std::sync::Arc::get_mut(&mut initial.slice).unwrap();
    slice.resolution = [256, 256];
    slice.fixed.reserve_exact(60_000);
    initial.reference = Some(PolicySource::model(&run, 0, None).unwrap());
    if let PolicyPin::Synthetic(pin) = &mut initial.pin {
        pin.as_dense_mut().unwrap().values.reserve_exact(60_000);
    }
    let mut latest = initial.clone();
    if let PolicyPin::Synthetic(pin) = &mut latest.pin {
        pin.as_dense_mut().expect("dense diagnostic input").values[0] = 0.321;
        pin.as_dense_mut().unwrap().values.reserve_exact(60_000);
    }
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_policy(initial).unwrap();
    wait(&mut worker);
    assert!(worker.policy_metrics().prepared_bytes <= rl_viewer::budgets::POLICY_FRAME);
    let retry = latest.clone(); // A compact latest pin shares the exact config/source metadata.
    let held_point = worker
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .clone();
    worker.submit_policy(latest).unwrap();
    finish(&mut worker);
    assert!(worker.policy_error().unwrap().contains("2 MiB"));
    assert!(std::sync::Arc::ptr_eq(
        worker.policy_frame().unwrap().selected.as_ref().unwrap(),
        &held_point
    ));
    drop(held_point);
    assert!(
        worker.policy_metrics().prepared_bytes <= rl_viewer::budgets::POLICY_FRAME,
        "committed frame includes the original grid pin and latest desired pin owners: {:?}",
        worker.policy_metrics()
    );
    worker.submit_policy(retry).unwrap();
    wait(&mut worker);
    assert_eq!(worker.policy_metrics().grid_batches, 64);
    assert!(worker.policy_metrics().prepared_bytes <= rl_viewer::budgets::POLICY_FRAME);
}
