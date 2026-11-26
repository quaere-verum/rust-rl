use super::*;
use rl::diagnostics::schema::*;
use std::collections::BTreeMap;

fn probes(values: Vec<f32>, masks: Vec<u8>) -> Chunk {
    let n = values.len() / 5;
    let mut context = RecordContext::new("run", 0);
    context.probe_set = Some("all".into());
    Chunk {
        context,
        record: Record::ProbeSet(ProbeSetRecord {
            id: "all".into(),
            population: ProbePopulation::FixedReference,
            methodology: "complete finite inputs".into(),
            observations: "x".into(),
            action_masks: Some("m".into()),
            fixed_actions: None,
            row_ids: "ids".into(),
            histories: BTreeMap::new(),
        }),
        arrays: BTreeMap::from([
            ("x".into(), Array::f32(vec![n, 5], values).unwrap()),
            ("m".into(), Array::u8(vec![n, 2], masks).unwrap()),
            (
                "ids".into(),
                Array::u64(vec![n], (0..n as u64).collect()).unwrap(),
            ),
        ]),
    }
}

#[test]
fn exact_full_inputs_include_time_phase_cooldown_and_masks() {
    let p = probes(
        vec![
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0.5, 0., 0., 1., 0., 0.5, 0., 0., 1., 1., 0.5,
        ],
        vec![1, 1, 1, 1, 1, 1, 1, 0],
    );
    let a = Arc::new(Accounting::default());
    let index = ExactIndex::build(&p, &a, &|| false).unwrap();
    for row in 0..4 {
        let o = crate::policy::prepared::exact_observation(&p, row).unwrap();
        assert_eq!(index.lookup(&p, &o).unwrap(), Some(row));
    }
    let mut o = crate::policy::prepared::exact_observation(&p, 0).unwrap();
    o.as_dense_mut().expect("dense diagnostic input").values[0] = -0.;
    assert_eq!(index.lookup(&p, &o).unwrap(), Some(0));
    o.as_dense_mut().expect("dense diagnostic input").values[4] = 0.75;
    assert_eq!(index.lookup(&p, &o).unwrap(), None);
    o.as_dense_mut().expect("dense diagnostic input").values[4] = 0.; // terminal inputs are absent
    assert_eq!(index.lookup(&p, &o).unwrap(), None);
    o.as_dense_mut().expect("dense diagnostic input").values[4] = 1.;
    o.as_dense_mut()
        .expect("dense diagnostic input")
        .action_mask = None;
    assert_eq!(index.lookup(&p, &o).unwrap(), None);
    o.as_dense_mut()
        .expect("dense diagnostic input")
        .action_mask = Some(vec![true, false]);
    assert_eq!(index.lookup(&p, &o).unwrap(), None);
}

#[test]
fn duplicate_keys_and_cancelled_builds_do_not_publish_or_leak() {
    let p = probes(vec![0.; 10], vec![1; 4]);
    let a = Arc::new(Accounting::default());
    assert!(ExactIndex::build(&p, &a, &|| false).is_err());
    assert_eq!(a.snapshot().prepared_bytes, 0);
    let p = probes(
        (0..43_264)
            .flat_map(|n| [n as f32, 0., 0., 0., 1.])
            .collect(),
        vec![1; 86_528],
    );
    let calls = std::cell::Cell::new(0);
    assert!(
        ExactIndex::build(&p, &a, &|| {
            calls.set(calls.get() + 1);
            calls.get() > 200
        })
        .is_err()
    );
    assert_eq!(a.snapshot().prepared_bytes, 0);
    let index = Arc::new(ExactIndex::build(&p, &a, &|| false).unwrap());
    assert_eq!(index.rows.len(), 43_264);
    assert!(index.charge() <= MAX_INDEX_BYTES);
    let retained = index.clone();
    drop(index);
    assert!(a.snapshot().prepared_bytes > 0);
    drop(retained);
    assert_eq!(a.snapshot().prepared_bytes, 0);
}

#[test]
fn missing_quality_is_unknown_and_all_authored_kinds_are_distinct() {
    assert_eq!(quality_label(None), "Unknown reference quality");
    assert_eq!(
        quality_label(Some(ReferenceKind::Learned)),
        "Learned reference"
    );
    assert_eq!(
        quality_label(Some(ReferenceKind::Heuristic)),
        "Heuristic benchmark"
    );
    assert_eq!(
        quality_label(Some(ReferenceKind::Analytical)),
        "Analytical benchmark"
    );
    assert_eq!(
        quality_label(Some(ReferenceKind::VerifiedOptimal)),
        "Verified optimal reference"
    );
}

fn context() -> ReferenceContext {
    use rl::{core::action::ActionSpace, diagnostics::value::ValueConventions};
    ReferenceContext {
        environment: "finite fixture".into(),
        observation_dim: 5,
        typed_observation_space: None,
        action_space: ActionSpace::Discrete { count: 2 },
        conventions: ValueConventions {
            dynamics: "fixture v1".into(),
            observation: "complete five coordinates".into(),
            action: "two decisions".into(),
            reward_units: "reward".into(),
            reward_convention: "larger better".into(),
            discount: 0.5,
            horizon: "64 decisions".into(),
            terminal_treatment: "zero terminal continuation".into(),
            action_constraints: "explicit mask".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 64 },
        environment_config: BTreeMap::from([
            ("horizon".into(), ConfigValue::Integer(64)),
            ("arrival".into(), ConfigValue::Real(0.25)),
        ]),
        markov_observation: true,
        decision_time_coordinate: Some(4),
    }
}
fn manifest() -> RunManifest {
    let c = context();
    let mut m = RunManifest::new("run", "fixture", "test", 5, c.action_space.clone(), 0.5).unwrap();
    m.environment = c.environment.clone();
    m.reward_convention = c.conventions.reward_convention.clone();
    attach_manifest_context(&mut m, &c).unwrap();
    m
}
fn declaration_for(
    role: &str,
    name: &str,
    finite: Option<ReferenceTableKind>,
    rows: usize,
) -> ReferenceDeclaration {
    ReferenceDeclaration {
        version: 1,
        run_id: "run".into(),
        revision: 0,
        model_role: role.into(),
        context: context(),
        sources: BTreeMap::from([(
            name.into(),
            ReferenceQuality {
                kind: if finite.is_some() {
                    ReferenceKind::VerifiedOptimal
                } else {
                    ReferenceKind::Learned
                },
                method: "fixture backward induction".into(),
                assumptions: vec!["declared finite fixture".into()],
            },
        )]),
        finite_table: finite.map(|kind| FiniteReferenceTable {
            probe_set: role.into(),
            rows,
            coverage: ReferenceCoverage::AllLiveStates,
            kind,
        }),
    }
}
fn scientific_chunks(role: &str, values: Vec<f32>, finite: bool) -> Vec<Chunk> {
    use rl::diagnostics::value::{
        self, ContinuationIdentity, CriticDeclaration, ValueCriticKind, ValueDeclaration,
    };
    let rows = values.len() / 5;
    let mut p = probes(values, vec![1; rows * 2]);
    p.context.identity.model_role = role.into();
    p.context.probe_set = Some(role.into());
    let Record::ProbeSet(meta) = &mut p.record else {
        unreachable!()
    };
    meta.id = role.into();
    let mut policy = Chunk {
        context: p.context.clone(),
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: Some("m".into()),
            stochastic: false,
        }),
        arrays: BTreeMap::from([
            (
                "a".into(),
                Array::u64(vec![rows], vec![u64::from(!finite); rows]).unwrap(),
            ),
            (
                "m".into(),
                Array::u8(vec![rows, 2], vec![1; rows * 2]).unwrap(),
            ),
        ]),
    };
    attach_declaration(
        &mut policy,
        &declaration_for(
            role,
            "controller",
            finite.then_some(ReferenceTableKind::DiscretePolicy),
            rows,
        ),
    )
    .unwrap();
    let mut critic = Chunk {
        context: p.context.clone(),
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Discrete { values: "q".into() },
            action_condition: ActionCondition::AllDiscreteActions,
            actions: None,
            continuation_policy: Some("greedy".into()),
            continuation_stochastic: Some(false),
        }),
        arrays: BTreeMap::from([(
            "q".into(),
            Array::f64(
                vec![rows, 2],
                (0..rows)
                    .flat_map(|row| {
                        if finite {
                            [row as f64 * 2., row as f64 * 2. + 1.]
                        } else {
                            [-1., 0.]
                        }
                    })
                    .collect(),
            )
            .unwrap(),
        )]),
    };
    value::attach_declaration(
        &mut critic,
        &ValueDeclaration {
            version: 1,
            run_id: "run".into(),
            revision: 0,
            model_role: role.into(),
            conventions: Some(context().conventions),
            critics: BTreeMap::from([(
                "q".into(),
                CriticDeclaration {
                    kind: ValueCriticKind::DiscreteQ,
                    learned_baseline: None,
                    actor: None,
                    continuation: Some(ContinuationIdentity {
                        identity: "greedy".into(),
                        stochastic: false,
                    }),
                    twin: None,
                    utility: None,
                },
            )]),
        },
    )
    .unwrap();
    attach_declaration(
        &mut critic,
        &declaration_for(
            role,
            "q",
            finite.then_some(ReferenceTableKind::OptimalDiscreteActionValues),
            rows,
        ),
    )
    .unwrap();
    vec![p, policy, critic]
}
fn recording() -> (tempfile::TempDir, crate::data::RunIndex) {
    use rl::diagnostics::storage::*;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("run");
    let mut writer = Recorder::create(&root, manifest(), StorageConfig::default()).unwrap();
    for c in scientific_chunks(
        "online",
        vec![0., 0., 0., 0., 0.375, 43263., 0., 0., 0., 0.375],
        false,
    )
    .into_iter()
    .chain(scientific_chunks(
        "oracle",
        (0..43264)
            .flat_map(|n| [n as f32, 0., 0., 0., 0.375])
            .collect(),
        true,
    )) {
        writer.append(&c).unwrap();
    }
    writer.finish().unwrap();
    (
        dir,
        crate::data::RunIndex::open(root, ReadLimits::default()).unwrap(),
    )
}
pub(crate) fn model_switch_recording() -> (tempfile::TempDir, crate::data::RunIndex) {
    use rand::SeedableRng;
    use rl::{
        diagnostics::{models::SnapshotBuilder, storage::*, value},
        nn::critics::scalar::DiscreteQ,
    };
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("run");
    let mut writer = Recorder::create(&root, manifest(), StorageConfig::default()).unwrap();
    let current = scientific_chunks(
        "online",
        vec![0., 0., 0., 0., 0.375, 1., 0., 0., 0., 0.375],
        false,
    );
    let mut conventions = value::declaration(&current[2]).unwrap().unwrap();
    let critic = conventions.critics.remove("q").unwrap();
    conventions.critics.insert("value".into(), critic);
    for c in current.into_iter().chain(scientific_chunks(
        "oracle",
        vec![0., 0., 0., 0., 0.375, 1., 0., 0., 0., 0.375],
        true,
    )) {
        writer.append(&c).unwrap();
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(8);
    let mut builder = SnapshotBuilder::new(
        "cp",
        RecordContext::new("run", 0),
        5,
        context().action_space,
    )
    .unwrap();
    builder
        .discrete("value", &DiscreteQ::new(5, 2, &[4], &mut rng).unwrap())
        .unwrap();
    let mut snapshot = builder.finish().unwrap();
    value::attach_declaration(&mut snapshot, &conventions).unwrap();
    attach_declaration(&mut snapshot, &declaration_for("online", "value", None, 0)).unwrap();
    writer.append(&snapshot).unwrap();
    writer.finish().unwrap();
    (
        dir,
        crate::data::RunIndex::open(root, ReadLimits::default()).unwrap(),
    )
}
pub(crate) fn state_value_reference_recording() -> (tempfile::TempDir, crate::data::RunIndex) {
    state_value_reference_recording_with_inputs(
        vec![0., 0., 0., 0., 0.375, 1., 0., 0., 0., 0.375],
        2,
        0,
    )
}
pub(crate) fn state_value_reference_recording_with_inputs(
    inputs: Vec<f32>,
    reference_rows: usize,
    manifest_padding: usize,
) -> (tempfile::TempDir, crate::data::RunIndex) {
    use rl::diagnostics::{storage::*, value};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("run");
    let mut manifest = manifest();
    manifest
        .metadata
        .insert("fixture-description".into(), "a".repeat(manifest_padding));
    let mut writer = Recorder::create(&root, manifest, StorageConfig::default()).unwrap();
    let rows = inputs.len() / 5;
    let mut current = scientific_chunks("online", inputs.clone(), false);
    let mut d = value::declaration(&current[2]).unwrap().unwrap();
    let mut critic = d.critics.remove("q").unwrap();
    critic.kind = value::ValueCriticKind::StateValue;
    critic.continuation = Some(value::ContinuationIdentity {
        identity: "learner-policy".into(),
        stochastic: true,
    });
    d.critics.insert("value".into(), critic);
    let mut state_value = Chunk {
        context: current[2].context.clone(),
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Scalar { values: "v".into() },
            action_condition: ActionCondition::StateValue,
            actions: None,
            continuation_policy: Some("learner-policy".into()),
            continuation_stochastic: Some(true),
        }),
        arrays: BTreeMap::from([(
            "v".into(),
            Array::f64(vec![rows, 1], (0..rows).map(|i| -(i as f64) - 2.).collect()).unwrap(),
        )]),
    };
    value::attach_declaration(&mut state_value, &d).unwrap();
    attach_declaration(
        &mut state_value,
        &declaration_for("online", "value", None, 0),
    )
    .unwrap();
    current[2] = state_value;
    let mut reference_inputs = inputs;
    reference_inputs.extend((rows..reference_rows).flat_map(|i| [i as f32, 0., 0., 0., 0.375]));
    for c in current
        .into_iter()
        .chain(scientific_chunks("oracle", reference_inputs, true))
    {
        writer.append(&c).unwrap();
    }
    writer.finish().unwrap();
    (
        dir,
        crate::data::RunIndex::open(root, ReadLimits::default()).unwrap(),
    )
}
fn frame(
    run: &crate::data::RunIndex,
    resolver: &mut crate::resolver::ProbeResolver,
    record: usize,
    probe: usize,
) -> crate::resolver::ResolvedRecord {
    let key = run.key(record).unwrap();
    let probe_key = run.key(probe).unwrap();
    crate::resolver::ResolvedRecord {
        chunk: resolver
            .lease_record(run, &key, Arc::new(run.recording().load(record).unwrap()))
            .unwrap(),
        probes: resolver
            .lease_record(
                run,
                &probe_key,
                Arc::new(run.recording().load(probe).unwrap()),
            )
            .unwrap(),
        key,
        probe_key,
        row: 0,
    }
}

#[test]
fn typed_contexts_reject_changed_configuration_and_real_discount_legacy_stays_compatible() {
    let m = manifest();
    let original = ReferenceInfo::new(&m, None).unwrap();
    let mut other = m.clone();
    let mut c = context();
    c.environment_config
        .insert("arrival".into(), ConfigValue::Real(0.75));
    other.metadata.insert(
        REFERENCE_CONTEXT_METADATA_KEY.into(),
        serde_json::to_string(&c).unwrap(),
    );
    assert!(
        original
            .compatible(&ReferenceInfo::new(&other, None).unwrap())
            .is_err()
    );
    let mut other = m.clone();
    other.discount = 0.9;
    assert!(ReferenceInfo::new(&other, None).is_err());
    let mut legacy = m.clone();
    legacy.metadata.clear();
    let unknown = ReferenceInfo::new(&legacy, None).unwrap();
    assert!(unknown.compatible(&original).is_ok());
    let table = scientific_chunks("oracle", vec![0., 0., 0., 0., 1.], true);
    let exact = ReferenceInfo::new(&m, Some(&table[1])).unwrap();
    assert!(unknown.compatible(&exact).is_err());
}

#[test]
fn privileged_provider_cannot_be_compared_as_a_same_information_overlay() {
    let manifest = manifest();
    let learner = ReferenceInfo::new(&manifest, None).unwrap();
    let mut policy =
        scientific_chunks("reference:privileged", vec![0., 0., 0., 0., 0.375], false).remove(1);
    let specification = ReferenceSpec {
        id: "privileged".into(),
        context: context(),
        quality: ReferenceQuality {
            kind: ReferenceKind::Heuristic,
            method: "privileged fixture".into(),
            assumptions: vec![],
        },
        target: ReferenceTarget::BenchmarkPolicy,
        lifetime: ReferenceLifetime::Fixed,
        information: ReferenceInformation::PrivilegedSimulatorState,
        uncertainty: None,
    };
    let bytes = serde_json::to_vec(&specification).unwrap();
    policy.arrays.insert(
        "metadata/reference-provider.v1".into(),
        Array::u8(vec![bytes.len()], bytes).unwrap(),
    );
    let reference = ReferenceInfo::new(&manifest, Some(&policy)).unwrap();
    let error = learner.compatible(&reference).unwrap_err().to_string();
    assert!(error.contains("privileged simulator"), "{error}");
}

#[test]
fn full_table_policy_and_value_use_only_bounded_current_rows_and_reuse_index() {
    use crate::{
        policy::{
            prepared::{self, PolicyPin, PolicyRequest, PolicySource},
            slice::{SliceConfig, SliceMode},
        },
        value::{
            prepared as vp,
            slice::{ValueCoordinate, ValueSlice},
        },
    };
    use rl::{
        core::action::Action,
        diagnostics::value::{ComparisonCondition, ValueOutput},
    };
    let (_dir, run) = recording();
    let mut resolver = crate::resolver::ProbeResolver::new(12 * 1024 * 1024);
    let current_frame = frame(&run, &mut resolver, 1, 0);
    let reference_frame = frame(&run, &mut resolver, 4, 3);
    let current = PolicySource::recorded(&run, &current_frame, "controller").unwrap();
    let reference = PolicySource::recorded_controller(&run, &reference_frame, "oracle").unwrap();
    let o = crate::policy::prepared::exact_observation(&current_frame.probes, 0).unwrap();
    let population = vec![
        o.clone(),
        crate::policy::prepared::exact_observation(&current_frame.probes, 1).unwrap(),
    ];
    let mut slice = SliceConfig::new(current.semantics(), &o, &population).unwrap();
    slice.mode = SliceMode::OneDimensional;
    slice.y = None;
    slice.x = 0;
    let r = PolicyRequest::new(
        current,
        Some(reference.clone()),
        0,
        slice.clone(),
        PolicyPin::Recorded {
            key: current_frame.key.clone(),
            probe_key: current_frame.probe_key.clone(),
            row: 1,
        },
        None,
    )
    .unwrap();
    let a = Arc::new(Accounting::default());
    let owner = prepared::RequestOwner::new(r, &a).unwrap();
    let prepared::Completion::Grid(grid) = prepared::execute(
        &prepared::Work::Grid(owner.clone()),
        &a,
        || false,
        |_, _| panic!("recorded table must not restore a model"),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert_eq!(grid.population_count, 2);
    assert_eq!(grid.values.len(), 2);
    assert_eq!(
        (grid.reference_at(0), grid.reference_at(1)),
        (Some(1.), Some(1.))
    );
    let index = reference
        .reference_info
        .index
        .lock()
        .unwrap()
        .as_ref()
        .unwrap()
        .clone();
    reference.prepare_reference(&a, &|| false).unwrap();
    assert!(Arc::ptr_eq(
        &index,
        reference
            .reference_info
            .index
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
    ));
    let prepared::Completion::Point(point) = prepared::execute(
        &prepared::Work::Point {
            request: owner,
            grid: grid.clone(),
            previous: None,
        },
        &a,
        || false,
        |_, _| panic!(),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert_eq!(point.reference_value, Some(1.));
    let full_as_current = PolicyRequest::new(
        reference.clone(),
        None,
        0,
        slice,
        PolicyPin::Recorded {
            key: reference_frame.key.clone(),
            probe_key: reference_frame.probe_key.clone(),
            row: 0,
        },
        None,
    )
    .unwrap();
    let full_owner = prepared::RequestOwner::new(full_as_current, &a).unwrap();
    assert!(
        prepared::execute(
            &prepared::Work::Grid(full_owner),
            &a,
            || false,
            |_, _| panic!()
        )
        .is_err()
    );
    drop(point);
    drop(grid);
    drop(reference);
    drop(index);
    drop(a);

    let current = vp::ValueSource::recorded(&run, &frame(&run, &mut resolver, 2, 0), "q").unwrap();
    let reference =
        vp::ValueSource::recorded(&run, &frame(&run, &mut resolver, 5, 3), "q").unwrap();
    let mut slice = ValueSlice::new(current.semantics(), &o, &population).unwrap();
    slice.mode = SliceMode::OneDimensional;
    slice.y = None;
    slice.x = ValueCoordinate::Observation(0);
    slice.state.mode = SliceMode::OneDimensional;
    slice.state.y = None;
    let mut r = vp::ValueRequest::new(
        current,
        ValueOutput::ExpectedReturn,
        slice,
        vp::ValuePin::Input {
            observation: population[1].clone(),
            action: Some(Action::Discrete(0)),
        },
    )
    .unwrap();
    r.reference_value = Some(reference.clone());
    let a = Arc::new(Accounting::default());
    let owner = vp::RequestOwner::new(r.clone(), &a).unwrap();
    let vp::Completion::Grid(grid) = vp::execute(
        &vp::Work::Grid(owner.clone()),
        &a,
        || false,
        &mut |_, _, _| panic!(),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert!(grid.reference_error.is_none(), "{:?}", grid.reference_error);
    assert_eq!(grid.values.len(), 2);
    assert!(grid.reference_series.is_none());
    assert_eq!(grid.reference_values, vec![Some(1.), Some(86527.)]);
    let vp::Completion::Point(point) = vp::execute(
        &vp::Work::Point {
            request: owner,
            grid: grid.clone(),
        },
        &a,
        || false,
        &mut |_, _, _| panic!(),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert_eq!(point.reference_value, Some(86526.)); // fixed action Q*, not V*
    r.condition = ComparisonCondition::ControllerValues;
    r.pin = vp::ValuePin::State(population[1].clone());
    let owner = vp::RequestOwner::new(r, &a).unwrap();
    let vp::Completion::Point(point) = vp::execute(
        &vp::Work::Point {
            request: owner,
            grid,
        },
        &a,
        || false,
        &mut |_, _, _| panic!(),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert_eq!(point.reference_value, Some(86527.)); // max legal Q*, independent of stored action 0
}

#[test]
fn full_table_is_one_async_source_and_role_changes_preserve_old_frame_leases() {
    use crate::{comparison::Comparison, data::Loader, resolver::ResolveStatus};
    use std::time::{Duration, Instant};
    let (_dir, run) = recording();
    let mut comparison = Comparison::new();
    comparison.reset_reference_to(&run).unwrap();
    let serial = comparison.reference_interaction_serial();
    assert!(comparison.set_reference_role("missing").is_err());
    assert_eq!(comparison.reference_interaction_serial(), serial);
    comparison.set_reference_role("oracle").unwrap();
    let mut loader = Loader::new(64 * 1024 * 1024, || {}).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let frame = loop {
        loader.poll();
        let result = comparison.policy_controller(&mut loader, None).unwrap();
        if result.status == ResolveStatus::Ready {
            break result.record.unwrap();
        }
        assert_eq!(
            result.status,
            ResolveStatus::Loading,
            "{:?}",
            result.warnings
        );
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(frame.key.sequence, 4);
    assert_eq!(frame.probe_key.sequence, 3);
    assert_eq!(Rows::new(&frame.probes).unwrap().count, 43_264);
    let source =
        crate::policy::prepared::PolicySource::recorded_controller(&run, &frame, "oracle").unwrap();
    let generation = Some(comparison.reference_interaction_serial());
    assert!(comparison.quality_source_is_current(generation, source.identity()));
    assert!(!comparison.quality_source_is_current(None, source.identity()));
    // Even choosing the same requested revision starts a new source-selection
    // generation: an older request cannot advertise its verified caption.
    comparison.set_reference_revision(0).unwrap();
    assert!(!comparison.quality_source_is_current(generation, source.identity()));
    comparison.set_reference_role("online").unwrap();
    assert!(!comparison.quality_source_is_current(
        Some(comparison.reference_interaction_serial()),
        source.identity()
    ));
    assert!(
        comparison.resident_bytes() > 0,
        "old frame raw arrays must remain charged"
    );
    drop(source);
    drop(frame);
    assert_eq!(comparison.resident_bytes(), 0);
}

#[test]
fn worker_generation_latest_pin_and_retirement_keep_index_with_external_frame() {
    use crate::{
        interactive::Interactive,
        value::{
            prepared::{ValuePin, ValueRequest, ValueSource},
            slice::ValueSlice,
        },
    };
    use rl::diagnostics::value::ValueOutput;
    use std::time::{Duration, Instant};
    let (_dir, run) = recording();
    let mut resolver = crate::resolver::ProbeResolver::new(12 * 1024 * 1024);
    let current = ValueSource::recorded(&run, &frame(&run, &mut resolver, 2, 0), "q").unwrap();
    let reference = ValueSource::recorded(&run, &frame(&run, &mut resolver, 5, 3), "q").unwrap();
    let input = Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
    let slice = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
    let mut request = ValueRequest::new(
        current,
        ValueOutput::ExpectedReturn,
        slice,
        ValuePin::State(input.clone()),
    )
    .unwrap();
    request.reference_value = Some(reference);
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.submit_value(request.clone()).unwrap();
    request.pin = ValuePin::State(Observation::dense(
        vec![43263., 0., 0., 0., 0.375],
        input.dense_mask().map(<[bool]>::to_vec),
    ));
    interactive.submit_value(request.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.value_pending() {
        interactive.poll_value().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        interactive.value_error().is_none(),
        "{:?}",
        interactive.value_error()
    );
    let held = interactive.value_frame().unwrap().grid.clone();
    assert_eq!(
        interactive
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_value,
        Some(86527.)
    );
    let info = &request.reference_value.as_ref().unwrap().reference_info;
    assert!(info.index.lock().unwrap().is_some());
    drop(request);
    interactive.retire_session();
    while interactive.retirement_status().0 != interactive.retirement_status().1 {
        interactive.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(interactive.value_metrics().prepared_bytes > 0);
    drop(held);
    assert_eq!(interactive.value_metrics().prepared_bytes, 0);
}

#[test]
fn immutable_model_metadata_is_bound_once_including_legacy_absence_and_restore_charge_retires() {
    use crate::{interactive::Interactive, policy::prepared::PolicySource};
    use rand::SeedableRng;
    use rl::{
        diagnostics::{models::SnapshotBuilder, storage::*},
        nn::critics::scalar::DiscreteQ,
    };
    use std::time::{Duration, Instant};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("run");
    let mut writer = Recorder::create(&root, manifest(), StorageConfig::default()).unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
    for declared in [true, false] {
        let mut builder = SnapshotBuilder::new(
            "cp",
            RecordContext::new("run", 0),
            5,
            context().action_space,
        )
        .unwrap();
        builder
            .discrete("q", &DiscreteQ::new(5, 2, &[4], &mut rng).unwrap())
            .unwrap();
        let mut chunk = builder.finish().unwrap();
        if declared {
            attach_declaration(&mut chunk, &declaration_for("online", "q", None, 0)).unwrap();
        }
        writer.append(&chunk).unwrap();
    }
    writer.finish().unwrap();
    let run = crate::data::RunIndex::open(root, ReadLimits::default()).unwrap();
    for index in 0..2 {
        let source = PolicySource::model(&run, index, Some("q")).unwrap();
        let unbound = source.clone();
        let source = source
            .bind_metadata(&run, || {
                Ok(Some(Arc::new(run.recording().load(index).unwrap())))
            })
            .unwrap()
            .unwrap();
        assert!(!source.same(&unbound));
        let retained = source.reference_info.clone();
        let metadata = source.shared_metadata();
        let source = source
            .bind_metadata(&run, || {
                panic!("an idle source must not read or decode metadata again")
            })
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&retained, &source.reference_info));
        assert_eq!(metadata, source.shared_metadata());
    }
    let expected = declaration(&run.recording().load(0).unwrap())
        .unwrap()
        .unwrap()
        .retained_charge();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.validate_snapshot(&run, 0).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = interactive.poll_snapshot_validation().unwrap() {
            result.unwrap();
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(interactive.policy_metrics().prepared_bytes >= expected);
    interactive.retire_session();
    while interactive.retirement_status().0 != interactive.retirement_status().1 {
        interactive.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(interactive.policy_metrics().prepared_bytes, 0);
}

#[test]
fn recorded_view_metadata_counts_allocations_private_handles_and_held_frames() {
    use crate::value::{
        ViewSourceCharges,
        prepared::{Completion, RequestOwner, ValuePin, ValueRequest, ValueSource, Work, execute},
        slice::ValueSlice,
    };
    use rl::diagnostics::value::ValueOutput;
    let (_dir, run) = state_value_reference_recording();
    let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
    let record = frame(&run, &mut resolver, 2, 0);
    let source = ValueSource::recorded(&run, &record, "value").unwrap();
    let clone = source.clone();
    let mut charges = ViewSourceCharges::default();
    assert_eq!(charges.charge(&source), source.metadata_bytes);
    let private = charges.charge(&clone);
    assert!(private >= std::mem::size_of::<ValueSource>());
    assert!(private < source.metadata_bytes);
    let mut larger = clone.clone();
    larger.names.reserve(5);
    larger.names[0].reserve(2048);
    larger.role.reserve(2048);
    larger.identity.name.reserve(2048);
    larger
        .actor
        .as_mut()
        .into_iter()
        .for_each(|s| s.reserve(2048));
    let crate::value::Data::Recorded(frames) = &mut larger.data else {
        unreachable!()
    };
    frames.reserve(5);
    assert!(charges.charge(&larger) > private + 6144);

    let independent = ValueSource::recorded(&run, &record, "value").unwrap();
    assert_eq!(source.identity(), independent.identity());
    assert_eq!(
        serde_json::to_string(source.manifest.as_ref()).unwrap(),
        serde_json::to_string(independent.manifest.as_ref()).unwrap()
    );
    assert_eq!(charges.charge(&independent), independent.metadata_bytes);
    let mut retained = vec![source.clone(), independent.clone()];
    for component in 0..3 {
        let mut partial = source.clone();
        match component {
            0 => partial.manifest = Arc::new((*partial.manifest).clone()),
            1 => {
                partial.declaration = partial
                    .declaration
                    .as_ref()
                    .map(|d| Arc::new((**d).clone()))
            }
            _ => {
                partial.reference_info =
                    ReferenceInfo::new(&partial.manifest, Some(&record.chunk)).unwrap()
            }
        }
        assert_eq!(charges.charge(&partial), partial.metadata_bytes);
        retained.push(partial);
    }
    // All unique allocations stay alive, including beyond the stack ledger's
    // bound. Overflow conservatively retains the full charge on repeat calls.
    for _ in 0..32 {
        let mut next = source.clone();
        next.manifest = Arc::new((*next.manifest).clone());
        assert_eq!(charges.charge(&next), next.metadata_bytes);
        retained.push(next);
    }
    let overflow = retained.last().unwrap();
    assert_eq!(charges.charge(overflow), overflow.metadata_bytes);

    // View estimates never remove prepared-owner charges. Independently
    // decoded replacement metadata overlaps the held old frame until drop.
    let accounting = Arc::new(Accounting::default());
    let prepare = |source: ValueSource| {
        let input = crate::policy::prepared::exact_observation(&record.probes, 0).unwrap();
        let slice = ValueSlice::new(source.semantics(), &input, &[]).unwrap();
        let request = ValueRequest::new(
            source,
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::State(input),
        )
        .unwrap();
        let owner = RequestOwner::new(request, &accounting).unwrap();
        let Completion::Grid(grid) = execute(
            &Work::Grid(owner),
            &accounting,
            || false,
            &mut |_, _, _| panic!(),
        )
        .unwrap()
        .unwrap() else {
            panic!()
        };
        grid
    };
    let held_old = prepare(source);
    let old_bytes = accounting.snapshot().prepared_bytes;
    let replacement = prepare(independent);
    assert!(accounting.snapshot().prepared_bytes > old_bytes);
    let overlap = accounting.snapshot().prepared_bytes;
    drop(replacement);
    assert!(accounting.snapshot().prepared_bytes < overlap);
    assert_eq!(accounting.snapshot().prepared_bytes, old_bytes);
    drop(held_old);
    assert_eq!(accounting.snapshot().prepared_bytes, 0);
}

#[test]
fn decision_loss_requires_current_policy_and_never_uses_critic_scalar() {
    use crate::value::{prepared as vp, slice::ValueSlice};
    use rl::diagnostics::value::ValueOutput;
    let (_dir, run) = recording();
    let mut resolver = crate::resolver::ProbeResolver::new(12 * 1024 * 1024);
    let current = vp::ValueSource::recorded(&run, &frame(&run, &mut resolver, 2, 0), "q").unwrap();
    let reference =
        vp::ValueSource::recorded(&run, &frame(&run, &mut resolver, 5, 3), "q").unwrap();
    let policy = crate::policy::prepared::PolicySource::recorded(
        &run,
        &frame(&run, &mut resolver, 1, 0),
        "controller",
    )
    .unwrap();
    let input = Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
    let slice = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
    let mut request = vp::ValueRequest::new(
        current,
        ValueOutput::ExpectedReturn,
        slice,
        vp::ValuePin::State(input),
    )
    .unwrap();
    request.output = ValueOutput::DecisionLoss;
    request.reference_value = Some(reference);
    request.current_policy = Some(policy);
    let a = Arc::new(Accounting::default());
    let owner = vp::RequestOwner::new(request, &a).unwrap();
    let vp::Completion::Grid(grid) = vp::execute(
        &vp::Work::Grid(owner.clone()),
        &a,
        || false,
        &mut |_, _, _| panic!("recorded policy needs no model"),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert_eq!(grid.values, vec![0.]);
    assert!(grid.reference_values.is_empty());
    let vp::Completion::Point(point) = vp::execute(
        &vp::Work::Point {
            request: owner,
            grid,
        },
        &a,
        || false,
        &mut |_, _, _| panic!(),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert_eq!(point.decision_loss.unwrap().loss, 0.);
    assert_eq!(point.decision_loss.unwrap().optimal_value, 1.);
}

pub(crate) mod decision_tests {
    use super::*;
    use crate::{
        interactive::Interactive,
        policy::prepared::PolicySource,
        value::{
            prepared as vp,
            slice::{ValueCoordinate, ValueSlice},
        },
    };
    use rand::SeedableRng;
    use rl::{
        diagnostics::{models::SnapshotBuilder, storage::*, value::ValueOutput},
        nn::{matrix, policy::Policy, vector},
    };
    use std::time::{Duration, Instant};

    pub(crate) fn fixture() -> (tempfile::TempDir, crate::data::RunIndex) {
        fixture_with_measurements(true)
    }
    pub(crate) fn fixture_with_measurements(
        at_ten: bool,
    ) -> (tempfile::TempDir, crate::data::RunIndex) {
        fixture_population(at_ten, false)
    }
    pub(crate) fn fixture_population(
        at_ten: bool,
        grid: bool,
    ) -> (tempfile::TempDir, crate::data::RunIndex) {
        fixture_population_context(at_ten, grid, false)
    }
    pub(crate) fn fixture_population_context(
        at_ten: bool,
        grid: bool,
        mismatched_oracle: bool,
    ) -> (tempfile::TempDir, crate::data::RunIndex) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("run");
        let mut manifest = manifest();
        let mut fixture_context = context();
        if mismatched_oracle {
            fixture_context
                .environment_config
                .insert("arrival".into(), ConfigValue::Real(0.75));
            manifest.metadata.insert(
                REFERENCE_CONTEXT_METADATA_KEY.into(),
                serde_json::to_string(&fixture_context).unwrap(),
            );
        }
        let mut writer = Recorder::create(&root, manifest, StorageConfig::default()).unwrap();
        let append = |writer: &mut Recorder, chunk: &Chunk| {
            let mut chunk = chunk.clone();
            if let Some(mut d) = declaration(&chunk).unwrap() {
                d.context = fixture_context.clone();
                chunk.arrays.remove(REFERENCE_METADATA_KEY);
                attach_declaration(&mut chunk, &d).unwrap();
            }
            writer.append(&chunk).unwrap();
        };
        let inputs: Vec<f32> = if grid {
            (0..169)
                .flat_map(|n| [(n % 13) as f32 / 12., (n / 13) as f32 / 12., 0., 0., 0.375])
                .collect()
        } else {
            vec![0., 0., 0., 0., 0.375, 1., 0., 0., 0., 0.375]
        };
        let rows = inputs.len() / 5;
        let mut current = scientific_chunks("online", inputs.clone(), false);
        if let Record::Policy(p) = &mut current[1].record {
            p.parameters = PolicyParameters::Categorical {
                probabilities: "p".into(),
            };
            p.stochastic = false;
        }
        current[1].arrays.insert(
            "p".into(),
            Array::f64(vec![rows, 2], [0.25, 0.75].repeat(rows)).unwrap(),
        );
        let mut changed_critic = current[2].clone();
        changed_critic.arrays.insert(
            "q".into(),
            Array::f64(vec![rows, 2], [999., -999.].repeat(rows)).unwrap(),
        );
        let mut oracle_inputs = inputs;
        oracle_inputs.extend([0., 0., 0., 0., 0.5, 0., 0., 1., 0., 0.375]);
        let mut oracle = scientific_chunks("oracle", oracle_inputs, true);
        let mut q = [-2., -5.].repeat(rows);
        q.extend([-10., -14., -1., -9.]);
        oracle[2]
            .arrays
            .insert("q".into(), Array::f64(vec![rows + 2, 2], q).unwrap());
        for c in current.into_iter().chain(oracle) {
            append(&mut writer, &c);
        }
        for (k, p) in [(0, 0.25_f32), (10, 0.75_f32)] {
            let mut policy = Policy::new(
                5,
                context().action_space,
                &[],
                &mut rand_chacha::ChaCha8Rng::seed_from_u64(k),
            )
            .unwrap();
            policy.network.layers[0].weight = policy.network.layers[0]
                .weight
                .clone()
                .map(|_| matrix(vec![0.; 10], 5, 2));
            policy.network.layers[0].bias = policy.network.layers[0]
                .bias
                .take()
                .map(|b| b.map(|_| vector(&[p.ln(), (1. - p).ln()])));
            let mut builder = SnapshotBuilder::new(
                &format!("cp-{k}"),
                RecordContext::new("run", k),
                5,
                context().action_space,
            )
            .unwrap();
            builder.policy("controller", &policy).unwrap();
            let mut chunk = builder.finish().unwrap();
            let mut d = declaration_for("online", "controller", None, 0);
            d.revision = k;
            attach_declaration(&mut chunk, &d).unwrap();
            append(&mut writer, &chunk);
        }
        append(&mut writer, &changed_critic);
        if at_ten {
            let mut at_ten = changed_critic;
            let mut value_d = rl::diagnostics::value::declaration(&at_ten)
                .unwrap()
                .unwrap();
            value_d.revision = 10;
            let mut ref_d = declaration(&at_ten).unwrap().unwrap();
            ref_d.revision = 10;
            at_ten
                .arrays
                .remove(rl::diagnostics::value::VALUE_METADATA_KEY);
            at_ten.arrays.remove(REFERENCE_METADATA_KEY);
            at_ten.context.identity.learner_update_revision = 10;
            at_ten.context.counters.updates = 10;
            rl::diagnostics::value::attach_declaration(&mut at_ten, &value_d).unwrap();
            attach_declaration(&mut at_ten, &ref_d).unwrap();
            append(&mut writer, &at_ten);
        }
        let mut trajectory_context = RecordContext::new("run", 0);
        trajectory_context.trajectory = Some("off-probe".into());
        writer
            .append(&Chunk {
                context: trajectory_context,
                record: Record::Trajectory(TrajectoryRecord {
                    render: None,
                    id: "off-probe".into(),
                    environment: "fixture".into(),
                    observations: "x".into(),
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
                        continuation_policy: "controller".into(),
                        stochastic_policy: false,
                        discount: 0.9,
                        independent_future_streams: true,
                        seed_methodology: "explicit".into(),
                        truncation_handling: "terminal".into(),
                        complete_returns: true,
                    },
                }),
                arrays: BTreeMap::from([
                    (
                        "x".into(),
                        Array::f32(vec![2, 5], vec![0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.])
                            .unwrap(),
                    ),
                    ("a".into(), Array::u64(vec![1], vec![0]).unwrap()),
                    ("r".into(), Array::f32(vec![1], vec![0.]).unwrap()),
                    ("t".into(), Array::u8(vec![1], vec![1]).unwrap()),
                    ("u".into(), Array::u8(vec![1], vec![0]).unwrap()),
                    ("m".into(), Array::u8(vec![2, 2], vec![1; 4]).unwrap()),
                ]),
            })
            .unwrap();
        writer.finish().unwrap();
        (
            dir,
            crate::data::RunIndex::open(root, ReadLimits::default()).unwrap(),
        )
    }
    pub(crate) fn request(
        run: &crate::data::RunIndex,
        resolver: &mut crate::resolver::ProbeResolver,
        critic: usize,
        policy: usize,
    ) -> vp::ValueRequest {
        let current =
            vp::ValueSource::recorded(run, &frame(run, resolver, critic, 0), "q").unwrap();
        let reference = vp::ValueSource::recorded(run, &frame(run, resolver, 5, 3), "q").unwrap();
        let current_policy = if policy == 1 {
            PolicySource::recorded(run, &frame(run, resolver, 1, 0), "caption is not subject")
                .unwrap()
        } else {
            PolicySource::model(run, policy, Some("controller"))
                .unwrap()
                .with_metadata(run, &run.recording().load(policy).unwrap())
                .unwrap()
        };
        let input = Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
        let mut slice = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
        slice.x = ValueCoordinate::Observation(0);
        slice.y = None;
        slice.mode = crate::policy::slice::SliceMode::OneDimensional;
        slice.resolution = [16, 1];
        slice.state.ranges[0] = [0., 1.];
        let mut r = vp::ValueRequest::new(
            current,
            ValueOutput::DecisionLoss,
            slice,
            vp::ValuePin::Input {
                observation: input,
                action: Some(rl::core::action::Action::Discrete(0)),
            },
        )
        .unwrap();
        r.requested_revision = current_policy.identity().revision;
        r.reference_value = Some(reference);
        r.current_policy = Some(current_policy);
        r
    }
    fn settle(i: &mut Interactive) {
        let end = Instant::now() + Duration::from_secs(15);
        while i.value_pending() {
            i.poll_value().unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(i.value_error().is_none(), "{:?}", i.value_error());
    }
    #[test]
    fn categorical_parameters_override_argmax_and_critic_values_are_irrelevant() {
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let mut i = Interactive::new(|| {}).unwrap();
        for critic in [2, 8] {
            let r = request(&run, &mut resolver, critic, 1);
            i.submit_value(r).unwrap();
            settle(&mut i);
            let f = i.value_frame().unwrap();
            assert_eq!(f.grid.values, vec![2.25, 2.25]);
            let p = f.selected.as_ref().unwrap();
            assert_eq!(p.value, 2.25);
            assert_eq!(p.decision_loss.unwrap().expected_action_value, -4.25);
            assert_eq!(p.decision_loss.unwrap().optimal_value, -2.);
            assert!(p.profile.is_none());
            assert!(p.reference_value.is_none());
        }
    }
    #[test]
    fn actual_checkpoint_changes_loss_reuses_oracle_and_off_probe_pin_reuses_grid() {
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let mut r = request(&run, &mut resolver, 2, 6);
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        settle(&mut i);
        let old = i.value_frame().unwrap().clone();
        assert!((old.selected.as_ref().unwrap().value - 2.25).abs() < 1e-6);
        let index = r
            .reference_value
            .as_ref()
            .unwrap()
            .reference_info
            .index
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        let old_charge = i.value_metrics().prepared_bytes;
        let p = PolicySource::model(&run, 7, Some("controller"))
            .unwrap()
            .with_metadata(&run, &run.recording().load(7).unwrap())
            .unwrap();
        r.current_policy = Some(p);
        r.requested_revision = 10;
        i.submit_value(r.clone()).unwrap();
        assert!((i.value_frame().unwrap().selected.as_ref().unwrap().value - 2.25).abs() < 1e-6);
        settle(&mut i);
        assert!((i.value_frame().unwrap().selected.as_ref().unwrap().value - 0.75).abs() < 1e-6);
        assert_eq!(
            i.value_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .current_policy()
                .unwrap()
                .identity()
                .revision,
            10
        );
        assert!(Arc::ptr_eq(
            &index,
            r.reference_value
                .as_ref()
                .unwrap()
                .reference_info
                .index
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
        ));
        let grid = i.value_frame().unwrap().grid.clone();
        let batches = i.value_metrics().grid_batches;
        r.pin = vp::ValuePin::State(Observation::dense(
            vec![0., 0., 0., 0., 0.5],
            Some(vec![true, true]),
        ));
        i.submit_value(r.clone()).unwrap();
        settle(&mut i);
        assert!(Arc::ptr_eq(&grid, &i.value_frame().unwrap().grid));
        assert_eq!(batches, i.value_metrics().grid_batches);
        assert!((i.value_frame().unwrap().selected.as_ref().unwrap().value - 1.).abs() < 1e-6);
        assert_eq!(
            i.value_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .decision_loss
                .unwrap()
                .optimal_value,
            -10.
        );
        let mut slice = (*r.slice).clone();
        slice.state.fixed[4] = 0.5;
        r.slice = Arc::new(slice);
        i.submit_value(r.clone()).unwrap();
        settle(&mut i);
        assert!(i.value_frame().unwrap().grid.values.is_empty());
        assert!(i.value_frame().unwrap().selected.is_some());
        assert!(i.value_metrics().prepared_bytes > old_charge);
        drop(r);
        i.retire_session();
        let end = Instant::now() + Duration::from_secs(10);
        while i.retirement_status().0 != i.retirement_status().1 {
            i.housekeeping();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(i.value_metrics().prepared_bytes > 0);
        drop(old);
        drop(grid);
        drop(index);
        assert_eq!(i.value_metrics().prepared_bytes, 0);
    }
    #[test]
    fn replacement_oracle_payload_retires_previous_loss_and_index() {
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let mut r = request(&run, &mut resolver, 2, 1);
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        settle(&mut i);
        let old = i.value_frame().unwrap().clone();
        let old_info = r.reference_value.as_ref().unwrap().reference_info.clone();
        let mut replacement = frame(&run, &mut resolver, 5, 3);
        let mut chunk = run.recording().load(5).unwrap();
        chunk.arrays.insert(
            "q".into(),
            Array::f64(vec![4, 2], vec![-2., -10., -2., -10., -10., -14., -1., -9.]).unwrap(),
        );
        replacement.chunk = resolver
            .lease_record(&run, &replacement.key, Arc::new(chunk))
            .unwrap();
        r.reference_value = Some(vp::ValueSource::recorded(&run, &replacement, "q").unwrap());
        i.submit_value(r.clone()).unwrap();
        settle(&mut i);
        assert_eq!(old.selected.as_ref().unwrap().value, 2.25);
        assert_eq!(
            i.value_frame().unwrap().selected.as_ref().unwrap().value,
            6.
        );
        assert!(
            r.reference_value
                .as_ref()
                .unwrap()
                .reference_info
                .prepared()
        );
        assert!(!Arc::ptr_eq(
            &old_info,
            &r.reference_value.as_ref().unwrap().reference_info
        ));
    }
    #[test]
    fn derived_fast_path_rejects_changed_requested_checkpoint_and_withdraws_old_frame() {
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let mut r = request(&run, &mut resolver, 2, 6);
        let mut i = Interactive::new(|| {}).unwrap();
        i.submit_value(r.clone()).unwrap();
        settle(&mut i);
        assert!(i.value_frame().is_some());
        r.requested_revision = 9;
        assert!(i.submit_value(r).is_err());
        assert!(i.value_frame().is_none());
    }
    #[test]
    fn missing_exact_context_mask_time_policy_or_oracle_refuses_loss() {
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let base = request(&run, &mut resolver, 2, 6);
        let mut i = Interactive::new(|| {}).unwrap();
        for mode in 0..8 {
            let mut r = base.clone();
            match mode {
                0 => r.current_policy = None,
                1 => r.reference_value = None,
                2 => r.requested_revision = 9,
                _ => {
                    let mut o =
                        Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
                    match mode {
                        3 => o.as_dense_mut().expect("dense diagnostic input").values[2] = 0.5,
                        4 => o.as_dense_mut().expect("dense diagnostic input").values[3] = 1.,
                        5 => o.as_dense_mut().expect("dense diagnostic input").values[4] = 0.4,
                        6 => {
                            o.as_dense_mut()
                                .expect("dense diagnostic input")
                                .action_mask = Some(vec![true, false])
                        }
                        _ => {
                            o.as_dense_mut().expect("dense diagnostic input").values[4] = 0.;
                            o.as_dense_mut()
                                .expect("dense diagnostic input")
                                .action_mask = Some(vec![false, false]);
                        }
                    }
                    r.pin = vp::ValuePin::State(o);
                }
            }
            let submitted = i.submit_value(r);
            if submitted.is_ok() {
                let end = Instant::now() + Duration::from_secs(10);
                while i.value_pending() {
                    i.poll_value().unwrap();
                    assert!(Instant::now() < end);
                    std::thread::sleep(Duration::from_millis(1));
                }
                assert!(i.value_error().is_some(), "case {mode}");
            }
            assert!(
                i.value_frame().is_none(),
                "case {mode} must withdraw derived output"
            );
        }
    }
    #[test]
    fn exact_slice_does_not_mix_nearby_times_and_cancellation_never_publishes() {
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let mut r = request(&run, &mut resolver, 2, 1);
        let mut slice = (*r.slice).clone();
        slice.state.fixed[4] = 0.374;
        slice.state.tolerance = 1.;
        r.slice = Arc::new(slice);
        let a = Arc::new(Accounting::default());
        let owner = vp::RequestOwner::new(r, &a).unwrap();
        assert!(
            vp::execute(
                &vp::Work::Grid(owner.clone()),
                &a,
                || true,
                &mut |_, _, _| panic!()
            )
            .unwrap()
            .is_none()
        );
        let vp::Completion::Grid(grid) = vp::execute(
            &vp::Work::Grid(owner.clone()),
            &a,
            || false,
            &mut |_, _, _| panic!(),
        )
        .unwrap()
        .unwrap() else {
            panic!()
        };
        assert!(grid.values.is_empty());
        drop(owner);
        drop(grid);
        assert_eq!(a.snapshot().prepared_bytes, 0);
    }
    #[test]
    fn derived_controls_restore_original_axes_conditioning_and_display() {
        use crate::{
            policy::{slice::SliceMode, view::Display},
            value::{slice::ActionConditioning, view::ValueView},
        };
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let r = request(&run, &mut resolver, 2, 1);
        for display in [Display::Current, Display::Reference, Display::Difference] {
            for mode in [SliceMode::OneDimensional, SliceMode::TwoDimensional] {
                let mut view = ValueView::new();
                view.display = display;
                let mut c = (*r.slice).clone();
                c.x = ValueCoordinate::DiscreteAction;
                c.mode = mode;
                c.y =
                    (mode == SliceMode::TwoDimensional).then_some(ValueCoordinate::Observation(1));
                c.resolution = if c.y.is_some() { [13, 13] } else { [16, 1] };
                c.conditioning = ActionConditioning::Fixed(rl::core::action::Action::Discrete(1));
                let before = c.clone();
                view.sync_decision_controls(&mut c, &ValueOutput::DecisionLoss);
                c.validate(r.current.semantics()).unwrap();
                assert!(c.state_only());
                assert_eq!(view.display, Display::Current);
                assert_eq!(c.conditioning, ActionConditioning::Policy);
                c.mode = if mode == SliceMode::OneDimensional {
                    SliceMode::TwoDimensional
                } else {
                    SliceMode::OneDimensional
                };
                c.y = (c.mode == SliceMode::TwoDimensional)
                    .then_some(ValueCoordinate::Observation(1));
                c.resolution = if c.y.is_some() { [17, 17] } else { [19, 1] };
                view.sync_decision_controls(&mut c, &ValueOutput::ExpectedReturn);
                assert_eq!(c, before);
                c.validate(r.current.semantics()).unwrap();
                assert_eq!(view.display, display);
            }
        }
        // A state/action plot may have no second state coordinate available.
        let mut c = (*r.slice).clone();
        c.state.y = None;
        c.state.mode = SliceMode::OneDimensional;
        c.state.resolution[1] = 1;
        c.mode = SliceMode::TwoDimensional;
        c.x = ValueCoordinate::DiscreteAction;
        c.y = Some(ValueCoordinate::Observation(0));
        let mut view = ValueView::new();
        view.sync_decision_controls(&mut c, &ValueOutput::DecisionLoss);
        assert_eq!(c.mode, SliceMode::OneDimensional);
        c.validate(r.current.semantics()).unwrap();
    }
    #[test]
    fn derived_controls_restore_does_not_reinstate_now_illegal_fixed_action() {
        use crate::value::{slice::ActionConditioning, view::ValueView};
        let (_d, run) = fixture();
        let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
        let r = request(&run, &mut resolver, 2, 1);
        let mut c = (*r.slice).clone();
        c.conditioning = ActionConditioning::Fixed(rl::core::action::Action::Discrete(1));
        let mut view = ValueView::new();
        view.sync_decision_controls(&mut c, &ValueOutput::DecisionLoss);
        c.state = crate::policy::slice::SliceConfig::new(
            r.current.semantics(),
            &Observation::dense(vec![0., 0., 0., 1., 0.375], Some(vec![true, false])),
            &[],
        )
        .unwrap();
        view.sync_decision_controls(&mut c, &ValueOutput::ExpectedReturn);
        assert_eq!(c.conditioning, ActionConditioning::Policy);
        c.validate(r.current.semantics()).unwrap();
    }
}
