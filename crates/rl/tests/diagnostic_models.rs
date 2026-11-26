#![cfg(all(feature = "diagnostics", feature = "neural"))]
use burn::{
    module::Param,
    tensor::{Device, Tensor, TensorData},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{models::*, schema::*},
    nn::{
        critics::scalar::QValue,
        matrix,
        network::{Mlp, Network},
        policy::Deterministic,
        probes::{IntervenedMlp, Intervention, InterventionValue},
        to_vec, vector,
    },
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

struct PackingAllocations;
thread_local! {
    static PACKING_SIZES: Cell<Option<(usize, usize, usize, usize)>> = const { Cell::new(None) };
    static CAPTURE_LIFETIME: Cell<Option<CaptureLifetime>> = const { Cell::new(None) };
}
#[derive(Clone, Copy, Debug, Default)]
struct CaptureLifetime {
    joined_size: usize,
    later_size: usize,
    joined_pointer: Option<usize>,
    joined_live: bool,
    joined_reallocated: bool,
    later_calls: usize,
    overlapping_calls: usize,
}
fn record_capture_allocation(pointer: *mut u8, size: usize) {
    let _ = CAPTURE_LIFETIME.try_with(|state| {
        if let Some(mut trace) = state.get() {
            if size == trace.joined_size && trace.joined_pointer.is_none() {
                trace.joined_pointer = Some(pointer as usize);
                trace.joined_live = true;
            }
            if size == trace.later_size {
                trace.later_calls += 1;
                trace.overlapping_calls += usize::from(trace.joined_live);
            }
            state.set(Some(trace));
        }
    });
}
fn record_capture_release(pointer: *mut u8, reallocated: bool) {
    let _ = CAPTURE_LIFETIME.try_with(|state| {
        if let Some(mut trace) = state.get() {
            if trace.joined_live && trace.joined_pointer == Some(pointer as usize) {
                trace.joined_live = false;
                trace.joined_reallocated = reallocated;
            }
            state.set(Some(trace));
        }
    });
}
fn capture_lifetime<T>(joined_size: usize, later_size: usize, f: impl FnOnce() -> T) -> (T, usize) {
    CAPTURE_LIFETIME.with(|state| {
        state.set(Some(CaptureLifetime {
            joined_size,
            later_size,
            ..Default::default()
        }))
    });
    let result = f();
    let trace = CAPTURE_LIFETIME.with(|state| state.replace(None).unwrap());
    assert!(
        trace.joined_pointer.is_some(),
        "fixture must allocate the shared joined payload"
    );
    assert!(!trace.joined_live, "shared joined payload must be released");
    assert!(
        !trace.joined_reallocated,
        "fixture expects an unchanged joined backing allocation"
    );
    assert!(
        trace.later_calls > 0,
        "fixture must allocate later-layer activation payloads"
    );
    (result, trace.overlapping_calls)
}
fn record_packing_allocation(size: usize) {
    let _ = PACKING_SIZES.try_with(|counts| {
        if let Some((observation, joined, o, j)) = counts.get() {
            counts.set(Some((
                observation,
                joined,
                o + usize::from(size == observation),
                j + usize::from(size == joined),
            )));
        }
    });
}
// SAFETY: System receives every original pointer/layout. The thread-local
// observer records requested sizes and pointer identities without allocating
// or dereferencing the observed pointers.
unsafe impl GlobalAlloc for PackingAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_packing_allocation(layout.size());
            record_capture_allocation(pointer, layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_packing_allocation(layout.size());
            record_capture_allocation(pointer, layout.size());
        }
        pointer
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, size) };
        if !replacement.is_null() {
            record_capture_release(pointer, true);
            record_packing_allocation(size);
            record_capture_allocation(replacement, size);
        }
        replacement
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_capture_release(pointer, false);
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static PACKING_ALLOCATOR: PackingAllocations = PackingAllocations;

fn assert_chunk_bits(actual: &Chunk, expected: &Chunk) {
    assert_eq!(
        serde_json::to_value(&actual.context).unwrap(),
        serde_json::to_value(&expected.context).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&actual.record).unwrap(),
        serde_json::to_value(&expected.record).unwrap()
    );
    assert_eq!(
        actual.arrays.keys().collect::<Vec<_>>(),
        expected.arrays.keys().collect::<Vec<_>>()
    );
    for (name, a) in &actual.arrays {
        let b = &expected.arrays[name];
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
        match a.dtype() {
            DType::F32 => assert_eq!(
                a.as_f32()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                b.as_f32()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            ),
            DType::F64 => assert_eq!(
                a.as_f64()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                b.as_f64()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            ),
            DType::U64 => assert_eq!(a.as_u64().unwrap(), b.as_u64().unwrap()),
            DType::U8 => assert_eq!(a.as_u8().unwrap(), b.as_u8().unwrap()),
        }
    }
}
fn assert_prediction_bits(actual: &InspectionResult, expected: &InspectionResult) {
    assert_eq!(actual.actions.len(), expected.actions.len());
    for (a, b) in actual.actions.iter().zip(&expected.actions) {
        match (a, b) {
            (Action::Continuous(a), Action::Continuous(b)) => assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            ),
            _ => assert_eq!(a, b),
        }
    }
    assert_chunk_bits(&actual.probes, &expected.probes);
    assert_chunk_bits(&actual.policy, &expected.policy);
    assert_eq!(
        actual.critics.keys().collect::<Vec<_>>(),
        expected.critics.keys().collect::<Vec<_>>()
    );
    for (name, chunk) in &actual.critics {
        assert_chunk_bits(chunk, &expected.critics[name]);
    }
}

// Full capture retains the original allocating path. Every other request must
// preserve its complete observable predictions and exactly its requested keys.
fn assert_capture_variants(bundle: &ModelBundle, request: &InspectionRequest) {
    let mut full = request.clone();
    for (name, layout) in bundle.activation_networks() {
        if name == request.policy
            || request.critics.iter().any(|c| {
                name == *c || name == format!("{c}.first") || name == format!("{c}.second")
            })
        {
            full.layers
                .insert(name, layout.layers().into_iter().map(|(n, _)| n).collect());
        }
    }
    let reference = bundle.inspect(&full).unwrap();
    let mut missing = request.clone();
    missing.layers.clear();
    let actual = bundle.inspect(&missing).unwrap();
    assert_prediction_bits(&actual, &reference);
    assert!(actual.activations.is_empty());
    for (name, layers) in &full.layers {
        let mut subset = missing.clone();
        subset.layers.insert(name.clone(), layers.clone());
        let actual = bundle.inspect(&subset).unwrap();
        assert_prediction_bits(&actual, &reference);
        assert_eq!(actual.activations.keys().collect::<Vec<_>>(), vec![name]);
        assert_chunk_bits(&actual.activations[name], &reference.activations[name]);
        subset.layers.insert(name.clone(), vec![]);
        let actual = bundle.inspect(&subset).unwrap();
        assert_prediction_bits(&actual, &reference);
        assert_eq!(actual.activations.keys().collect::<Vec<_>>(), vec![name]);
        let Record::Activations(record) = &actual.activations[name].record else {
            panic!("activation record")
        };
        assert!(record.layers.is_empty());
        assert!(actual.activations[name].arrays.is_empty());
    }
}

fn network() -> Mlp {
    let mut m = Mlp::new(2, &[2], 1, &mut ChaCha8Rng::seed_from_u64(1)).unwrap();
    m.layers[0].weight = Param::from_tensor(matrix(vec![1., 0., 0., 1.], 2, 2));
    m.layers[0].bias = Some(Param::from_tensor(vector(&[0., 0.])));
    m.layers[1].weight = Param::from_tensor(matrix(vec![1., 2.], 2, 1));
    m.layers[1].bias = Some(Param::from_tensor(vector(&[0.])));
    m
}
fn parameter_bits(network: &Mlp) -> Vec<Vec<u32>> {
    network
        .layers
        .iter()
        .flat_map(|layer| {
            std::iter::once(to_vec(layer.weight.val()).unwrap())
                .chain(layer.bias.as_ref().map(|bias| to_vec(bias.val()).unwrap()))
        })
        .map(|values| values.into_iter().map(f32::to_bits).collect())
        .collect()
}
#[test]
fn metadata_validation_needs_no_weight_payload_and_checks_policy_input_width() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let policy = Deterministic::from_network(network(), space.clone()).unwrap();
    let mut builder =
        SnapshotBuilder::new("metadata", RecordContext::new("models", 7), 2, space).unwrap();
    builder.deterministic("real-policy", &policy).unwrap();
    let mut chunk = builder.finish().unwrap();
    chunk.arrays.clear();
    let Record::ModelSnapshot(meta) = &mut chunk.record else {
        panic!()
    };
    assert!(snapshot_configuration(meta, ModelLimits::default()).is_ok());
    meta.configuration["models"]["real-policy"]["network"]["widths"][0] = serde_json::json!(3);
    assert!(snapshot_configuration(meta, ModelLimits::default()).is_err());
}
#[test]
fn metadata_preflight_rejects_layout_format_names_and_budgets_before_restoration() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let policy = Deterministic::from_network(network(), space.clone()).unwrap();
    let mut builder =
        SnapshotBuilder::new("metadata", RecordContext::new("models", 7), 2, space).unwrap();
    builder.deterministic("policy", &policy).unwrap();
    let chunk = builder.finish().unwrap();
    let Record::ModelSnapshot(meta) = &chunk.record else {
        panic!()
    };
    let reject = |change: fn(&mut ModelSnapshotRecord)| {
        let mut meta = meta.clone();
        change(&mut meta);
        assert!(snapshot_configuration(&meta, ModelLimits::default()).is_err());
    };
    reject(|m| m.architecture = "other".into());
    reject(|m| m.configuration["format_version"] = serde_json::json!(99));
    reject(|m| m.configuration["pack_format"] = serde_json::json!("other"));
    reject(|m| {
        m.configuration["models"]["policy"]["network"]["widths"] = serde_json::json!(vec![1; 66])
    });
    reject(|m| {
        m.configuration["models"]["policy"]["network"]["widths"] = serde_json::json!([2, 2, 2])
    });
    reject(|m| {
        m.models.remove("policy");
    });
    reject(|m| {
        m.configuration["action_space"] =
            serde_json::to_value(ActionSpace::Discrete { count: 1 }).unwrap()
    });
    reject(|m| {
        let definition = m.configuration["models"]["policy"].clone();
        m.configuration["models"] = serde_json::json!({" invalid ":definition});
    });
    assert!(
        snapshot_configuration(
            meta,
            ModelLimits {
                max_parameters: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        snapshot_configuration(
            meta,
            ModelLimits {
                max_model_bytes: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}
#[test]
fn viewer_retention_limit_does_not_become_an_inference_metadata_limit() {
    let mut builder = SnapshotBuilder::new(
        "large-metadata",
        RecordContext::new("models", 1),
        2,
        ActionSpace::Unbounded { dimensions: 1 },
    )
    .unwrap();
    builder
        .opaque("external", &network(), &"x".repeat(70 * 1024))
        .unwrap();
    let chunk = builder.finish().unwrap();
    let Record::ModelSnapshot(meta) = &chunk.record else {
        panic!()
    };
    assert!(snapshot_configuration(meta, ModelLimits::default()).is_ok());
    assert!(ModelBundle::load(&chunk, ModelLimits::default()).is_ok());
}
#[test]
fn pre_and_post_interventions_have_distinct_meaning_and_never_change_saved_weights() {
    let m = network();
    let x = matrix(vec![1., 2.], 1, 2);
    assert_eq!(to_vec(m.forward(x.clone())).unwrap(), vec![5.]);
    let pre = IntervenedMlp::new(
        &m,
        vec![Intervention {
            layer: "layers.0.pre".into(),
            neuron: 1,
            value: InterventionValue::Add(-3.),
        }],
    )
    .unwrap();
    assert_eq!(to_vec(pre.forward(x.clone())).unwrap(), vec![1.]);
    let post = IntervenedMlp::new(
        &m,
        vec![Intervention {
            layer: "layers.0.post".into(),
            neuron: 1,
            value: InterventionValue::Set(-1.),
        }],
    )
    .unwrap();
    assert_eq!(to_vec(post.forward(x.clone())).unwrap(), vec![-1.]);
    assert_eq!(to_vec(m.forward(x)).unwrap(), vec![5.]);
    assert!(
        IntervenedMlp::new(
            &m,
            vec![Intervention {
                layer: "layers.9.post".into(),
                neuron: 0,
                value: InterventionValue::Set(0.)
            }]
        )
        .is_err()
    );
}
#[test]
fn model_only_snapshot_reconstructs_policy_critic_and_counterfactuals() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let policy = Deterministic::from_network(network(), space.clone()).unwrap();
    let mut critic = QValue::new(2, 1, 1, &[], &mut ChaCha8Rng::seed_from_u64(2)).unwrap();
    critic.network.layers[0].weight = Param::from_tensor(matrix(vec![1., 0., 2.], 3, 1));
    let source_parameters = (
        parameter_bits(&policy.network),
        parameter_bits(&critic.network),
    );
    let mut builder =
        SnapshotBuilder::new("model-k7", RecordContext::new("models", 7), 2, space).unwrap();
    builder.deterministic("policy", &policy).unwrap();
    builder.scalar("critic", &critic).unwrap();
    let chunk = builder.finish().unwrap();
    let mut invalid = chunk.clone();
    if let Record::ModelSnapshot(meta) = &mut invalid.record {
        meta.configuration["observation_dim"] = serde_json::json!(usize::MAX);
    }
    assert!(ModelBundle::load(&invalid, ModelLimits::default()).is_err());
    let Record::ModelSnapshot(meta) = &chunk.record else {
        panic!()
    };
    assert_eq!(meta.models.len(), 2);
    assert!(chunk.array_bytes() > 0);
    assert!(serde_json::to_string(meta).unwrap().len() < 10_000);
    let loaded = ModelBundle::load(&chunk, ModelLimits::default()).unwrap();
    let mut request = InspectionRequest::new("policy", vec![Observation::new(vec![1., 2.])]);
    request.critics = vec!["critic".into()];
    request.layers.insert(
        "policy".into(),
        vec!["layers.0.pre".into(), "layers.0.post".into()],
    );
    let baseline = loaded.inspect(&request).unwrap();
    let actor = loaded.evaluation_actor("policy").unwrap();
    assert_eq!(
        actor
            .act(
                &request.observations,
                rl::core::actor::ActMode::Deterministic,
                &mut ChaCha8Rng::seed_from_u64(request.seed),
            )
            .unwrap(),
        baseline.actions
    );
    assert!(loaded.evaluation_actor("critic").is_err());
    assert!(loaded.evaluation_actor("missing").is_err());
    assert_capture_variants(&loaded, &request);
    assert_eq!(baseline.actions, vec![Action::Continuous(vec![5.])]);
    let Record::Critic(c) = &baseline.critics["critic"].record else {
        panic!()
    };
    let CriticRepresentation::Scalar { values } = &c.representation else {
        panic!()
    };
    assert_eq!(
        baseline.critics["critic"].arrays[values].as_f32().unwrap(),
        &[11.]
    );
    request.interventions.insert(
        "policy".into(),
        vec![Intervention {
            layer: "layers.0.post".into(),
            neuron: 1,
            value: InterventionValue::Set(-1.),
        }],
    );
    let changed = loaded.inspect(&request).unwrap();
    assert_eq!(changed.actions, vec![Action::Continuous(vec![-1.])]);
    let Record::Critic(c) = &changed.critics["critic"].record else {
        panic!()
    };
    let CriticRepresentation::Scalar { values } = &c.representation else {
        panic!()
    };
    assert_eq!(
        changed.critics["critic"].arrays[values].as_f32().unwrap(),
        &[-1.]
    );
    request.layers.clear();
    let intervention_only = loaded.inspect(&request).unwrap();
    assert_prediction_bits(&intervention_only, &changed);
    assert!(intervention_only.activations.is_empty());
    request.interventions.clear();
    assert_prediction_bits(&loaded.inspect(&request).unwrap(), &baseline);
    assert_eq!(
        (
            parameter_bits(&policy.network),
            parameter_bits(&critic.network)
        ),
        source_parameters
    );
    request.observations[0].as_dense_mut().unwrap().values[0] = 2.;
    assert_eq!(
        loaded.inspect(&request).unwrap().actions,
        vec![Action::Continuous(vec![6.])]
    );
    request.observations[0].as_dense_mut().unwrap().values[0] = f32::NAN;
    assert!(loaded.inspect(&request).is_err());
    let mut corrupt = chunk.clone();
    if let Record::ModelSnapshot(m) = &mut corrupt.record {
        m.architecture = "unknown-v999".into();
    }
    assert!(ModelBundle::load(&corrupt, ModelLimits::default()).is_err());
    assert!(
        ModelBundle::load(
            &chunk,
            ModelLimits {
                max_parameters: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn all_builtin_snapshot_families_validate_inputs_and_preserve_predictions() {
    use rl::{
        core::actor::ActMode,
        nn::{
            critics::{
                quantile::QuantileGrid,
                scalar::{DiscreteQ, Value},
            },
            policy::Policy,
        },
    };
    let mut rng = ChaCha8Rng::seed_from_u64(55);
    for discrete in [true, false] {
        let space = if discrete {
            ActionSpace::Discrete { count: 3 }
        } else {
            ActionSpace::Box {
                low: vec![-2.],
                high: vec![2.],
            }
        };
        let mut builder =
            SnapshotBuilder::new("all", RecordContext::new("test", 0), 2, space.clone()).unwrap();
        let mut policy = Policy::new(2, space.clone(), &[4], &mut rng).unwrap();
        policy.log_std = Param::from_tensor(vector(&[-0.8]));
        builder.policy("policy", &policy).unwrap();
        builder
            .value("value", &Value::new(2, &[4], &mut rng).unwrap())
            .unwrap();
        let mut request = InspectionRequest::new(
            "policy",
            vec![Observation::dense(
                vec![0.2, 0.5],
                discrete.then(|| vec![false, true, false]),
            )],
        );
        request.mode = ActMode::Sample;
        request.critics = vec!["value".into()];
        if discrete {
            builder
                .discrete("q", &DiscreteQ::new(2, 3, &[4], &mut rng).unwrap())
                .unwrap();
            request.critics.push("q".into());
        } else {
            builder
                .quantiles(
                    "quantiles",
                    &QValue::new(2, 1, 4, &[4], &mut rng).unwrap(),
                    &QuantileGrid::uniform(4).unwrap(),
                )
                .unwrap();
            builder
                .moments(
                    "moments",
                    &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
                    &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
                )
                .unwrap();
            request
                .critics
                .extend(["quantiles".into(), "moments".into()]);
            request.fixed_actions = Some(vec![Action::Continuous(vec![1.])]);
        }
        let chunk = builder.finish().unwrap();
        let bundle = ModelBundle::load(&chunk, ModelLimits::default()).unwrap();
        let a = bundle.inspect(&request).unwrap();
        let actor = bundle.evaluation_actor("policy").unwrap();
        assert_eq!(
            actor
                .act(
                    &request.observations,
                    request.mode,
                    &mut ChaCha8Rng::seed_from_u64(request.seed),
                )
                .unwrap(),
            a.actions
        );
        let b = bundle.inspect(&request).unwrap();
        assert_prediction_bits(&a, &b);
        assert_capture_variants(&bundle, &request);
        if !discrete {
            // Exercise sampled current-policy critic actions as well as fixed ones.
            request.fixed_actions = None;
            assert_capture_variants(&bundle, &request);
            request.fixed_actions = Some(vec![Action::Continuous(vec![1.])]);
        }
        if discrete {
            assert_eq!(a.actions, vec![Action::Discrete(1)]);
            request.policy = "q".into();
            request.mode = ActMode::Deterministic;
            assert_capture_variants(&bundle, &request);
            assert_eq!(
                bundle.inspect(&request).unwrap().actions,
                vec![Action::Discrete(1)]
            );
            request.observations[0].as_dense_mut().unwrap().action_mask = Some(vec![false; 3]);
            assert!(bundle.inspect(&request).is_err());
        } else {
            assert!(matches!(
                &a.critics["moments"].record,
                Record::Critic(CriticRecord {
                    representation: CriticRepresentation::RawMoments { .. },
                    ..
                })
            ));
            request.fixed_actions = Some(vec![Action::Continuous(vec![3.])]);
            assert!(bundle.inspect(&request).is_err());
        }
    }
    let custom = network();
    let mut builder = SnapshotBuilder::new(
        "opaque",
        RecordContext::new("test", 0),
        2,
        ActionSpace::Unbounded { dimensions: 1 },
    )
    .unwrap();
    builder
        .opaque("custom", &custom, "external adapter required")
        .unwrap();
    let bundle = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
    assert!(bundle.activation_networks().is_empty());
    assert!(
        bundle
            .inspect(&InspectionRequest::new(
                "custom",
                vec![Observation::new(vec![1., 2.])]
            ))
            .is_err()
    );
}

#[test]
fn capture_packing_occurs_only_for_requested_networks_and_moments_share_join() {
    let (batch, width) = (32, 257);
    let mut rng = ChaCha8Rng::seed_from_u64(61);
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut builder = SnapshotBuilder::new(
        "packing",
        RecordContext::new("test", 0),
        width,
        space.clone(),
    )
    .unwrap();
    builder
        .deterministic(
            "policy",
            &Deterministic::new(width, space, &[], &mut rng).unwrap(),
        )
        .unwrap();
    builder
        .scalar("scalar", &QValue::new(width, 1, 1, &[], &mut rng).unwrap())
        .unwrap();
    builder
        .moments(
            "moments",
            &QValue::new(width, 1, 1, &[], &mut rng).unwrap(),
            &QValue::new(width, 1, 1, &[], &mut rng).unwrap(),
        )
        .unwrap();
    let bundle = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
    let mut request =
        InspectionRequest::new("policy", vec![Observation::new(vec![0.25; width]); batch]);
    request.critics = vec!["scalar".into(), "moments".into()];
    // Empty names still require a valid input and an empty chunk, without
    // activation-forward allocations obscuring the capture packing sizes.
    let mut counts = |keys: &[&str]| {
        request.layers = keys.iter().map(|k| ((*k).into(), vec![])).collect();
        drop(bundle.inspect(&request).unwrap());
        PACKING_SIZES.with(|c| c.set(Some((batch * width * 4, batch * (width + 1) * 4, 0, 0))));
        let result = bundle.inspect(&request).unwrap();
        let (_, _, observations, joins) = PACKING_SIZES.with(|c| c.replace(None).unwrap());
        assert_eq!(result.activations.len(), keys.len());
        (observations, joins)
    };
    let missing = counts(&[]);
    let policy = counts(&["policy"]);
    let scalar = counts(&["scalar"]);
    let first = counts(&["moments.first"]);
    let second = counts(&["moments.second"]);
    let both = counts(&["moments.first", "moments.second"]);
    let all = counts(&["policy", "scalar", "moments.first", "moments.second"]);
    eprintln!(
        "capture payload allocation calls (observation, joined): missing={missing:?}, policy={policy:?}, scalar={scalar:?}, first={first:?}, second={second:?}, both={both:?}, all={all:?}"
    );
    assert!(
        policy.0 > missing.0,
        "missing capture must avoid observation payload packing: {missing:?} vs {policy:?}"
    );
    assert_eq!(missing.1, policy.1, "policy capture needs no action concat");
    assert!(
        scalar.1 > policy.1,
        "scalar capture alone must introduce its capture concat: {scalar:?} vs {policy:?}"
    );
    assert_eq!(first, second);
    assert_eq!(
        first.1,
        policy.1 + 1,
        "one moment capture adds exactly one concat"
    );
    assert_eq!(
        first.0, policy.0,
        "moment capture shares observation packing"
    );
    assert_eq!(first, both, "both moment captures must share one concat");
    assert!(
        all.1 > both.1,
        "all capture adds the independent scalar concat"
    );
    assert_eq!(
        all.0, policy.0,
        "all captures share one observation packing"
    );
}

#[test]
fn nonempty_final_moment_capture_releases_join_as_early_as_original_owning_path() {
    use rl::diagnostics::probes;
    let (batch, width, later_width) = (32, 257, 211);
    let mut rng = ChaCha8Rng::seed_from_u64(76);
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let policy = Deterministic::new(width, space.clone(), &[], &mut rng).unwrap();
    let first = QValue::new(width, 1, 1, &[17, later_width], &mut rng).unwrap();
    let second = QValue::new(width, 1, 1, &[17, later_width], &mut rng).unwrap();
    let mut builder = SnapshotBuilder::new(
        "capture-lifetime",
        RecordContext::new("test", 0),
        width,
        space,
    )
    .unwrap();
    builder.deterministic("policy", &policy).unwrap();
    builder.moments("moments", &first, &second).unwrap();
    let bundle = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
    let mut request =
        InspectionRequest::new("policy", vec![Observation::new(vec![0.25; width]); batch]);
    request.critics = vec!["moments".into()];
    request.fixed_actions = Some(vec![Action::Continuous(vec![0.75]); batch]);
    let first = IntervenedMlp::new(&first.network, vec![]).unwrap();
    let second = IntervenedMlp::new(&second.network, vec![]).unwrap();
    let mut context = bundle.context().clone();
    context.probe_set = Some("interactive".into());
    for both in [true, false] {
        request.layers.clear();
        request
            .layers
            .insert("moments.second".into(), vec!["layers.1.post".into()]);
        if both {
            request
                .layers
                .insert("moments.first".into(), vec!["layers.1.post".into()]);
        }
        drop(bundle.inspect(&request).unwrap());
        // Original ownership: keep one shared concat through the first
        // capture, then move its final owner into the second capture. The
        // watched later-layer payload cannot collide with weights or inputs.
        let reference = || {
            let input = Tensor::<2>::from_data(
                TensorData::new(vec![0.25_f32; batch * width], [batch, width]),
                &Device::flex(),
            );
            let actions = Tensor::<2>::from_data(
                TensorData::new(vec![0.75_f32; batch], [batch, 1]),
                &Device::flex(),
            );
            let joined = Tensor::cat(vec![input.clone(), actions.clone()], 1);
            let mut captures = std::collections::BTreeMap::new();
            if both {
                captures.insert(
                    "moments.first".to_owned(),
                    probes::activations(
                        context.clone(),
                        "moments.first",
                        &first,
                        joined.clone(),
                        &["layers.1.post"],
                    )
                    .unwrap(),
                );
            }
            captures.insert(
                "moments.second".to_owned(),
                probes::activations(
                    context.clone(),
                    "moments.second",
                    &second,
                    joined,
                    &["layers.1.post"],
                )
                .unwrap(),
            );
            captures
        };
        let (expected, original_overlap) =
            capture_lifetime(batch * (width + 1) * 4, batch * later_width * 4, reference);
        let (actual, candidate_overlap) =
            capture_lifetime(batch * (width + 1) * 4, batch * later_width * 4, || {
                bundle.inspect(&request).unwrap()
            });
        assert_eq!(
            actual.activations.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (name, expected) in &expected {
            assert_chunk_bits(&actual.activations[name], expected);
        }
        if both {
            assert!(
                original_overlap > 0,
                "first capture intentionally retains the shared input for the second"
            );
        } else {
            assert_eq!(
                original_overlap, 0,
                "original final owner releases the join before later-layer payloads"
            );
        }
        eprintln!(
            "nonempty moment capture lifetime, both={both}: original={original_overlap}, candidate={candidate_overlap} later-payload allocations while shared join live"
        );
        assert_eq!(
            candidate_overlap, original_overlap,
            "both={both}: final capture must not extend shared joined-payload lifetime"
        );
    }
}

#[test]
fn inspection_preserves_first_error_with_paired_invalid_inputs() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let policy = Deterministic::from_network(network(), space.clone()).unwrap();
    let mut builder =
        SnapshotBuilder::new("errors", RecordContext::new("test", 0), 2, space).unwrap();
    builder.deterministic("policy", &policy).unwrap();
    let bundle = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
    let invalid = Intervention {
        layer: "layers.0.post".into(),
        neuron: 99,
        value: InterventionValue::Set(f32::NAN),
    };
    let error = |request: &InspectionRequest, message: &str| match bundle.inspect(request) {
        Err(rl::Error::Config(actual)) => assert_eq!(actual, message),
        Err(other) => panic!("expected Config({message:?}), got {other:?}"),
        Ok(_) => panic!("invalid inspection succeeded"),
    };
    for capture in [false, true] {
        for bad_observation in [vec![1.], vec![f32::NAN, 2.]] {
            let mut request =
                InspectionRequest::new("policy", vec![Observation::new(bad_observation)]);
            if capture {
                request
                    .layers
                    .insert("policy".into(), vec!["layers.0.post".into()]);
            }
            request
                .interventions
                .insert("policy".into(), vec![invalid.clone()]);
            request.fixed_actions = Some(vec![]);
            error(
                &request,
                "invalid observation shape or nonfinite observation",
            );
            request.interventions.insert("unknown".into(), vec![]);
            error(
                &request,
                "inspection requests an unsupported or unselected activation network",
            );
            request.interventions.remove("unknown");
            request.observations[0] = Observation::new(vec![1., 2.]);
            error(&request, "probe actions and observations differ in length");
            request
                .layers
                .insert("policy".into(), vec!["unsupported".into()]);
            error(&request, "unsupported activation policy/unsupported");
            request.layers.clear();
            request.fixed_actions = None;
            error(&request, "invalid intervention neuron or value");
        }
    }
}
