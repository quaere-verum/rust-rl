#![cfg(feature = "diagnostics")]
#[cfg(feature = "neural")]
use rand::SeedableRng;
#[cfg(feature = "neural")]
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        policy_slice::{PolicySliceSpec, SliceConfig},
        schema::RunManifest,
        visualization::EnvironmentSemantics,
    },
};
#[cfg(feature = "neural")]
use rl::{
    diagnostics::{
        models::{ModelBundle, ModelLimits, SnapshotBuilder},
        policy_numerics::{self, SelectedDistribution},
        policy_surface::{SurfaceLimits, evaluate_policy_batch, evaluate_surface},
        schema::RecordContext,
    },
    nn::policy::{Deterministic, Policy},
};

fn manifest(space: ActionSpace) -> RunManifest {
    RunManifest::new("surface", "ordinary", "test", 2, space, 0.9).unwrap()
}
fn spec(m: &RunManifest) -> PolicySliceSpec {
    let mut config = SliceConfig::new(
        EnvironmentSemantics::new(m),
        &Observation::new(vec![0.25, -0.5]),
        &[],
    )
    .unwrap();
    config.resolution = [3, 2];
    config.action = 0;
    PolicySliceSpec::from_config("online", "actor", &config)
}

#[test]
fn slice_spec_round_trips_editable_settings_and_rejects_bad_versions_and_conditioning() {
    let m = manifest(ActionSpace::Unbounded { dimensions: 2 });
    let e = EnvironmentSemantics::new(&m);
    let original = spec(&m);
    let json = serde_json::to_vec(&original).unwrap();
    let decoded: PolicySliceSpec = serde_json::from_slice(&json).unwrap();
    let config = decoded.to_config(e).unwrap();
    assert_eq!(
        PolicySliceSpec::from_config("online", "actor", &config),
        original
    );
    let mut invalid = decoded.clone();
    invalid.version += 1;
    assert!(invalid.to_config(e).is_err());
    invalid = decoded.clone();
    invalid.y = Some(invalid.x);
    assert!(invalid.to_config(e).is_err());
    invalid = decoded;
    invalid.action_mask = Some(vec![true, false]);
    assert!(invalid.to_config(e).is_err());
}

#[cfg(feature = "neural")]
#[test]
fn complete_surface_and_direct_evaluation_agree_for_every_component_and_policy_family() {
    for (space, stochastic) in [
        (ActionSpace::Unbounded { dimensions: 2 }, false),
        (ActionSpace::Discrete { count: 3 }, true),
        (ActionSpace::Unbounded { dimensions: 2 }, true),
        (
            ActionSpace::Box {
                low: vec![-3., -1.],
                high: vec![7., 2.],
            },
            true,
        ),
    ] {
        let m = manifest(space.clone());
        let mut builder =
            SnapshotBuilder::new("test", RecordContext::new("surface", 7), 2, space.clone())
                .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        if stochastic {
            builder
                .policy(
                    "actor",
                    &Policy::new(2, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
        } else {
            builder
                .deterministic(
                    "actor",
                    &Deterministic::new(2, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
        }
        let bundle = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
        let spec = spec(&m);
        let config = spec.to_config(EnvironmentSemantics::new(&m)).unwrap();
        let mut progress = Vec::new();
        let surface = evaluate_surface(
            &bundle,
            &spec,
            EnvironmentSemantics::new(&m),
            SurfaceLimits {
                max_batch: 2,
                ..Default::default()
            },
            || false,
            |p| progress.push((p.completed_rows, p.retained_bytes)),
        )
        .unwrap();
        assert_eq!(surface.rows(), 6);
        assert_eq!(progress.iter().map(|p| p.0).collect::<Vec<_>>(), [2, 4, 6]);
        assert!(progress.windows(2).all(|w| w[0].1 < w[1].1));
        assert_eq!(progress.last().unwrap().1, surface.owned_bytes());
        let observations: Vec<_> = (0..6).map(|i| config.observation_at(i).unwrap()).collect();
        let direct = evaluate_policy_batch(&bundle, "actor", &observations).unwrap();
        let direct = direct.checked(&space).unwrap();
        for (row, input) in observations.iter().enumerate() {
            assert_eq!(surface.input(row).unwrap(), *input);
            for action in 0..surface.action_count() {
                let expected = policy_numerics::project(&direct, row, action).unwrap();
                assert!((surface.value(row, action).unwrap() - expected).abs() <= 2e-6);
                let actual = surface.distribution(row, action).unwrap();
                assert_eq!(
                    std::mem::discriminant(&actual),
                    std::mem::discriminant(
                        &policy_numerics::selected_distribution(&direct, row, action).unwrap()
                    )
                );
                if let SelectedDistribution::Gaussian(g) = actual {
                    assert_eq!(g.bounds.is_some(), matches!(space, ActionSpace::Box { .. }));
                    assert!((g.expected_action().unwrap() - expected).abs() <= 2e-6);
                }
            }
        }
        assert!(surface.value(6, 0).is_err());
        assert!(surface.value(0, surface.action_count()).is_err());
        assert!(surface.owned_bytes() <= SurfaceLimits::default().max_surface_bytes);
    }
}

#[cfg(feature = "neural")]
#[test]
fn masked_and_history_slices_remain_explicit_and_limits_precede_inference() {
    let m = manifest(ActionSpace::Discrete { count: 2 });
    let e = EnvironmentSemantics::new(&m);
    let anchor = Observation::dense(vec![0., 1.], Some(vec![false, true]));
    let config = SliceConfig::new(e, &anchor, &[]).unwrap();
    let exported = PolicySliceSpec::from_config("online", "actor", &config);
    assert_eq!(
        exported.to_config(e).unwrap().anchor_mask(),
        Some([false, true].as_slice())
    );
    assert!(exported.to_config(e).unwrap().observation_at(0).is_err());
    let history = PolicySliceSpec::from_config("online", "actor", &config.with_history());
    assert!(history.to_config(e).unwrap().observation_at(0).is_err());
    let mut builder = SnapshotBuilder::new(
        "test",
        RecordContext::new("surface", 0),
        2,
        m.action_space.clone(),
    )
    .unwrap();
    builder
        .policy(
            "actor",
            &Policy::new(
                2,
                m.action_space.clone(),
                &[],
                &mut ChaCha8Rng::seed_from_u64(3),
            )
            .unwrap(),
        )
        .unwrap();
    let model = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
    assert!(
        evaluate_surface(
            &model,
            &exported,
            e,
            SurfaceLimits::default(),
            || false,
            |_| panic!("unsupported grid was evaluated")
        )
        .is_err()
    );
    let valid = spec(&m);
    for limits in [
        SurfaceLimits {
            max_batch: 0,
            ..Default::default()
        },
        SurfaceLimits {
            max_surface_bytes: 1,
            ..Default::default()
        },
        SurfaceLimits {
            max_scratch_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(
            evaluate_surface(
                &model,
                &valid,
                e,
                limits,
                || false,
                |_| panic!("inadmissible work was evaluated")
            )
            .is_err()
        );
    }
    assert!(
        evaluate_surface(
            &model,
            &valid,
            e,
            SurfaceLimits::default(),
            || true,
            |_| panic!("cancelled work progressed")
        )
        .is_err()
    );
}

#[cfg(feature = "neural")]
#[test]
fn greedy_discrete_surfaces_keep_choices_distinct_from_stochastic_probabilities() {
    use rl::nn::critics::scalar::DiscreteQ;
    let m = manifest(ActionSpace::Discrete { count: 3 });
    let mut builder = SnapshotBuilder::new(
        "greedy",
        RecordContext::new("surface", 7),
        2,
        m.action_space.clone(),
    )
    .unwrap();
    builder
        .discrete(
            "actor",
            &DiscreteQ::new(2, 3, &[4], &mut ChaCha8Rng::seed_from_u64(19)).unwrap(),
        )
        .unwrap();
    let model = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
    let surface = evaluate_surface(
        &model,
        &spec(&m),
        EnvironmentSemantics::new(&m),
        SurfaceLimits::default(),
        || false,
        |_| {},
    )
    .unwrap();
    for row in 0..surface.rows() {
        let (batch, local) = surface.checked_row(row).unwrap();
        let choice = batch.actions().as_u64().unwrap()[local] as usize;
        for action in 0..3 {
            assert_eq!(
                surface.value(row, action).unwrap(),
                f64::from(action == choice)
            );
            assert_eq!(
                surface.distribution(row, action).unwrap(),
                SelectedDistribution::Deterministic
            );
        }
    }
    let inputs = [Observation::dense(
        vec![0., 1.],
        Some(vec![false, true, false]),
    )];
    let evaluated = evaluate_policy_batch(&model, "actor", &inputs).unwrap();
    let batch = evaluated.checked(&m.action_space).unwrap();
    assert_eq!(
        batch.effective_mask(0).unwrap(),
        Some(vec![false, true, false])
    );
    assert_eq!(batch.actions().as_u64().unwrap(), [1]);
}
