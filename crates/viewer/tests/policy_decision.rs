use rl::{core::action::ActionSpace, diagnostics::schema::*};
use rl_viewer::policy_change::PolicyBatch;
use std::collections::BTreeMap;

fn gaussian(mu: f64, log_std: f64, action: f64) -> Chunk {
    let mut context = RecordContext::new("run", 7);
    context.probe_set = Some("fixed".into());
    Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Gaussian {
                mean: "m".into(),
                log_std: "s".into(),
                tanh_bounded: true,
            },
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: true,
        }),
        arrays: BTreeMap::from([
            ("a".into(), Array::f64(vec![1, 1], vec![action]).unwrap()),
            ("m".into(), Array::f64(vec![1, 1], vec![mu]).unwrap()),
            ("s".into(), Array::f64(vec![1], vec![log_std]).unwrap()),
        ]),
    }
}
#[cfg(feature = "screenshots")]
fn trajectory() -> Chunk {
    Chunk {
        context: RecordContext::new("run", 7),
        record: Record::Trajectory(TrajectoryRecord {
            render: None,
            id: "episode".into(),
            environment: "unknown".into(),
            observations: "o".into(),
            actions: "a".into(),
            rewards: "r".into(),
            terminated: "t".into(),
            truncated: "u".into(),
            simulation_times: None,
            action_masks: None,
            frames: BTreeMap::new(),
            evaluation: EvaluationMetadata {
                conditioning: Conditioning::InitialScenario,
                initial_action: ActionCondition::CurrentPolicy,
                continuation_policy: "online".into(),
                stochastic_policy: true,
                discount: 0.99,
                independent_future_streams: false,
                seed_methodology: "recorded".into(),
                truncation_handling: "recorded".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            (
                "o".into(),
                Array::f32(vec![4, 1], vec![1., 2., 3., 4.]).unwrap(),
            ),
            ("a".into(), Array::u64(vec![3], vec![2, 1, 0]).unwrap()),
            ("r".into(), Array::f64(vec![3], vec![2., -1., 4.]).unwrap()),
            ("t".into(), Array::u8(vec![3], vec![0, 0, 1]).unwrap()),
            ("u".into(), Array::u8(vec![3], vec![0, 0, 0]).unwrap()),
        ]),
    }
}

#[cfg(feature = "screenshots")]
#[test]
fn capture_trajectory_selection_rejects_invalid_sources_and_survives_training_time() {
    use rl::diagnostics::storage::*;
    use rl_viewer::{
        data::SelectedInput,
        ui::{Laboratory, View},
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "run",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 3 },
            0.99,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    writer.append(&trajectory()).unwrap();
    writer
        .append(&Chunk {
            context: RecordContext::new("run", 20),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        })
        .unwrap();
    let ctx = eframe::egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    for (sequence, step) in [
        ("bad", "0"),
        ("99", "0"),
        ("1", "0"),
        ("0", "4"),
        ("0", "bad"),
    ] {
        assert!(
            app.select_capture_trajectory(sequence, step).is_err(),
            "{sequence} {step}"
        );
    }
    app.select_capture_trajectory("0", "1").unwrap();
    app.select_revision(20).unwrap();
    assert_eq!(
        app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 0,
            step: 1
        }
    );
    // The retired history cards no longer render. Capture still preserves the
    // exact trajectory source and decision across each supported workspace.
    let run = rl_viewer::data::RunIndex::open(&path, ReadLimits::default()).unwrap();
    let source = run.recording().load(run.by_sequence(0).unwrap()).unwrap();
    assert_eq!(source.context.identity.learner_update_revision, 7);
    assert_eq!(source.arrays["a"].to_f64()[1], 1.);
    assert_eq!(source.arrays["r"].to_f64()[1], -1.);
    for view in [View::PolicyMap, View::Value, View::Network] {
        app.select_view(view);
        let mut output = ctx.run_ui(
            eframe::egui::RawInput {
                screen_rect: Some(eframe::egui::Rect::from_min_size(
                    eframe::egui::Pos2::ZERO,
                    eframe::egui::vec2(1440., 900.),
                )),
                ..Default::default()
            },
            |ui| app.show(ui),
        );
        output.textures_delta.clear();
        assert!(app.notice().is_none(), "{view:?}: {:?}", app.notice());
        assert_eq!(app.selection().revision, 20);
        assert_eq!(
            app.selection().input,
            SelectedInput::TrajectoryStep {
                sequence: 0,
                step: 1
            }
        );
    }
}

#[test]
fn gaussian_selection_preserves_marginals_and_policy_changes_ignore_recorded_samples() {
    use rl_viewer::policy::numerics::{
        SelectedDistribution, current_reference_kl, selected_distribution,
    };
    let space = ActionSpace::Box {
        low: vec![-2.],
        high: vec![4.],
    };
    let a = gaussian(0., 0., 3.5);
    let b = gaussian(0., 0., -1.);
    let a = PolicyBatch::new(&a, None, &space).unwrap();
    let b = PolicyBatch::new(&b, None, &space).unwrap();
    let SelectedDistribution::Gaussian(marginal) = selected_distribution(&a, 0, 0).unwrap() else {
        panic!()
    };
    assert_eq!(marginal.cdf(1.).unwrap(), 0.5);
    for (expected_action, target_probability) in
        [(-1.78444409686107, 0.05), (3.78444409686107, 0.95)]
    {
        assert!(marginal.cdf(expected_action - 1e-9).unwrap() <= target_probability);
        assert!(target_probability <= marginal.cdf(expected_action + 1e-9).unwrap());
    }
    assert_eq!(current_reference_kl(&a, 0, &b, 0).unwrap(), Some(0.));
}

#[test]
fn current_gaussian_cdf_preserves_median_when_variance_changes() {
    use rl_viewer::policy::numerics::{
        SelectedDistribution, current_reference_kl, selected_distribution,
    };
    let space = ActionSpace::Box {
        low: vec![-2.],
        high: vec![4.],
    };
    let a = gaussian(0.7, 0., -1.);
    let b = gaussian(0.7, 1., 3.);
    let a = PolicyBatch::new(&a, None, &space).unwrap();
    let b = PolicyBatch::new(&b, None, &space).unwrap();
    for batch in [&a, &b] {
        let SelectedDistribution::Gaussian(marginal) = selected_distribution(batch, 0, 0).unwrap()
        else {
            panic!()
        };
        assert!(marginal.cdf(2.81310333135 - 1e-10).unwrap() <= 0.5);
        assert!(0.5 <= marginal.cdf(2.81310333135 + 1e-10).unwrap());
    }
    assert!(current_reference_kl(&a, 0, &b, 0).unwrap().unwrap() > 0.);
    let median = |batch: &PolicyBatch<'_>| {
        let SelectedDistribution::Gaussian(marginal) = selected_distribution(batch, 0, 0).unwrap()
        else {
            panic!()
        };
        let [low, high] = marginal.bounds.unwrap();
        (low + high) / 2. + (high - low) / 2. * marginal.mean.tanh()
    };
    assert_eq!(median(&a), median(&b));

    // Unbounded expected actions equal the latent mean for every variance.
    // Bounded expected actions depend on variance, even with an equal median.
    let mut a = gaussian(0.7, 0., -1.);
    let mut b = gaussian(0.7, 1., 3.);
    for chunk in [&mut a, &mut b] {
        let Record::Policy(meta) = &mut chunk.record else {
            panic!()
        };
        let PolicyParameters::Gaussian { tanh_bounded, .. } = &mut meta.parameters else {
            panic!()
        };
        *tanh_bounded = false;
    }
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let a = PolicyBatch::new(&a, None, &space).unwrap();
    let b = PolicyBatch::new(&b, None, &space).unwrap();
    assert_eq!(
        rl_viewer::policy::numerics::project(&a, 0, 0).unwrap(),
        rl_viewer::policy::numerics::project(&b, 0, 0).unwrap()
    );
}

#[test]
fn current_categorical_distribution_retains_masked_support_and_rejects_family_change() {
    use rl_viewer::policy::numerics::{
        SelectedDistribution, current_reference_kl, selected_distribution,
    };
    let mut chunk = gaussian(0., 0., 0.);
    chunk
        .arrays
        .insert("a".into(), Array::u64(vec![1], vec![1]).unwrap());
    chunk.arrays.insert(
        "p".into(),
        Array::f64(vec![1, 3], vec![0., 0.25, 0.75]).unwrap(),
    );
    chunk
        .arrays
        .insert("mask".into(), Array::u8(vec![1, 3], vec![0, 1, 1]).unwrap());
    let Record::Policy(meta) = &mut chunk.record else {
        panic!()
    };
    meta.parameters = PolicyParameters::Categorical {
        probabilities: "p".into(),
    };
    meta.masks = Some("mask".into());
    let space = ActionSpace::Discrete { count: 3 };
    let batch = PolicyBatch::new(&chunk, None, &space).unwrap();
    assert!(
        matches!(selected_distribution(&batch, 0, 0).unwrap(), SelectedDistribution::Categorical { probabilities, valid } if probabilities == [0.,0.25,0.75] && valid == [false,true,true])
    );
    let mut deterministic = chunk.clone();
    let Record::Policy(meta) = &mut deterministic.record else {
        panic!()
    };
    meta.parameters = PolicyParameters::Deterministic;
    meta.stochastic = false;
    let deterministic = PolicyBatch::new(&deterministic, None, &space).unwrap();
    assert!(current_reference_kl(&batch, 0, &deterministic, 0).is_err());
}
