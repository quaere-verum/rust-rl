//! Retained model outputs on immutable, genuinely masked recorded inputs.
#![cfg(feature = "screenshots")]
#[path = "support/policy_ui.rs"]
mod support;
use eframe::egui;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::SnapshotBuilder, probes::ProbeSet, schema::*, storage::*},
    nn::{critics::scalar::DiscreteQ, matrix, vector},
};
use rl_viewer::{data::SelectedInput, ui::View};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

fn fixture() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 3 };
    let manifest =
        RunManifest::new("masked-playback", "ordinary", "DDQN", 2, space.clone(), 0.9).unwrap();
    let mut w = Recorder::create(d.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("masked-playback", 3);
    context.probe_set = Some("frozen".into());
    let probes = ProbeSet::new(
        "frozen",
        ProbePopulation::FixedReference,
        "genuine typed masked inputs",
        vec![
            Observation::dense(vec![0., 0.], Some(vec![true, true, false])),
            Observation::dense(vec![0.25, 0.], Some(vec![false, true, true])),
        ],
        None,
        space.clone(),
    )
    .unwrap()
    .record(context.clone())
    .unwrap();
    w.append(&probes).unwrap();
    // Deliberately different stored stochastic predictions: a model-backed
    // discrete-Q view must never reuse these old probabilities as its output.
    w.append(&Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Categorical {
                probabilities: "p".into(),
            },
            stochastic: true,
            masks: Some("m".into()),
            entropy: None,
            entropy_method: None,
        }),
        arrays: BTreeMap::from([
            ("a".into(), Array::u64(vec![2], vec![1, 2]).unwrap()),
            (
                "p".into(),
                Array::f64(vec![2, 3], vec![0.25, 0.75, 0., 0., 0.25, 0.75]).unwrap(),
            ),
            (
                "m".into(),
                Array::u8(vec![2, 3], vec![1, 1, 0, 0, 1, 1]).unwrap(),
            ),
        ]),
    })
    .unwrap();
    for (k, bias) in [(3, [3., 2., 1.]), (17, [1., 2., 4.]), (41, [5., 3., 1.])] {
        let mut q = DiscreteQ::new(2, 3, &[], &mut ChaCha8Rng::seed_from_u64(k)).unwrap();
        q.network.layers[0].weight = q.network.layers[0]
            .weight
            .clone()
            .map(|_| matrix(vec![0.; 6], 2, 3));
        q.network.layers[0].bias = q.network.layers[0]
            .bias
            .take()
            .map(|p| p.map(|_| vector(&bias)));
        let mut b = SnapshotBuilder::new(
            "actual",
            RecordContext::new("masked-playback", k),
            2,
            space.clone(),
        )
        .unwrap();
        b.discrete("actual-q", &q).unwrap();
        w.append(&b.finish().unwrap()).unwrap();
    }
    w.finish().unwrap();
    d
}
fn settle(h: &mut support::Harness) {
    h.settle();
    let until = Instant::now() + Duration::from_secs(10);
    while !h.app.snapshot_preparation_idle() {
        h.frame(vec![]);
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    h.settle();
}
fn assert_row(h: &support::Harness, k: u64, value: f64) {
    let f = h.app.policy_frame().unwrap();
    let p = f.selected.as_ref().unwrap();
    assert_eq!(f.grid.current.revision, k);
    assert_eq!(f.requested_revision, k);
    assert!(!f.grid.current.recorded);
    assert!(!f.grid.regular());
    assert_eq!(f.grid.population_count, 2);
    assert_eq!(p.input.dense_values(), [0.25, 0.]);
    assert_eq!(p.input.dense_mask(), Some(&[false, true, true][..]));
    assert!(matches!(
        p.recorded_input,
        Some(SelectedInput::ProbeRow {
            row: 1,
            sequence: 0,
            ..
        })
    ));
    assert_eq!(p.value, value);
    assert!(f.grid.values.iter().all(|&v| v == value));
    assert!(matches!(
        p.distribution,
        rl_viewer::policy::numerics::SelectedDistribution::Deterministic
    ));
}
#[cfg(feature = "screenshots")]
#[test]
fn masked_population_and_original_row_survive_nonconsecutive_model_navigation_and_reference() {
    let d = fixture();
    let mut h = support::Harness::new(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    assert!(
        !h.app.policy_frame().unwrap().grid.current.recorded,
        "catalog discovery must replace stored predictions even at the same learner revision"
    );
    h.app.select_revision(41).unwrap();
    settle(&mut h);
    assert!(!h.has("model playback unavailable"));
    h.click("Next sample");
    settle(&mut h);
    assert_row(&h, 41, 1.);
    h.click("‹");
    settle(&mut h);
    assert_row(&h, 17, 0.);
    h.click("‹");
    settle(&mut h);
    assert_row(&h, 3, 1.);
    h.click("›");
    settle(&mut h);
    assert_row(&h, 17, 0.);
    h.click_last("Reference");
    settle(&mut h);
    let p = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(p.reference_value, Some(1.));
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .grid
            .reference
            .as_ref()
            .unwrap()
            .revision,
        3
    );
    let before = h.app.policy_metrics();
    let uploads = h.app.policy_pane().texture_uploads;
    for _ in 0..50 {
        h.frame(vec![]);
    }
    assert_eq!(before.grid_batches, h.app.policy_metrics().grid_batches);
    assert_eq!(before.point_batches, h.app.policy_metrics().point_batches);
    assert_eq!(before.restores, h.app.policy_metrics().restores);
    assert_eq!(uploads, h.app.policy_pane().texture_uploads);
    // Foreground requests supersede one another; the original row/mask does not.
    for k in [41, 3, 17, 41] {
        h.app.select_revision(k).unwrap();
        h.frame(vec![]);
    }
    settle(&mut h);
    assert_row(&h, 41, 1.);
    h.app.select_view(View::Value);
    h.frame(vec![]);
    assert!(h.app.policy_frame().is_none());
    assert_eq!(h.app.policy_pane().texture_bytes(), 0);
}

#[test]
fn no_axis_masked_model_evaluates_only_the_selected_original_row() {
    let d = fixture();
    let path = d.path().join("run/manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let observations:Vec<_>=(0..2).map(|i|serde_json::json!({"id":format!("c{i}"),"label":format!("Category {i}"),"kind":{"Categorical":{"categories":[{"code":0.,"label":"Zero"},{"code":0.25,"label":"Quarter"}]}},"coordinate":false})).collect();
    manifest["environment_descriptor"] = serde_json::json!({"version":1,"observations":observations,"actions":{"Discrete":{"choices":[{"label":"A"},{"label":"B"},{"label":"C"}]}},"clocks":{"learner_revision":{"label":"Revision"},"decision_step":{"label":"Step"}},"reward":{"label":"Reward"}});
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let mut h = support::Harness::new(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    h.app.select_revision(41).unwrap();
    settle(&mut h);
    assert!(!h.has("model playback unavailable"));
    h.click("Next sample");
    settle(&mut h);
    let frame = h.app.policy_frame().unwrap();
    assert_eq!(
        frame.grid.config.mode,
        rl_viewer::policy::slice::SliceMode::SamplesOnly
    );
    assert_eq!(frame.grid.population_count, 2);
    assert!(frame.grid.values.is_empty());
    assert_eq!(frame.selected.as_ref().unwrap().value, 1.);
    assert_eq!(h.app.policy_metrics().grid_batches, 0);
    assert!(h.app.policy_metrics().point_batches > 0);
    h.click("‹");
    settle(&mut h);
    let point = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.value, 0.);
    assert_eq!(point.input.dense_values(), [0.25, 0.]);
    assert_eq!(point.input.dense_mask(), Some(&[false, true, true][..]));
    assert_eq!(h.app.policy_metrics().grid_batches, 0);
    assert!(h.app.policy_pane().main_rect.is_none());
}
