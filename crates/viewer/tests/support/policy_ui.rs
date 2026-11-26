#![allow(dead_code)]
use eframe::egui;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::SnapshotBuilder, probes::ProbeSet, schema::*, storage::*},
    nn::{critics::scalar::QValue, network::Mlp, policy::Deterministic, vector},
};
use rl_viewer::ui::{Laboratory, View};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
pub fn fixture(dim: usize, actions: usize, model: bool) -> tempfile::TempDir {
    let space = if actions > 0 {
        ActionSpace::Discrete { count: actions }
    } else {
        ActionSpace::Unbounded { dimensions: 2 }
    };
    fixture_with_space(dim, actions, model, space, 0)
}
pub fn positive_model_fixture() -> tempfile::TempDir {
    fixture_with_space(
        3,
        0,
        true,
        ActionSpace::Box {
            low: vec![0.1; 2],
            high: vec![0.9; 2],
        },
        0,
    )
}
pub fn policy_with_critic_fixture() -> tempfile::TempDir {
    fixture_with_space(
        3,
        0,
        true,
        ActionSpace::Box {
            low: vec![-1.; 2],
            high: vec![1.; 2],
        },
        1,
    )
}
pub fn policy_with_ambiguous_critics_fixture() -> tempfile::TempDir {
    fixture_with_space(
        3,
        0,
        true,
        ActionSpace::Box {
            low: vec![-1.; 2],
            high: vec![1.; 2],
        },
        2,
    )
}
fn fixture_with_space(
    dim: usize,
    actions: usize,
    model: bool,
    space: ActionSpace,
    critic_count: usize,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let manifest =
        RunManifest::new("policy-ui", "ordinary", "unknown", dim, space.clone(), 0.9).unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        manifest.clone(),
        StorageConfig::default(),
    )
    .unwrap();
    let mut context = RecordContext::new("policy-ui", 3);
    context.probe_set = Some("fixed".into());
    let observations = vec![
        Observation::new(vec![0.; dim]),
        Observation::new(vec![0.5; dim]),
    ];
    let mut probes = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "actual fixture inputs",
        observations,
        None,
        space.clone(),
    )
    .unwrap()
    .record(context.clone())
    .unwrap();
    // Effective masks exist ONLY in PolicyRecord, exercising the accepted initializer.
    if let Record::ProbeSet(p) = &mut probes.record
        && let Some(key) = p.action_masks.take()
    {
        probes.arrays.remove(&key);
    }
    writer.append(&probes).unwrap();
    for revision in [3, 7] {
        context.identity.learner_update_revision = revision;
        context.counters.updates = revision;
        let invalid = if revision == 7 {
            actions.saturating_sub(1)
        } else {
            0
        };
        let (parameters, mut arrays) = if actions > 0 {
            (
                PolicyParameters::Categorical {
                    probabilities: "p".into(),
                },
                BTreeMap::from([
                    (
                        "a".into(),
                        Array::u64(vec![2], vec![u64::from(revision == 3); 2]).unwrap(),
                    ),
                    (
                        "p".into(),
                        Array::f64(
                            vec![2, actions],
                            (0..2 * actions)
                                .map(|i| {
                                    if i % actions == invalid {
                                        0.
                                    } else {
                                        1. / (actions - 1) as f64
                                    }
                                })
                                .collect(),
                        )
                        .unwrap(),
                    ),
                ]),
            )
        } else {
            (
                PolicyParameters::Deterministic,
                BTreeMap::from([(
                    "a".into(),
                    Array::f32(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
                )]),
            )
        };
        if actions > 0 {
            arrays.insert(
                "m".into(),
                Array::u8(
                    vec![2, actions],
                    (0..2 * actions)
                        .map(|i| u8::from(i % actions != invalid))
                        .collect(),
                )
                .unwrap(),
            );
        }
        writer
            .append(&Chunk {
                context: context.clone(),
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters,
                    stochastic: false,
                    masks: (actions > 0).then(|| "m".into()),
                    entropy: None,
                    entropy_method: None,
                }),
                arrays,
            })
            .unwrap();
        if model {
            let network = Mlp::new(dim, &[4], 2, &mut ChaCha8Rng::seed_from_u64(revision)).unwrap();
            let policy = Deterministic::from_network(network, space.clone()).unwrap();
            let mut snapshot = SnapshotBuilder::new(
                "real",
                RecordContext::new("policy-ui", revision),
                dim,
                space.clone(),
            )
            .unwrap();
            snapshot.deterministic("actor", &policy).unwrap();
            for index in 0..critic_count {
                let mut critic = QValue::new(
                    dim,
                    2,
                    1,
                    &[],
                    &mut ChaCha8Rng::seed_from_u64(revision + 10),
                )
                .unwrap();
                critic.network.layers[0].bias = critic.network.layers[0]
                    .bias
                    .take()
                    .map(|bias| bias.map(|_| vector(&[revision as f32])));
                snapshot
                    .scalar(if index == 0 { "critic" } else { "critic2" }, &critic)
                    .unwrap();
            }
            let mut snapshot = snapshot.finish().unwrap();
            if critic_count > 0 {
                use rl::diagnostics::value::{
                    ContinuationIdentity, CriticDeclaration, ValueConventions, ValueCriticKind,
                    ValueDeclaration, attach_declaration,
                };
                let model_role = snapshot.context.identity.model_role.clone();
                let declaration = CriticDeclaration {
                    kind: ValueCriticKind::ScalarQ,
                    learned_baseline: None,
                    actor: Some(ContinuationIdentity {
                        identity: "fixture-actor".into(),
                        stochastic: false,
                    }),
                    continuation: Some(ContinuationIdentity {
                        identity: "fixture-continuation".into(),
                        stochastic: false,
                    }),
                    twin: None,
                    utility: None,
                };
                let mut critics = BTreeMap::from([("critic".into(), declaration.clone())]);
                if critic_count > 1 {
                    critics.insert("critic2".into(), declaration);
                }
                attach_declaration(
                    &mut snapshot,
                    &ValueDeclaration {
                        version: 1,
                        run_id: manifest.run_id.clone(),
                        revision,
                        model_role,
                        conventions: Some(ValueConventions {
                            dynamics: "fixture".into(),
                            observation: "fixture".into(),
                            action: "fixture".into(),
                            reward_units: "reward".into(),
                            reward_convention: manifest.reward_convention.clone(),
                            discount: manifest.discount,
                            horizon: "fixture".into(),
                            terminal_treatment: "fixture".into(),
                            action_constraints: "fixture".into(),
                        }),
                        critics,
                    },
                )
                .unwrap();
            }
            writer.append(&snapshot).unwrap();
        }
    }
    let observations = [vec![0.; dim], vec![0.; dim], vec![999.; dim]].concat();
    writer
        .append(&Chunk {
            context: RecordContext::new("policy-ui", 7),
            record: Record::Trajectory(TrajectoryRecord {
                render: None,
                id: "visits".into(),
                environment: "ordinary".into(),
                observations: "obs".into(),
                actions: "a".into(),
                rewards: "r".into(),
                terminated: "done".into(),
                truncated: "cut".into(),
                simulation_times: None,
                action_masks: None,
                frames: BTreeMap::new(),
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: "recorded".into(),
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
                    "obs".into(),
                    Array::f32(vec![3, dim], observations).unwrap(),
                ),
                (
                    "a".into(),
                    if actions > 0 {
                        Array::u64(vec![2], vec![0, 0]).unwrap()
                    } else {
                        Array::f32(vec![2, 2], vec![0.; 4]).unwrap()
                    },
                ),
                ("r".into(), Array::f32(vec![2], vec![0., 0.]).unwrap()),
                ("done".into(), Array::u8(vec![2], vec![0, 1]).unwrap()),
                ("cut".into(), Array::u8(vec![2], vec![0, 0]).unwrap()),
            ]),
        })
        .unwrap();
    writer.finish().unwrap();
    dir
}
/// Independent real rows: the last current input has no matching reference row.
/// Gaussian fixtures use the same worker preparation as ordinary recorded policies.
pub fn comparison_fixture(gaussian_reference_log_std: Option<f64>) -> tempfile::TempDir {
    comparison_fixture_impl(gaussian_reference_log_std, false)
}
pub fn varying_gaussian_fixture() -> tempfile::TempDir {
    comparison_fixture_impl(Some(0.), true)
}
fn comparison_fixture_impl(
    gaussian_reference_log_std: Option<f64>,
    varying: bool,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let space = if gaussian_reference_log_std.is_some() {
        ActionSpace::Unbounded { dimensions: 1 }
    } else {
        ActionSpace::Box {
            low: vec![0.],
            high: vec![1.],
        }
    };
    let manifest = RunManifest::new(
        "policy-compare",
        "ordinary",
        "unknown",
        2,
        space.clone(),
        0.9,
    )
    .unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for revision in [3, 7] {
        let mut context = RecordContext::new("policy-compare", revision);
        context.probe_set = Some("fixed".into());
        let xs = [0., 0.02, if revision == 3 { 0.7 } else { 0.5 }];
        writer
            .append(
                &ProbeSet::new(
                    "fixed",
                    ProbePopulation::FixedReference,
                    "independent exact comparison inputs",
                    xs.into_iter()
                        .map(|x| Observation::new(vec![x, 0.]))
                        .collect(),
                    None,
                    space.clone(),
                )
                .unwrap()
                .record(context.clone())
                .unwrap(),
            )
            .unwrap();
        let (parameters, arrays) = if let Some(reference_log_std) = gaussian_reference_log_std {
            (
                PolicyParameters::Gaussian {
                    mean: "mu".into(),
                    log_std: "std".into(),
                    tanh_bounded: false,
                },
                BTreeMap::from([
                    ("a".into(), Array::f64(vec![3, 1], vec![0.; 3]).unwrap()),
                    ("mu".into(), Array::f64(vec![3, 1], vec![0.; 3]).unwrap()),
                    (
                        "std".into(),
                        Array::f64(
                            if varying { vec![3, 1] } else { vec![1] },
                            if varying {
                                vec![
                                    if revision == 3 { 0.25f64.ln() } else { 0. },
                                    0.125f64.ln(),
                                    0.,
                                ]
                            } else {
                                vec![if revision == 3 { reference_log_std } else { 0. }]
                            },
                        )
                        .unwrap(),
                    ),
                ]),
            )
        } else {
            (
                PolicyParameters::Deterministic,
                BTreeMap::from([(
                    "a".into(),
                    Array::f64(
                        vec![3, 1],
                        if revision == 3 {
                            vec![0.9, 0.5, 0.7]
                        } else {
                            vec![0.8, 0.2, 0.6]
                        },
                    )
                    .unwrap(),
                )]),
            )
        };
        writer
            .append(&Chunk {
                context,
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters,
                    stochastic: gaussian_reference_log_std.is_some(),
                    masks: None,
                    entropy: None,
                    entropy_method: None,
                }),
                arrays,
            })
            .unwrap();
    }
    writer.finish().unwrap();
    dir
}
/// A retained model with a genuinely sparse recorded reference at slice endpoints.
pub fn mixed_reference_fixture(matches: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest = RunManifest::new("mixed", "ordinary", "unknown", 2, space.clone(), 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for revision in [3, 7] {
        let mut context = RecordContext::new("mixed", revision);
        context.probe_set = Some("fixed".into());
        let observations = if revision == 3 {
            vec![
                Observation::new(vec![0., 0.]),
                Observation::new(vec![-1., 0.]),
                Observation::new(vec![if matches == 2 { 1. } else { 0.7 }, 0.]),
            ]
        } else {
            vec![
                Observation::new(vec![0., 0.]),
                Observation::new(vec![1., 0.]),
            ]
        };
        writer
            .append(
                &ProbeSet::new(
                    "fixed",
                    ProbePopulation::FixedReference,
                    "exact endpoints",
                    observations,
                    None,
                    space.clone(),
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
                    actions: "a".into(),
                    parameters: PolicyParameters::Deterministic,
                    stochastic: false,
                    masks: None,
                    entropy: None,
                    entropy_method: None,
                }),
                arrays: BTreeMap::from([(
                    "a".into(),
                    if revision == 3 {
                        Array::f64(vec![3, 1], vec![0.4, 0.3, 0.8]).unwrap()
                    } else {
                        Array::f64(vec![2, 1], vec![0.3, 0.8]).unwrap()
                    },
                )]),
            })
            .unwrap();
        if revision == 7 {
            let network = Mlp::new(2, &[4], 1, &mut ChaCha8Rng::seed_from_u64(7)).unwrap();
            let policy = Deterministic::from_network(network, space.clone()).unwrap();
            let mut snapshot = SnapshotBuilder::new(
                "real",
                RecordContext::new("mixed", revision),
                2,
                space.clone(),
            )
            .unwrap();
            snapshot.deterministic("actor", &policy).unwrap();
            writer.append(&snapshot.finish().unwrap()).unwrap();
        }
    }
    writer.finish().unwrap();
    dir
}
pub struct Harness {
    pub ctx: egui::Context,
    pub app: Laboratory,
    pub time: f64,
    pub size: egui::Vec2,
    pub text: Vec<(String, egui::Rect)>,
    pub shapes: Vec<egui::Shape>,
    pub tooltip_text: Vec<(String, egui::Rect)>,
}
impl Harness {
    pub fn new(dir: &tempfile::TempDir, size: egui::Vec2) -> Self {
        let ctx = egui::Context::default();
        let mut app =
            Laboratory::from_context(&ctx, dir.path().join("run").display().to_string()).unwrap();
        app.select_view(View::PolicyMap);
        Self {
            ctx,
            app,
            time: 0.,
            size,
            text: vec![],
            shapes: vec![],
            tooltip_text: vec![],
        }
    }
    pub fn frame(&mut self, events: Vec<egui::Event>) {
        self.time += 0.1;
        self.tooltip_text.clear();
        let mut output = self.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, self.size)),
                time: Some(self.time),
                events,
                ..Default::default()
            },
            |ui| {
                self.app.show(ui);
                // Capture actual tooltip-layer paint before Context drains it.
                // Exclude labels merely present elsewhere in the UI and clipped text.
                let layers = ui.ctx().memory(|m| {
                    m.layer_ids()
                        .filter(|l| l.order == egui::Order::Tooltip)
                        .collect::<Vec<_>>()
                });
                ui.ctx().graphics(|g| {
                    for layer in layers {
                        if let Some(paint) = g.get(layer) {
                            for shape in paint.all_entries() {
                                let mut labels = Vec::new();
                                collect(&shape.shape, &mut labels);
                                self.tooltip_text.extend(
                                    labels
                                        .into_iter()
                                        .filter(|(_, rect)| shape.clip_rect.intersects(*rect)),
                                );
                            }
                        }
                    }
                });
            },
        );
        self.text.clear();
        for shape in &output.shapes {
            collect(&shape.shape, &mut self.text);
        }
        self.shapes = output.shapes.iter().map(|s| s.shape.clone()).collect();
        output.textures_delta.clear();
        assert!(self.app.notice().is_none(), "{:?}", self.app.notice());
        let pane = self.app.policy_pane();
        if pane.state_reservation_bytes() > 0 {
            assert!(
                pane.owned_state_bytes() <= pane.state_reservation_bytes(),
                "owned={} reserved={}",
                pane.owned_state_bytes(),
                pane.state_reservation_bytes()
            );
        }
        assert!(self.app.policy_metrics().prepared_peak <= 4 * 1024 * 1024);
    }
    pub fn settle(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            self.frame(vec![]);
            if self.app.policy_pane().ready() {
                // Paint controls whose availability changed with the just-published frame.
                self.frame(vec![]);
                break;
            }
            assert!(
                Instant::now() < deadline,
                "policy did not settle: {:?}",
                self.app.policy_pane().status()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    pub fn find(&self, label: &str) -> egui::Pos2 {
        self.text
            .iter()
            .find(|(s, _)| s == label)
            .unwrap_or_else(|| {
                panic!(
                    "Missing {label:?}: {:?}",
                    self.text.iter().map(|(s, _)| s).collect::<Vec<_>>()
                )
            })
            .1
            .center()
    }
    pub fn click_at(&mut self, pos: egui::Pos2) {
        self.frame(vec![egui::Event::PointerMoved(pos), button(pos, true)]);
        self.frame(vec![button(pos, false)]);
        self.frame(vec![]);
    }
    pub fn click_last(&mut self, label: &str) {
        let pos = self
            .text
            .iter()
            .rev()
            .find(|(s, _)| s == label)
            .unwrap_or_else(|| panic!("Missing last {label:?}: {:?}", self.text))
            .1
            .center();
        self.click_at(pos);
    }
    pub fn click(&mut self, label: &str) {
        let pos = self.find(label);
        self.click_at(pos);
    }
    pub fn set_reference_request(&mut self, revision: u64) {
        if !self.text.iter().any(|(s, _)| s == "k") {
            self.click("Reference source");
        }
        let revision_rect = |h: &Self| {
            let row = h
                .text
                .iter()
                .find(|(s, _)| s == "k")
                .unwrap_or_else(|| panic!("missing reference k after popup: {:?}", h.text))
                .1;
            h.text
                .iter()
                .find(|(s, r)| {
                    s.parse::<u64>().is_ok()
                        && r.left() > row.right()
                        && (r.center().y - row.center().y).abs() < 4.
                })
                .unwrap_or_else(|| panic!("missing reference revision editor: {:?}", h.text))
                .1
        };
        let pos = revision_rect(self).center();
        self.click_at(pos);
        assert!(
            self.ctx
                .memory(|m| m.focused())
                .and_then(|id| self.ctx.read_response(id))
                .is_some_and(|response| response.rect.contains(pos)),
            "reference revision editor did not receive focus at {pos:?}: {:?}",
            self.text
        );
        self.frame(vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
            egui::Event::Text(revision.to_string()),
        ]);
        let edited = revision_rect(self);
        assert!(
            self.text
                .iter()
                .any(|(s, r)| s == &revision.to_string() && *r == edited),
            "reference revision edit was not accepted: {:?}",
            self.text
        );
        self.frame(vec![egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        // Enter can close a menu. Escape closes any remaining popup without
        // accidentally reopening it by toggling its anchor button.
        self.frame(vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        self.frame(vec![]);
        assert!(!self.text.iter().any(|(s, _)| s == "k"));
    }
    pub fn has(&self, label: &str) -> bool {
        self.text.iter().any(|(s, _)| s.contains(label))
    }
}
pub fn button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}
fn collect(s: &egui::epaint::Shape, out: &mut Vec<(String, egui::Rect)>) {
    match s {
        egui::epaint::Shape::Text(t) => out.push((
            t.galley.text().into(),
            egui::Rect::from_min_size(t.pos, t.galley.size()),
        )),
        egui::epaint::Shape::Vec(v) => {
            for s in v {
                collect(s, out)
            }
        }
        _ => {}
    }
}
