use std::{
    cell::{Cell, RefCell},
    num::NonZeroUsize,
};

use rl::{
    Result,
    algorithms::{fqi, lspi},
    approx::{
        FeatureMap, FeatureSpec, FittedQ, FixedLocal, GateAssignment, GateConfig, GateDescriptor,
        LocalLayout, StatisticalActor,
        batch::{BatchObjective, BatchOptions, DecisionSpan, FactualBatch},
        regression::{
            DiagonalPenalty, LinearModel, LogisticTargetKind, Predictor, WeightedLogistic,
            WeightedRidge,
        },
    },
    core::{
        action::{Action, ActionSpace},
        env::{EpisodeStart, Observation, PublicHistory},
        reward::Reward,
        structured::FeatureIdentity,
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
    data::temporal::EpisodeStore,
    planning::guidance::PolicyGuide,
};

fn spec(name: &str, width: usize) -> FeatureSpec {
    FeatureSpec {
        namespace: name.into(),
        version: 1,
        terms: (0..width)
            .map(|i| FeatureIdentity::new(format!("x{i}"), "identity"))
            .collect(),
        configuration: vec![],
        representation: None,
    }
}

#[derive(Clone)]
struct Map {
    spec: FeatureSpec,
    encode_one: fn(&PublicHistory, usize) -> Vec<f64>,
}

impl FeatureMap for Map {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }
    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        Ok(actions
            .iter()
            .map(|a| match a {
                Action::Discrete(a) => (self.encode_one)(history, *a),
                _ => panic!("expected a discrete action"),
            })
            .collect())
    }
}

fn obs(state: f32, mask: [bool; 2]) -> Observation {
    Observation::dense(vec![state], Some(mask.into()))
}

fn both(state: f32) -> Observation {
    obs(state, [true, true])
}

fn store() -> EpisodeStore {
    EpisodeStore::new(128, 1, ActionSpace::Discrete { count: 2 }).unwrap()
}

#[derive(Clone)]
struct Move {
    action: usize,
    reward: Reward,
    next: Observation,
    terminated: bool,
    truncated: bool,
}

fn finish(action: usize, reward: f32) -> Move {
    Move {
        action,
        reward: reward.into(),
        next: obs(-1., [false, false]),
        terminated: true,
        truncated: false,
    }
}

fn episode(store: &mut EpisodeStore, id: u64, mut observation: Observation, moves: &[Move]) {
    store
        .begin_episode(id, &EpisodeStart::fresh(observation.clone()))
        .unwrap();
    for (timestep, step) in moves.iter().enumerate() {
        store
            .push(&TemporalTransition {
                episode_id: id,
                timestep,
                observation,
                action: Action::Discrete(step.action),
                reward: step.reward.clone(),
                next_observation: step.next.clone(),
                terminated: step.terminated,
                truncated: step.truncated,
                behavior: BehaviorMetadata {
                    revision: 12,
                    log_prob: Some(-0.5),
                },
            })
            .unwrap();
        observation = step.next.clone();
    }
}

fn close(value: f64, reference: f64) {
    assert!((value - reference).abs() < 1e-9, "{value} != {reference}");
}

fn fqi_config(sweeps: usize, horizon: usize, target: fqi::TargetMode<'_>) -> fqi::FqiConfig<'_> {
    fqi::FqiConfig {
        sweeps: NonZeroUsize::new(sweeps).unwrap(),
        horizon: NonZeroUsize::new(horizon).unwrap(),
        target,
        clip_successor: false,
    }
}

fn tabular(states: usize) -> Map {
    Map {
        spec: spec("test.tabular", states * 2),
        encode_one: |_, _| unreachable!(),
    }
}

fn tabular_three() -> Map {
    Map {
        encode_one: |h, a| {
            let mut result = vec![0.; 6];
            result[h.observation().dense_values()[0] as usize * 2 + a] = 1.;
            result
        },
        ..tabular(3)
    }
}

fn ridge() -> WeightedRidge {
    WeightedRidge {
        penalty: DiagonalPenalty::Uniform(0.),
        ..Default::default()
    }
}

struct CountedMap {
    base: Map,
    new_spec: FeatureSpec,
    changed: Cell<bool>,
    requests: RefCell<Vec<(u64, Vec<Action>)>>,
}

impl CountedMap {
    fn new() -> Self {
        let base = tabular_three();
        let mut new_spec = base.spec.clone();
        new_spec.version += 1;
        Self {
            base,
            new_spec,
            changed: Cell::new(false),
            requests: RefCell::new(Vec::new()),
        }
    }
}

impl FeatureMap for CountedMap {
    fn spec(&self) -> &FeatureSpec {
        if self.changed.get() {
            &self.new_spec
        } else {
            self.base.spec()
        }
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        self.requests
            .borrow_mut()
            .push((history.timestep(), actions.to_vec()));
        self.base.encode(history, actions)
    }
}

fn cache_fixture() -> EpisodeStore {
    let mut tape = store();
    episode(
        &mut tape,
        0,
        both(0.),
        &[
            Move {
                action: 0,
                reward: 0.0.into(),
                next: both(1.),
                terminated: false,
                truncated: false,
            },
            finish(1, 1.),
        ],
    );
    tape
}

#[test]
fn factual_features_reuse_executed_actions_and_successors_across_fits() {
    let tape = cache_fixture();
    let features = CountedMap::new();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 0.9 },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let config = fqi_config(2, 1, fqi::TargetMode::Greedy);
    fqi::fit(&batch, &ridge(), config, fqi::Initial::Zero).unwrap();
    lspi::fit(&batch, &lspi::LspiConfig::default(), None).unwrap();
    assert_eq!(
        *features.requests.borrow(),
        vec![
            (0, vec![Action::Discrete(0)]),
            (1, vec![Action::Discrete(1)]),
            (1, vec![Action::Discrete(0)]),
        ]
    );
}

#[test]
fn bound_batch_rejects_a_changed_feature_revision() {
    let tape = cache_fixture();
    let features = CountedMap::new();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 0.9 },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let config = fqi_config(2, 1, fqi::TargetMode::Greedy);
    fqi::fit(&batch, &ridge(), config, fqi::Initial::Zero).unwrap();
    features.changed.set(true);
    assert!(fqi::fit(&batch, &ridge(), config, fqi::Initial::Zero).is_err());
}

struct Fixed([f64; 2]);
impl PolicyGuide<PublicHistory, Action> for Fixed {
    fn prior(&self, _: &PublicHistory, actions: &[Action]) -> Result<Vec<f64>> {
        Ok(actions
            .iter()
            .map(|a| {
                let Action::Discrete(a) = a else {
                    unreachable!()
                };
                self.0[*a]
            })
            .collect())
    }
}

#[test]
fn fqi_sweeps_are_frozen_and_maximize_only_over_legal_actions() {
    let mut tape = store();
    episode(
        &mut tape,
        0,
        both(0.),
        &[
            Move {
                action: 0,
                reward: Reward::from_primitives(vec![1., 2.]).unwrap(),
                next: obs(1., [false, true]),
                terminated: false,
                truncated: false,
            },
            finish(1, 4.),
        ],
    );
    episode(&mut tape, 1, both(0.), &[finish(1, 2.5)]);
    episode(&mut tape, 2, both(1.), &[finish(0, 100.)]);

    let features = tabular_three();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 0.5 },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let once = fqi::fit(
        &batch,
        &ridge(),
        fqi_config(1, 1, fqi::TargetMode::Greedy),
        fqi::Initial::Zero,
    )
    .unwrap();
    close(
        once.model
            .predictor()
            .predict(&[1., 0., 0., 0., 0., 0.])
            .unwrap(),
        2.,
    );
    let twice = fqi::fit(
        &batch,
        &ridge(),
        fqi_config(2, 1, fqi::TargetMode::Greedy),
        fqi::Initial::Zero,
    )
    .unwrap();
    let actor = StatisticalActor::new(features, twice.model, 2, 0.).unwrap();
    let history = PublicHistory::new(both(0.));
    close(actor.action_values(&history).unwrap()[0].1, 3.);
    assert_eq!(actor.greedy(&history).unwrap(), Action::Discrete(0));
}

#[test]
fn returns_keep_physical_duration_recorded_final_frame_and_terminal_precedence() {
    let mut tape = store();
    let first = Move {
        action: 0,
        reward: Reward::from_primitives(vec![1., 2.]).unwrap(),
        next: both(1.),
        terminated: false,
        truncated: false,
    };
    let truncation = Move {
        action: 1,
        reward: 4_f32.into(),
        next: both(2.),
        terminated: false,
        truncated: true,
    };
    episode(
        &mut tape,
        10,
        both(0.),
        &[first.clone(), truncation.clone()],
    );
    episode(
        &mut tape,
        11,
        both(0.),
        &[
            first.clone(),
            Move {
                terminated: true,
                ..truncation
            },
        ],
    );
    episode(&mut tape, 12, both(0.), &[first]);

    let features = tabular_three();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 0.5 },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let n = NonZeroUsize::new(3).unwrap();
    let truncated = batch.n_step(0, n).unwrap();
    close(truncated.reward, 3.);
    close(truncated.discount, 0.125);
    assert_eq!(truncated.duration, 3);
    assert_eq!(truncated.successor.observation().dense_values(), &[2.]);
    assert_eq!(batch.monte_carlo(0).unwrap(), None);
    let terminated = batch.n_step(2, n).unwrap();
    close(terminated.reward, 3.);
    assert_eq!(terminated.discount, 0.);
    assert_eq!(batch.monte_carlo(2).unwrap(), Some(3.));
    let incomplete = batch.n_step(4, n).unwrap();
    close(incomplete.reward, 2.);
    close(incomplete.discount, 0.25);
    assert_eq!(incomplete.successor.observation().dense_values(), &[1.]);
    assert_eq!(batch.monte_carlo(4).unwrap(), None);
    assert_eq!(batch.rows()[4].step.behavior.revision, 12);
    assert_eq!(batch.rows()[4].episode_id, 12);
}

#[test]
fn selected_decision_span_keeps_forced_rewards_without_fabricating_actions() {
    let mut tape = store();
    episode(
        &mut tape,
        1,
        both(0.),
        &[
            Move {
                action: 1,
                reward: 1_f32.into(),
                next: obs(1., [true, false]),
                terminated: false,
                truncated: false,
            },
            finish(0, 2.),
        ],
    );
    let features = tabular_three();
    let batch = FactualBatch::from_store_with(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 0.5 },
            span: DecisionSpan::NextSelected,
        },
        |point| Ok((point.legal.len() > 1).then_some(2.)),
    )
    .unwrap();
    assert_eq!(batch.rows().len(), 1);
    assert_eq!(batch.rows()[0].step.action, Action::Discrete(1));
    assert_eq!(batch.rows()[0].weight, 2.);
    let target = batch.n_step(0, NonZeroUsize::MIN).unwrap();
    assert_eq!(target.duration, 2);
    assert_eq!(target.discount, 0.);
    close(target.reward, 2.);
}

#[test]
fn mixed_endpoints_equal_their_controls_and_interior_requires_factual_completion() {
    let mut tape = store();
    episode(
        &mut tape,
        0,
        both(0.),
        &[
            Move {
                action: 0,
                reward: 0_f32.into(),
                next: both(1.),
                terminated: false,
                truncated: false,
            },
            finish(0, 2.),
        ],
    );
    episode(&mut tape, 1, both(1.), &[finish(1, 4.)]);
    episode(
        &mut tape,
        2,
        both(2.),
        &[Move {
            action: 1,
            reward: 2_f32.into(),
            next: both(1.),
            terminated: false,
            truncated: true,
        }],
    );
    let features = tabular_three();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 1. },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let fitted = |target| {
        fqi::fit(
            &batch,
            &ridge(),
            fqi_config(2, 1, target),
            fqi::Initial::Zero,
        )
        .unwrap()
        .model
        .predictor()
        .coefficients()
        .to_vec()
    };
    let fitted_greedy = fitted(fqi::TargetMode::Greedy);
    let factual = fitted(fqi::TargetMode::MonteCarlo);
    assert_eq!(
        fitted(fqi::TargetMode::Mixed { mc_weight: 0. }),
        fitted_greedy
    );
    assert_eq!(fitted(fqi::TargetMode::Mixed { mc_weight: 1. }), factual);
    close(fitted_greedy[0], 4.);
    close(fitted_greedy[5], 6.);
    close(factual[0], 2.);
    let half = fitted(fqi::TargetMode::Mixed { mc_weight: 0.5 });
    close(half[0], 3.);
    close(half[5], 0.);
    close(fitted(fqi::TargetMode::Fixed(&Fixed([0.25, 0.75])))[0], 3.5);
    close(
        fqi::fit(
            &batch,
            &ridge(),
            fqi_config(1, 2, fqi::TargetMode::Greedy),
            fqi::Initial::Zero,
        )
        .unwrap()
        .model
        .predictor()
        .coefficients()[0],
        2.,
    );
}

#[test]
fn lstdq_matches_exact_fixed_policy_and_lspi_changes_to_legal_greedy_choice() {
    let mut tape = store();
    episode(
        &mut tape,
        0,
        both(0.),
        &[
            Move {
                action: 0,
                reward: 1_f32.into(),
                next: both(1.),
                terminated: false,
                truncated: false,
            },
            finish(0, 0.),
        ],
    );
    episode(&mut tape, 1, both(0.), &[finish(1, 2.)]);
    episode(&mut tape, 2, both(1.), &[finish(1, 4.)]);
    let features = tabular_three();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 0.5 },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let config = lspi::LspiConfig {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..Default::default()
    };
    let evaluation = lspi::evaluate(&batch, &config, &Fixed([0.25, 0.75])).unwrap();
    let coefficients = evaluation.model.predictor().coefficients();
    close(coefficients[0], 2.5);
    close(coefficients[1], 2.);
    close(coefficients[2], 0.);
    close(coefficients[3], 4.);
    assert_eq!(evaluation.solve.rank, 4);

    let result = lspi::fit(&batch, &config, None).unwrap();
    assert!(result.stable);
    assert_eq!(result.iterations[0].changed_actions, 1);
    let actor = StatisticalActor::new(features, result.model, 6, 0.).unwrap();
    assert_eq!(
        actor.greedy(&PublicHistory::new(both(0.))).unwrap(),
        Action::Discrete(0)
    );
    assert_eq!(
        actor
            .greedy(&PublicHistory::new(obs(1., [true, false])))
            .unwrap(),
        Action::Discrete(0)
    );
}

#[test]
fn logistic_distinguishes_bootstraps_from_factual_terminal_success() {
    let mut tape = store();
    episode(&mut tape, 0, both(0.), &[finish(0, 1.)]);
    episode(&mut tape, 1, both(0.), &[finish(1, -1.)]);
    let features = tabular_three();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::TerminalSuccess {
                success: 1.,
                failure: -1.,
            },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let model = fqi::fit(
        &batch,
        &WeightedLogistic::default(),
        fqi_config(1, 1, fqi::TargetMode::MonteCarlo),
        fqi::Initial::Zero,
    )
    .unwrap()
    .model;
    assert_eq!(
        model.predictor().target_kind(),
        LogisticTargetKind::FactualBinary
    );
    let positive = [1., 0., 0., 0., 0., 0.];
    let negative = [0., 1., 0., 0., 0., 0.];
    assert!(model.signed_terminal_return(&positive).unwrap() > 0.9);
    assert!(model.signed_terminal_return(&negative).unwrap() < -0.9);
    assert!(
        fqi::fit(
            &batch,
            &WeightedLogistic::default(),
            fqi_config(1, 1, fqi::TargetMode::Greedy),
            fqi::Initial::Zero,
        )
        .is_err()
    );
    let soft = fqi::fit(
        &batch,
        &WeightedLogistic {
            target_kind: LogisticTargetKind::SoftLabels,
            ..Default::default()
        },
        fqi_config(1, 1, fqi::TargetMode::Greedy),
        fqi::Initial::Zero,
    )
    .unwrap();
    assert_eq!(
        soft.model.predictor().target_kind(),
        LogisticTargetKind::SoftLabels
    );
    assert_eq!(tape.episode(1).unwrap().steps[0].reward.total(), -1.);
}

#[test]
fn terminal_success_clipping_changes_only_the_successor_backup() {
    let mut tape = store();
    episode(
        &mut tape,
        0,
        both(0.),
        &[
            Move {
                action: 0,
                reward: 0_f32.into(),
                next: obs(1., [true, false]),
                terminated: false,
                truncated: false,
            },
            finish(0, 1.),
        ],
    );
    let features = tabular_three();
    let objective = BatchObjective::TerminalSuccess {
        success: 1.,
        failure: -1.,
    };
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective,
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    let initial = FittedQ::new(
        features.spec.clone(),
        batch.spaces().clone(),
        objective,
        LinearModel::new(vec![0., 0., 3., 100., 0., 0.]).unwrap(),
    )
    .unwrap();
    let config = fqi_config(1, 1, fqi::TargetMode::Greedy);
    let raw = fqi::fit(&batch, &ridge(), config, fqi::Initial::Fitted(&initial)).unwrap();
    let bounded = fqi::fit(
        &batch,
        &ridge(),
        fqi::FqiConfig {
            clip_successor: true,
            ..config
        },
        fqi::Initial::Fitted(&initial),
    )
    .unwrap();
    close(raw.model.predictor().coefficients()[0], 3.);
    close(bounded.model.predictor().coefficients()[0], 1.);
    close(initial.predictor().coefficients()[2], 3.);

    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 1. },
            span: DecisionSpan::Single,
        },
    )
    .unwrap();
    assert!(
        fqi::fit(
            &batch,
            &ridge(),
            fqi::FqiConfig {
                clip_successor: true,
                ..config
            },
            fqi::Initial::Zero
        )
        .is_err()
    );
}

#[test]
fn lstdq_weights_apply_to_the_normalized_reward_and_moment_equations() {
    let mut tape = store();
    episode(&mut tape, 0, both(0.), &[finish(0, 1.)]);
    episode(&mut tape, 1, both(0.), &[finish(0, 9.)]);
    let features = Map {
        spec: spec("test.constant", 1),
        encode_one: |_, _| vec![1.],
    };
    let batch = FactualBatch::from_store_with(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 1. },
            span: DecisionSpan::Single,
        },
        |point| Ok(Some(if point.episode_id == 0 { 1. } else { 3. })),
    )
    .unwrap();
    let solution = lspi::evaluate(
        &batch,
        &lspi::LspiConfig {
            penalty: DiagonalPenalty::Uniform(0.5),
            ..Default::default()
        },
        &Fixed([0.25, 0.75]),
    )
    .unwrap();
    close(solution.model.predictor().coefficients()[0], 7. / 1.5);
}

#[test]
fn lstdq_keeps_a_representable_moment_when_the_weight_fraction_underflows() {
    let mut tape = store();
    episode(&mut tape, 0, both(0.), &[finish(0, 0.)]);
    episode(&mut tape, 1, both(0.), &[finish(1, 1.)]);
    let features = Map {
        spec: spec("test.scaled_action", 1),
        encode_one: |_, a| vec![a as f64 * 1e210],
    };
    let batch = FactualBatch::from_store_with(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 1. },
            span: DecisionSpan::Single,
        },
        |point| Ok(Some(if point.episode_id == 0 { 1e308 } else { 1e-100 })),
    )
    .unwrap();
    let solution = lspi::evaluate(
        &batch,
        &lspi::LspiConfig {
            penalty: DiagonalPenalty::Uniform(0.),
            ..Default::default()
        },
        &Fixed([0.25, 0.75]),
    )
    .unwrap();
    close(solution.model.predictor().predict(&[1e210]).unwrap(), 1.);
}

#[derive(Clone)]
struct Region(FeatureSpec);
impl GateDescriptor for Region {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }
    fn describe(&self, h: &PublicHistory) -> Result<Vec<f64>> {
        Ok(vec![f64::from(h.observation().dense_values()[0])])
    }
}

#[test]
fn soft_gate_identity_and_hard_gates_recover_context_dependent_ranking() {
    let affine = Map {
        spec: spec("test.affine", 3),
        encode_one: |history, a| {
            vec![
                1.,
                f64::from(history.observation().dense_values()[0]),
                a as f64,
            ]
        },
    };
    let config = GateConfig {
        centers: vec![vec![0.], vec![1.]],
        scale: vec![vec![1.]],
        assignment: GateAssignment::Gaussian { sigma: 0.5 },
    };
    let gaussian = FixedLocal::new(
        affine.clone(),
        Region(spec("test.region", 1)),
        config.clone(),
        LocalLayout::GlobalPlusLocal,
    )
    .unwrap();
    let center = PublicHistory::new(both(0.));
    let row = gaussian
        .encode(&center, &[Action::Discrete(1)])
        .unwrap()
        .remove(0);
    let expected = 1. / (1. + (-2_f64).exp());
    assert_eq!(&row[..3], &[1., 0., 1.]);
    close(row[3], expected);
    close(row[5], expected);
    close(row[3] + row[6], 1.);
    close(row[5] + row[8], 1.);
    assert_eq!(
        gaussian
            .penalty(
                &DiagonalPenalty::PerCoefficient(vec![0., 1., 1.]),
                &DiagonalPenalty::Uniform(2.),
            )
            .unwrap(),
        DiagonalPenalty::PerCoefficient(vec![0., 1., 1., 2., 2., 2., 2., 2., 2.]),
    );

    let local = FixedLocal::new(
        affine.clone(),
        Region(spec("test.region", 1)),
        GateConfig {
            assignment: GateAssignment::Nearest,
            ..config
        },
        LocalLayout::Local,
    )
    .unwrap();
    let mut tape = store();
    episode(&mut tape, 0, both(0.), &[finish(0, 1.)]);
    episode(&mut tape, 1, both(0.), &[finish(1, 0.)]);
    episode(&mut tape, 2, both(1.), &[finish(0, 0.)]);
    episode(&mut tape, 3, both(1.), &[finish(1, 1.)]);
    let options = BatchOptions {
        objective: BatchObjective::TerminalSuccess {
            success: 1.,
            failure: 0.,
        },
        span: DecisionSpan::Single,
    };
    let cfg = fqi_config(1, 1, fqi::TargetMode::MonteCarlo);
    let global_fit = fqi::fit(
        &FactualBatch::from_store(&tape, &affine, options).unwrap(),
        &ridge(),
        cfg,
        fqi::Initial::Zero,
    )
    .unwrap();
    let local_fit = fqi::fit(
        &FactualBatch::from_store(&tape, &local, options).unwrap(),
        &ridge(),
        cfg,
        fqi::Initial::Zero,
    )
    .unwrap();
    let global = StatisticalActor::new(affine, global_fit.model, 1, 0.).unwrap();
    let fitted = StatisticalActor::new(local, local_fit.model, 1, 0.).unwrap();
    let other = PublicHistory::new(both(1.));
    assert_eq!(
        global.greedy(&center).unwrap(),
        global.greedy(&other).unwrap()
    );
    assert_eq!(fitted.greedy(&center).unwrap(), Action::Discrete(0));
    assert_eq!(fitted.greedy(&other).unwrap(), Action::Discrete(1));
}
