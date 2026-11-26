#[path = "support/planning_doors.rs"]
mod doors;

#[cfg(feature = "neural")]
use burn::tensor::Device;
#[cfg(feature = "neural")]
use doors::model_config;
use doors::{Doors, Public, SCAN, exact_values, frame, guess};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::approx::batch::{BatchOptions, DecisionSpan, FactualBatch};
#[cfg(feature = "neural")]
use rl::nn::structured::{FrozenRecurrentFeatures, PackLimits, RecurrentLightweightQ};
use rl::{
    Result,
    algorithms::fqi,
    approx::{
        FeatureMap, FeatureSpec, FittedQ, StatisticalActor,
        regression::{DiagonalPenalty, LinearModel, WeightedRidge},
    },
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::{Checkpoint, Snapshot},
        env::{EpisodeStart, Observation, ObservationSpace, PublicHistory},
        reward::{BatchObjective, Reward},
        stateful::{HistoryInput, PreviousInput, StatefulActor},
        structured::FeatureIdentity,
        temporal::{BehaviorMetadata, TemporalSink, TemporalSpaces, TemporalTransition},
    },
    data::temporal::EpisodeStore,
    envs::delayed_bit::DelayedBit,
    models::predictive::{PlanningModel, PlanningReturns, PredictiveModel, Transition},
    planning::{
        guidance::PolicyGuide,
        policy_evaluation::{PolicyEvaluationConfig, evaluate_reference_policy},
    },
    train::{
        checkpoint::{load_run, save_run},
        statistical::StatisticalRunSnapshot,
    },
};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};

fn root() -> PublicHistory {
    PublicHistory::new(frame(Public { reading: None }))
}

fn identify_model() -> TemporalSpaces {
    doors::spaces()
}

fn conditional(action: ConditionalAction) -> Action {
    Action::Conditional(action.into())
}

fn public(observation: &Observation) -> Public {
    let Observation::Structured(frame) = observation else {
        unreachable!()
    };
    Public {
        reading: match frame.scalars[0] {
            0. => None,
            x => Some(x > 0.),
        },
    }
}

#[derive(Clone)]
struct DoorFeatures(FeatureSpec);

impl DoorFeatures {
    fn new() -> Self {
        Self(FeatureSpec {
            namespace: "test.statistical.doors".into(),
            version: 1,
            terms: [
                "scan",
                "guess_left_prior",
                "guess_right_prior",
                "left_given_left",
                "right_given_left",
                "left_given_right",
                "right_given_right",
            ]
            .into_iter()
            .map(|name| FeatureIdentity::new(name, "indicator"))
            .collect(),
            configuration: vec!["public reading, semantic guess".into()],
            representation: None,
        })
    }
}

impl FeatureMap for DoorFeatures {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        let reading = public(history.observation()).reading;
        Ok(actions
            .iter()
            .map(|action| {
                let column = if *action == conditional(SCAN) {
                    0
                } else {
                    let side = usize::from(*action == conditional(guess(true)));
                    (match reading {
                        None => 1,
                        Some(false) => 3,
                        Some(true) => 5,
                    }) + side
                };
                let mut features = vec![0.; 7];
                features[column] = 1.;
                features
            })
            .collect())
    }
}

fn record_step(
    store: &mut EpisodeStore,
    id: u64,
    history: &PublicHistory,
    action: Action,
    reward: Reward,
    next: Observation,
    terminated: bool,
) -> Result<PublicHistory> {
    store.push(&TemporalTransition {
        episode_id: id,
        timestep: usize::try_from(history.timestep()).unwrap(),
        observation: history.observation().clone(),
        action: action.clone(),
        reward: reward.clone(),
        next_observation: next.clone(),
        terminated,
        truncated: false,
        behavior: BehaviorMetadata::default(),
    })?;
    history.append(action, reward, next)
}

fn factual_door_bank() -> Result<EpisodeStore> {
    let spaces = identify_model();
    let mut store = EpisodeStore::new(512, spaces.observations, spaces.actions)?;
    let mut id = 0;
    // Enumerate actual latent/reading outcomes with 17:3 counts instead of
    // inserting expected action values as training targets.
    for hidden in [false, true] {
        for choice in [false, true] {
            let initial = root();
            store.begin_episode(id, &EpisodeStart::fresh(initial.observation().clone()))?;
            record_step(
                &mut store,
                id,
                &initial,
                conditional(guess(choice)),
                Reward::One(if hidden == choice { 1. } else { 0. }),
                initial.observation().clone(),
                true,
            )?;
            id += 1;
            for reading in [false, true] {
                let next = frame(Public {
                    reading: Some(reading),
                });
                for _ in 0..if reading == hidden { 17 } else { 3 } {
                    store.begin_episode(id, &EpisodeStart::fresh(initial.observation().clone()))?;
                    let successor = record_step(
                        &mut store,
                        id,
                        &initial,
                        conditional(SCAN),
                        Reward::from_primitives(vec![0., 0.])?,
                        next.clone(),
                        false,
                    )?;
                    record_step(
                        &mut store,
                        id,
                        &successor,
                        conditional(guess(choice)),
                        Reward::One(if hidden == choice { 1. } else { 0. }),
                        next.clone(),
                        true,
                    )?;
                    id += 1;
                }
            }
        }
    }
    Ok(store)
}

/// Same predictive law as the existing doors fixture, exposing the actual
/// hypothetical public prefix instead of passing a hidden world to the actor.
struct HistoryDoors;
impl PlanningModel for HistoryDoors {
    type Context = PublicHistory;
    type Action = ConditionalAction;
    type Observation = bool;

    fn actions(&self, history: &PublicHistory, actions: &mut Vec<ConditionalAction>) {
        Doors.actions(&public(history.observation()), actions);
    }

    fn planning_returns(&self) -> PlanningReturns {
        doors::returns()
    }
}

impl PredictiveModel for HistoryDoors {
    type Hidden = bool;

    fn sample_root(&self, history: &PublicHistory, rng: &mut ChaCha8Rng) -> Result<bool> {
        Doors.sample_root(&public(history.observation()), rng)
    }

    fn rollout(&self, history: &PublicHistory) -> Option<ConditionalAction> {
        Doors.rollout(&public(history.observation()))
    }

    fn simulate(
        &self,
        history: &PublicHistory,
        hidden: &mut bool,
        action: &ConditionalAction,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<PublicHistory, bool>> {
        let outcome = Doors.simulate(&public(history.observation()), hidden, action, rng)?;
        Ok(Transition {
            observation: outcome.observation,
            reward: outcome.reward,
            duration: outcome.duration,
            next: outcome
                .next
                .map(|state| {
                    history.append(
                        conditional(*action),
                        Reward::from_primitives(vec![0.; usize::from(outcome.duration.get())])?,
                        frame(state),
                    )
                })
                .transpose()?,
        })
    }
}

#[test]
fn factual_fqi_learns_information_gathering_and_supplies_a_contingent_planning_policy() -> Result<()>
{
    let tape = factual_door_bank()?;
    let features = DoorFeatures::new();
    let batch = FactualBatch::from_store(
        &tape,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 0.9 },
            span: DecisionSpan::Single,
        },
    )?;
    let fit = fqi::fit(
        &batch,
        &WeightedRidge {
            penalty: DiagonalPenalty::Uniform(0.),
            ..Default::default()
        },
        fqi::FqiConfig {
            sweeps: NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        },
        fqi::Initial::Zero,
    )?;
    let actor = StatisticalActor::new(features, fit.model, 1, 0.)?;
    for ((_, q), truth) in actor
        .action_values(&root())?
        .into_iter()
        .zip(exact_values(0.5, true, 2))
    {
        assert!((q - truth).abs() < 1e-9);
    }
    assert_eq!(actor.greedy(&root())?, conditional(SCAN));
    for reading in [false, true] {
        let history = root().append(
            conditional(SCAN),
            Reward::from_primitives(vec![0., 0.])?,
            frame(Public {
                reading: Some(reading),
            }),
        )?;
        assert_eq!(actor.greedy(&history)?, conditional(guess(reading)));
    }
    let evaluation = evaluate_reference_policy(
        &HistoryDoors,
        &root(),
        &[SCAN, guess(false), guess(true)],
        &actor,
        PolicyEvaluationConfig {
            samples_per_action: NonZeroU32::new(2048).unwrap(),
            max_transitions: NonZeroU32::new(9000).unwrap(),
            termination_bound: NonZeroU16::new(2).unwrap(),
            epsilon: 0.1,
        },
        &mut ChaCha8Rng::seed_from_u64(190),
    )?
    .outcome
    .unwrap();
    assert!((evaluation.actions[0].mean - 0.6885).abs() < 0.04);
    assert!(evaluation.actions[0].mean > evaluation.actions[1].mean);
    Ok(())
}

#[test]
fn ties_are_semantic_and_lane_random_streams_resume_independently() -> Result<()> {
    let features = DoorFeatures::new();
    let model = FittedQ::new(
        features.spec().clone(),
        identify_model(),
        BatchObjective::Physical { gamma: 0.9 },
        LinearModel::new(vec![0., 1., 1., 0., 0., 0., 0.])?,
    )?;
    let mut actor = StatisticalActor::new(features.clone(), model, 41, 0.3)?;
    let checkpoint = actor.checkpoint()?;
    let restored = StatisticalActor::<_, LinearModel>::restore_with_features(
        features.clone(),
        &serde_json::from_slice(&serde_json::to_vec(&checkpoint)?)?,
    )?;
    let mut incompatible = features.clone();
    incompatible.0.version += 1;
    assert!(
        StatisticalActor::<_, LinearModel>::restore_with_features(incompatible, &checkpoint)
            .is_err()
    );
    let mut bad_record = checkpoint;
    bad_record.metadata["epsilon"] = 2.0.into();
    assert!(actor.restore(&bad_record).is_err());
    assert_eq!(actor.revision(), 41);
    let initial = root();
    let Observation::Structured(frame) = initial.observation() else {
        unreachable!()
    };
    let mut shuffled = (**frame).clone();
    shuffled.candidates.reverse();
    let shuffled = PublicHistory::new(Observation::Structured(shuffled.into()));
    assert_eq!(actor.greedy(&initial)?, conditional(guess(false)));
    assert_eq!(restored.greedy(&shuffled)?, conditional(guess(false)));
    let probabilities = actor.epsilon_greedy(&initial)?;
    for ((_, probability), expected) in probabilities.into_iter().zip([0.1, 0.45, 0.45]) {
        assert!((probability - expected).abs() < 1e-12);
    }
    let ordered = [guess(true), SCAN, guess(false)];
    assert_eq!(restored.prior(&shuffled, &ordered)?, vec![0., 0., 1.]);
    let h = [initial, shuffled];
    let mut rngs = [ChaCha8Rng::seed_from_u64(11), ChaCha8Rng::seed_from_u64(21)];
    let mut replay_rngs = rngs.clone();
    let s = h
        .iter()
        .map(|h| actor.reconstruct_history(&h.inputs()))
        .collect::<Result<Vec<_>>>()?;
    for _ in 0..24 {
        for lane in [1, 0] {
            let a = actor.decide(
                &s[lane],
                h[lane].observation(),
                ActMode::Sample,
                &mut rngs[lane],
            )?;
            let b = restored.decide(
                &s[1 - lane],
                h[1 - lane].observation(),
                ActMode::Sample,
                &mut replay_rngs[lane],
            )?;
            assert_eq!(a.action, b.action);
            assert_eq!(a.log_prob, b.log_prob);
        }
    }
    for (mut ran, mut replayed) in rngs.into_iter().zip(replay_rngs) {
        assert_eq!(ran.next_u64(), replayed.next_u64());
    }
    Ok(())
}

#[test]
#[cfg(feature = "neural")]
fn frozen_recurrent_features_use_causal_duration_and_bind_checkpoint_coefficients() -> Result<()> {
    let device = Device::default();
    let build = |seed| {
        RecurrentLightweightQ::new(
            model_config(),
            &mut ChaCha8Rng::seed_from_u64(seed),
            &device,
        )
    };
    let model = build(1)?;
    let checkpoint = model.checkpoint()?;
    let features = FrozenRecurrentFeatures::new(model, PackLimits::default())?;
    let mut recovered = build(2)?;
    recovered.restore(&checkpoint)?;
    let restored_features = FrozenRecurrentFeatures::new(recovered, PackLimits::default())?;
    assert_eq!(features.spec(), restored_features.spec());
    let final_observation = frame(Public {
        reading: Some(true),
    });
    let short = root().append(conditional(SCAN), 0., final_observation.clone())?;
    let long = root().append(
        conditional(SCAN),
        Reward::from_primitives(vec![0., 0.])?,
        final_observation,
    )?;
    let actions = long
        .observation()
        .finite_actions(&identify_model().actions)?;
    let original = features.encode(&long, &actions)?;
    assert_eq!(original, restored_features.encode(&long, &actions)?);
    assert_ne!(original, features.encode(&short, &actions)?);
    let reversed: Vec<_> = actions.iter().cloned().rev().collect();
    assert_eq!(
        original.into_iter().rev().collect::<Vec<_>>(),
        features.encode(&long, &reversed)?
    );

    let fitted = FittedQ::new(
        features.spec().clone(),
        identify_model(),
        BatchObjective::Physical { gamma: 0.9 },
        LinearModel::new(vec![1.; features.spec().dimension()])?,
    )?;
    let actor = StatisticalActor::new(features, fitted, 3, 0.)?;
    let record = actor.checkpoint()?;
    let resumed =
        StatisticalActor::<_, LinearModel>::restore_with_features(restored_features, &record)?;
    assert_eq!(actor.action_values(&long)?, resumed.action_values(&long)?);
    assert!(
        StatisticalActor::<_, LinearModel>::restore_with_features(
            FrozenRecurrentFeatures::new(build(3)?, PackLimits::default())?,
            &record,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn lower_value_rmse_can_reverse_the_greedy_ranking() -> Result<()> {
    let features = DoorFeatures::new();
    let truth = exact_values(0.5, true, 2);
    let incumbent = [3., 2., 2.];
    let challenger = [0.55, 0.60, 0.49];
    let evaluate = |predictions: [f64; 3]| -> Result<(f64, f64)> {
        let mut coefficients = predictions.to_vec();
        // Both complete policies correctly use the scan result if they scan.
        coefficients.extend([0.85, 0.15, 0.15, 0.85]);
        let fitted = FittedQ::new(
            features.spec().clone(),
            identify_model(),
            BatchObjective::Physical { gamma: 0.9 },
            LinearModel::new(coefficients)?,
        )?;
        let actor = StatisticalActor::new(features.clone(), fitted, 1, 0.)?;
        let ranking = actor.action_values(&root())?;
        let rmse = (ranking
            .iter()
            .zip(&truth)
            .map(|((_, prediction), exact)| (prediction - exact).powi(2))
            .sum::<f64>()
            / 3.)
            .sqrt();
        let selected = actor.greedy(&root())?;
        let index = ranking.iter().position(|(a, _)| *a == selected).unwrap();
        Ok((rmse, truth[index]))
    };
    let (incumbent_rmse, incumbent_return) = evaluate(incumbent)?;
    let (challenger_rmse, challenger_return) = evaluate(challenger)?;
    assert!(challenger_rmse < incumbent_rmse);
    assert!(challenger_return < incumbent_return);
    Ok(())
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct DelayedCue(FeatureSpec);

impl FeatureMap for DelayedCue {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        let cue = history.inputs()[0].observation.dense_values()[0];
        Ok(actions
            .iter()
            .map(|action| vec![f64::from((*action == Action::Discrete(1)) == (cue > 0.))])
            .collect())
    }
}

type DelayedRun = StatisticalRunSnapshot<SyncVectorEnv<DelayedBit>, DelayedCue, LinearModel>;
type DelayedActor = StatisticalActor<DelayedCue, LinearModel>;

fn delayed_statistical_run(
    delay: usize,
    capacity: usize,
) -> Result<(DelayedRun, DelayedActor, DelayedCue)> {
    let features = DelayedCue(FeatureSpec {
        namespace: "test.statistical.delayed_cue".into(),
        version: 1,
        terms: vec![FeatureIdentity::new(
            "matching_action",
            "first public cue and action id",
        )],
        configuration: vec![],
        representation: None,
    });
    let spaces = TemporalSpaces {
        observations: ObservationSpace::Dense(5),
        actions: ActionSpace::Discrete { count: 2 },
    };
    let fitted = FittedQ::new(
        features.spec().clone(),
        spaces.clone(),
        BatchObjective::Physical { gamma: 1. },
        LinearModel::new(vec![1.])?,
    )?;
    let actor = StatisticalActor::new(features.clone(), fitted, 17, 0.2)?;
    let environments = SyncVectorEnv::new(vec![
        DelayedBit::new(delay, None, false, 3)?,
        DelayedBit::new(delay, None, false, 5)?,
    ])?;
    let run = DelayedRun {
        collector: StatefulCollector::new(environments, actor.clone(), 59)?,
        episodes: EpisodeStore::new(capacity, spaces.observations, spaces.actions)?,
    };
    Ok((run, actor, features))
}

#[test]
fn statistical_run_checkpoint_replays_the_collector_and_rejects_a_substituted_prefix() -> Result<()>
{
    let (mut run, mut actor, _) = delayed_statistical_run(4, 64)?;
    run.validate_snapshot()?;
    let mut foreign_tape = run.episodes.clone();
    foreign_tape.begin_episode(
        0,
        &EpisodeStart::fresh(run.collector.observations()[0].clone()),
    )?;
    let mut wire = serde_json::to_value(&run)?;
    wire["episodes"] = serde_json::to_value(&foreign_tape)?;
    let invalid: DelayedRun = serde_json::from_value(wire)?;
    assert!(invalid.validate_snapshot().is_err());
    run.collector
        .collect(2, ActMode::Sample, &mut run.episodes)?;
    let h = run.collector.lane_state(0).unwrap().as_ref().unwrap();
    // A scalar total cannot prove how many ticks elapsed, even for dense inputs.
    assert!(
        actor
            .assimilate(
                &Some(h.clone()),
                HistoryInput {
                    observation: h.observation(),
                    previous: Some(PreviousInput::from_total(&Action::Discrete(0), 0.)),
                }
            )
            .is_err()
    );
    let substituted = PublicHistory::new(h.observation().clone());
    let mut wire = serde_json::to_value(&run)?;
    wire["collector"]["agents"][0]["state"] = serde_json::to_value(Some(substituted))?;
    let invalid: DelayedRun = serde_json::from_value(wire)?;
    assert!(invalid.validate_snapshot().is_err());

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("statistical-run.json");
    save_run(&path, &actor, &run)?;
    let mut restored: DelayedRun = load_run(&path, &mut actor)?;
    let live = run
        .collector
        .collect(8, ActMode::Sample, &mut run.episodes)?;
    let resumed = restored
        .collector
        .collect(8, ActMode::Sample, &mut restored.episodes)?;
    assert_eq!(live, resumed);
    assert_eq!(
        run.episodes.episodes().collect::<Vec<_>>(),
        restored.episodes.episodes().collect::<Vec<_>>()
    );
    run.validate_snapshot()?;
    restored.validate_snapshot()
}

#[test]
fn evicted_closed_statistical_lane_resumes_at_its_next_genuine_reset() -> Result<()> {
    let (mut run, mut actor, _) = delayed_statistical_run(2, 4)?;
    run.collector
        .collect(3, ActMode::Sample, &mut run.episodes)?;
    assert_eq!(run.collector.lane_needs_reset(0), Some(true));
    let evicted = run.collector.lane_episode(0).unwrap();
    assert!(run.episodes.episode(evicted).is_none());

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("closed-lane.run");
    save_run(&path, &actor, &run)?;
    let mut resumed: DelayedRun = load_run(&path, &mut actor)?;
    assert_eq!(
        run.collector
            .collect(1, ActMode::Sample, &mut run.episodes)?,
        resumed
            .collector
            .collect(1, ActMode::Sample, &mut resumed.episodes)?,
    );
    assert_eq!(
        run.episodes.episodes().collect::<Vec<_>>(),
        resumed.episodes.episodes().collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
fn fitted_statistical_policy_uses_native_publication_and_keeps_open_lane_history() -> Result<()> {
    let (mut run, actor, features) = delayed_statistical_run(2, 64)?;
    run.collector
        .collect(4, ActMode::Sample, &mut run.episodes)?;
    let revision = actor.revision();
    assert_eq!(run.collector.lane_revision(0), Some(revision));
    assert_eq!(run.collector.lane_needs_reset(0), Some(false));
    let batch = FactualBatch::from_store(
        &run.episodes,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 1. },
            span: DecisionSpan::Single,
        },
    )?;
    let fitted = fqi::fit(
        &batch,
        &WeightedRidge::default(),
        fqi::FqiConfig {
            target: fqi::TargetMode::MonteCarlo,
            sweeps: NonZeroUsize::MIN,
            ..Default::default()
        },
        fqi::Initial::Zero,
    )?;
    let next = StatisticalActor::new(features, fitted.model, revision + 1, 0.2)?;
    run.collector.publish(next)?;
    assert_eq!(run.collector.latest_revision(), revision + 1);
    assert_eq!(run.collector.lane_revision(0), Some(revision));
    run.collector
        .collect(4, ActMode::Sample, &mut run.episodes)?;
    assert_eq!(run.collector.lane_revision(0), Some(revision + 1));
    run.validate_snapshot()
}
