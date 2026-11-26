//! Opt-in bounded genuine training; preserves the original generic proof.
use super::*;
use rl::diagnostics::{models::SnapshotBuilder, visualization::*};
use std::time::Instant;

pub(super) fn record(path: &std::path::Path) -> Result<()> {
    let started = Instant::now();
    let space = ActionSpace::Discrete { count: 2 };
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let config = PpoConfig {
        epochs: 1,
        minibatch_size: 4,
        ..Default::default()
    };
    let trainer_config = OnPolicyConfig {
        rollout_steps: 4,
        ..Default::default()
    };
    let mut learner = Ppo::new(
        ActorCritic {
            policy: Policy::new(2, space.clone(), &[4], &mut rng)?,
            value: Value::new(2, &[4], &mut rng)?,
        },
        config.clone(),
        SEED + 1,
    )?;
    let mut trainer = OnPolicyTrainer::new(
        OnPolicyCollector::new(SyncVectorEnv::new(vec![CounterBalance::new()])?, SEED + 2)?,
        trainer_config.clone(),
    )?;
    let run_id = "categorical-counterbalance-seed7301";
    let mut manifest = RunManifest::new(
        run_id,
        "bounded categorical CounterBalance",
        "PPO",
        2,
        space.clone(),
        0.99,
    )?;
    manifest.observation_labels = vec!["Position".into(), "Decision step / 4".into()];
    manifest.metadata.insert("configuration".into(), format!("PPO={config:?}; trainer={trainer_config:?}; networks=[2,4,2]/[2,4,1]; lanes=1; budget=48 vector/environment steps; seeds model=7301 learner=7302 collector=7303 probes=7304 evaluation_policy=7305 evaluation_env=[101,102]; evaluation=Sample,max_steps4,episodes2,discount0.99; snapshots=[0,4,12]"));
    manifest.environment_descriptor = Some(EnvironmentDescriptor {
        version: 1,
        observations: vec![
            ObservationField { id: "position".into(), label: "Position".into(), unit: None, description: Some("Starts at ±0.5; action changes position by ±0.25; range includes terminal observations".into()), kind: ObservationType::Numerical, range: Some([-1.5, 1.5]), coordinate: true },
            ObservationField { id: "episode_fraction".into(), label: "Decision step / 4".into(), unit: None, description: Some("Actual values 0,.25,.5,.75,1; 1 terminal. Intermediate grid coordinates are model-input counterfactuals".into()), kind: ObservationType::Numerical, range: Some([0., 1.]), coordinate: true },
        ],
        actions: ActionDescriptor::Discrete { choices: vec![
            DiscreteChoice { label: "Left".into(), meaning: Some("position -= 0.25".into()) },
            DiscreteChoice { label: "Right".into(), meaning: Some("position += 0.25".into()) },
        ] },
        clocks: ClockDescriptor { learner_revision: QuantityLabel { label: "Completed learner updates".into(), unit: None }, decision_step: QuantityLabel { label: "Environment decision steps".into(), unit: None }, simulation: None },
        reward: RewardDescriptor { label: "Reward".into(), unit: None, description: Some("-abs(position) after action; original environment units; larger is better".into()), components: vec![] },
        frames: vec![], preferences: VisualizationPreferences { coordinates: vec!["position".into(), "episode_fraction".into()], ..Default::default() },
    });
    let probes = ProbeSet::new(
        "ordinary-inputs",
        ProbePopulation::FixedReference,
        "three fixed CounterBalance observations",
        vec![
            Observation::new(vec![-0.5, 0.]),
            Observation::new(vec![0., 0.5]),
            Observation::new(vec![0.5, 0.]),
        ],
        None,
        space.clone(),
    )?;
    let mut recorder =
        Recorder::create(path, manifest, StorageConfig::default())?.with_snapshot_retention(3)?;
    let mut probe_rng = ChaCha8Rng::seed_from_u64(SEED + 3);
    let mut snapshots = 0;
    let mut diagnostics_seconds = 0.;
    let mut capture = |learner: &Ppo,
                       context: &RecordContext,
                       due: Due,
                       writer: &mut Recorder|
     -> Result<()> {
        if !due.snapshots && !due.probes {
            return Ok(());
        }
        let clock = Instant::now();
        let revision = context.identity.learner_update_revision;
        if learner.updates() != revision {
            return Err(Error::State(
                "actual PPO counter differs from diagnostic boundary".into(),
            ));
        }
        let policy = learner.policy();
        if due.snapshots {
            let mut snapshot = SnapshotBuilder::new(
                &format!("ppo-k{revision}"),
                context.clone(),
                2,
                space.clone(),
            )?;
            snapshot.policy("policy", &policy)?;
            writer.append(&snapshot.finish()?)?;
            snapshots += 1;
        }
        if due.probes {
            writer.append(&probes.record(context.clone())?)?;
            writer.append(&probes::policy(
                context.clone(),
                &policy,
                &probes,
                ActMode::Sample,
                &mut probe_rng,
            )?)?;
            let evaluation = evaluate(
                |_| Ok(CounterBalance::new()),
                &StochasticActor::new(&policy),
                &EvaluationConfig {
                    seeds: vec![101, 102],
                    policy_seed: SEED + 4,
                    max_steps: 4,
                    trajectories: 2,
                    policy_id: "recorded PPO policy".into(),
                    mode: ActMode::Sample,
                    discount: 0.99,
                    ..Default::default()
                },
            )?;
            writer.append(&evaluation.empirical_record(context.clone(), None)?)?;
            for j in 0..2 {
                writer.append(&evaluation.trajectory_record(
                    j,
                    context.clone(),
                    &format!("evaluation-{revision}-{j}"),
                    "CounterBalance",
                )?)?;
            }
            println!(
                "checkpoint actual_revision={revision} learner_updates={} learner_call={} collection_cycle={} snapshots={snapshots} evaluation_steps={} diagnostic_seconds={:.6} total_seconds={:.6}",
                learner.updates(),
                context.learner_call,
                context.collection_cycle,
                evaluation.environment_steps,
                clock.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64()
            );
        }
        diagnostics_seconds += clock.elapsed().as_secs_f64();
        Ok(())
    };
    capture(
        &learner,
        &RecordContext::new(run_id, 0),
        Due {
            snapshots: true,
            probes: true,
            ..Default::default()
        },
        &mut recorder,
    )?;
    let mut observer = DiagnosticObserver::new(
        recorder,
        RecordingSchedule {
            metrics: Some(Interval { first: 1, every: 1 }),
            probes: Some(Interval { first: 4, every: 8 }),
            snapshots: Some(Interval { first: 4, every: 8 }),
            monte_carlo: None,
        },
        capture,
    );
    let summary =
        trainer.train_observed(&mut learner, Budget::vector_steps(48), None, &mut observer)?;
    observer.finish()?;
    println!(
        "complete learner_updates={} counters={:?} collection_seconds={} learning_seconds={} diagnostic_seconds={diagnostics_seconds} total_seconds={} storage=finished",
        learner.updates(),
        summary.counters,
        summary.collection_seconds,
        summary.learning_seconds,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
