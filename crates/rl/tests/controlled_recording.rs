#![cfg(all(feature = "diagnostics", feature = "neural"))]

#[path = "support/controlled/environment.rs"]
mod environment;

use environment::ContextLottery;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        analysis::match_conditional_rows,
        evaluation::{DeterministicActor, EvaluationConfig, evaluate_conditional_with_frames},
        models::{ModelBundle, ModelLimits, SnapshotBuilder},
        probes::{self, ProbeSet},
        schema::*,
        storage::*,
    },
    nn::probes::{CriticProbe, QuantileProbe},
    objectives::quantile::QuantileLoss,
    prelude::*,
};

const RUN: &str = "quantile-conditional-recording";
const POLICY: &str = "D4PG deterministic";
const SAMPLES: u64 = 16;

fn record_checkpoint(
    writer: &mut Recorder,
    learner: &D4pg,
    probes: &ProbeSet,
    mut context: RecordContext,
) -> Result<()> {
    let policy = learner.policy();
    let critic = learner.critic();
    let checkpoint = format!("d4pg-{}", learner.updates());
    context.identity.checkpoint_id = Some(checkpoint.clone());

    let mut snapshot = SnapshotBuilder::new(
        &checkpoint,
        context.clone(),
        3,
        probes.action_space().clone(),
    )?;
    snapshot.deterministic("policy", &policy)?;
    snapshot.quantiles("critic", &critic, learner.grid())?;
    writer.append(&snapshot.finish()?)?;

    context.probe_set = Some(probes.id().into());
    let actions = probes.fixed_actions().unwrap();
    let action_values = actions
        .iter()
        .map(|action| {
            let Action::Continuous(values) = action else {
                unreachable!()
            };
            values[0]
        })
        .collect();
    let forecast = QuantileProbe::new(&critic, learner.grid())?.predict(
        probes.observations().into(),
        rl::nn::matrix(action_values, actions.len(), 1),
    )?;
    writer.append(&probes::critic_for_policy(
        context.clone(),
        forecast,
        ActionCondition::Fixed,
        Some(actions),
        POLICY,
        ActMode::Deterministic,
    )?)?;

    let actor = DeterministicActor::new(&policy);
    for (row, (lottery_context, action)) in [1, -1].into_iter().zip(actions).enumerate() {
        let first_seed = 1000 + (row as u64) * SAMPLES;
        let mut env = ContextLottery::new(lottery_context, 0);
        env.reset(Some(17))?;
        let evaluation = evaluate_conditional_with_frames(
            &env,
            &actor,
            Some(action),
            &EvaluationConfig {
                seeds: (first_seed..first_seed + SAMPLES).collect(),
                policy_seed: 31,
                policy_id: POLICY.into(),
                discount: 1.,
                max_steps: 4,
                ..Default::default()
            },
            |_| {},
        )?;
        writer.append(&evaluation.empirical_record(context.clone(), Some(row as u64))?)?;
    }
    Ok(())
}

#[test]
fn trained_quantile_recording_reopens_with_exact_conditioning_and_provenance() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("recording");
    let space = ContextLottery::new(1, 7).action_space().clone();
    let probes = ProbeSet::new(
        "fixed-actions",
        ProbePopulation::FixedReference,
        "Reset observations for each context with fixed initial exposure",
        vec![
            Observation::new(vec![0., 0., 1.]),
            Observation::new(vec![0., 0., -1.]),
        ],
        Some(vec![
            Action::Continuous(vec![0.25]),
            Action::Continuous(vec![0.75]),
        ]),
        space.clone(),
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut config = D4pgConfig::new(10)?;
    config.loss = QuantileLoss::Absolute;
    let mut learner = D4pg::new(
        Deterministic::new(3, space.clone(), &[8], &mut rng)?,
        QValue::new(3, 1, 10, &[8], &mut rng)?,
        config,
        11,
    )?;
    let envs = vec![ContextLottery::new(1, 13), ContextLottery::new(-1, 19)];
    let mut trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(envs)?, 23)?,
        OffPolicyConfig {
            warmup_transitions: 16,
            replay_capacity: 128,
            batch_size: 16,
            n_step: 3,
            gamma: 1.,
            updates_per_transition: 0.25,
            ..Default::default()
        },
        29,
    )?;
    let mut manifest =
        RunManifest::new(RUN, "conditional recording fixture", "D4PG", 3, space, 1.)?;
    let provenance = serde_json::json!({"initialize":7,"learner":11,"collector":23,"replay":29});
    manifest
        .metadata
        .insert("provenance".into(), provenance.to_string());
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default())?
        .with_buffering(BufferConfig::default())?;

    let initial = RecordContext::new(RUN, 0);
    writer.append(&probes.record(initial.clone())?)?;
    record_checkpoint(&mut writer, &learner, &probes, initial)?;
    trainer.train(&mut learner, Budget::vector_steps(32), None)?;
    assert_eq!(trainer.counters().env_steps, 64);
    let revision = learner.updates();
    assert!(revision > 0);
    let mut final_context = RecordContext::new(RUN, revision);
    final_context.counters = *trainer.counters();
    record_checkpoint(&mut writer, &learner, &probes, final_context)?;
    writer.finish()?;

    let recording = Recording::open(&path, ReadLimits::default())?;
    assert!(recording.complete());
    assert!(recording.issues().is_empty());
    assert_eq!(
        recording.manifest().metadata["provenance"],
        provenance.to_string()
    );
    let chunks = (0..recording.entries().len())
        .map(|i| recording.load(i))
        .collect::<Result<Vec<_>>>()?;
    let definition = chunks
        .iter()
        .find(|chunk| matches!(chunk.record, Record::ProbeSet(_)))
        .unwrap();
    assert_eq!(definition.arrays["row_ids"].as_u64()?, &[0, 1]);
    assert_eq!(
        definition.arrays["observations"].as_f32()?,
        &[0., 0., 1., 0., 0., -1.]
    );

    let snapshots = chunks
        .iter()
        .filter(|chunk| matches!(chunk.record, Record::ModelSnapshot(_)))
        .map(|chunk| {
            ModelBundle::load(chunk, ModelLimits::default())?;
            Ok(chunk.context.identity.learner_update_revision)
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(snapshots, [0, revision]);

    let mut rows = Vec::new();
    for sample in &chunks {
        let Record::Empirical(empirical) = &sample.record else {
            continue;
        };
        assert!(matches!(
            empirical.evaluation.initial_action,
            ActionCondition::Fixed
        ));
        assert!(
            empirical.evaluation.complete_returns
                && empirical.evaluation.independent_future_streams
        );
        assert_eq!(sample.arrays[&empirical.returns].len(), SAMPLES as usize);
        let critic = chunks
            .iter()
            .find(|chunk| {
                matches!(chunk.record, Record::Critic(_))
                    && chunk.context.identity == sample.context.identity
            })
            .unwrap();
        let row = empirical.probe_row.unwrap() as usize;
        match_conditional_rows(
            critic,
            definition,
            row,
            sample,
            definition,
            recording.manifest(),
        )?;

        let mut wrong = sample.clone();
        wrong.arrays.insert(
            empirical.initial_action.clone().unwrap(),
            Array::f32(vec![1, 1], vec![0.5])?,
        );
        assert!(
            match_conditional_rows(
                critic,
                definition,
                row,
                &wrong,
                definition,
                recording.manifest()
            )
            .is_err()
        );
        rows.push((sample.context.identity.learner_update_revision, row));
    }
    assert_eq!(rows, [(0, 0), (0, 1), (revision, 0), (revision, 1)]);
    Ok(())
}
