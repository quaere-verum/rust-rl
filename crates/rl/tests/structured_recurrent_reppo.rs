#![cfg(feature = "neural")]

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;
#[allow(dead_code)]
#[path = "support/delayed_candidate.rs"]
mod delayed;

use burn::tensor::Device;
use delayed::{DELAY, DelayedCandidate, continue_action, model_config};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::{
        recurrent_reppo::{RecurrentReppo, StructuredPathwiseActor},
        reppo::ReppoConfig,
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        checkpoint::{Checkpoint, LearnerRecord},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        stateful::StatefulLearner,
    },
    data::{minibatch::shuffled_batches, temporal::SequenceConfig},
    nn::{
        structured::{
            PackLimits, RecurrentLightweightDistributionalQ, RecurrentLightweightPolicy,
            StructuredHistoryInput,
        },
        to_vec,
    },
    objectives::hl_gauss::HlGaussConfig,
    train::{
        checkpoint::{load_run, save_run},
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};
use std::sync::Arc;

type Run = TemporalOnPolicyTrainer<SyncVectorEnv<DelayedCandidate>, StructuredPathwiseActor>;

fn learner(seed: u64) -> RecurrentReppo<RecurrentLightweightPolicy> {
    learner_with_q_limit(seed, None)
}

fn learner_with_q_limit(
    seed: u64,
    q_limit: Option<usize>,
) -> RecurrentReppo<RecurrentLightweightPolicy> {
    let config = ReppoConfig {
        lambda: 0.,
        epochs: 1,
        minibatch_size: 8,
        auxiliary_coefficient: 0.1,
        hl_gauss: HlGaussConfig {
            min: -2.,
            max: 2.,
            bins: 7,
            sigma_ratio: 0.75,
        },
        ..Default::default()
    };
    let device = Device::flex().autodiff();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let policy = RecurrentLightweightPolicy::new(model_config(), &mut rng, &device).unwrap();
    let mut critic_config = model_config();
    critic_config.hidden_width += 3;
    if let Some(limit) = q_limit {
        critic_config.max_observations = limit;
    }
    let critic = RecurrentLightweightDistributionalQ::new(
        critic_config,
        config.hl_gauss.bins,
        true,
        &mut rng,
        &device,
    )
    .unwrap();
    RecurrentReppo::new_structured(policy, critic, config, seed).unwrap()
}

#[test]
fn structured_reppo_bootstraps_the_complete_successor_law_after_partial_and_sealed_restore() {
    let mut original = learner(19);
    let mut run = Run::new(
        SyncVectorEnv::new(vec![DelayedCandidate::new()]).unwrap(),
        &original,
        TemporalOnPolicyConfig {
            collection: OnPolicyConfig {
                rollout_steps: DELAY,
                gamma: 0.5,
                gae_lambda: None,
                normalize_advantages: false,
                ..Default::default()
            },
            sequence: SequenceConfig {
                burn_in: 0,
                learn_len: 2,
                n_step: 1,
            },
            ..Default::default()
        },
        21,
    )
    .unwrap();
    for _ in 0..DELAY - 2 {
        run.collect_tick(&original).unwrap();
    }
    let temp = tempfile::tempdir().unwrap();
    let partial_path = temp.path().join("partial");
    save_run(&partial_path, &original, &run).unwrap();
    let mut partial_learner = learner(31);
    let mut partial: Run = load_run(&partial_path, &mut partial_learner).unwrap();
    for _ in 0..2 {
        assert_eq!(
            run.collect_tick(&original).unwrap(),
            partial.collect_tick(&partial_learner).unwrap()
        );
    }
    let batch = run.seal(&original).unwrap();
    let preview = original.target_preview(batch).unwrap();
    let partial_batch = partial.seal(&partial_learner).unwrap();
    assert_eq!(batch.rows, partial_batch.rows);
    assert_eq!(
        partial_learner.target_preview(partial_batch).unwrap(),
        preview
    );
    let row = batch.rows.last().unwrap();
    let sample = &batch.sequences.samples[row.sample];
    let inputs = sample.exact_history_inputs().unwrap().collect::<Vec<_>>();
    let history = [StructuredHistoryInput {
        inputs: &inputs,
        learn_start: sample.learn_start,
    }];
    let next = row.time + 1;
    let Observation::Structured(frame) = &sample.observations[next] else {
        panic!("fixture successor must be structured");
    };
    let policy = original.policy();
    let p = policy
        .evaluate_histories(&history, PackLimits::default())
        .unwrap()[0]
        .complete_probabilities_f64(next)
        .unwrap();
    assert_eq!(p.len(), 2);
    assert!(p.iter().all(|probability| *probability > 0.01));
    let queries = frame
        .candidates
        .iter()
        .map(|candidate| (0, next, candidate.action))
        .collect::<Vec<_>>();
    let critic = original.critic();
    let q = critic
        .evaluate_histories_at(&history, &[(0, next)], PackLimits::default())
        .unwrap()
        .query(&queries)
        .unwrap();
    let q = to_vec(original.config().hl_gauss.decode(q.logits).unwrap()).unwrap();
    let alpha = f64::from(original.multipliers().0);
    let expected = 0.25
        * (p[0] * (f64::from(q[0]) - alpha * p[0].ln())
            + p[1] * (f64::from(q[1]) - alpha * p[1].ln()));
    assert!((f64::from(*preview.last().unwrap()) - expected).abs() < 1e-6);

    let mut shuffle_rng: ChaCha8Rng =
        serde_json::from_value(original.checkpoint().unwrap().metadata["rng"].clone()).unwrap();
    shuffled_batches(batch.sequences.samples.len(), 4, &mut shuffle_rng).unwrap();
    let path = temp.path().join("sealed");
    save_run(&path, &original, &run).unwrap();
    let mut restored = learner(41);
    let mut resumed: Run = load_run(&path, &mut restored).unwrap();
    assert_eq!(
        restored
            .target_preview(resumed.pending_batch().unwrap())
            .unwrap(),
        preview
    );
    let expected_report = run.update(&mut original).unwrap();
    let resumed_report = resumed.update(&mut restored).unwrap();
    assert_eq!(expected_report.metrics, resumed_report.metrics);
    assert!(expected_report.metrics["auxiliary_loss"] > 0.);
    assert_eq!(run.counters(), resumed.counters());
    let record = restored.checkpoint().unwrap();
    assert_eq!(
        record.metadata["rng"],
        serde_json::to_value(shuffle_rng).unwrap()
    );
    let expected_record = checkpoint_eq::normalized(&original.checkpoint().unwrap());
    assert!(
        checkpoint_eq::normalized(&record) == expected_record,
        "structured REPPO model, optimizer, RNG, or collection state differs after resume"
    );
    let mut corrupted = record.clone();
    corrupted.metadata["optimizer_steps"]["critic"] = serde_json::json!({});
    assert!(restored.restore(&corrupted).is_err());
    assert!(checkpoint_eq::normalized(&restored.checkpoint().unwrap()) == expected_record);
    let mut missing_predictor = record.clone();
    omit_adam(&mut missing_predictor, "critic", |name| {
        name.starts_with("predictor.")
    });
    let missing_predictor_error = restored.restore(&missing_predictor).unwrap_err();
    assert!(
        missing_predictor_error
            .to_string()
            .contains("predictor Adam membership"),
        "unexpected omitted predictor error: {missing_predictor_error}"
    );
    assert!(checkpoint_eq::normalized(&restored.checkpoint().unwrap()) == expected_record);
    for (owner, path) in [
        ("policy", "operation_head.layers.1.weight"),
        ("critic", "head.layers.1.bias"),
    ] {
        let mut missing_head = record.clone();
        omit_adam(&mut missing_head, owner, |name| name == path);
        let error = restored.restore(&missing_head).unwrap_err();
        assert!(
            error.to_string().contains("required Adam parameter clock"),
            "{error}"
        );
        assert!(checkpoint_eq::normalized(&restored.checkpoint().unwrap()) == expected_record);
    }
}

fn omit_adam(record: &mut LearnerRecord, owner: &str, omit: impl Fn(&str) -> bool) {
    use burn::optim::OptimizerRecord;
    use burn_pack::{Bytes, Reader, Writer};
    let mut model = serde_json::to_value(&record.models[owner]).unwrap();
    let bytes = serde_json::from_value(model["bytes"].take()).unwrap();
    let model = Reader::from_bytes(Bytes::from_bytes_vec(bytes)).unwrap();
    let omitted: std::collections::BTreeSet<_> = model
        .into_tensors()
        .unwrap()
        .into_iter()
        .filter(|tensor| omit(&tensor.name))
        .map(|tensor| tensor.param_id.unwrap())
        .collect();
    assert!(!omitted.is_empty());
    let moments =
        Reader::from_bytes(Bytes::from_bytes_vec(record.optimizers[owner].clone())).unwrap();
    let scalars = moments.scalars().clone();
    let metadata = moments.metadata().clone();
    let tensors: Vec<_> = moments
        .into_tensors()
        .unwrap()
        .into_iter()
        .filter(|tensor| !omitted.contains(&tensor.param_id.unwrap()))
        .collect();
    let mut writer = Writer::new(tensors);
    for (key, value) in scalars {
        let id = key.split_once('.').unwrap().0.parse::<u64>().unwrap();
        if !omitted.contains(&id) {
            writer = writer.with_scalar(&key, value);
        }
    }
    for (key, value) in metadata {
        writer = writer.with_metadata(&key, &value);
    }
    let packed = OptimizerRecord::from_bytes(writer.into_bytes().unwrap()).unwrap();
    let packed = rl::algorithms::OptimizerConfig::default()
        .build()
        .unwrap()
        .load_record(packed)
        .to_record()
        .into_bytes()
        .unwrap();
    record.optimizers.insert(owner.into(), packed.to_vec());
    record.metadata["optimizer_owners"][owner]
        .as_array_mut()
        .unwrap()
        .retain(|id| !omitted.contains(&id.as_u64().unwrap()));
    for id in omitted {
        record.metadata["optimizer_steps"][owner]
            .as_object_mut()
            .unwrap()
            .remove(&id.to_string());
    }
}

struct TerminalDecision {
    env: DelayedCandidate,
    choose: bool,
}

impl TerminalDecision {
    fn new(choose: bool) -> Self {
        Self {
            env: DelayedCandidate::new(),
            choose,
        }
    }

    fn strip_notebook(observation: &mut Observation) {
        let Observation::Structured(frame) = observation else {
            unreachable!("delayed task uses structured observations")
        };
        Arc::make_mut(frame).events.clear();
    }
}

impl Env for TerminalDecision {
    fn observation_space(&self) -> ObservationSpace {
        self.env.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        self.env.action_space()
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        let start = self.env.reset(seed)?;
        if !self.choose {
            return Ok(start);
        }
        let mut frame = start.observation;
        for _ in 0..DELAY {
            frame = self.env.step(&continue_action())?.observation;
        }
        Self::strip_notebook(&mut frame);
        Ok(EpisodeStart::fresh(frame))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        let mut step = self.env.step(action)?;
        step.terminated = true;
        if self.choose {
            Self::strip_notebook(&mut step.observation);
        }
        Ok(step)
    }
}

#[test]
fn structured_reppo_rejects_terminal_batch_beyond_q_history_budget_before_mutation() {
    let mut learner = learner_with_q_limit(47, Some(1));
    let mut run = TemporalOnPolicyTrainer::new(
        SyncVectorEnv::new(vec![TerminalDecision::new(false)]).unwrap(),
        &learner,
        TemporalOnPolicyConfig {
            collection: OnPolicyConfig {
                rollout_steps: 1,
                gae_lambda: None,
                ..Default::default()
            },
            sequence: SequenceConfig {
                burn_in: 0,
                learn_len: 1,
                n_step: 1,
            },
            ..Default::default()
        },
        22,
    )
    .unwrap();
    run.collect_tick(&learner).unwrap();
    let batch = run.seal(&learner).unwrap();
    let before = checkpoint_eq::normalized(&learner.checkpoint().unwrap());
    assert!(learner.update(batch).is_err());
    assert!(checkpoint_eq::normalized(&learner.checkpoint().unwrap()) == before);
}

#[test]
fn structured_reppo_preflights_complete_q_budget_for_each_scheduled_minibatch() {
    let device = Device::flex().autodiff();
    for exceeds_budget in [false, true] {
        // Lane 2 has one enabled key. A two-window minibatch fits only when
        // it includes lane 2. Compare two schedules differing in epoch two.
        let seed = (0..64)
            .find(|&seed| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let first_fits = shuffled_batches(3, 2, &mut rng).unwrap()[0].contains(&2);
                let second_exceeds = !shuffled_batches(3, 2, &mut rng).unwrap()[0].contains(&2);
                first_fits && second_exceeds == exceeds_budget
            })
            .unwrap();
        let config = ReppoConfig {
            epochs: 2,
            minibatch_size: 2,
            hl_gauss: HlGaussConfig {
                bins: 101,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut rng = ChaCha8Rng::seed_from_u64(47);
        let policy = RecurrentLightweightPolicy::new(model_config(), &mut rng, &device).unwrap();
        let mut critic_config = model_config();
        // Three histogram rows fit; four (both choice lanes) exceed the cap.
        critic_config.model.max_activation_elements = 350;
        let critic = RecurrentLightweightDistributionalQ::new(
            critic_config,
            config.hl_gauss.bins,
            false,
            &mut rng,
            &device,
        )
        .unwrap();
        let mut learner = RecurrentReppo::new_structured(policy, critic, config, seed).unwrap();
        let mut run = TemporalOnPolicyTrainer::new(
            SyncVectorEnv::new(vec![
                TerminalDecision::new(true),
                TerminalDecision::new(true),
                TerminalDecision::new(false),
            ])
            .unwrap(),
            &learner,
            TemporalOnPolicyConfig {
                collection: OnPolicyConfig {
                    rollout_steps: 1,
                    gae_lambda: None,
                    ..Default::default()
                },
                sequence: SequenceConfig {
                    burn_in: 0,
                    learn_len: 1,
                    n_step: 1,
                },
                ..Default::default()
            },
            22,
        )
        .unwrap();
        run.collect_tick(&learner).unwrap();
        let batch = run.seal(&learner).unwrap();
        let before = checkpoint_eq::normalized(&learner.checkpoint().unwrap());
        let update = learner.update(batch);
        if exceeds_budget {
            assert!(
                update
                    .unwrap_err()
                    .to_string()
                    .contains("lightweight activation budget exceeded")
            );
            assert_eq!(learner.updates(), 0);
            assert!(checkpoint_eq::normalized(&learner.checkpoint().unwrap()) == before);
        } else {
            assert_eq!(update.unwrap().updates, 4);
        }
    }
}
