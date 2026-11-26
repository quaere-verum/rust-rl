use super::*;
use burn::{
    module::{Module, Param},
    tensor::{Tensor, activation::log_softmax},
};
use rl::nn::{
    checkpoint::RecordBurnExt,
    structured::{BuildCandidatePolicy, CandidatePolicy, InputSchema, PackLimits},
    to_vec,
};
use std::hash::Hash;

/// A complete-action softmax with one trainable score per operation.
#[derive(Module, Debug)]
struct OperationPolicy {
    scores: Param<Tensor<1>>,
    #[module(skip)]
    schema: InputSchema,
}

impl CandidatePolicy for OperationPolicy {
    type Configuration = InputSchema;

    fn configuration(&self) -> &InputSchema {
        &self.schema
    }
    fn validate_configuration(config: &InputSchema) -> rl::Result<()> {
        config.validate()
    }
    fn input_schema(&self) -> &InputSchema {
        &self.schema
    }
    fn model_device(&self) -> Device {
        self.scores.val().device()
    }

    fn complete_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        _: PackLimits,
    ) -> rl::Result<Vec<Vec<f64>>> {
        let values = to_vec(self.scores.val())?;
        Ok(samples
            .iter()
            .map(|sample| {
                let mut mass: Vec<_> = sample
                    .candidates
                    .iter()
                    .map(|candidate| {
                        if candidate.enabled {
                            f64::from(values[candidate.operation]).exp()
                        } else {
                            0.
                        }
                    })
                    .collect();
                let sum: f64 = mass.iter().sum();
                mass.iter_mut().for_each(|value| *value /= sum);
                mass
            })
            .collect())
    }

    fn complete_log_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        keys: &[Vec<K>],
        _: PackLimits,
    ) -> rl::Result<Vec<Tensor<1>>> {
        Ok(samples
            .iter()
            .zip(keys)
            .map(|(sample, keys)| {
                let enabled: Vec<_> = sample
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.enabled)
                    .collect();
                let logits = Tensor::cat(
                    enabled
                        .iter()
                        .map(|candidate| self.scores.val().narrow(0, candidate.operation, 1))
                        .collect(),
                    0,
                );
                let logs = log_softmax(logits, 0);
                Tensor::cat(
                    keys.iter()
                        .map(|key| {
                            let index = enabled
                                .iter()
                                .position(|candidate| candidate.key == *key)
                                .unwrap();
                            logs.clone().narrow(0, index, 1)
                        })
                        .collect(),
                    0,
                )
            })
            .collect())
    }
}

impl BuildCandidatePolicy for OperationPolicy {
    fn from_configuration(
        config: InputSchema,
        _: &mut ChaCha8Rng,
        device: &Device,
    ) -> rl::Result<Self> {
        config.validate()?;
        Ok(Self {
            scores: Param::from_tensor(Tensor::zeros([config.operations.len()], device)),
            schema: config,
        })
    }
}

impl Checkpoint for OperationPolicy {
    fn checkpoint(&self) -> rl::Result<LearnerRecord> {
        let mut record = LearnerRecord::new("test-operation-policy-v1", &self.schema)?;
        record.model("policy", self)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> rl::Result<()> {
        record.check("test-operation-policy-v1")?;
        assert_eq!(serde_json::to_value(&self.schema)?, record.metadata);
        let restored = if self.scores.val().is_require_grad() {
            record.load_model("policy", self)?
        } else {
            record.load_model_inference("policy", self)?
        };
        *self = restored;
        Ok(())
    }
}

#[test]
fn mpo_fits_and_resumes_a_policy_without_a_structured_encoder() {
    let case = fixtures().cases.into_iter().next().unwrap();
    let record = adapt_record(case.record, false);
    let model: StructuredModelConfig =
        serde_json::from_value(record.metadata["policy"].clone()).unwrap();
    let schema = model.encoder.input.clone();
    let observations = ObservationSpace::Structured(Arc::new(schema.observation_schema()));
    let actions = ActionSpace::Conditional(Arc::new(schema.action_grammar()));
    let device = Device::flex().autodiff();
    let build = |seed| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let policy =
            OperationPolicy::from_configuration(schema.clone(), &mut rng, &device).unwrap();
        let q = StructuredQ::new(model.clone(), &mut rng, &device).unwrap();
        CandidateMpo::new(
            policy,
            q,
            CandidateDdqnConfig::default(),
            CandidateMpoConfig::default(),
            observations.clone(),
            actions.clone(),
        )
        .unwrap()
    };
    let batch = sequence(&case.batch);
    let mut learner = build(14);
    let first = learner.update(&batch).unwrap();
    assert_eq!(first.metrics["projection_accepted"], 1.);
    let saved = learner.checkpoint().unwrap();
    let mut restored = build(27);
    restored.restore(&saved).unwrap();
    let a = learner.update(&batch).unwrap();
    let b = restored.update(&batch).unwrap();
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
    assert_eq!(
        checkpoint_eq::normalized(&learner.checkpoint().unwrap()),
        checkpoint_eq::normalized(&restored.checkpoint().unwrap())
    );

    let actor = learner
        .behavior_actor(&CandidateMpoBehavior::default(), 5)
        .unwrap();
    let restored_actor: CandidateMpoActor<OperationPolicy> =
        serde_json::from_slice(&serde_json::to_vec(&actor).unwrap()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(6);
    let mut restored_rng = rng.clone();
    for _ in 0..8 {
        let a = actor
            .decide(
                &(),
                &batch.samples[0].observations[0],
                ActMode::Sample,
                &mut rng,
            )
            .unwrap();
        let b = restored_actor
            .decide(
                &(),
                &batch.samples[0].observations[0],
                ActMode::Sample,
                &mut restored_rng,
            )
            .unwrap();
        assert_eq!(a.action, b.action);
        assert_eq!(a.log_prob, b.log_prob);
    }
}
