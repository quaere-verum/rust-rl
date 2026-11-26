//! Mathematical policy capability over factual complete-action candidates.
//! Architectures own feature packing and probability arithmetic; algorithms own
//! policy improvement, sampling, replay and optimizer state.
use super::{
    InputSchema, PackLimits, StructuredBatch, StructuredModelConfig, StructuredPolicy,
    StructuredSample,
};
use crate::{
    Result,
    core::{
        action::{ConditionalAction, ConditionalGreedy},
        checkpoint::Checkpoint,
        ensure,
    },
    nn::{distributions::Distribution, to_vec},
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, de::DeserializeOwned};
use std::{collections::HashSet, hash::Hash};

pub trait CandidatePolicy: Module + Checkpoint {
    /// Complete semantic and architectural identity, including non-tensor choices.
    type Configuration: Clone + PartialEq + Serialize + DeserializeOwned;
    fn configuration(&self) -> &Self::Configuration;
    fn validate_configuration(config: &Self::Configuration) -> Result<()>;
    fn input_schema(&self) -> &InputSchema;
    /// Derive from the current module's tensors, including restored execution mode.
    fn model_device(&self) -> Device;
    /// Exact complete-action probabilities in each input's candidate order.
    /// Disabled rows have zero mass. Enabled rows retain positive f64 mass.
    fn complete_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        limits: PackLimits,
    ) -> Result<Vec<Vec<f64>>>;
    /// Differentiable log likelihood of each requested complete semantic key.
    /// Return one tensor per input, in exactly the requested key order.
    fn complete_log_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        keys: &[Vec<K>],
        limits: PackLimits,
    ) -> Result<Vec<Tensor<1>>>;

    /// Deterministic deployment maximizes the operation, then its argument.
    /// Models with separate heads can override this to compare their logits directly.
    fn greedy_action(
        &self,
        sample: &StructuredSample<ConditionalAction>,
        limits: PackLimits,
    ) -> Result<ConditionalAction> {
        let mut probabilities =
            candidate_probabilities_checked(self, std::slice::from_ref(sample), limits)?;
        let (key, _) = ConditionalGreedy::Hierarchical.select(
            sample
                .candidates
                .iter()
                .map(|row| row.key)
                .zip(probabilities.pop().expect("one validated policy row")),
        )?;
        Ok(key)
    }
}

/// Optional reconstruction capability for serialized behavior actors.
pub trait BuildCandidatePolicy: CandidatePolicy {
    fn from_configuration(
        config: Self::Configuration,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self>;
}

impl CandidatePolicy for StructuredPolicy {
    type Configuration = StructuredModelConfig;

    fn configuration(&self) -> &Self::Configuration {
        self.config()
    }

    fn validate_configuration(config: &Self::Configuration) -> Result<()> {
        config.validate()
    }

    fn input_schema(&self) -> &InputSchema {
        &self.config().encoder.input
    }

    fn model_device(&self) -> Device {
        self.encoder.conv1.weight.val().device()
    }

    fn complete_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        limits: PackLimits,
    ) -> Result<Vec<Vec<f64>>> {
        cnn_probabilities(self, samples, limits)
    }

    fn complete_log_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        keys: &[Vec<K>],
        limits: PackLimits,
    ) -> Result<Vec<Tensor<1>>> {
        ensure(
            keys.len() == samples.len(),
            "candidate policy key row count differs",
        )?;
        let batch = StructuredBatch::pack_with_validated_schema(
            self.input_schema(),
            samples,
            limits,
            &self.model_device(),
        )?;
        let evaluation = self.evaluate(&batch)?;
        keys.iter()
            .enumerate()
            .map(|(row, keys)| evaluation.log_prob_many(row, keys))
            .collect()
    }

    fn greedy_action(
        &self,
        sample: &StructuredSample<ConditionalAction>,
        limits: PackLimits,
    ) -> Result<ConditionalAction> {
        let batch = StructuredBatch::pack_with_validated_schema(
            self.input_schema(),
            std::slice::from_ref(sample),
            limits,
            &self.model_device(),
        )?;
        Ok(self.valid().evaluate(&batch)?.greedy(0)?.key)
    }
}

impl BuildCandidatePolicy for StructuredPolicy {
    fn from_configuration(
        config: Self::Configuration,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::new(config, rng, device)
    }
}

fn validate_samples<P: CandidatePolicy, K: Clone + Eq + Hash>(
    policy: &P,
    samples: &[StructuredSample<K>],
) -> Result<()> {
    ensure(!samples.is_empty(), "empty candidate policy batch")?;
    for sample in samples {
        sample.validate(policy.input_schema())?;
        ensure(
            sample.candidates.iter().any(|candidate| candidate.enabled),
            "empty live candidate policy support",
        )?;
    }
    Ok(())
}

pub(crate) fn candidate_probabilities_checked<P: CandidatePolicy, K: Clone + Eq + Hash>(
    policy: &P,
    samples: &[StructuredSample<K>],
    limits: PackLimits,
) -> Result<Vec<Vec<f64>>> {
    validate_samples(policy, samples)?;
    let rows = policy.complete_probabilities(samples, limits)?;
    ensure(
        rows.len() == samples.len(),
        "candidate policy probability row count differs",
    )?;
    for (probabilities, sample) in rows.iter().zip(samples) {
        ensure(
            probabilities.len() == sample.candidates.len(),
            "candidate policy probability shape differs",
        )?;
        if !probabilities
            .iter()
            .zip(&sample.candidates)
            .all(|(&p, candidate)| {
                p.is_finite() && if candidate.enabled { p > 0. } else { p == 0. }
            })
            || (probabilities.iter().sum::<f64>() - 1.).abs() > 1e-10
        {
            return Err(crate::Error::Numerical(
                "invalid candidate policy probability mass or support".into(),
            ));
        }
    }
    Ok(rows)
}

pub(crate) fn candidate_log_probabilities_checked<P: CandidatePolicy, K: Clone + Eq + Hash>(
    policy: &P,
    samples: &[StructuredSample<K>],
    keys: &[Vec<K>],
    limits: PackLimits,
) -> Result<Vec<Tensor<1>>> {
    validate_samples(policy, samples)?;
    ensure(
        keys.len() == samples.len(),
        "candidate policy key row count differs",
    )?;
    for (sample, keys) in samples.iter().zip(keys) {
        let enabled: HashSet<_> = sample
            .candidates
            .iter()
            .filter(|candidate| candidate.enabled)
            .map(|candidate| &candidate.key)
            .collect();
        ensure(
            !keys.is_empty() && keys.iter().all(|key| enabled.contains(key)),
            "candidate policy likelihood key is unavailable",
        )?;
    }
    let rows = policy.complete_log_probabilities(samples, keys, limits)?;
    ensure(
        rows.len() == samples.len(),
        "candidate policy likelihood row count differs",
    )?;
    for (row, keys) in rows.iter().zip(keys) {
        ensure(
            row.dims() == [keys.len()],
            "candidate policy likelihood shape differs",
        )?;
    }
    // Both callers feed these rows directly to distribution_loss, which checks
    // finite likelihoods while validating the detached target distribution.
    Ok(rows)
}

/// f64 host softmax of the actual f32 logits. No probability flooring: loss of
/// positive support through extreme underflow is a diagnostic error.
pub(super) fn categorical(distribution: &Distribution, row: usize) -> Result<Vec<f64>> {
    let Distribution::Categorical { logits, masks } = distribution else {
        return Err(crate::Error::Config(
            "MPO requires categorical heads".into(),
        ));
    };
    let [batch, width] = logits.dims();
    ensure(row < batch, "MPO categorical row out of range")?;
    let values = to_vec(
        logits
            .clone()
            .slice([row..row + 1, 0..width])
            .reshape([width]),
    )?;
    let mask = &masks[row * width..(row + 1) * width];
    ensure(mask.iter().any(|x| *x), "empty MPO categorical support")?;
    if !values.iter().all(|v| v.is_finite()) {
        return Err(crate::Error::Numerical(
            "nonfinite MPO categorical logits".into(),
        ));
    }
    let max = values
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(&v, _)| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<_> = values
        .iter()
        .zip(mask)
        .map(|(&v, &m)| if m { (f64::from(v) - max).exp() } else { 0. })
        .collect();
    let sum: f64 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    if p.iter().zip(mask).any(|(&p, &m)| m && p == 0.) {
        return Err(crate::Error::Numerical(
            "MPO categorical support underflow".into(),
        ));
    }
    Ok(p)
}
fn cnn_probabilities<K: Clone + Eq + Hash>(
    policy: &StructuredPolicy,
    samples: &[StructuredSample<K>],
    limits: PackLimits,
) -> Result<Vec<Vec<f64>>> {
    let batch = StructuredBatch::pack_with_validated_schema(
        &policy.config().encoder.input,
        samples,
        limits,
        &Device::flex(),
    )?;
    let model = policy.valid();
    let evaluation = model.evaluate(&batch)?;
    samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let operations = categorical(evaluation.operations(), i)?;
            let mut p = vec![0.; s.candidates.len()];
            for (op, &op_p) in operations.iter().enumerate().filter(|(_, p)| **p > 0.) {
                if let Some(conditional) = evaluation.conditional(i, op)? {
                    let targets = categorical(&conditional.distribution, 0)?;
                    for (&row, &target) in conditional.rows.iter().zip(&targets) {
                        p[row] = op_p * target;
                    }
                } else {
                    let row = s
                        .candidates
                        .iter()
                        .position(|c| c.operation == op && c.enabled)
                        .expect("validated no-argument branch");
                    p[row] = op_p;
                }
            }
            let sum: f64 = p.iter().sum();
            for v in &mut p {
                *v /= sum;
            }
            if p.iter()
                .zip(&s.candidates)
                .any(|(&p, c)| c.enabled && p == 0.)
            {
                return Err(crate::Error::Numerical(
                    "MPO joint support underflow".into(),
                ));
            }
            Ok(p)
        })
        .collect()
}
