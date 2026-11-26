use super::{StructuredBatch, StructuredEncoder, StructuredModelConfig, heads};
use crate::{
    Error, Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
    },
    nn::{
        checkpoint::RecordBurnExt,
        network::{Mlp, Network},
        to_vec,
    },
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;

/// Complete-action Q representation only. No backups, replay or learner.
#[derive(Module, Debug)]
pub struct StructuredQ {
    pub encoder: StructuredEncoder,
    pub head: Mlp,
    #[module(skip)]
    config: StructuredModelConfig,
}
pub struct CandidateValues<'a, K> {
    pub keys: Vec<&'a K>,
    pub values: Tensor<1>,
    operations: Vec<usize>,
    enabled: Vec<bool>,
    operation_count: usize,
}
impl<'a, K> CandidateValues<'a, K> {
    /// Bind a shared head's differentiable values to the factual candidate rows.
    pub fn from_candidates(
        candidates: &'a [super::Candidate<K>],
        values: Tensor<1>,
        operation_count: usize,
    ) -> Result<Self> {
        ensure(
            values.dims() == [candidates.len()]
                && operation_count > 0
                && candidates.iter().all(|c| c.operation < operation_count),
            "candidate value tensor/operation shape mismatch",
        )?;
        Ok(Self {
            keys: candidates.iter().map(|c| &c.key).collect(),
            values,
            operations: candidates.iter().map(|c| c.operation).collect(),
            enabled: candidates.iter().map(|c| c.enabled).collect(),
            operation_count,
        })
    }
    /// Exact differentiable maxima of this same complete-action value tensor.
    /// Absent operations have no value; disabled candidates never enter maxima.
    pub fn operation_maxima(&self) -> Vec<Option<Tensor<1>>> {
        (0..self.operation_count)
            .map(|op| {
                let rows: Vec<_> = self
                    .operations
                    .iter()
                    .enumerate()
                    .filter(|&(i, &o)| o == op && self.enabled[i])
                    .map(|(i, _)| self.values.clone().narrow(0, i, 1))
                    .collect();
                if rows.is_empty() {
                    None
                } else {
                    Some(Tensor::cat(rows, 0).max())
                }
            })
            .collect()
    }
    pub fn greedy_key(&self) -> Result<&K> {
        let values = to_vec(self.values.clone())?;
        ensure(
            values.iter().all(|v| v.is_finite()),
            "nonfinite structured Q value",
        )?;
        let index = (0..values.len())
            .filter(|&i| self.enabled[i])
            .max_by(|&i, &j| values[i].total_cmp(&values[j]))
            .ok_or_else(|| Error::Action("no available Q candidate".into()))?;
        Ok(self.keys[index])
    }
}
impl StructuredQ {
    pub fn with_initialization(
        mut self,
        initialization: crate::nn::critics::CriticInitialization,
    ) -> Result<Self> {
        initialization.apply(&mut self.head)?;
        Ok(self)
    }

    pub fn new(
        config: StructuredModelConfig,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            encoder: StructuredEncoder::new(config.encoder.clone(), rng, device)?,
            head: Mlp::new_on_device(config.joint_width(), &[config.head_width], 1, rng, device)?,
            config,
        })
    }
    pub fn config(&self) -> &StructuredModelConfig {
        &self.config
    }
    pub fn evaluate<'a, K>(
        &self,
        batch: &'a StructuredBatch<K>,
    ) -> Result<Vec<CandidateValues<'a, K>>> {
        self.evaluate_with(batch, || self.encoder.forward(batch))
    }

    pub(crate) fn evaluate_with<'a, K>(
        &self,
        batch: &'a StructuredBatch<K>,
        encode: impl FnOnce() -> Result<super::EncodedWorld>,
    ) -> Result<Vec<CandidateValues<'a, K>>> {
        heads::check_live(batch, &self.config)?;
        let count = batch
            .candidates
            .iter()
            .try_fold(0usize, |n, c| n.checked_add(c.len()))
            .ok_or_else(|| Error::Shape("candidate batch overflow".into()))?;
        self.config.check_rows(count, self.config.joint_width())?;
        let world = encode()?;
        let mut inputs = Vec::with_capacity(count);
        for (sample, candidates) in batch.candidates.iter().enumerate() {
            for c in candidates {
                inputs.push(Tensor::cat(
                    vec![
                        heads::query(&world, &self.config, sample, c.operation),
                        heads::target(&world, &self.config, sample, c),
                    ],
                    1,
                ));
            }
        }
        let values = self.head.forward(Tensor::cat(inputs, 0)).reshape([count]);
        let mut offset = 0;
        Ok(batch
            .candidates
            .iter()
            .map(|candidates| {
                let start = offset;
                offset += candidates.len();
                CandidateValues {
                    keys: candidates.iter().map(|c| &c.key).collect(),
                    values: values.clone().narrow(0, start, offset - start),
                    operations: candidates.iter().map(|c| c.operation).collect(),
                    enabled: candidates.iter().map(|c| c.enabled).collect(),
                    operation_count: self.config.encoder.input.operations.len(),
                }
            })
            .collect())
    }
}
impl Checkpoint for StructuredQ {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut r = LearnerRecord::new("structured-q-v2", &self.config)?;
        r.model("q", self)?;
        Ok(r)
    }
    fn restore(&mut self, r: &LearnerRecord) -> Result<()> {
        r.check("structured-q-v2")?;
        let c: StructuredModelConfig = serde_json::from_value(r.metadata.clone())?;
        ensure(
            c == self.config,
            "structured Q checkpoint configuration mismatch",
        )?;
        *self = super::checkpoint::restore(r, "q", self)?;
        Ok(())
    }
}
