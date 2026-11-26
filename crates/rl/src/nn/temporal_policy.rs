//! Differentiable stochastic policies evaluated from chronological public history.
use super::{
    distributions::Distribution,
    network::{Mlp, Network},
    temporal::TemporalEncoder,
    temporal_models::{self, TemporalBackbone, TemporalModelArchitecture},
};
use crate::{
    Result,
    core::{
        action::ActionSpace, ensure, env::ObservationBatch, stateful::HistorySpec,
        stateful_on_policy::DistributionSnapshot,
    },
    data::temporal::SequenceBatch,
};
use burn::{
    module::{Module, Param},
    tensor::{Device, Tensor, TensorData},
};
use rand_chacha::ChaCha8Rng;

/// A temporal capability, deliberately separate from observation-only StochasticPolicy.
pub trait TemporalStochasticPolicy: Module {
    fn action_space(&self) -> &ActionSpace;
    fn history_spec(&self) -> &HistorySpec;
    fn distribution(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
    ) -> Result<Distribution>;
}

#[derive(Module, Debug)]
pub struct TemporalPolicy<E: TemporalEncoder = TemporalBackbone, H: Network = Mlp> {
    pub encoder: E,
    pub head: H,
    pub log_std: Param<Tensor<1>>,
    #[module(skip)]
    spec: HistorySpec,
    #[module(skip)]
    observation_only: bool,
}
impl TemporalPolicy {
    /// None selects a genuinely memoryless observation-only representation.
    pub fn new(
        spec: HistorySpec,
        recurrent_hidden: Option<usize>,
        head_hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        spec.validate()?;
        let encoder = TemporalBackbone::for_history(&spec, recurrent_hidden, rng, false)?;
        let head = Mlp::new(
            encoder.output_dim(),
            head_hidden,
            policy_width(&spec.action_space),
            rng,
        )?;
        Self::from_parts(encoder, head, spec, recurrent_hidden.is_none())
    }
    pub fn architecture(&self) -> Result<TemporalModelArchitecture> {
        let architecture = temporal_models::architecture(
            &self.encoder,
            &self.head,
            &self.spec,
            self.observation_only,
        )?;
        ensure(
            architecture.output_dim == policy_width(&self.spec.action_space)
                && self.log_std.val().dims() == [self.spec.action_space.dimensions()],
            "temporal policy output or standard deviation shape mismatch",
        )?;
        Ok(architecture)
    }
    pub fn from_architecture(
        architecture: &TemporalModelArchitecture,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Self::from_architecture_with_mode(architecture, rng, false)
    }
    pub(crate) fn from_architecture_with_mode(
        architecture: &TemporalModelArchitecture,
        rng: &mut ChaCha8Rng,
        inference: bool,
    ) -> Result<Self> {
        architecture.validate()?;
        ensure(
            architecture.version == 1
                && architecture.head_input_extra == 0
                && architecture.output_dim == policy_width(&architecture.spec.action_space),
            "temporal policy architecture action width mismatch",
        )?;
        let (encoder, head) = architecture.parts(rng, inference)?;
        let device = temporal_models::module_device(&head);
        Ok(Self {
            encoder,
            head,
            log_std: Param::from_tensor(Tensor::zeros(
                [architecture.spec.action_space.dimensions()],
                &device,
            )),
            spec: architecture.spec.clone(),
            observation_only: architecture.hidden_dim.is_none(),
        })
    }
}
impl<E: TemporalEncoder, H: Network> TemporalPolicy<E, H> {
    pub fn from_parts(
        encoder: E,
        head: H,
        spec: HistorySpec,
        observation_only: bool,
    ) -> Result<Self> {
        spec.validate()?;
        ensure(
            encoder.input_dim()
                == if observation_only {
                    spec.observation_dim
                } else {
                    spec.input_dim()
                }
                && encoder.output_dim() > 0
                && head.input_dim() == encoder.output_dim()
                && head.output_dim() == policy_width(&spec.action_space),
            "temporal policy encoder/head dimensions mismatch",
        )?;
        let device = temporal_models::module_device(&head);
        let log_std = Param::from_tensor(Tensor::zeros([spec.action_space.dimensions()], &device));
        Ok(Self {
            encoder,
            head,
            log_std,
            spec,
            observation_only,
        })
    }
    pub fn spec(&self) -> &HistorySpec {
        &self.spec
    }
    pub fn action_space(&self) -> &ActionSpace {
        &self.spec.action_space
    }
    pub fn observation_only(&self) -> bool {
        self.observation_only
    }
    pub fn device(&self) -> Device {
        temporal_models::module_device(self)
    }
    /// Read a post-observation representation without advancing the encoder.
    pub fn distribution_from_features(
        &self,
        features: Tensor<2>,
        observations: ObservationBatch<'_>,
    ) -> Result<Distribution> {
        self.distribution_from_features_using(features, observations, |x| self.head.forward(x))
    }
    pub(crate) fn distribution_from_features_using(
        &self,
        features: Tensor<2>,
        observations: ObservationBatch<'_>,
        forward: impl FnOnce(Tensor<2>) -> Tensor<2>,
    ) -> Result<Distribution> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::PolicyEvaluation);
        ensure(
            !observations.is_empty()
                && features.dims() == [observations.len(), self.encoder.output_dim()],
            "temporal policy feature shape mismatch",
        )?;
        for observation in observations.iter() {
            observation.validate(self.spec.observation_dim, &self.spec.action_space, true)?;
        }
        let output = forward(features);
        ensure(
            output.dims() == [observations.len(), policy_width(&self.spec.action_space)],
            "temporal policy head output mismatch",
        )?;
        match &self.spec.action_space {
            ActionSpace::Discrete { count } => {
                let mut masks = Vec::with_capacity(observations.len() * count);
                for observation in observations.iter() {
                    if let Some(mask) = observation.dense_mask() {
                        masks.extend_from_slice(mask);
                    } else {
                        masks.resize(masks.len() + count, true);
                    }
                }
                Ok(Distribution::Categorical {
                    logits: output,
                    masks,
                })
            }
            space => {
                let d = space.dimensions();
                let log_std = self
                    .log_std
                    .val()
                    .clamp(-20., 2.)
                    .reshape([1, d])
                    .repeat_dim(0, observations.len());
                let bounds = if let ActionSpace::Box { low, high } = space {
                    Some((low.clone(), high.clone()))
                } else {
                    None
                };
                Ok(Distribution::Gaussian {
                    mean: output,
                    log_std,
                    bounds,
                })
            }
        }
    }
}
impl<E: TemporalEncoder, H: Network> TemporalStochasticPolicy for TemporalPolicy<E, H> {
    fn action_space(&self) -> &ActionSpace {
        &self.spec.action_space
    }
    fn history_spec(&self) -> &HistorySpec {
        &self.spec
    }
    fn distribution(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
    ) -> Result<Distribution> {
        let features = temporal_models::selected_features(
            &self.encoder,
            batch,
            &self.spec,
            self.observation_only,
            &self.device(),
            tokens,
        )?;
        let observations: Vec<_> = tokens
            .iter()
            .map(|&(s, t)| &batch.samples[s].observations[t])
            .collect();
        self.distribution_from_features(features, ObservationBatch::from_refs(&observations))
    }
}
fn policy_width(space: &ActionSpace) -> usize {
    match space {
        ActionSpace::Discrete { count } => *count,
        _ => space.dimensions(),
    }
}

impl DistributionSnapshot {
    /// Tensor conversion belongs to the neural layer; the core DTO remains backend-independent.
    pub fn to_distribution(&self, device: &Device) -> Result<Distribution> {
        distribution_from_snapshots(std::slice::from_ref(self), device)
    }
    pub fn from_distribution(distribution: &Distribution, row: usize) -> Result<Self> {
        let snapshot = match distribution {
            Distribution::Categorical { logits, masks } => {
                let [n, k] = logits.dims();
                ensure(
                    row < n && n.checked_mul(k) == Some(masks.len()),
                    "invalid categorical snapshot row",
                )?;
                Self::Categorical {
                    logits: super::to_vec(logits.clone().slice([row..row + 1, 0..k]))?,
                    masks: masks[row * k..(row + 1) * k].to_vec(),
                }
            }
            Distribution::Gaussian {
                mean,
                log_std,
                bounds,
            } => {
                let [n, d] = mean.dims();
                ensure(
                    row < n && log_std.dims() == [n, d],
                    "invalid Gaussian snapshot row",
                )?;
                Self::Gaussian {
                    mean: super::to_vec(mean.clone().slice([row..row + 1, 0..d]))?,
                    log_std: super::to_vec(log_std.clone().slice([row..row + 1, 0..d]))?,
                    bounds: bounds.clone(),
                }
            }
        };
        snapshot.validate()?;
        Ok(snapshot)
    }
}

/// Stack behavior rows in caller order without normalizing or losing support information.
pub fn distribution_from_snapshots(
    snapshots: &[DistributionSnapshot],
    device: &Device,
) -> Result<Distribution> {
    ensure(!snapshots.is_empty(), "empty distribution snapshot batch")?;
    for snapshot in snapshots {
        snapshot.validate()?;
    }
    let n = snapshots.len();
    match &snapshots[0] {
        DistributionSnapshot::Categorical { logits, .. } => {
            let k = logits.len();
            let mut values = Vec::new();
            let mut masks = Vec::new();
            for snapshot in snapshots {
                let DistributionSnapshot::Categorical {
                    logits,
                    masks: row_masks,
                } = snapshot
                else {
                    return Err(crate::Error::Shape(
                        "mixed distribution snapshot families".into(),
                    ));
                };
                ensure(logits.len() == k, "distribution snapshot widths differ")?;
                values.extend_from_slice(logits);
                masks.extend_from_slice(row_masks);
            }
            Ok(Distribution::Categorical {
                logits: Tensor::from_data(TensorData::new(values, [n, k]), device),
                masks,
            })
        }
        DistributionSnapshot::Gaussian { mean, bounds, .. } => {
            let d = mean.len();
            let mut means = Vec::new();
            let mut deviations = Vec::new();
            for snapshot in snapshots {
                let DistributionSnapshot::Gaussian {
                    mean,
                    log_std,
                    bounds: row_bounds,
                } = snapshot
                else {
                    return Err(crate::Error::Shape(
                        "mixed distribution snapshot families".into(),
                    ));
                };
                ensure(
                    mean.len() == d && row_bounds == bounds,
                    "Gaussian snapshot transforms differ",
                )?;
                means.extend_from_slice(mean);
                deviations.extend_from_slice(log_std);
            }
            Ok(Distribution::Gaussian {
                mean: Tensor::from_data(TensorData::new(means, [n, d]), device),
                log_std: Tensor::from_data(TensorData::new(deviations, [n, d]), device),
                bounds: bounds.clone(),
            })
        }
        DistributionSnapshot::Conditional { .. } => Err(crate::Error::Shape(
            "a numerical distribution requires categorical or Gaussian snapshots".into(),
        )),
    }
}
