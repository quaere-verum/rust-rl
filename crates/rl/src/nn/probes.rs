//! Explicit, optional inference probes. No probe runs inside ordinary forwards.
//!
//! Callers supply identities at inspection/checkpoint boundaries. Predictions
//! remain separate from observed episode returns; no registry or history is kept.
use super::{
    critics::{moments::Moments, quantile::QuantileGrid, scalar::ActionValue},
    network::{Mlp, Network},
    target::frozen_copy,
};
use crate::{
    Error, Result,
    core::{ensure, env::ObservationBatch},
};
use burn::tensor::{Tensor, activation::relu};

pub use crate::core::recording::ProbeIdentity;

/// Actual observed episode returns, distinct from any critic's prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalReturns {
    pub identity: ProbeIdentity,
    pub episode_returns: Vec<f64>,
}

/// A detached activation with shape `[batch, layer_width]`.
#[derive(Debug, Clone)]
pub struct NamedActivation {
    pub name: String,
    pub tensor: Tensor<2>,
}

/// Opt-in capability independent of [`Network`]. Implementations return only
/// requested activations, in request order, without changing live model state.
pub trait ActivationProbe {
    fn activations(&self, input: Tensor<2>, names: &[&str]) -> Result<Vec<NamedActivation>>;
    /// Streaming extension. Custom probes inherit a correct collecting fallback;
    /// large networks can override it to release each layer after inspection.
    fn visit_activations(
        &self,
        input: Tensor<2>,
        names: &[&str],
        mut visitor: impl FnMut(NamedActivation) -> Result<()>,
    ) -> Result<()>
    where
        Self: Sized,
    {
        for activation in self.activations(input, names)? {
            visitor(activation)?;
        }
        Ok(())
    }
}

fn activation_name(name: &str, layers: usize) -> Result<(usize, bool)> {
    let mut parts = name.split('.');
    let prefix = parts.next();
    let index = parts.next().and_then(|value| value.parse::<usize>().ok());
    let stage = parts.next();
    if let (Some("layers"), Some(index), Some(stage), None) = (prefix, index, stage, parts.next())
        && index < layers
        && matches!(stage, "pre" | "post")
        && name == format!("layers.{index}.{stage}")
    {
        return Ok((index, stage == "post"));
    }
    Err(Error::Config(format!(
        "unsupported activation name: {name}"
    )))
}

impl ActivationProbe for Mlp {
    /// Stable names are `layers.0.pre`, `layers.0.post`, and so on. `pre` is
    /// the affine output; `post` includes the hidden-layer ReLU. The final layer
    /// is linear, so its pre/post values coincide. Duplicate names are allowed.
    /// An empty request performs no forward; evaluation stops at the last
    /// requested layer, and unrequested intermediate activations are not saved.
    fn activations(&self, input: Tensor<2>, names: &[&str]) -> Result<Vec<NamedActivation>> {
        let mut out = Vec::with_capacity(names.len());
        self.visit_activations(input, names, |a| {
            out.push(a);
            Ok(())
        })?;
        // Preserve the original request-order contract, including duplicate names.
        Ok(names
            .iter()
            .map(|&name| {
                let i = out.iter().position(|a| a.name == name).unwrap();
                out.remove(i)
            })
            .collect())
    }
    fn visit_activations(
        &self,
        input: Tensor<2>,
        names: &[&str],
        mut visitor: impl FnMut(NamedActivation) -> Result<()>,
    ) -> Result<()> {
        ensure(
            input.dims()[0] > 0 && input.dims()[1] == self.input_dim(),
            "invalid activation probe input shape",
        )?;
        let requests: Vec<_> = names
            .iter()
            .map(|name| activation_name(name, self.layers.len()))
            .collect::<Result<_>>()?;
        let Some(last) = requests.iter().map(|request| request.0).max() else {
            return Ok(());
        };
        let snapshot = frozen_copy(self);
        let mut x = input.without_autodiff();
        for (layer_index, layer) in snapshot.layers.iter().enumerate().take(last + 1) {
            x = layer.forward(x);
            for (index, &(requested, post)) in requests.iter().enumerate() {
                if requested == layer_index && !post {
                    visitor(NamedActivation {
                        name: names[index].into(),
                        tensor: x.clone().without_autodiff(),
                    })?;
                }
            }
            if layer_index + 1 < snapshot.layers.len() {
                x = relu(x);
            }
            for (index, &(requested, post)) in requests.iter().enumerate() {
                if requested == layer_index && post {
                    visitor(NamedActivation {
                        name: names[index].into(),
                        tensor: x.clone().without_autodiff(),
                    })?;
                }
            }
        }
        Ok(())
    }
}

/// An explicitly computational intervention applied to every input row. Post-
/// activation replacement may leave the range of physically attainable states.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum InterventionValue {
    Set(f32),
    Add(f32),
    Scale(f32),
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Intervention {
    pub layer: String,
    pub neuron: usize,
    pub value: InterventionValue,
}
/// An isolated inference copy of a built-in MLP with temporary activation
/// interventions. Saved parameters and the source model are never modified.
#[derive(burn::module::Module, Debug)]
pub struct IntervenedMlp {
    base: Mlp,
    #[module(skip)]
    interventions: Vec<Intervention>,
}
impl IntervenedMlp {
    pub fn new(base: &Mlp, interventions: Vec<Intervention>) -> Result<Self> {
        ensure(!base.layers.is_empty(), "cannot inspect an empty MLP")?;
        for intervention in &interventions {
            let (layer, _) = activation_name(&intervention.layer, base.layers.len())?;
            let value = match intervention.value {
                InterventionValue::Set(v)
                | InterventionValue::Add(v)
                | InterventionValue::Scale(v) => v,
            };
            ensure(
                intervention.neuron < base.layers[layer].weight.val().dims()[1]
                    && value.is_finite(),
                "invalid intervention neuron or value",
            )?;
        }
        Ok(Self {
            base: frozen_copy(base),
            interventions,
        })
    }
    fn stage(&self, mut x: Tensor<2>, name: &str) -> Tensor<2> {
        let n = x.dims()[0];
        for intervention in self.interventions.iter().filter(|i| i.layer == name) {
            let j = intervention.neuron;
            let value = match intervention.value {
                InterventionValue::Set(value) => Tensor::full([n, 1], value, &x.device()),
                InterventionValue::Add(value) => x.clone().slice([0..n, j..j + 1]) + value,
                InterventionValue::Scale(value) => x.clone().slice([0..n, j..j + 1]) * value,
            };
            x = x.slice_assign([0..n, j..j + 1], value);
        }
        x
    }
    fn run(
        &self,
        mut x: Tensor<2>,
        names: &[&str],
        last: usize,
    ) -> (Tensor<2>, Vec<NamedActivation>) {
        let mut recorded = std::collections::BTreeMap::new();
        x = x.without_autodiff();
        for (i, layer) in self.base.layers.iter().enumerate().take(last + 1) {
            let pre = format!("layers.{i}.pre");
            let post = format!("layers.{i}.post");
            x = self.stage(layer.forward(x), &pre);
            if names.contains(&pre.as_str()) {
                recorded.insert(pre, x.clone());
            }
            if i + 1 < self.base.layers.len() {
                x = relu(x);
            }
            x = self.stage(x, &post);
            if names.contains(&post.as_str()) {
                recorded.insert(post, x.clone());
            }
        }
        let values = names
            .iter()
            .map(|name| NamedActivation {
                name: (*name).into(),
                tensor: recorded[*name].clone().without_autodiff(),
            })
            .collect();
        (x, values)
    }
}
impl Network for IntervenedMlp {
    fn input_dim(&self) -> usize {
        self.base.input_dim()
    }
    fn output_dim(&self) -> usize {
        self.base.output_dim()
    }
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.run(input, &[], self.base.layers.len() - 1).0
    }
}
impl ActivationProbe for IntervenedMlp {
    fn activations(&self, input: Tensor<2>, names: &[&str]) -> Result<Vec<NamedActivation>> {
        ensure(
            input.dims()[0] > 0 && input.dims()[1] == self.input_dim(),
            "invalid intervention input dimensions",
        )?;
        let last = names
            .iter()
            .map(|name| activation_name(name, self.base.layers.len()).map(|v| v.0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max();
        Ok(last.map_or_else(Vec::new, |last| self.run(input, names, last).1))
    }
}

/// Detached predictions. None of these variants represents sampled returns.
#[derive(Debug, Clone)]
// These opt-in transient results own their tensors and are never stored per
// rollout row. Keeping tensors directly accessible avoids variant-specific boxes.
#[allow(clippy::large_enum_variant)]
pub enum CriticPrediction {
    /// `[batch]` scalar action values.
    Scalar(Tensor<1>),
    /// `[batch, action_count]` action values; these are not probabilities.
    Discrete(Tensor<2>),
    /// `[batch, quantile_count]` values and shared `[quantile_count]` metadata.
    Quantiles {
        values: Tensor<2>,
        locations: Vec<f32>,
        masses: Vec<f32>,
    },
    /// `[batch, atom_count]` probabilities and shared fixed support.
    CategoricalReturn {
        atoms: Vec<f32>,
        probabilities: Tensor<2>,
    },
    /// Both tensors have shape `[batch]`; `raw_second` is `E[G²]`, not variance.
    Moments {
        mean: Tensor<1>,
        raw_second: Tensor<1>,
    },
}

/// State-value inspection does not require fictitious action inputs.
pub struct StateValueProbe<'a, V: super::critics::scalar::StateValue>(pub &'a V);
impl<V: super::critics::scalar::StateValue> StateValueProbe<'_, V> {
    pub fn predict(&self, obs: ObservationBatch<'_>) -> Result<CriticPrediction> {
        let values = frozen_copy(self.0).values(obs)?;
        ensure(
            values.dims() == [obs.len()],
            "invalid state-value probe output",
        )?;
        Ok(CriticPrediction::Scalar(values.without_autodiff()))
    }
}
pub struct DiscreteProbe<'a, Q: super::critics::scalar::DiscreteActionValue>(pub &'a Q);
impl<Q: super::critics::scalar::DiscreteActionValue> DiscreteProbe<'_, Q> {
    pub fn predict(&self, obs: ObservationBatch<'_>) -> Result<CriticPrediction> {
        let model = frozen_copy(self.0);
        let values = model.action_values(obs)?;
        ensure(
            values.dims() == [obs.len(), model.action_count()],
            "invalid discrete critic probe output",
        )?;
        Ok(CriticPrediction::Discrete(values.without_autodiff()))
    }
}

/// A detached distribution and selected actions, produced with caller-owned
/// diagnostic randomness. No training random stream is borrowed or reseeded.
#[derive(Debug, Clone)]
pub struct PolicyPrediction {
    pub distribution: super::distributions::Distribution,
    pub actions: Vec<crate::core::action::Action>,
    pub entropy: Tensor<1>,
}
pub struct PolicyProbe<'a, P: super::policy::StochasticPolicy>(pub &'a P);
impl<P: super::policy::StochasticPolicy> PolicyProbe<'_, P> {
    pub fn predict(
        &self,
        obs: ObservationBatch<'_>,
        mode: crate::core::actor::ActMode,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<PolicyPrediction> {
        let distribution = frozen_copy(self.0).distribution(obs)?;
        let actions = distribution.sample(mode, rng)?;
        let entropy = distribution.entropy(rng)?.without_autodiff();
        Ok(PolicyPrediction {
            distribution,
            actions,
            entropy,
        })
    }
}

/// Explicit action-value inspection. Actions have shape `[batch, action_dim]`.
/// Each call evaluates an independent synchronized inference snapshot, detaches
/// its outputs, and leaves live parameters and running statistics unchanged.
pub trait CriticProbe {
    fn predict(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<CriticPrediction>;
}

fn prediction<Q: ActionValue>(
    critic: &Q,
    observations: ObservationBatch<'_>,
    actions: Tensor<2>,
) -> Result<Tensor<2>> {
    ensure(
        !observations.is_empty() && actions.dims() == [observations.len(), critic.action_dim()],
        "invalid critic probe input shape",
    )?;
    let result = critic.values(observations, actions.without_autodiff())?;
    ensure(
        result.dims() == [observations.len(), critic.output_dim()],
        "invalid critic probe prediction shape",
    )?;
    Ok(result.without_autodiff())
}

/// Adapter for a critic with one scalar output.
pub struct ScalarProbe<'a, Q: ActionValue> {
    critic: &'a Q,
}
impl<'a, Q: ActionValue> ScalarProbe<'a, Q> {
    pub fn new(critic: &'a Q) -> Result<Self> {
        ensure(critic.output_dim() == 1, "scalar probe requires one output")?;
        Ok(Self { critic })
    }
}
impl<Q: ActionValue> CriticProbe for ScalarProbe<'_, Q> {
    fn predict(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<CriticPrediction> {
        Ok(CriticPrediction::Scalar(
            prediction(&frozen_copy(self.critic), observations, actions)?.squeeze_dim(1),
        ))
    }
}

/// Adapter pairing quantile outputs with their validated probability grid.
pub struct QuantileProbe<'a, Q: ActionValue> {
    critic: &'a Q,
    grid: &'a QuantileGrid,
}
impl<'a, Q: ActionValue> QuantileProbe<'a, Q> {
    pub fn new(critic: &'a Q, grid: &'a QuantileGrid) -> Result<Self> {
        grid.validate()?;
        ensure(
            critic.output_dim() == grid.len(),
            "quantile probe grid/output mismatch",
        )?;
        Ok(Self { critic, grid })
    }
}
impl<Q: ActionValue> CriticProbe for QuantileProbe<'_, Q> {
    fn predict(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<CriticPrediction> {
        Ok(CriticPrediction::Quantiles {
            values: prediction(&frozen_copy(self.critic), observations, actions)?,
            locations: self.grid.locations(),
            masses: self.grid.masses(),
        })
    }
}

/// Adapter interpreting action-value heads as logits over fixed return atoms.
pub struct CategoricalReturnProbe<'a, Q: ActionValue> {
    critic: &'a Q,
    support: &'a crate::objectives::categorical::CategoricalSupport,
}
impl<'a, Q: ActionValue> CategoricalReturnProbe<'a, Q> {
    pub fn new(
        critic: &'a Q,
        support: &'a crate::objectives::categorical::CategoricalSupport,
    ) -> Result<Self> {
        ensure(
            critic.output_dim() == support.atoms().len(),
            "categorical probe support/output mismatch",
        )?;
        Ok(Self { critic, support })
    }
}
impl<Q: ActionValue> CriticProbe for CategoricalReturnProbe<'_, Q> {
    fn predict(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<CriticPrediction> {
        let logits = prediction(&frozen_copy(self.critic), observations, actions)?;
        Ok(CriticPrediction::CategoricalReturn {
            atoms: self.support.atoms().to_vec(),
            probabilities: self.support.probabilities(logits)?,
        })
    }
}

/// Adapter for separately represented first and raw second moment critics.
pub struct MomentsProbe<'a, Q: ActionValue> {
    critic: &'a Moments<Q>,
}
impl<'a, Q: ActionValue> MomentsProbe<'a, Q> {
    pub fn new(critic: &'a Moments<Q>) -> Result<Self> {
        ensure(
            critic.first.output_dim() == 1
                && critic.second.output_dim() == 1
                && critic.first.action_dim() == critic.second.action_dim(),
            "moment probe requires matching scalar critics",
        )?;
        Ok(Self { critic })
    }
}
impl<Q: ActionValue> CriticProbe for MomentsProbe<'_, Q> {
    fn predict(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<CriticPrediction> {
        let first = frozen_copy(&self.critic.first);
        let second = frozen_copy(&self.critic.second);
        Ok(CriticPrediction::Moments {
            mean: prediction(&first, observations, actions.clone())?.squeeze_dim(1),
            raw_second: prediction(&second, observations, actions)?.squeeze_dim(1),
        })
    }
}
