//! Burn-backed model capture, restoration and isolated inference.
use super::super::{
    network_geometry,
    probes::{self, ProbeSet},
    reference,
    schema::*,
    value,
};
use super::{
    ARCHITECTURE, CategoricalQLayout, MlpLayout, ModelDefinition, ModelKind, ModelLimits,
    PACK_FORMAT, ResidualAnchor, ResidualOutput, SnapshotConfiguration, snapshot_configuration,
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        checkpoint::{LearnerRecord, ModelRecord},
        ensure,
        env::{Observation, ObservationBatch},
    },
    nn::{
        self,
        checkpoint::RecordBurnExt,
        critics::{
            categorical::CategoricalQ,
            moments::Moments,
            quantile::QuantileGrid,
            residual::{ResidualQ, ResidualScale},
            scalar::{ActionValue, DiscreteQ, QValue, Value},
        },
        distributions::Distribution,
        gaussian_policy::GaussianPolicy,
        network::{Mlp, Network},
        policy::{Deterministic, Policy, StochasticPolicy},
        probes::{
            CategoricalReturnProbe, CriticPrediction, CriticProbe, DiscreteProbe, IntervenedMlp,
            Intervention, MomentsProbe, QuantileProbe, ScalarProbe, StateValueProbe,
        },
    },
    objectives::{categorical::CategoricalSupport, hl_gauss::HlGaussConfig},
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};

mod reppo;
pub(crate) use reppo::{
    categorical_activations, categorical_prediction, categorical_prediction_bytes,
};

#[cfg(test)]
mod lifetime_tests {
    use super::*;
    use burn::module::{ModuleVisitor, Param};

    struct PlainParameters(usize);
    impl ModuleVisitor for PlainParameters {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            self.0 += 1;
            assert!(
                !parameter.val().is_autodiff(),
                "historical construction/restoration entered autodiff"
            );
        }
    }
    #[test]
    fn historical_template_is_plain_before_record_restoration() {
        let layout = MlpLayout {
            widths: vec![2, 4, 1],
            biases: vec![true, true],
        };
        let template = layout.build(&mut ChaCha8Rng::seed_from_u64(7)).unwrap();
        let mut witness = PlainParameters(0);
        template.visit(&mut witness);
        assert_eq!(witness.0, 4);
        let policy =
            Policy::from_network_inference(template, ActionSpace::Unbounded { dimensions: 1 })
                .unwrap();
        let mut witness = PlainParameters(0);
        policy.visit(&mut witness);
        assert_eq!(
            witness.0, 5,
            "include stochastic log_std before restoration"
        );
        let mut record = LearnerRecord::new("inference-witness", ()).unwrap();
        record.model("policy", &policy).unwrap();
        let restored = record.load_model_inference("policy", &policy).unwrap();
        let mut witness = PlainParameters(0);
        restored.visit(&mut witness);
        assert_eq!(witness.0, 5);
        assert!(!restored.log_std.val().is_require_grad());
    }
    #[test]
    fn all_builtin_restored_parameters_are_plain_including_both_moments() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        for discrete in [false, true] {
            let space = if discrete {
                ActionSpace::Discrete { count: 3 }
            } else {
                ActionSpace::Unbounded { dimensions: 1 }
            };
            let mut builder =
                SnapshotBuilder::new("witness", RecordContext::new("test", 0), 2, space.clone())
                    .unwrap();
            builder
                .policy(
                    "policy",
                    &Policy::new(2, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
            builder
                .value("value", &Value::new(2, &[4], &mut rng).unwrap())
                .unwrap();
            builder
                .categorical(
                    "categorical",
                    &CategoricalQ::new(2, space.clone(), 101, &[4], &mut rng).unwrap(),
                    &HlGaussConfig::default(),
                    true,
                )
                .unwrap();
            if discrete {
                builder
                    .discrete("discrete", &DiscreteQ::new(2, 3, &[4], &mut rng).unwrap())
                    .unwrap();
            } else {
                builder
                    .state_gaussian(
                        "state-gaussian",
                        &GaussianPolicy::new(2, space.clone(), &[4], &mut rng).unwrap(),
                    )
                    .unwrap();
                builder
                    .deterministic(
                        "deterministic",
                        &Deterministic::new(2, space, &[4], &mut rng).unwrap(),
                    )
                    .unwrap();
                builder
                    .scalar("scalar", &QValue::new(2, 1, 1, &[4], &mut rng).unwrap())
                    .unwrap();
                builder
                    .quantiles(
                        "quantiles",
                        &QValue::new(2, 1, 3, &[4], &mut rng).unwrap(),
                        &QuantileGrid::uniform(3).unwrap(),
                    )
                    .unwrap();
                builder
                    .moments(
                        "moments",
                        &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
                        &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
                    )
                    .unwrap();
            }
            let bundle =
                ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
            for model in bundle.models.values() {
                let mut witness = PlainParameters(0);
                let expected = match model {
                    Loaded::Policy(model) => {
                        model.visit(&mut witness);
                        5
                    }
                    Loaded::StateGaussian(model) => {
                        model.visit(&mut witness);
                        4
                    }
                    Loaded::Deterministic(model) => {
                        model.visit(&mut witness);
                        4
                    }
                    Loaded::Value(model) => {
                        model.visit(&mut witness);
                        4
                    }
                    Loaded::Discrete(model) => {
                        model.visit(&mut witness);
                        4
                    }
                    Loaded::Q(model) => {
                        model.visit(&mut witness);
                        4
                    }
                    Loaded::Residual(model) => {
                        model.visit(&mut witness);
                        9
                    }
                    Loaded::ResidualMoments(model) => {
                        model.first.visit(&mut witness);
                        model.second.visit(&mut witness);
                        18
                    }
                    Loaded::Moments(model) => {
                        model.first.visit(&mut witness);
                        model.second.visit(&mut witness);
                        8
                    }
                    Loaded::Categorical(model) => {
                        model.visit(&mut witness);
                        10
                    }
                    Loaded::Opaque => unreachable!(),
                };
                assert_eq!(witness.0, expected);
            }
        }
    }
    #[test]
    fn historical_inspection_preserves_restored_bytes_and_seeded_policy_samples() {
        let mut rng = ChaCha8Rng::seed_from_u64(22);
        let space = ActionSpace::Unbounded { dimensions: 1 };
        let mut builder = SnapshotBuilder::new(
            "invariance",
            RecordContext::new("test", 3),
            2,
            space.clone(),
        )
        .unwrap();
        builder
            .policy("policy", &Policy::new(2, space, &[4], &mut rng).unwrap())
            .unwrap();
        builder
            .value("v", &Value::new(2, &[4], &mut rng).unwrap())
            .unwrap();
        builder
            .scalar("q", &QValue::new(2, 1, 1, &[4], &mut rng).unwrap())
            .unwrap();
        builder
            .quantiles(
                "z",
                &QValue::new(2, 1, 3, &[4], &mut rng).unwrap(),
                &QuantileGrid::uniform(3).unwrap(),
            )
            .unwrap();
        builder
            .moments(
                "m",
                &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
                &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
            )
            .unwrap();
        let bundle = ModelBundle::load(&builder.finish().unwrap(), ModelLimits::default()).unwrap();
        let serialize_loaded = || {
            let mut record = LearnerRecord::new("witness", ()).unwrap();
            for (name, model) in &bundle.models {
                match model {
                    Loaded::Policy(model) => record.model(name, model).unwrap(),
                    Loaded::Value(model) => record.model(name, model).unwrap(),
                    Loaded::Q(model) => record.model(name, model).unwrap(),
                    Loaded::Moments(model) => {
                        record
                            .model(&format!("{name}.first"), &model.first)
                            .unwrap();
                        record
                            .model(&format!("{name}.second"), &model.second)
                            .unwrap();
                    }
                    _ => unreachable!(),
                }
            }
            record
                .models
                .into_iter()
                .map(|(name, model)| (name, model.bytes))
                .collect::<BTreeMap<_, _>>()
        };
        let before = serialize_loaded();
        let observations = vec![Observation::new(vec![0.2, 0.7])];
        let mut policy_request = InspectionRequest::new("policy", observations.clone());
        policy_request.mode = ActMode::Sample;
        policy_request.seed = 999;
        let sample = bundle.inspect(&policy_request).unwrap().actions;
        let request = CriticInspectionRequest {
            critics: vec!["v".into(), "q".into(), "z".into(), "m".into()],
            observations,
            fixed_actions: Some(vec![Action::Continuous(vec![0.3])]),
        };
        bundle.inspect_critics(&request).unwrap();
        policy_request.critics = request.critics.clone();
        policy_request.fixed_actions = request.fixed_actions.clone();
        let original = bundle.inspect(&policy_request).unwrap();
        for (name, layout) in bundle.activation_networks() {
            let last = layout.widths.len() - 2;
            policy_request.interventions.insert(
                name.clone(),
                vec![Intervention {
                    layer: format!("layers.{last}.post"),
                    neuron: 0,
                    value: crate::nn::probes::InterventionValue::Set(2.),
                }],
            );
            let changed = bundle.inspect(&policy_request).unwrap();
            assert!(changed.activations.is_empty());
            policy_request
                .layers
                .insert(name.clone(), vec![format!("layers.{last}.post")]);
            let captured = bundle.inspect(&policy_request).unwrap();
            let equal_chunk = |a: &Chunk, b: &Chunk| {
                assert_eq!(
                    serde_json::to_vec(&a.context).unwrap(),
                    serde_json::to_vec(&b.context).unwrap()
                );
                assert_eq!(
                    serde_json::to_vec(&a.record).unwrap(),
                    serde_json::to_vec(&b.record).unwrap()
                );
                for (key, array) in &a.arrays {
                    let other = &b.arrays[key];
                    assert_eq!(array.shape(), other.shape());
                    assert_eq!(
                        array
                            .to_f64()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        other
                            .to_f64()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
                assert_eq!(a.arrays.len(), b.arrays.len());
            };
            equal_chunk(&changed.policy, &captured.policy);
            equal_chunk(&changed.probes, &captured.probes);
            for (critic, chunk) in &changed.critics {
                equal_chunk(chunk, &captured.critics[critic]);
            }
            let activation = &captured.activations[&name];
            assert!(
                activation
                    .arrays
                    .values()
                    .any(|a| a.as_f32().is_ok_and(|v| v[0] == 2.))
            );
            if name != "policy" {
                let critic = name.split('.').next().unwrap();
                let Record::Critic(record) = &changed.critics[critic].record else {
                    panic!()
                };
                let values = match &record.representation {
                    CriticRepresentation::Scalar { values }
                    | CriticRepresentation::Quantiles { values, .. } => values,
                    CriticRepresentation::RawMoments { mean, raw_second } => {
                        if name.ends_with(".first") {
                            mean
                        } else {
                            raw_second
                        }
                    }
                    _ => unreachable!(),
                };
                assert_eq!(
                    changed.critics[critic].arrays[values].as_f32().unwrap()[0],
                    2.
                );
            }
            policy_request.interventions.clear();
            policy_request.layers.clear();
            let cleared = bundle.inspect(&policy_request).unwrap();
            equal_chunk(&cleared.policy, &original.policy);
            for (name, chunk) in &cleared.critics {
                equal_chunk(chunk, &original.critics[name]);
            }
        }
        assert_eq!(
            serialize_loaded(),
            before,
            "actual restored parameter bytes changed"
        );
        assert_eq!(bundle.inspect(&policy_request).unwrap().actions, sample);
    }
}
impl MlpLayout {
    fn capture(model: &Mlp) -> Result<Self> {
        ensure(!model.layers.is_empty(), "empty snapshot network")?;
        let mut widths = vec![model.input_dim()];
        let mut biases = vec![];
        for layer in &model.layers {
            let [input, output] = layer.weight.val().dims();
            ensure(
                input == *widths.last().unwrap() && output > 0,
                "invalid MLP layer connectivity",
            )?;
            ensure(
                layer
                    .bias
                    .as_ref()
                    .is_none_or(|p| p.val().dims() == [output]),
                "invalid MLP bias shape",
            )?;
            widths.push(output);
            biases.push(layer.bias.is_some());
        }
        ensure(
            *widths.last().unwrap() == model.output_dim(),
            "invalid MLP output layout",
        )?;
        Ok(Self { widths, biases })
    }
    fn build(&self, rng: &mut ChaCha8Rng) -> Result<Mlp> {
        self.parameters()?;
        let mut m = Mlp::new_inference(
            self.widths[0],
            &self.widths[1..self.widths.len() - 1],
            *self.widths.last().unwrap(),
            rng,
        )?;
        for (layer, &bias) in m.layers.iter_mut().zip(&self.biases) {
            if !bias {
                layer.bias = None;
            }
        }
        Ok(m)
    }
}

pub struct SnapshotBuilder {
    id: String,
    context: RecordContext,
    config: SnapshotConfiguration,
    record: LearnerRecord,
}
impl SnapshotBuilder {
    pub fn new(
        id: &str,
        mut context: RecordContext,
        observation_dim: usize,
        action_space: ActionSpace,
    ) -> Result<Self> {
        ensure(
            !id.is_empty() && observation_dim > 0,
            "invalid model snapshot identity/dimensions",
        )?;
        action_space.validate()?;
        context.identity.checkpoint_id = Some(id.into());
        context.probe_set = None;
        Ok(Self {
            id: id.into(),
            context,
            config: SnapshotConfiguration {
                format_version: 1,
                pack_format: PACK_FORMAT.into(),
                observation_dim,
                action_space,
                models: BTreeMap::new(),
            },
            record: LearnerRecord::new("inference-only", ())?,
        })
    }
    fn add<M: Module>(
        &mut self,
        name: &str,
        model: &M,
        kind: ModelKind,
        network: Option<MlpLayout>,
    ) -> Result<()> {
        ensure(
            !name.is_empty() && !self.record.models.contains_key(name),
            "duplicate/empty snapshot model name",
        )?;
        self.record.model(name, model)?;
        self.config.models.insert(
            name.into(),
            ModelDefinition {
                kind,
                network,
                second: None,
            },
        );
        Ok(())
    }
    pub fn policy(&mut self, name: &str, model: &Policy) -> Result<()> {
        ensure(
            model.space == self.config.action_space
                && model.network.input_dim() == self.config.observation_dim,
            "snapshot policy specifications differ",
        )?;
        self.add(
            name,
            model,
            ModelKind::StochasticPolicy,
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    pub fn state_gaussian(&mut self, name: &str, model: &GaussianPolicy) -> Result<()> {
        ensure(
            model.action_space() == &self.config.action_space
                && model.network.input_dim() == self.config.observation_dim,
            "snapshot state-dependent Gaussian policy specifications differ",
        )?;
        self.add(
            name,
            model,
            ModelKind::StateGaussianPolicy,
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    pub fn deterministic(&mut self, name: &str, model: &Deterministic) -> Result<()> {
        ensure(
            model.space == self.config.action_space
                && model.network.input_dim() == self.config.observation_dim,
            "snapshot policy specifications differ",
        )?;
        self.add(
            name,
            model,
            ModelKind::DeterministicPolicy,
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    pub fn value(&mut self, name: &str, model: &Value) -> Result<()> {
        ensure(
            model.network.input_dim() == self.config.observation_dim
                && model.network.output_dim() == 1,
            "snapshot value dimensions differ",
        )?;
        self.add(
            name,
            model,
            ModelKind::StateValue,
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    fn check_q(&self, model: &QValue, outputs: usize) -> Result<()> {
        ensure(
            !matches!(self.config.action_space, ActionSpace::Discrete { .. })
                && model.observation_dim == self.config.observation_dim
                && model.action_dim == self.config.action_space.dimensions()
                && model.network.input_dim() == model.observation_dim + model.action_dim
                && model.network.output_dim() == outputs,
            "snapshot critic specifications differ",
        )
    }
    pub fn scalar(&mut self, name: &str, model: &QValue) -> Result<()> {
        self.check_q(model, 1)?;
        self.add(
            name,
            model,
            ModelKind::ScalarQ,
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    pub fn quantiles(&mut self, name: &str, model: &QValue, grid: &QuantileGrid) -> Result<()> {
        grid.validate()?;
        self.check_q(model, grid.len())?;
        self.add(
            name,
            model,
            ModelKind::QuantileQ { grid: grid.clone() },
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    pub fn categorical_return(
        &mut self,
        name: &str,
        model: &QValue,
        support: &CategoricalSupport,
    ) -> Result<()> {
        ensure(
            support.atoms().len() <= value::MAX_VALUE_HEADS,
            "categorical return support exceeds snapshot limits",
        )?;
        self.check_q(model, support.atoms().len())?;
        self.add(
            name,
            model,
            ModelKind::CategoricalReturnQ {
                support: support.clone(),
            },
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    fn residual_anchor(&self, model: &ResidualQ, outputs: usize) -> Result<Box<ResidualAnchor>> {
        self.check_q(model.inner(), outputs)?;
        let anchor = ResidualAnchor {
            version: 1,
            identity: model.anchor_identity()?,
            network: MlpLayout::capture(&model.anchor().network)?,
            action_space: model.anchor().space.clone(),
            scale: model.scales()?,
        };
        anchor.parameters(
            self.config.observation_dim,
            self.config.action_space.dimensions(),
        )?;
        Ok(Box::new(anchor))
    }

    pub fn residual_scalar(&mut self, name: &str, model: &ResidualQ) -> Result<()> {
        let anchor = self.residual_anchor(model, 1)?;
        self.add(
            name,
            model,
            ModelKind::ResidualQ {
                anchor,
                output: ResidualOutput::Scalar,
            },
            Some(MlpLayout::capture(&model.inner().network)?),
        )
    }

    pub fn residual_quantiles(
        &mut self,
        name: &str,
        model: &ResidualQ,
        grid: &QuantileGrid,
    ) -> Result<()> {
        grid.validate()?;
        let anchor = self.residual_anchor(model, grid.len())?;
        self.add(
            name,
            model,
            ModelKind::ResidualQ {
                anchor,
                output: ResidualOutput::Quantile { grid: grid.clone() },
            },
            Some(MlpLayout::capture(&model.inner().network)?),
        )
    }

    pub fn residual_moments(
        &mut self,
        name: &str,
        first: &ResidualQ,
        second: &ResidualQ,
    ) -> Result<()> {
        let anchor = self.residual_anchor(first, 1)?;
        let second_anchor = self.residual_anchor(second, 1)?;
        let second_key = format!("{name}/raw_second");
        ensure(
            !self.record.models.contains_key(&second_key),
            "duplicate second-moment model key",
        )?;
        self.add(
            name,
            first,
            ModelKind::ResidualQ {
                anchor,
                output: ResidualOutput::Moments {
                    second_key: second_key.clone(),
                    second_anchor,
                },
            },
            Some(MlpLayout::capture(&first.inner().network)?),
        )?;
        self.record.model(&second_key, second)?;
        self.config.models.get_mut(name).unwrap().second =
            Some(MlpLayout::capture(&second.inner().network)?);
        Ok(())
    }
    /// Capture the complete normalized encoder, logit head and auxiliary predictor.
    pub fn categorical(
        &mut self,
        name: &str,
        model: &CategoricalQ,
        support: &HlGaussConfig,
        auxiliary: bool,
    ) -> Result<()> {
        let (observation_dim, encoder_widths) = model.diagnostic_layout();
        let action_dim = match self.config.action_space {
            ActionSpace::Discrete { count } => count,
            _ => self.config.action_space.dimensions(),
        };
        ensure(
            observation_dim == self.config.observation_dim
                && model.action_dim() == action_dim
                && model.output_dim() == support.bins,
            "snapshot categorical critic dimensions differ",
        )?;
        let layout = CategoricalQLayout {
            encoder_widths,
            auxiliary,
        };
        layout.parameters(observation_dim, &self.config.action_space, support)?;
        self.add(
            name,
            model,
            ModelKind::CategoricalQ {
                layout,
                support: support.clone(),
            },
            None,
        )
    }
    pub fn moments(&mut self, name: &str, first: &QValue, second: &QValue) -> Result<()> {
        self.check_q(first, 1)?;
        self.check_q(second, 1)?;
        let second_key = format!("{name}/raw_second");
        ensure(
            !self.record.models.contains_key(&second_key),
            "duplicate second-moment model key",
        )?;
        self.add(
            name,
            first,
            ModelKind::MomentQ {
                second_key: second_key.clone(),
            },
            Some(MlpLayout::capture(&first.network)?),
        )?;
        self.record.model(&second_key, second)?;
        self.config.models.get_mut(name).unwrap().second =
            Some(MlpLayout::capture(&second.network)?);
        Ok(())
    }
    pub fn discrete(&mut self, name: &str, model: &DiscreteQ) -> Result<()> {
        ensure(
            self.config.action_space
                == ActionSpace::Discrete {
                    count: model.network.output_dim(),
                }
                && model.network.input_dim() == self.config.observation_dim,
            "snapshot discrete critic dimensions differ",
        )?;
        self.add(
            name,
            model,
            ModelKind::DiscreteQ,
            Some(MlpLayout::capture(&model.network)?),
        )
    }
    /// Preserve custom module inference state without claiming a built-in loader
    /// or intervention capability. A custom application may supply its own adapter.
    pub fn opaque<M: Module>(&mut self, name: &str, model: &M, description: &str) -> Result<()> {
        self.add(
            name,
            model,
            ModelKind::Opaque {
                description: description.into(),
            },
            None,
        )
    }
    pub fn finish(self) -> Result<Chunk> {
        ensure(!self.config.models.is_empty(), "empty model snapshot")?;
        let mut arrays = BTreeMap::new();
        let mut models = BTreeMap::new();
        for (name, record) in self.record.models {
            let key = format!("models/{name}");
            let len = record.bytes.len();
            arrays.insert(key.clone(), Array::u8(vec![len], record.bytes)?);
            models.insert(
                name,
                ModelArtifact {
                    bytes: key,
                    signature: record.signature,
                    parameter_shapes: record.parameter_shapes,
                },
            );
        }
        let mut capabilities = vec![
            "builtin-mlp inference".into(),
            "temporary pre/post activation interventions".into(),
            "opaque custom models require an external adapter".into(),
        ];
        if self
            .config
            .models
            .values()
            .any(|model| matches!(model.kind, ModelKind::CategoricalQ { .. }))
        {
            capabilities.push("normalized HL-Gauss critic inference and representations".into());
        }
        let chunk = Chunk {
            context: self.context,
            record: Record::ModelSnapshot(ModelSnapshotRecord {
                id: self.id,
                architecture: ARCHITECTURE.into(),
                configuration: serde_json::to_value(self.config)?,
                models,
                capabilities,
            }),
            arrays,
        };
        chunk.validate()?;
        Ok(chunk)
    }
}
#[allow(clippy::large_enum_variant)]
enum Loaded {
    Policy(Policy),
    StateGaussian(GaussianPolicy),
    Deterministic(Deterministic),
    Value(Value),
    Q(QValue),
    Residual(ResidualQ),
    ResidualMoments(Moments<ResidualQ>),
    Moments(Moments<QValue>),
    Categorical(CategoricalQ),
    Discrete(DiscreteQ),
    Opaque,
}
pub struct ModelBundle {
    configuration: SnapshotConfiguration,
    context: RecordContext,
    models: BTreeMap<String, Loaded>,
    limits: ModelLimits,
    value_declaration: Option<value::ValueDeclaration>,
    return_target: Option<value::ReturnTarget>,
    reference_declaration: Option<reference::ReferenceDeclaration>,
}
/// Isolated critic inference with no policy dependency or random stream.
#[derive(Debug, Clone)]
pub struct CriticInspectionRequest {
    pub critics: Vec<String>,
    pub observations: Vec<Observation>,
    pub fixed_actions: Option<Vec<Action>>,
}
pub struct CriticInspectionResult {
    pub probes: Chunk,
    pub critics: BTreeMap<String, Chunk>,
}
#[derive(Debug, Clone)]
pub struct InspectionRequest {
    pub policy: String,
    pub critics: Vec<String>,
    pub observations: Vec<Observation>,
    pub fixed_actions: Option<Vec<Action>>,
    pub mode: ActMode,
    pub seed: u64,
    pub layers: BTreeMap<String, Vec<String>>,
    pub interventions: BTreeMap<String, Vec<Intervention>>,
}
impl InspectionRequest {
    pub fn new(policy: impl Into<String>, observations: Vec<Observation>) -> Self {
        Self {
            policy: policy.into(),
            critics: vec![],
            observations,
            fixed_actions: None,
            mode: ActMode::Deterministic,
            seed: 0,
            layers: BTreeMap::new(),
            interventions: BTreeMap::new(),
        }
    }
}
pub struct InspectionResult {
    pub actions: Vec<Action>,
    pub probes: Chunk,
    pub policy: Chunk,
    pub critics: BTreeMap<String, Chunk>,
    pub activations: BTreeMap<String, Chunk>,
}
impl ModelBundle {
    pub fn load(chunk: &Chunk, limits: ModelLimits) -> Result<Self> {
        chunk.validate()?;
        let value_declaration = value::declaration(chunk)?;
        let return_target = value::return_target(chunk)?;
        let reference_declaration = reference::declaration(chunk)?;
        let Record::ModelSnapshot(snapshot) = &chunk.record else {
            return Err(Error::Config("expected model-only snapshot".into()));
        };
        let config = snapshot_configuration(snapshot, limits)?;
        let mut record = LearnerRecord::new("inference-only", ())?;
        let mut bytes = 0usize;
        for (name, artifact) in &snapshot.models {
            let payload = chunk.arrays[&artifact.bytes].as_u8()?;
            bytes = bytes
                .checked_add(payload.len())
                .ok_or_else(|| Error::Shape("snapshot byte count overflow".into()))?;
            ensure(
                bytes <= limits.max_model_bytes,
                "snapshot exceeds model byte budget",
            )?;
            record.models.insert(
                name.clone(),
                ModelRecord {
                    signature: artifact.signature.clone(),
                    bytes: payload.to_vec(),
                    parameter_shapes: artifact.parameter_shapes.clone(),
                },
            );
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut models = BTreeMap::new();
        for (name, definition) in &config.models {
            let network = definition
                .network
                .as_ref()
                .map(|l| l.build(&mut rng))
                .transpose()?;
            let model = match &definition.kind {
                ModelKind::Opaque { .. } => Loaded::Opaque,
                ModelKind::StochasticPolicy => Loaded::Policy(record.load_model_inference(
                    name,
                    &Policy::from_network_inference(network.unwrap(), config.action_space.clone())?,
                )?),
                ModelKind::StateGaussianPolicy => {
                    Loaded::StateGaussian(record.load_model_inference(
                        name,
                        &GaussianPolicy::from_network(
                            network.unwrap(),
                            config.action_space.clone(),
                        )?,
                    )?)
                }
                ModelKind::DeterministicPolicy => {
                    Loaded::Deterministic(record.load_model_inference(
                        name,
                        &Deterministic::from_network(
                            network.unwrap(),
                            config.action_space.clone(),
                        )?,
                    )?)
                }
                ModelKind::StateValue => {
                    let network = network.unwrap();
                    Loaded::Value(record.load_model_inference(name, &Value { network })?)
                }
                ModelKind::DiscreteQ => {
                    let network = network.unwrap();
                    Loaded::Discrete(record.load_model_inference(name, &DiscreteQ { network })?)
                }
                ModelKind::CategoricalQ { layout, support } => {
                    let template = CategoricalQ::new_inference(
                        config.observation_dim,
                        config.action_space.clone(),
                        support.bins,
                        &layout.encoder_widths,
                        &mut rng,
                    )?;
                    Loaded::Categorical(record.load_model_inference(name, &template)?)
                }
                ModelKind::ResidualQ { anchor, output } => {
                    let mut load = |key: &str, network, anchor: &ResidualAnchor| {
                        let template = ResidualQ::new(
                            QValue {
                                network,
                                observation_dim: config.observation_dim,
                                action_dim: config.action_space.dimensions(),
                            },
                            &Deterministic::from_network(
                                anchor.network.build(&mut rng)?,
                                anchor.action_space.clone(),
                            )?,
                            ResidualScale::Diagonal(anchor.scale.clone()),
                        )?;
                        let model = record.load_model_inference(key, &template)?;
                        ensure(
                            model.anchor_identity()? == anchor.identity
                                && model.scales()? == anchor.scale,
                            "historical residual anchor identity or scale differs from the snapshot",
                        )?;
                        Ok::<_, Error>(model)
                    };
                    let first = load(name, network.unwrap(), anchor)?;
                    if let ResidualOutput::Moments {
                        second_key,
                        second_anchor,
                    } = output
                    {
                        let second_network = definition
                            .second
                            .as_ref()
                            .unwrap()
                            .build(&mut ChaCha8Rng::seed_from_u64(0))?;
                        let second = load(second_key, second_network, second_anchor)?;
                        Loaded::ResidualMoments(Moments { first, second })
                    } else {
                        Loaded::Residual(first)
                    }
                }
                ModelKind::ScalarQ
                | ModelKind::QuantileQ { .. }
                | ModelKind::MomentQ { .. }
                | ModelKind::CategoricalReturnQ { .. } => {
                    let network = network.unwrap();
                    let model = record.load_model_inference(
                        name,
                        &QValue {
                            network,
                            observation_dim: config.observation_dim,
                            action_dim: config.action_space.dimensions(),
                        },
                    )?;
                    if let ModelKind::MomentQ { second_key } = &definition.kind {
                        let network = definition.second.as_ref().unwrap().build(&mut rng)?;
                        let second = record.load_model_inference(
                            second_key,
                            &QValue {
                                network,
                                observation_dim: config.observation_dim,
                                action_dim: config.action_space.dimensions(),
                            },
                        )?;
                        Loaded::Moments(Moments {
                            first: model,
                            second,
                        })
                    } else {
                        Loaded::Q(model)
                    }
                }
            };
            models.insert(name.clone(), model);
        }
        Ok(Self {
            configuration: config,
            context: chunk.context.clone(),
            models,
            limits,
            value_declaration,
            return_target,
            reference_declaration,
        })
    }
    pub fn reference_declaration(&self) -> Option<&reference::ReferenceDeclaration> {
        self.reference_declaration.as_ref()
    }
    fn project_reference(&self, name: &str, chunk: &mut Chunk) -> Result<()> {
        if let Some(source) = &self.reference_declaration
            && source.sources.contains_key(name)
        {
            let mut declaration = source.clone();
            declaration.sources.retain(|key, _| key == name);
            reference::attach_declaration(chunk, &declaration)?;
        }
        Ok(())
    }
    fn project_critic_value(&self, name: &str, chunk: &mut Chunk) -> Result<()> {
        if let Some(source) = &self.value_declaration
            && let Some(critic) = source.critics.get(name)
        {
            if let Record::Critic(record) = &mut chunk.record {
                record.continuation_policy =
                    critic.continuation.as_ref().map(|id| id.identity.clone());
                record.continuation_stochastic =
                    critic.continuation.as_ref().map(|id| id.stochastic);
            }
            let mut declaration = source.clone();
            declaration.critics.retain(|key, _| key == name);
            value::attach_declaration(chunk, &declaration)?;
        }
        if let Some(target) = self.return_target {
            value::attach_return_target(chunk, target)?;
        }
        Ok(())
    }
    pub fn configuration(&self) -> &SnapshotConfiguration {
        &self.configuration
    }
    pub fn context(&self) -> &RecordContext {
        &self.context
    }

    /// Derivative of a restored scalar residual critic in environment-action coordinates.
    /// Returns no derivative at a ReLU kink or for another representation.
    pub fn residual_action_gradient(
        &self,
        name: &str,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<reference::ActionGradient>> {
        let Some(Loaded::Residual(model)) = self.models.get(name) else {
            return Ok(None);
        };
        if model.output_dim() != 1
            || self.configuration.models[name]
                .kind
                .quantile_grid()
                .is_some()
        {
            return Ok(None);
        }
        self.admit_inspection(std::iter::once(name), std::slice::from_ref(observation))?;
        super::super::adapters::residual_action_gradient(
            model,
            observation,
            action,
            &self.configuration.action_space,
        )
    }

    /// Create an isolated actor from a restored feedforward policy for an
    /// independent evaluation. The caller must retain [`Self::context`] with
    /// the results; the actor itself does not carry recording identity.
    /// Unsupported model roles remain explicit errors.
    pub fn evaluation_actor(&self, name: &str) -> Result<Box<dyn Actor>> {
        use crate::diagnostics::evaluation::{DeterministicActor, StochasticActor};
        match self.models.get(name) {
            Some(Loaded::Policy(model)) => Ok(Box::new(StochasticActor::new(model))),
            Some(Loaded::StateGaussian(model)) => Ok(Box::new(StochasticActor::new(model))),
            Some(Loaded::Deterministic(model)) => Ok(Box::new(DeterministicActor::new(model))),
            Some(_) => Err(Error::Config(format!(
                "Model '{name}' is not a feedforward policy"
            ))),
            None => Err(Error::Config(format!("No model named '{name}'"))),
        }
    }
    pub fn limits(&self) -> ModelLimits {
        self.limits
    }

    /// Recover the actual stochastic distribution, including the action-space
    /// transform. Bounded likelihood queries use the distribution's latent
    /// sample contract, including at rounded action endpoints.
    pub fn policy_distribution(
        &self,
        name: &str,
        observations: &[Observation],
    ) -> Result<Distribution> {
        self.admit_inspection(std::iter::once(name), observations)?;
        match self.models.get(name) {
            Some(Loaded::Policy(model)) => model.distribution(observations.into()),
            Some(Loaded::StateGaussian(model)) => model.distribution(observations.into()),
            _ => Err(Error::Config(format!("no stochastic policy named {name}"))),
        }
    }
    fn admit_inspection<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
        observations: &[Observation],
    ) -> Result<()> {
        ensure(
            !observations.is_empty() && observations.len() <= self.limits.max_batch,
            "inspection batch exceeds limits",
        )?;
        for observation in observations {
            observation.validate(
                self.configuration.observation_dim,
                &self.configuration.action_space,
                true,
            )?;
        }
        for name in names {
            let definition = self
                .configuration
                .models
                .get(name)
                .ok_or_else(|| Error::Config(format!("snapshot lacks model {name}")))?;
            for shape in definition.network.iter().chain(&definition.second).chain(
                definition
                    .kind
                    .residual_anchors()
                    .map(|anchor| &anchor.network),
            ) {
                let width = *shape.widths.iter().max().expect("validated network");
                let size = width
                    .checked_mul(observations.len())
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(|| Error::Shape("inspection tensor overflow".into()))?;
                ensure(
                    size <= self.limits.max_model_bytes,
                    "inspection intermediate exceeds memory budget",
                )?;
            }
            if let ModelKind::CategoricalQ { layout, support } = &definition.kind {
                let width = layout.work_width(
                    self.configuration.observation_dim,
                    &self.configuration.action_space,
                    support,
                )?;
                let bytes = width
                    .checked_mul(observations.len())
                    .and_then(|n| n.checked_mul(16))
                    .ok_or_else(|| Error::Shape("categorical inspection tensor overflow".into()))?;
                ensure(
                    bytes <= self.limits.max_model_bytes,
                    "categorical inspection intermediate exceeds memory budget",
                )?;
            }
        }
        Ok(())
    }
    pub fn inspect_critics(
        &self,
        request: &CriticInspectionRequest,
    ) -> Result<CriticInspectionResult> {
        ensure(
            !request.critics.is_empty() && request.critics.len() <= self.limits.max_models,
            "critic request is empty or exceeds model admission",
        )?;
        let names: BTreeSet<_> = request.critics.iter().map(String::as_str).collect();
        ensure(
            names.len() == request.critics.len(),
            "duplicate critic request",
        )?;
        self.admit_inspection(names.iter().copied(), &request.observations)?;
        let mut output_bytes = 0usize;
        for name in &names {
            let definition = &self.configuration.models[*name];
            let kind = definition.kind.critic_kind().ok_or_else(|| {
                Error::Config("model has no supported critic inspection adapter".into())
            })?;
            if let Some(declaration) = &self.value_declaration {
                ensure(
                    declaration
                        .critics
                        .get(*name)
                        .is_none_or(|c| c.learned_baseline != Some(false)),
                    "selected learned baseline is disabled",
                )?;
            }
            if !matches!(
                kind,
                value::ValueCriticKind::StateValue | value::ValueCriticKind::DiscreteQ
            ) {
                ensure(
                    request.fixed_actions.is_some(),
                    "action-conditioned critic requires explicit complete actions",
                )?;
            }
            for shape in definition.network.iter().chain(&definition.second) {
                let width = *shape.widths.last().expect("validated network");
                output_bytes = width
                    .checked_mul(request.observations.len())
                    .and_then(|n| n.checked_mul(8))
                    .and_then(|n| output_bytes.checked_add(n))
                    .ok_or_else(|| Error::Shape("critic output size overflow".into()))?;
            }
            if let ModelKind::CategoricalQ { support, .. } = &definition.kind {
                output_bytes = output_bytes
                    .checked_add(categorical_prediction_bytes(
                        request.observations.len(),
                        support.bins,
                    )?)
                    .ok_or_else(|| Error::Shape("critic output size overflow".into()))?;
            }
            if let ModelKind::CategoricalReturnQ { support } = &definition.kind {
                output_bytes = output_bytes
                    .checked_add(support.atoms().len().saturating_mul(4))
                    .ok_or_else(|| Error::Shape("critic output size overflow".into()))?;
            }
        }
        ensure(
            output_bytes <= self.limits.max_model_bytes,
            "critic outputs exceed memory budget",
        )?;
        if let Some(actions) = &request.fixed_actions {
            ensure(
                actions.len() == request.observations.len(),
                "critic actions and observations differ in length",
            )?;
            for (action, observation) in actions.iter().zip(&request.observations) {
                self.configuration
                    .action_space
                    .check(action, observation.dense_mask())?;
            }
        }
        let set = ProbeSet::new(
            "interactive",
            ProbePopulation::FixedReference,
            "Explicit critic inputs and complete fixed actions; no actor or simulator evaluation",
            request.observations.clone(),
            request.fixed_actions.clone(),
            self.configuration.action_space.clone(),
        )?;
        let mut context = self.context.clone();
        context.probe_set = Some(set.id().into());
        let action_tensor = match &request.fixed_actions {
            Some(actions)
                if !matches!(
                    self.configuration.action_space,
                    ActionSpace::Discrete { .. }
                ) =>
            {
                let array = probes::actions_array(actions)?;
                Some(nn::matrix_on_device(
                    array.as_f32()?.to_vec(),
                    actions.len(),
                    self.configuration.action_space.dimensions(),
                    &Device::flex(),
                ))
            }
            _ => None,
        };
        let mut critics = BTreeMap::new();
        for name in &request.critics {
            if let Loaded::Categorical(model) = &self.models[name] {
                let mut chunk = self.categorical_critic(
                    context.clone(),
                    name,
                    model,
                    &set,
                    request
                        .fixed_actions
                        .as_deref()
                        .expect("admitted categorical actions"),
                    ActionCondition::Fixed,
                )?;
                self.project_reference(name, &mut chunk)?;
                critics.insert(name.clone(), chunk);
                continue;
            }
            let (prediction, condition, actions) = match &self.models[name] {
                Loaded::Value(model) => (
                    StateValueProbe(model).predict(set.observations().into())?,
                    ActionCondition::StateValue,
                    None,
                ),
                Loaded::Discrete(model) => (
                    DiscreteProbe(model).predict(set.observations().into())?,
                    ActionCondition::AllDiscreteActions,
                    None,
                ),
                Loaded::Q(model) => {
                    let actions = action_tensor
                        .as_ref()
                        .expect("admitted continuous actions")
                        .clone();
                    let prediction =
                        self.predict_dense_q(name, model, set.observations().into(), actions)?;
                    (
                        prediction,
                        ActionCondition::Fixed,
                        request.fixed_actions.as_deref(),
                    )
                }
                Loaded::Residual(_) | Loaded::ResidualMoments(_) => (
                    self.residual_prediction(
                        name,
                        set.observations().into(),
                        action_tensor
                            .as_ref()
                            .expect("admitted continuous actions")
                            .clone(),
                    )?,
                    ActionCondition::Fixed,
                    request.fixed_actions.as_deref(),
                ),
                Loaded::Moments(model) => (
                    MomentsProbe::new(model)?.predict(
                        set.observations().into(),
                        action_tensor
                            .as_ref()
                            .expect("admitted continuous actions")
                            .clone(),
                    )?,
                    ActionCondition::Fixed,
                    request.fixed_actions.as_deref(),
                ),
                _ => unreachable!("admitted critic kind"),
            };
            let mut chunk = probes::critic(context.clone(), prediction, condition, actions)?;
            self.project_critic_value(name, &mut chunk)?;
            self.project_reference(name, &mut chunk)?;
            chunk.validate()?;
            critics.insert(name.clone(), chunk);
        }
        Ok(CriticInspectionResult {
            probes: set.record(context)?,
            critics,
        })
    }
    fn categorical_critic(
        &self,
        context: RecordContext,
        name: &str,
        model: &CategoricalQ,
        set: &ProbeSet,
        actions: &[Action],
        condition: ActionCondition,
    ) -> Result<Chunk> {
        let ModelKind::CategoricalQ { support, .. } = &self.configuration.models[name].kind else {
            unreachable!("categorical critic kind is bound at load");
        };
        let mut chunk =
            categorical_prediction(context, model, support, set, actions, condition, None)?;
        self.project_critic_value(name, &mut chunk)?;
        Ok(chunk)
    }
    /// Extract an immutable numerical MLP for geometric inspection. No actor is
    /// required, so this also supports Q-only learners such as DDQN.
    ///
    /// The two MD4PG moment models are distinct representations, addressed by
    /// `name.first` and `name.second` respectively.
    pub fn numerical_network(&self, name: &str) -> Result<network_geometry::NumericalMlp> {
        let network = match self.models.get(name) {
            Some(Loaded::Policy(model)) => Some(&model.network),
            Some(Loaded::StateGaussian(model)) => Some(&model.network),
            Some(Loaded::Deterministic(model)) => Some(&model.network),
            Some(Loaded::Value(model)) => Some(&model.network),
            Some(Loaded::Discrete(model)) => Some(&model.network),
            Some(Loaded::Q(model)) => Some(&model.network),
            _ => {
                if let Some(parent) = name.strip_suffix(".first") {
                    match self.models.get(parent) {
                        Some(Loaded::Moments(model)) => Some(&model.first.network),
                        _ => None,
                    }
                } else if let Some(parent) = name.strip_suffix(".second") {
                    match self.models.get(parent) {
                        Some(Loaded::Moments(model)) => Some(&model.second.network),
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
        .ok_or_else(|| Error::Config(format!("no inspectable numerical MLP named {name}")))?;
        network_geometry::NumericalMlp::from_model(network)
    }
    /// Named activation targets; moments expose independent `.first` and `.second` networks.
    pub fn activation_networks(&self) -> BTreeMap<String, MlpLayout> {
        let mut out = BTreeMap::new();
        for (name, spec) in &self.configuration.models {
            if matches!(&spec.kind, ModelKind::ResidualQ { .. }) {
                continue;
            }
            if let Some(first) = &spec.network {
                if let Some(second) = &spec.second {
                    out.insert(format!("{name}.first"), first.clone());
                    out.insert(format!("{name}.second"), second.clone());
                } else {
                    out.insert(name.clone(), first.clone());
                }
            }
        }
        out
    }

    fn residual_prediction(
        &self,
        name: &str,
        observations: crate::core::env::ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<crate::nn::probes::CriticPrediction> {
        match &self.models[name] {
            Loaded::Residual(model) => {
                if let Some(grid) = self.configuration.models[name].kind.quantile_grid() {
                    QuantileProbe::new(model, grid)?.predict(observations, actions)
                } else {
                    ScalarProbe::new(model)?.predict(observations, actions)
                }
            }
            Loaded::ResidualMoments(model) => {
                MomentsProbe::new(model)?.predict(observations, actions)
            }
            _ => unreachable!("residual model selected"),
        }
    }
    fn predict_dense_q<Q: ActionValue>(
        &self,
        name: &str,
        model: &Q,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<CriticPrediction> {
        match &self.configuration.models[name].kind {
            ModelKind::QuantileQ { grid } => {
                QuantileProbe::new(model, grid)?.predict(observations, actions)
            }
            ModelKind::CategoricalReturnQ { support } => {
                CategoricalReturnProbe::new(model, support)?.predict(observations, actions)
            }
            _ => ScalarProbe::new(model)?.predict(observations, actions),
        }
    }
    pub fn inspect(&self, request: &InspectionRequest) -> Result<InspectionResult> {
        ensure(
            !request.observations.is_empty() && request.observations.len() <= self.limits.max_batch,
            "inspection batch exceeds limits",
        )?;
        let selected: BTreeSet<_> = std::iter::once(request.policy.as_str())
            .chain(request.critics.iter().map(String::as_str))
            .collect();
        let networks = self.activation_networks();
        for name in request.layers.keys().chain(request.interventions.keys()) {
            let categorical = self.configuration.models.get(name).is_some_and(|spec| {
                matches!(spec.kind, ModelKind::CategoricalQ { .. })
                    && !request.interventions.contains_key(name)
            });
            ensure(
                (networks.contains_key(name) || categorical)
                    && selected.iter().any(|&s| {
                        name == s || name == &format!("{s}.first") || name == &format!("{s}.second")
                    }),
                "inspection requests an unsupported or unselected activation network",
            )?;
        }
        let mut work = 0usize;
        self.admit_inspection(selected.iter().copied(), &request.observations)?;
        for (name, layers) in &request.layers {
            for layer in layers {
                let width = match self.configuration.models.get(name).map(|spec| &spec.kind) {
                    Some(ModelKind::CategoricalQ { layout, .. }) => layout
                        .activation_layers()
                        .into_iter()
                        .find_map(|(n, width)| (n == layer).then_some(width)),
                    _ => networks[name]
                        .layers()
                        .into_iter()
                        .find_map(|(n, width)| (&n == layer).then_some(width)),
                }
                .ok_or_else(|| Error::Config(format!("unsupported activation {name}/{layer}")))?;
                work = work
                    .checked_add(
                        width
                            .checked_mul(request.observations.len())
                            .and_then(|n| n.checked_mul(4))
                            .ok_or_else(|| Error::Shape("inspection activation overflow".into()))?,
                    )
                    .ok_or_else(|| Error::Shape("inspection activation overflow".into()))?;
            }
        }
        for name in &request.critics {
            let bytes = match &self.configuration.models[name].kind {
                ModelKind::CategoricalQ { support, .. } => {
                    categorical_prediction_bytes(request.observations.len(), support.bins)?
                }
                ModelKind::CategoricalReturnQ { support } => probes::categorical_return_bytes(
                    request.observations.len(),
                    support.atoms().len(),
                )?,
                _ => continue,
            };
            work = work
                .checked_add(bytes)
                .ok_or_else(|| Error::Shape("categorical inspection output overflow".into()))?;
        }
        ensure(
            work <= self.limits.max_model_bytes,
            "requested activations and categorical outputs exceed memory budget",
        )?;
        let set = ProbeSet::new(
            "interactive",
            ProbePopulation::FixedReference,
            "User-supplied neural inputs; interventions are computational counterfactuals, not simulator-state samples",
            request.observations.clone(),
            request.fixed_actions.clone(),
            self.configuration.action_space.clone(),
        )?;
        let mut context = self.context.clone();
        context.probe_set = Some(set.id().into());
        let mut rng = ChaCha8Rng::seed_from_u64(request.seed);
        let adapt = |name: &str, network: &Mlp| {
            IntervenedMlp::new(
                network,
                request.interventions.get(name).cloned().unwrap_or_default(),
            )
        };
        let policy = self
            .models
            .get(&request.policy)
            .ok_or_else(|| Error::Config("unknown snapshot policy".into()))?;
        let mut activations = BTreeMap::new();
        let mut capture = |name: &str,
                           network: &IntervenedMlp,
                           input: Option<Tensor<2>>,
                           actions: Option<&Tensor<2>>|
         -> Result<()> {
            if let Some(names) = request.layers.get(name) {
                let input = input.expect("requested captures have shared input");
                let input = match actions {
                    Some(actions) => Tensor::cat(vec![input, actions.clone()], 1),
                    None => input,
                };
                let refs: Vec<_> = names.iter().map(String::as_str).collect();
                activations.insert(
                    name.into(),
                    probes::activations(context.clone(), name, network, input, &refs)?,
                );
            }
            Ok(())
        };
        // Prediction adapters pack their own inputs; this shared tensor is
        // needed only for requested captures, including present empty keys.
        let input = if request
            .layers
            .keys()
            .any(|name| networks.contains_key(name))
        {
            Some(nn::observations_on_device(
                set.observations().into(),
                self.configuration.observation_dim,
                &Device::flex(),
            )?)
        } else {
            None
        };
        let mut policy = match policy {
            Loaded::Policy(model) => {
                let model = Policy {
                    network: adapt(&request.policy, &model.network)?,
                    log_std: model.log_std.clone(),
                    space: model.space.clone(),
                };
                capture(&request.policy, &model.network, input.clone(), None)?;
                probes::policy(context.clone(), &model, &set, request.mode, &mut rng)?
            }
            Loaded::StateGaussian(model) => {
                let model = GaussianPolicy::from_network(
                    adapt(&request.policy, &model.network)?,
                    model.action_space().clone(),
                )?;
                capture(&request.policy, &model.network, input.clone(), None)?;
                probes::policy(context.clone(), &model, &set, request.mode, &mut rng)?
            }
            Loaded::Deterministic(model) => {
                let model = Deterministic {
                    network: adapt(&request.policy, &model.network)?,
                    space: model.space.clone(),
                };
                capture(&request.policy, &model.network, input.clone(), None)?;
                probes::deterministic_policy(context.clone(), &model, &set)?
            }
            Loaded::Discrete(model) => {
                ensure(
                    request.mode == ActMode::Deterministic,
                    "Q-network snapshot supports masked greedy inference; exploration state is not a model distribution",
                )?;
                let model = DiscreteQ {
                    network: adapt(&request.policy, &model.network)?,
                };
                capture(&request.policy, &model.network, input.clone(), None)?;
                let prediction = DiscreteProbe(&model).predict(set.observations().into())?;
                let crate::nn::probes::CriticPrediction::Discrete(values) = prediction else {
                    unreachable!()
                };
                let k = values.dims()[1];
                let values = nn::to_vec(values)?;
                ensure(
                    values.iter().all(|v| v.is_finite()),
                    "nonfinite historical action values",
                )?;
                let actions: Vec<_> = set
                    .observations()
                    .iter()
                    .enumerate()
                    .map(|(i, o)| {
                        Action::Discrete(
                            (0..k)
                                .filter(|&j| o.dense_mask().is_none_or(|m| m[j]))
                                .max_by(|&a, &b| values[i * k + a].total_cmp(&values[i * k + b]))
                                .expect("validated live mask"),
                        )
                    })
                    .collect();
                Chunk {
                    context: context.clone(),
                    record: Record::Policy(PolicyRecord {
                        actions: "actions".into(),
                        parameters: PolicyParameters::Deterministic,
                        entropy: None,
                        entropy_method: None,
                        masks: None,
                        stochastic: false,
                    }),
                    arrays: BTreeMap::from([("actions".into(), probes::actions_array(&actions)?)]),
                }
            }
            _ => {
                return Err(Error::Config(
                    "selected model has no supported policy inference adapter".into(),
                ));
            }
        };
        let actions = decode_actions(&policy.arrays["actions"], &self.configuration.action_space)?;
        let critic_actions = request.fixed_actions.as_ref().unwrap_or(&actions);
        let condition = if request.fixed_actions.is_some() {
            ActionCondition::Fixed
        } else {
            ActionCondition::CurrentPolicy
        };
        let action_tensor = if matches!(
            self.configuration.action_space,
            ActionSpace::Discrete { .. }
        ) {
            None
        } else {
            let array = probes::actions_array(critic_actions)?;
            Some(nn::matrix_on_device(
                array.as_f32()?.to_vec(),
                critic_actions.len(),
                self.configuration.action_space.dimensions(),
                &Device::flex(),
            ))
        };
        let mut critics = BTreeMap::new();
        let mut categorical_representations = BTreeMap::new();
        for name in &request.critics {
            let model = self
                .models
                .get(name)
                .ok_or_else(|| Error::Config(format!("unknown historical critic {name}")))?;
            if let Loaded::Categorical(model) = model {
                let chunk = self.categorical_critic(
                    context.clone(),
                    name,
                    model,
                    &set,
                    critic_actions,
                    condition,
                )?;
                critics.insert(name.clone(), chunk);
                if let Some(layers) = request.layers.get(name).filter(|layers| !layers.is_empty()) {
                    let source = if request.fixed_actions.is_some() {
                        ActivationActionSource::FixedProbe
                    } else {
                        ActivationActionSource::Policy {
                            identity: context.identity.clone(),
                            model: request.policy.clone(),
                            stochastic: request.mode == ActMode::Sample,
                        }
                    };
                    let layers = layers.iter().map(String::as_str).collect::<Vec<_>>();
                    let chunk = reppo::activation_chunk(
                        context.clone(),
                        name,
                        model,
                        &set,
                        critic_actions,
                        source,
                        &layers,
                    )?;
                    categorical_representations.insert(name.clone(), chunk);
                }
                continue;
            }
            let (prediction, kind, actual) = match model {
                Loaded::Value(model) => {
                    let model = Value {
                        network: adapt(name, &model.network)?,
                    };
                    capture(name, &model.network, input.clone(), None)?;
                    (
                        StateValueProbe(&model).predict(set.observations().into())?,
                        ActionCondition::StateValue,
                        None,
                    )
                }
                Loaded::Discrete(model) => {
                    let model = DiscreteQ {
                        network: adapt(name, &model.network)?,
                    };
                    capture(name, &model.network, input.clone(), None)?;
                    (
                        DiscreteProbe(&model).predict(set.observations().into())?,
                        ActionCondition::AllDiscreteActions,
                        None,
                    )
                }
                Loaded::Q(model) => {
                    let model = QValue {
                        network: adapt(name, &model.network)?,
                        observation_dim: model.observation_dim,
                        action_dim: model.action_dim,
                    };
                    let a = action_tensor
                        .as_ref()
                        .ok_or_else(|| {
                            Error::Action("continuous critic needs continuous actions".into())
                        })?
                        .clone();
                    capture(name, &model.network, input.clone(), Some(&a))?;
                    let predicted =
                        self.predict_dense_q(name, &model, set.observations().into(), a)?;
                    (predicted, condition, Some(critic_actions.as_slice()))
                }
                Loaded::Residual(_) | Loaded::ResidualMoments(_) => (
                    self.residual_prediction(
                        name,
                        set.observations().into(),
                        action_tensor
                            .as_ref()
                            .ok_or_else(|| {
                                Error::Action("residual critic needs continuous actions".into())
                            })?
                            .clone(),
                    )?,
                    condition,
                    Some(critic_actions.as_slice()),
                ),
                Loaded::Moments(model) => {
                    let first_name = format!("{name}.first");
                    let second_name = format!("{name}.second");
                    let first = QValue {
                        network: adapt(&first_name, &model.first.network)?,
                        observation_dim: model.first.observation_dim,
                        action_dim: model.first.action_dim,
                    };
                    let second = QValue {
                        network: adapt(&second_name, &model.second.network)?,
                        observation_dim: model.second.observation_dim,
                        action_dim: model.second.action_dim,
                    };
                    let a = action_tensor
                        .as_ref()
                        .ok_or_else(|| {
                            Error::Action("moment critic needs continuous actions".into())
                        })?
                        .clone();
                    // Both sides share one capture concat when either is requested.
                    let x = if request.layers.contains_key(&first_name)
                        || request.layers.contains_key(&second_name)
                    {
                        Some(Tensor::cat(
                            vec![
                                input
                                    .as_ref()
                                    .expect("requested captures have shared input")
                                    .clone(),
                                a.clone(),
                            ],
                            1,
                        ))
                    } else {
                        None
                    };
                    capture(&first_name, &first.network, x.clone(), None)?;
                    // Transfer the final owner so the input can be released
                    // during this capture's forward pass, as in the original.
                    capture(&second_name, &second.network, x, None)?;
                    (
                        MomentsProbe::new(&Moments { first, second })?
                            .predict(set.observations().into(), a)?,
                        condition,
                        Some(critic_actions.as_slice()),
                    )
                }
                _ => {
                    return Err(Error::Config(format!(
                        "model {name} has no supported critic inspection adapter"
                    )));
                }
            };
            critics.insert(
                name.clone(),
                probes::critic_for_policy(
                    context.clone(),
                    prediction,
                    kind,
                    actual,
                    &request.policy,
                    request.mode,
                )?,
            );
        }
        // Interventions describe a counterfactual computation, not the authored model.
        if request.interventions.is_empty() {
            self.project_reference(&request.policy, &mut policy)?;
            for (name, chunk) in &mut critics {
                self.project_critic_value(name, chunk)?;
                self.project_reference(name, chunk)?;
            }
        } else if let Some(target) = self.return_target {
            for chunk in critics.values_mut() {
                value::attach_return_target(chunk, target)?;
            }
        }
        activations.extend(categorical_representations);
        Ok(InspectionResult {
            actions,
            probes: set.record(context)?,
            policy,
            critics,
            activations,
        })
    }
}
