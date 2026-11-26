//! Portable model snapshot metadata and optional historical neural inspection.
use super::schema::ModelSnapshotRecord;
use crate::{
    Error, Result,
    core::{action::ActionSpace, ensure},
    objectives::{
        categorical::CategoricalSupport, hl_gauss::HlGaussConfig, quantile_grid::QuantileGrid,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod categorical;
pub use categorical::CategoricalQLayout;
#[cfg(feature = "neural")]
mod neural;
#[cfg(feature = "neural")]
pub use neural::{
    CriticInspectionRequest, CriticInspectionResult, InspectionRequest, InspectionResult,
    ModelBundle, SnapshotBuilder,
};
#[cfg(feature = "neural")]
pub(crate) use neural::{
    categorical_activations, categorical_prediction, categorical_prediction_bytes,
};

const ARCHITECTURE: &str = "rust-rl/builtin-mlp/1";
const PACK_FORMAT: &str = "burn-pack/0.22.1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpLayout {
    pub widths: Vec<usize>,
    pub biases: Vec<bool>,
}
impl MlpLayout {
    fn parameters(&self) -> Result<usize> {
        ensure(
            self.widths.len() >= 2
                && self.widths.len() <= 65
                && self.biases.len() + 1 == self.widths.len()
                && self.widths.iter().all(|&n| n > 0),
            "invalid snapshot MLP layout",
        )?;
        self.widths
            .windows(2)
            .zip(&self.biases)
            .try_fold(0usize, |total, (pair, &bias)| {
                pair[0]
                    .checked_mul(pair[1])
                    .and_then(|n| n.checked_add(if bias { pair[1] } else { 0 }))
                    .and_then(|n| n.checked_add(total))
                    .ok_or_else(|| Error::Shape("snapshot parameter count overflow".into()))
            })
    }
    pub fn layers(&self) -> Vec<(String, usize)> {
        self.widths
            .iter()
            .skip(1)
            .enumerate()
            .flat_map(|(i, &width)| {
                [
                    (format!("layers.{i}.pre"), width),
                    (format!("layers.{i}.post"), width),
                ]
            })
            .collect()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelKind {
    StochasticPolicy,
    /// Built-in state-dependent Gaussian with `[means..., log_std...]` outputs.
    StateGaussianPolicy,
    DeterministicPolicy,
    StateValue,
    ScalarQ,
    QuantileQ {
        grid: QuantileGrid,
    },
    CategoricalQ {
        layout: CategoricalQLayout,
        support: HlGaussConfig,
    },
    CategoricalReturnQ {
        support: CategoricalSupport,
    },
    MomentQ {
        second_key: String,
    },
    ResidualQ {
        anchor: Box<ResidualAnchor>,
        output: ResidualOutput,
    },
    DiscreteQ,
    Opaque {
        description: String,
    },
}
impl ModelKind {
    pub fn is_stochastic_policy(&self) -> bool {
        matches!(self, Self::StochasticPolicy | Self::StateGaussianPolicy)
    }
    /// Scientific prediction represented by the model, independent of its coordinates.
    pub fn critic_kind(&self) -> Option<super::value::ValueCriticKind> {
        use super::value::ValueCriticKind as Kind;
        Some(match self {
            Self::StateValue => Kind::StateValue,
            Self::ScalarQ => Kind::ScalarQ,
            Self::DiscreteQ => Kind::DiscreteQ,
            Self::QuantileQ { .. } => Kind::QuantileQ,
            Self::MomentQ { .. } => Kind::MomentQ,
            Self::CategoricalQ { .. } => Kind::CategoricalQ,
            Self::CategoricalReturnQ { .. } => Kind::CategoricalReturnQ,
            Self::ResidualQ { output, .. } => match output {
                ResidualOutput::Scalar => Kind::ScalarQ,
                ResidualOutput::Quantile { .. } => Kind::QuantileQ,
                ResidualOutput::Moments { .. } => Kind::MomentQ,
            },
            _ => return None,
        })
    }

    pub fn quantile_grid(&self) -> Option<&QuantileGrid> {
        match self {
            Self::QuantileQ { grid }
            | Self::ResidualQ {
                output: ResidualOutput::Quantile { grid },
                ..
            } => Some(grid),
            _ => None,
        }
    }

    fn second_key(&self) -> Option<&str> {
        match self {
            Self::MomentQ { second_key }
            | Self::ResidualQ {
                output: ResidualOutput::Moments { second_key, .. },
                ..
            } => Some(second_key),
            _ => None,
        }
    }

    fn residual_anchors(&self) -> impl Iterator<Item = &ResidualAnchor> {
        match self {
            Self::ResidualQ { anchor, output } => [
                Some(anchor.as_ref()),
                match output {
                    ResidualOutput::Moments { second_anchor, .. } => Some(second_anchor.as_ref()),
                    _ => None,
                },
            ],
            _ => [None, None],
        }
        .into_iter()
        .flatten()
    }
}

/// Frozen environment-action policy and scale for the versioned input
/// convention `(action - anchor(observation)) / scale`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualAnchor {
    pub version: u32,
    pub identity: [u8; 32],
    pub network: MlpLayout,
    pub action_space: ActionSpace,
    pub scale: Vec<f32>,
}

impl ResidualAnchor {
    fn parameters(&self, observation_dim: usize, action_dim: usize) -> Result<usize> {
        self.action_space.validate()?;
        let parameters = self.network.parameters()?;
        ensure(
            self.version == 1
                && matches!(
                    self.action_space,
                    ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
                )
                && self.action_space.dimensions() == action_dim
                && self.network.widths[0] == observation_dim
                && self.network.widths.last() == Some(&action_dim)
                && self.scale.len() == action_dim
                && self
                    .scale
                    .iter()
                    .all(|scale| scale.is_finite() && *scale > 0.),
            "invalid historical residual critic anchor or scale",
        )?;
        parameters
            .checked_add(action_dim)
            .ok_or_else(|| Error::Shape("residual anchor size overflow".into()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResidualOutput {
    Scalar,
    Quantile {
        grid: QuantileGrid,
    },
    Moments {
        second_key: String,
        second_anchor: Box<ResidualAnchor>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub kind: ModelKind,
    pub network: Option<MlpLayout>,
    pub second: Option<MlpLayout>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfiguration {
    pub format_version: u32,
    pub pack_format: String,
    pub observation_dim: usize,
    pub action_space: ActionSpace,
    pub models: BTreeMap<String, ModelDefinition>,
}
#[derive(Debug, Clone, Copy)]
pub struct ModelLimits {
    pub max_parameters: usize,
    pub max_model_bytes: usize,
    pub max_models: usize,
    pub max_batch: usize,
}
impl Default for ModelLimits {
    fn default() -> Self {
        Self {
            max_parameters: 16 * 1024 * 1024,
            max_model_bytes: 64 * 1024 * 1024,
            max_models: 64,
            max_batch: 1024,
        }
    }
}
/// Validate configuration and named artifact references without constructing models or reading weights.
/// The metadata admission limit is the caller's model limit, independent of viewer retention.
pub fn snapshot_configuration(
    snapshot: &ModelSnapshotRecord,
    limits: ModelLimits,
) -> Result<SnapshotConfiguration> {
    preflight_configuration(&snapshot.configuration, limits)?;
    ensure(
        snapshot.architecture == ARCHITECTURE,
        "unsupported historical model architecture/version",
    )?;
    let config: SnapshotConfiguration =
        SnapshotConfiguration::deserialize(&snapshot.configuration)?;
    ensure(
        config.format_version == 1
            && config.pack_format == PACK_FORMAT
            && config.observation_dim > 0
            && !config.models.is_empty()
            && config.models.len() <= limits.max_models,
        "incompatible/oversized model snapshot configuration",
    )?;
    config.action_space.validate()?;
    let action_value_width = config
        .observation_dim
        .checked_add(config.action_space.dimensions())
        .ok_or_else(|| Error::Shape("historical action-value input dimension overflow".into()))?;
    let mut parameters = 0usize;
    let mut expected = BTreeSet::new();
    for (name, definition) in &config.models {
        ensure(
            expected.insert(name.as_str()),
            "duplicate model record reference",
        )?;
        if let Some(second_key) = definition.kind.second_key() {
            ensure(
                expected.insert(second_key),
                "duplicate model record reference",
            )?;
        }
        ensure(
            definition.network.is_none()
                == matches!(
                    definition.kind,
                    ModelKind::Opaque { .. } | ModelKind::CategoricalQ { .. }
                )
                && definition.second.is_some() == definition.kind.second_key().is_some(),
            "invalid model capability/layout declaration",
        )?;
        validate_definition(&config, definition, action_value_width)?;
        for layout in definition.network.iter().chain(&definition.second) {
            parameters = parameters
                .checked_add(layout.parameters()?)
                .ok_or_else(|| Error::Shape("model parameter overflow".into()))?;
        }
        for anchor in definition.kind.residual_anchors() {
            parameters = parameters
                .checked_add(
                    anchor.parameters(config.observation_dim, config.action_space.dimensions())?,
                )
                .ok_or_else(|| Error::Shape("model parameter overflow".into()))?;
        }
        if let ModelKind::CategoricalQ { layout, support } = &definition.kind {
            parameters = parameters
                .checked_add(layout.parameters(
                    config.observation_dim,
                    &config.action_space,
                    support,
                )?)
                .ok_or_else(|| Error::Shape("model parameter overflow".into()))?;
        }
        if matches!(definition.kind, ModelKind::StochasticPolicy) {
            parameters = parameters
                .checked_add(config.action_space.dimensions())
                .ok_or_else(|| Error::Shape("model parameter overflow".into()))?;
        }
    }
    ensure(
        parameters <= limits.max_parameters
            && expected
                .iter()
                .copied()
                .eq(snapshot.models.keys().map(String::as_str)),
        "snapshot model names or parameter budget mismatch",
    )?;
    Ok(config)
}
// Preflight the borrowed JSON tree before Serde allocates vectors or strings.
fn preflight_configuration(value: &serde_json::Value, limits: ModelLimits) -> Result<()> {
    fn visit(v: &serde_json::Value, depth: usize, remaining: &mut usize) -> Result<()> {
        let charge = std::mem::size_of::<serde_json::Value>();
        ensure(
            depth < 32 && *remaining >= charge,
            "snapshot metadata exceeds budget",
        )?;
        *remaining -= charge;
        match v {
            serde_json::Value::String(s) => {
                ensure(s.len() <= *remaining, "snapshot metadata exceeds budget")?;
                *remaining -= s.len();
            }
            serde_json::Value::Array(a) => {
                ensure(
                    a.len() <= *remaining / charge,
                    "snapshot metadata exceeds budget",
                )?;
                for v in a {
                    visit(v, depth + 1, remaining)?;
                }
            }
            serde_json::Value::Object(o) => {
                ensure(
                    o.len() <= *remaining / charge,
                    "snapshot metadata exceeds budget",
                )?;
                for (k, v) in o {
                    ensure(k.len() <= *remaining, "snapshot metadata exceeds budget")?;
                    *remaining -= k.len();
                    visit(v, depth + 1, remaining)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let models = value
        .get("models")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Error::Config("missing snapshot models".into()))?;
    ensure(
        !models.is_empty() && models.len() <= limits.max_models,
        "snapshot model count exceeds budget",
    )?;
    for (name, model) in models {
        ensure(
            !name.is_empty() && name.trim() == name && !name.chars().any(char::is_control),
            "invalid snapshot model name",
        )?;
        for key in ["network", "second"] {
            if let Some(layout) = model.get(key).filter(|v| !v.is_null()) {
                ensure(
                    layout
                        .get("widths")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|a| (2..=65).contains(&a.len()))
                        && layout
                            .get("biases")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(|a| a.len() <= 64),
                    "invalid snapshot MLP layout",
                )?;
            }
        }
        if let Some(categorical) = model.get("kind").and_then(|kind| kind.get("CategoricalQ")) {
            ensure(
                categorical
                    .get("layout")
                    .and_then(|layout| layout.get("encoder_widths"))
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|widths| (1..=64).contains(&widths.len())),
                "invalid normalized categorical critic layout",
            )?;
        }
        if let Some(categorical) = model
            .get("kind")
            .and_then(|kind| kind.get("CategoricalReturnQ"))
        {
            ensure(
                categorical
                    .get("support")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|atoms| {
                        (2..=super::value::MAX_VALUE_HEADS).contains(&atoms.len())
                    }),
                "invalid categorical return snapshot support",
            )?;
        }
    }
    let mut remaining = limits.max_model_bytes;
    visit(value, 0, &mut remaining)
}
fn validate_definition(
    config: &SnapshotConfiguration,
    definition: &ModelDefinition,
    action_width: usize,
) -> Result<()> {
    let Some(layout) = &definition.network else {
        return Ok(());
    };
    layout.parameters()?;
    let input = layout.widths[0];
    let output = *layout.widths.last().unwrap();
    let policy_width = match config.action_space {
        ActionSpace::Discrete { count } => count,
        _ => config.action_space.dimensions(),
    };
    let valid = match &definition.kind {
        ModelKind::StochasticPolicy => input == config.observation_dim && output == policy_width,
        ModelKind::StateGaussianPolicy => {
            let supported_space = match &config.action_space {
                ActionSpace::Unbounded { .. } => true,
                ActionSpace::Box { low, high } => low.iter().zip(high).all(|(&l, &h)| {
                    let scale = (h - l) * 0.5;
                    let center = (h + l) * 0.5;
                    scale.is_finite() && scale > 0. && center.is_finite()
                }),
                _ => false,
            };
            supported_space
                && input == config.observation_dim
                && policy_width.checked_mul(2) == Some(output)
        }
        ModelKind::DeterministicPolicy => {
            !matches!(config.action_space, ActionSpace::Discrete { .. })
                && input == config.observation_dim
                && output == policy_width
        }
        ModelKind::StateValue => input == config.observation_dim && output == 1,
        ModelKind::DiscreteQ => {
            input == config.observation_dim
                && config.action_space == ActionSpace::Discrete { count: output }
        }
        ModelKind::CategoricalReturnQ { support } => {
            matches!(
                config.action_space,
                ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
            ) && input == action_width
                && output == support.atoms().len()
        }
        ModelKind::ScalarQ
        | ModelKind::MomentQ { .. }
        | ModelKind::QuantileQ { .. }
        | ModelKind::ResidualQ { .. } => {
            let outputs = if let Some(grid) = definition.kind.quantile_grid() {
                grid.validate()?;
                grid.len()
            } else {
                1
            };
            !matches!(config.action_space, ActionSpace::Discrete { .. })
                && input == action_width
                && output == outputs
        }
        ModelKind::Opaque { .. } | ModelKind::CategoricalQ { .. } => false,
    };
    ensure(valid, "invalid historical model input/output layout")?;
    if let Some(second) = &definition.second {
        second.parameters()?;
        ensure(
            second.widths[0] == input && second.widths.last() == Some(&1),
            "invalid second-moment layout",
        )?;
    }
    Ok(())
}
