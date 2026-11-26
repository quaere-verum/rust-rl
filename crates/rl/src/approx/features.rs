//! Versioned action features and fixed mixtures of linear surfaces.

use crate::{
    Result,
    core::{
        action::Action,
        ensure,
        env::PublicHistory,
        structured::{FeatureIdentity, feature_fields},
        temporal::TemporalSpaces,
    },
};
use serde::{Deserialize, Serialize};

use super::regression::DiagonalPenalty;

/// Semantic identity of an ordered feature vector. Adapters must change this
/// descriptor when the transformations, scales, action identities, gating
/// descriptors, or learned representation change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureSpec {
    pub namespace: String,
    pub version: u32,
    pub terms: Vec<FeatureIdentity>,
    /// Canonically ordered transformation and source settings, including
    /// fixed normalizers or the revision of a supplied public-state model.
    pub configuration: Vec<String>,
    /// Content identity of an underlying frozen representation, when used.
    pub representation: Option<[u8; 32]>,
}

impl FeatureSpec {
    pub fn dimension(&self) -> usize {
        self.terms.len()
    }

    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version > 0
                && !self.namespace.trim().is_empty()
                && self.namespace.len() <= 4096
                && !self.terms.is_empty()
                && feature_fields(&self.terms)
                && self
                    .configuration
                    .iter()
                    .all(|value| !value.trim().is_empty()),
            "invalid feature specification",
        )
    }
}

/// A frozen, causal mapping from one public decision frame and complete
/// actions to feature rows in the supplied action order. Implementations may
/// share state encoding across the actions. A row's value depends on its own
/// action and history. Callers supply legal actions and check output widths
/// and finiteness at the fitting and inference boundaries.
pub trait FeatureMap {
    fn spec(&self) -> &FeatureSpec;
    fn validate_spaces(&self, _spaces: &TemporalSpaces) -> Result<()> {
        Ok(())
    }
    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>>;
}

pub(crate) fn checked_features(
    features: &dyn FeatureMap,
    history: &PublicHistory,
    actions: &[Action],
) -> Result<Vec<Vec<f64>>> {
    let encoded = features.encode(history, actions)?;
    ensure(
        encoded.len() == actions.len()
            && encoded.iter().all(|row| {
                row.len() == features.spec().dimension() && row.iter().all(|x| x.is_finite())
            }),
        "invalid feature rows",
    )?;
    Ok(encoded)
}

/// Prepend a constant feature to a global linear basis. Set the penalty for
/// coefficient zero to zero to fit an unpenalized intercept.
#[derive(Debug, Clone)]
pub struct Intercept<F> {
    base: F,
    spec: FeatureSpec,
}

impl<F: FeatureMap> Intercept<F> {
    pub fn new(base: F) -> Result<Self> {
        base.spec().validate()?;
        let mut terms = vec![FeatureIdentity::new("intercept", "constant 1")];
        terms.extend(base.spec().terms.iter().map(|term| {
            FeatureIdentity::new(format!("base/{}", term.name), term.transform.clone())
        }));
        let spec = FeatureSpec {
            namespace: "rust_rl.approx.intercept".into(),
            version: 1,
            terms,
            configuration: vec![serde_json::to_string(base.spec())?],
            representation: base.spec().representation,
        };
        spec.validate()?;
        Ok(Self { base, spec })
    }
}

impl<F: FeatureMap> FeatureMap for Intercept<F> {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn validate_spaces(&self, spaces: &TemporalSpaces) -> Result<()> {
        self.base.validate_spaces(spaces)
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        let mut rows = checked_features(&self.base, history, actions)?;
        for row in &mut rows {
            row.insert(0, 1.);
        }
        Ok(rows)
    }
}

/// Action-independent public gating coordinates, in declared order.
pub trait GateDescriptor {
    fn spec(&self) -> &FeatureSpec;
    fn describe(&self, history: &PublicHistory) -> Result<Vec<f64>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GateAssignment {
    /// Squared scaled distance; the first center wins an exact tie.
    Nearest,
    Gaussian {
        sigma: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateConfig {
    pub centers: Vec<Vec<f64>>,
    /// Rows of the fixed matrix S, applied before taking the Euclidean norm.
    pub scale: Vec<Vec<f64>>,
    pub assignment: GateAssignment,
}

impl GateConfig {
    fn validate(&self, dimension: usize) -> Result<()> {
        ensure(
            !self.centers.is_empty()
                && !self.scale.is_empty()
                && self
                    .centers
                    .iter()
                    .chain(&self.scale)
                    .all(|row| row.len() == dimension && row.iter().all(|value| value.is_finite()))
                && match self.assignment {
                    GateAssignment::Nearest => true,
                    GateAssignment::Gaussian { sigma } => sigma.is_finite() && sigma > 0.,
                },
            "invalid fixed gate configuration",
        )
    }

    fn weights(&self, descriptor: &[f64]) -> Result<Vec<f64>> {
        ensure(
            descriptor.len() == self.centers[0].len()
                && descriptor.iter().all(|value| value.is_finite()),
            "invalid gate descriptor",
        )?;
        let mut distances: Vec<f64> = self
            .centers
            .iter()
            .map(|center| {
                self.scale
                    .iter()
                    .map(|row| {
                        let scaled: f64 = row
                            .iter()
                            .zip(descriptor.iter().zip(center))
                            .map(|(scale, (coordinate, center))| scale * (coordinate - center))
                            .sum();
                        scaled * scaled
                    })
                    .sum()
            })
            .collect();
        ensure(
            distances.iter().all(|distance| distance.is_finite()),
            "fixed gate distance overflow",
        )?;
        let (nearest, &minimum) = distances
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("validated nonempty gates");
        match self.assignment {
            GateAssignment::Nearest => {
                distances.fill(0.);
                distances[nearest] = 1.;
            }
            GateAssignment::Gaussian { sigma } => {
                for distance in &mut distances {
                    let offset = (*distance - minimum).sqrt() / sigma;
                    *distance = (-0.5 * offset * offset).exp();
                }
                let normalizer: f64 = distances.iter().sum();
                for mass in &mut distances {
                    *mass /= normalizer;
                }
            }
        }
        Ok(distances)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocalLayout {
    /// Concatenate one gated base feature block per center.
    Local,
    /// Prepend the unchanged base features before the local corrections.
    GlobalPlusLocal,
}

/// Fixed local-linear features as in the archived Rust local Q-learning
/// study. The gates are action independent and their masses sum to one.
/// Coefficients remain globally linear, with no fitted partition.
#[derive(Debug, Clone)]
pub struct FixedLocal<F, G> {
    base: F,
    descriptor: G,
    gates: GateConfig,
    layout: LocalLayout,
    spec: FeatureSpec,
}

impl<F: FeatureMap, G: GateDescriptor> FixedLocal<F, G> {
    pub fn new(base: F, descriptor: G, gates: GateConfig, layout: LocalLayout) -> Result<Self> {
        base.spec().validate()?;
        descriptor.spec().validate()?;
        gates.validate(descriptor.spec().dimension())?;
        let mut terms = Vec::new();
        if layout == LocalLayout::GlobalPlusLocal {
            terms.extend(base.spec().terms.iter().map(|term| {
                FeatureIdentity::new(format!("global/{}", term.name), term.transform.clone())
            }));
        }
        for k in 0..gates.centers.len() {
            terms.extend(base.spec().terms.iter().map(|term| {
                FeatureIdentity::new(
                    format!("gate{k}/{}", term.name),
                    format!("gate{k} * ({})", term.transform),
                )
            }));
        }
        let spec = FeatureSpec {
            namespace: "rust_rl.approx.fixed_local".into(),
            version: 1,
            terms,
            configuration: vec![serde_json::to_string(&(
                base.spec(),
                descriptor.spec(),
                &gates,
                layout,
            ))?],
            representation: base.spec().representation,
        };
        spec.validate()?;
        Ok(Self {
            base,
            descriptor,
            gates,
            layout,
            spec,
        })
    }

    pub fn gate_weights(&self, history: &PublicHistory) -> Result<Vec<f64>> {
        self.gates.weights(&self.descriptor.describe(history)?)
    }

    /// Repeat the local block's penalty for each gate, independently of the
    /// optional global block. Per-coefficient penalties refer to the base order.
    pub fn penalty(
        &self,
        global: &DiagonalPenalty,
        local: &DiagonalPenalty,
    ) -> Result<DiagonalPenalty> {
        let width = self.base.spec().dimension();
        let mut penalty = match self.layout {
            LocalLayout::Local => Vec::new(),
            LocalLayout::GlobalPlusLocal => global.diagonal(width)?,
        };
        let local = local.diagonal(width)?;
        for _ in &self.gates.centers {
            penalty.extend_from_slice(&local);
        }
        Ok(DiagonalPenalty::PerCoefficient(penalty))
    }
}

impl<F: FeatureMap, G: GateDescriptor> FeatureMap for FixedLocal<F, G> {
    fn spec(&self) -> &FeatureSpec {
        &self.spec
    }

    fn validate_spaces(&self, spaces: &TemporalSpaces) -> Result<()> {
        self.base.validate_spaces(spaces)
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        let base = checked_features(&self.base, history, actions)?;
        let weights = self.gate_weights(history)?;
        Ok(base
            .iter()
            .map(|row| {
                let mut output = Vec::with_capacity(self.spec.dimension());
                if self.layout == LocalLayout::GlobalPlusLocal {
                    output.extend_from_slice(row);
                }
                for weight in &weights {
                    output.extend(row.iter().map(|value| weight * value));
                }
                output
            })
            .collect())
    }
}
