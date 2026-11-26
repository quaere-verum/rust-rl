//! Stable numerical input slices. Ranges describe model inputs, not reachability.
use crate::core::ensure;
use crate::{
    Result,
    core::{action::ActionSpace, env::Observation},
    diagnostics::visualization::{ActionDescriptor, EnvironmentSemantics, ObservationType},
};
use serde::{Deserialize, Serialize};

pub const MAX_SLICE_STATE_BYTES: usize = 256 * 1024;
pub const MAX_POPULATION_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_POPULATION_ROWS: usize = 8192;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceMode {
    /// No physical observation projection. Axis fields are dormant and must not be rendered.
    SamplesOnly,
    OneDimensional,
    TwoDimensional,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangeSource {
    Declared,
    ReferencePopulation,
    ModelInputDefault,
    UserConfigured,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridEligibility {
    /// Numerical observation axes with no masks or history; actions may be discrete.
    ContinuousFeedforward,
    Masked,
    HistoryBound,
    UnsupportedDiscrete,
}

pub const POLICY_SLICE_VERSION: u32 = 1;

/// Complete checkpoint-independent user input. Environment-derived category
/// metadata is deliberately reconstructed through `SliceConfig::new`, so an
/// authored file cannot redefine the encoding or relax categorical validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySliceSpec {
    pub version: u32,
    pub model_role: String,
    pub policy: String,
    pub mode: SliceMode,
    pub x: usize,
    pub y: Option<usize>,
    pub fixed: Vec<f32>,
    pub ranges: Vec<[f32; 2]>,
    pub range_sources: Vec<RangeSource>,
    pub resolution: [usize; 2],
    /// Preferred display component; numerical surfaces retain every component.
    pub action: usize,
    pub tolerance: f32,
    pub action_mask: Option<Vec<bool>>,
    pub history_bound: bool,
}
impl PolicySliceSpec {
    /// Exact evaluated grid/conditioning compatibility, independent of source,
    /// policy family and presentation-only preferred action/visitation tolerance.
    pub fn same_inputs(&self, other: &Self) -> bool {
        self.version == other.version
            && self.mode == other.mode
            && self.x == other.x
            && self.y == other.y
            && self.fixed == other.fixed
            && self.ranges == other.ranges
            && self.resolution == other.resolution
            && self.action_mask == other.action_mask
            && self.history_bound == other.history_bound
    }
    pub fn from_config(model_role: &str, policy: &str, config: &SliceConfig) -> Self {
        Self {
            version: POLICY_SLICE_VERSION,
            model_role: model_role.into(),
            policy: policy.into(),
            mode: config.mode,
            x: config.x,
            y: config.y,
            fixed: config.fixed.clone(),
            ranges: config.ranges.clone(),
            range_sources: config.range_sources.clone(),
            resolution: config.resolution,
            action: config.action,
            tolerance: config.tolerance,
            action_mask: config.anchor_mask.clone(),
            history_bound: config.eligibility == GridEligibility::HistoryBound,
        }
    }

    pub fn to_config(&self, environment: EnvironmentSemantics<'_>) -> Result<SliceConfig> {
        ensure(
            self.version == POLICY_SLICE_VERSION,
            "unsupported policy slice version",
        )?;
        ensure(
            [&self.model_role, &self.policy]
                .iter()
                .all(|s| !s.is_empty() && s.len() <= 128 && s.trim() == s.as_str()),
            "slice requires an explicit model role and named policy",
        )?;
        // Reject malformed imported vectors before duplicating their storage.
        let dim = environment.observation_dim();
        ensure(
            self.fixed.len() == dim
                && self.ranges.len() == dim
                && self.range_sources.len() == dim
                && dim
                    .saturating_mul(13)
                    .saturating_add(self.action_mask.as_ref().map_or(0, Vec::len))
                    .saturating_add(512)
                    <= MAX_SLICE_STATE_BYTES,
            "slice input/metadata exceeds dimensions or state byte budget",
        )?;
        ensure(
            self.action_mask.as_ref().is_none_or(|mask| {
                matches!(environment.action_space(), ActionSpace::Discrete { count } if mask.len() == *count && mask.iter().any(|v| *v))
            }),
            "slice mask differs from the action encoding",
        )?;
        let anchor = Observation::dense(self.fixed.clone(), self.action_mask.clone());
        let mut config = SliceConfig::new(environment, &anchor, &[])?;
        config.mode = self.mode;
        config.x = self.x;
        config.y = self.y;
        config.ranges.clone_from(&self.ranges);
        config.range_sources.clone_from(&self.range_sources);
        config.resolution = self.resolution;
        config.action = self.action;
        config.tolerance = self.tolerance;
        if self.history_bound {
            config = config.with_history();
        }
        config.validate(environment)?;
        Ok(config)
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct SliceConfig {
    pub mode: SliceMode,
    pub x: usize,
    pub y: Option<usize>,
    pub fixed: Vec<f32>,
    /// One persistent range per coordinate, retained when axes/checkpoints change.
    pub ranges: Vec<[f32; 2]>,
    pub range_sources: Vec<RangeSource>,
    pub resolution: [usize; 2],
    pub action: usize,
    /// Absolute model-input tolerance on non-axis numerical coordinates.
    pub tolerance: f32,
    categories: Vec<(usize, Vec<f32>)>,
    anchor_mask: Option<Vec<bool>>,
    eligibility: GridEligibility,
    action_count: usize,
}
impl SliceConfig {
    pub fn new(
        environment: EnvironmentSemantics<'_>,
        anchor: &Observation,
        population: &[Observation],
    ) -> Result<Self> {
        let dim = environment.observation_dim();
        let category_bytes = environment.descriptor().map_or(0, |d| {
            d.observations
                .iter()
                .map(|f| match &f.kind {
                    ObservationType::Categorical { categories } => {
                        64usize.saturating_add(categories.len().saturating_mul(4))
                    }
                    _ => 0,
                })
                .sum::<usize>()
        });
        let mask_bytes = anchor.dense_mask().map_or(0, <[bool]>::len);
        ensure(
            dim > 0
                && dim
                    .checked_mul(13)
                    .and_then(|b| b.checked_add(category_bytes))
                    .and_then(|b| b.checked_add(mask_bytes))
                    .and_then(|b| b.checked_add(512))
                    .is_some_and(|b| b <= MAX_SLICE_STATE_BYTES),
            "slice input/metadata exceeds state byte budget",
        )?;
        admit_population(dim, population)?;
        anchor.validate(dim, environment.action_space(), true)?;
        for row in population {
            row.validate(dim, environment.action_space(), true)?;
        }
        let categories = environment
            .descriptor()
            .map(|d| {
                d.observations
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| match &f.kind {
                        ObservationType::Categorical { categories } => {
                            Some((i, categories.iter().map(|c| c.code).collect()))
                        }
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let action_count = match environment.action_space() {
            ActionSpace::Discrete { count } => *count,
            s => s.dimensions(),
        };
        let action = match environment.descriptor().map(|d| &d.actions) {
            Some(ActionDescriptor::Continuous {
                preferred_component,
                ..
            }) => preferred_component.unwrap_or(0),
            _ => anchor
                .dense_mask()
                .and_then(|m| m.iter().position(|&v| v))
                .unwrap_or(0),
        };
        let ranges: (Vec<[f32; 2]>, Vec<RangeSource>) = (0..dim)
            .map(|i| {
                if let Some(range) = environment
                    .descriptor()
                    .and_then(|d| d.observations.get(i))
                    .and_then(|f| f.range)
                    .filter(|r| r[0] < r[1])
                {
                    return (range, RangeSource::Declared);
                }
                if !population.is_empty() {
                    let lo = population
                        .iter()
                        .map(|r| r.dense_values()[i])
                        .fold(f32::INFINITY, f32::min);
                    let hi = population
                        .iter()
                        .map(|r| r.dense_values()[i])
                        .fold(f32::NEG_INFINITY, f32::max);
                    if lo < hi {
                        return ([lo, hi], RangeSource::ReferencePopulation);
                    }
                }
                let v = f64::from(anchor.dense_values()[i]);
                let radius = (v.abs() * 0.1).max(1.);
                (
                    [
                        (v - radius).max(-f64::from(f32::MAX)) as f32,
                        (v + radius).min(f64::from(f32::MAX)) as f32,
                    ],
                    RangeSource::ModelInputDefault,
                )
            })
            .unzip();
        let eligibility = if anchor.dense_mask().is_some() {
            GridEligibility::Masked
        } else {
            GridEligibility::ContinuousFeedforward
        };
        let mut value = Self {
            mode: SliceMode::OneDimensional,
            x: 0,
            y: None,
            fixed: anchor.dense_values().to_vec(),
            ranges: ranges.0,
            range_sources: ranges.1,
            resolution: [256, 64],
            action,
            tolerance: 0.,
            categories,
            anchor_mask: anchor.dense_mask().map(<[bool]>::to_vec),
            eligibility,
            action_count,
        };
        let mut axes = [None, None];
        let mut add = |i: usize| {
            if value.numeric(i) && !axes.contains(&Some(i)) {
                if axes[0].is_none() {
                    axes[0] = Some(i);
                } else if axes[1].is_none() {
                    axes[1] = Some(i);
                }
            }
        };
        if let Some(d) = environment.descriptor() {
            for name in &d.preferences.coordinates {
                if let Some(i) = d.observations.iter().position(|f| f.id == *name) {
                    add(i);
                }
            }
            for (i, f) in d.observations.iter().enumerate() {
                if f.coordinate {
                    add(i);
                }
            }
        }
        for i in 0..dim {
            add(i);
        }
        value.x = axes[0].unwrap_or(0);
        value.y = axes[1];
        if axes[0].is_none() {
            value.mode = SliceMode::SamplesOnly;
        } else if value.y.is_some() {
            value.mode = SliceMode::TwoDimensional;
            value.resolution = [64, 64];
        }
        value.validate(environment)?;
        Ok(value)
    }
    pub fn numeric(&self, index: usize) -> bool {
        index < self.fixed.len() && !self.categories.iter().any(|(i, _)| *i == index)
    }
    fn structural(&self) -> Result<()> {
        ensure(
            self.owned_bytes() <= MAX_SLICE_STATE_BYTES,
            "slice retained state exceeds byte budget",
        )?;
        let n = self.fixed.len();
        ensure(
            n > 0
                && self.ranges.len() == n
                && self.range_sources.len() == n
                && self.fixed.iter().all(|v| v.is_finite())
                && self
                    .ranges
                    .iter()
                    .all(|[l, h]| l.is_finite() && h.is_finite() && l < h)
                && (self.mode == SliceMode::SamplesOnly
                    || (self.numeric(self.x)
                        && self.y.is_none_or(|y| self.numeric(y) && y != self.x)
                        && (self.mode == SliceMode::OneDimensional || self.y.is_some())))
                && self.action < self.action_count
                && self.tolerance.is_finite()
                && self.tolerance >= 0.,
            "invalid slice axes, values, ranges, action or tolerance",
        )?;
        ensure(
            self.categories
                .iter()
                .all(|(i, codes)| self.fixed.get(*i).is_some_and(|v| codes.contains(v))),
            "invalid categorical fixed input code",
        )?;
        self.len()?;
        Ok(())
    }
    pub fn validate(&self, environment: EnvironmentSemantics<'_>) -> Result<()> {
        ensure(
            self.fixed.len() == environment.observation_dim(),
            "slice/environment observation dimensions differ",
        )?;
        let actions = match environment.action_space() {
            ActionSpace::Discrete { count } => *count,
            s => s.dimensions(),
        };
        ensure(
            self.action_count == actions,
            "slice/environment action dimensions differ",
        )?;
        self.structural()
    }
    pub fn len(&self) -> Result<usize> {
        let [x, y] = self.resolution;
        let n = match self.mode {
            SliceMode::SamplesOnly => 0,
            SliceMode::OneDimensional => {
                ensure(
                    (2..=4096).contains(&x),
                    "1D slice resolution must be 2..4096",
                )?;
                x
            }
            SliceMode::TwoDimensional => {
                ensure(
                    (2..=256).contains(&x) && (2..=256).contains(&y),
                    "2D slice resolution must be 2..256 per axis",
                )?;
                x.checked_mul(y)
                    .ok_or_else(|| crate::Error::Shape("slice size overflow".into()))?
            }
        };
        Ok(n)
    }
    pub fn is_empty(&self) -> bool {
        self.mode == SliceMode::SamplesOnly
    }
    /// Axis changes are transactional and retain all fixed values/action/ranges.
    pub fn set_axes(
        &mut self,
        environment: EnvironmentSemantics<'_>,
        x: usize,
        y: Option<usize>,
    ) -> Result<()> {
        ensure(
            self.numeric(x) && y.is_none_or(|i| self.numeric(i) && i != x),
            "slice axes must be distinct numerical coordinates",
        )?;
        self.validate(environment)?;
        let mut proposed = self.clone();
        proposed.x = x;
        proposed.y = y;
        proposed.mode = if y.is_some() {
            SliceMode::TwoDimensional
        } else {
            SliceMode::OneDimensional
        };
        if y.is_some() && proposed.resolution.iter().any(|n| !(2..=256).contains(n)) {
            proposed.resolution = [64, 64];
        }
        proposed.validate(environment)?;
        *self = proposed;
        Ok(())
    }
    /// There is no history inference adapter. This restriction cannot be cleared.
    pub fn with_history(mut self) -> Self {
        self.eligibility = GridEligibility::HistoryBound;
        self
    }
    pub fn grid_eligibility(&self) -> GridEligibility {
        if self.mode == SliceMode::SamplesOnly && self.eligibility != GridEligibility::HistoryBound
        {
            GridEligibility::UnsupportedDiscrete
        } else {
            self.eligibility
        }
    }
    pub fn anchor_mask(&self) -> Option<&[bool]> {
        self.anchor_mask.as_deref()
    }
    /// All owned capacities, including nested categorical codes and the anchor
    /// mask. The containing owner must charge this alongside its other buffers.
    pub fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.fixed.capacity().saturating_mul(4))
            .saturating_add(self.ranges.capacity().saturating_mul(8))
            .saturating_add(
                self.range_sources
                    .capacity()
                    .saturating_mul(std::mem::size_of::<RangeSource>()),
            )
            .saturating_add(
                self.categories
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(usize, Vec<f32>)>()),
            )
            .saturating_add(
                self.categories
                    .iter()
                    .map(|(_, v)| v.capacity().saturating_mul(4))
                    .sum::<usize>(),
            )
            .saturating_add(self.anchor_mask.as_ref().map_or(0, Vec::capacity))
    }
    /// Local-bin methodology shared by the overlay and Selected count. Require
    /// exact categorical/mask equality, absolute tolerance on fixed numerical
    /// coordinates, and plot-range inclusion; x varies fastest. Bins are uniform
    /// half-open intervals except the final bin includes the high endpoint.
    pub fn visit_bin(&self, observation: &Observation) -> Result<Option<usize>> {
        self.structural()?;
        if self.mode == SliceMode::SamplesOnly {
            return Ok(None);
        }
        ensure(
            observation.dense_values().len() == self.fixed.len()
                && observation.dense_values().iter().all(|v| v.is_finite()),
            "invalid visitation observation",
        )?;
        if observation.dense_mask() != self.anchor_mask() {
            return Ok(None);
        }
        for (i, (&value, &fixed)) in observation
            .dense_values()
            .iter()
            .zip(&self.fixed)
            .enumerate()
        {
            if i == self.x || (self.mode == SliceMode::TwoDimensional && Some(i) == self.y) {
                continue;
            }
            let matches = if self
                .categories
                .binary_search_by_key(&i, |(index, _)| *index)
                .is_ok()
            {
                value == fixed
            } else {
                (f64::from(value) - f64::from(fixed)).abs() <= f64::from(self.tolerance)
            };
            if !matches {
                return Ok(None);
            }
        }
        let bin = |axis: usize, n: usize| -> Option<usize> {
            let value = observation.dense_values()[axis];
            let [l, h] = self.ranges[axis];
            if value < l || value > h {
                return None;
            }
            Some(
                (((f64::from(value) - f64::from(l)) / (f64::from(h) - f64::from(l)) * n as f64)
                    .floor() as usize)
                    .min(n - 1),
            )
        };
        let Some(x) = bin(self.x, self.resolution[0]) else {
            return Ok(None);
        };
        let y = if self.mode == SliceMode::TwoDimensional {
            let Some(y) = bin(self.y.expect("validated y"), self.resolution[1]) else {
                return Ok(None);
            };
            y
        } else {
            0
        };
        Ok(Some(y * self.resolution[0] + x))
    }
    /// Construct only this row, never the full grid. x varies fastest, endpoints
    /// included. Return the exact f32 values consumed by model inference.
    pub fn observation_at(&self, index: usize) -> Result<Observation> {
        self.structural()?;
        ensure(
            self.eligibility == GridEligibility::ContinuousFeedforward,
            "synthetic grid unavailable for masked or history-bound input",
        )?;
        ensure(index < self.len()?, "slice row outside grid")?;
        let mut values = self.fixed.clone();
        let nx = self.resolution[0];
        let coordinate = |axis: usize, i: usize, n: usize| {
            let [l, h] = self.ranges[axis];
            (f64::from(l) + (f64::from(h) - f64::from(l)) * i as f64 / (n - 1) as f64) as f32
        };
        values[self.x] = coordinate(self.x, index % nx, nx);
        if self.mode == SliceMode::TwoDimensional {
            let y = self.y.expect("validated y");
            values[y] = coordinate(y, index / nx, self.resolution[1]);
        }
        Ok(Observation::new(values))
    }
}
pub fn admit_population(dim: usize, population: &[Observation]) -> Result<()> {
    ensure(
        population.len() <= MAX_POPULATION_ROWS
            && population
                .len()
                .checked_mul(dim)
                .and_then(|v| v.checked_mul(4))
                .is_some_and(|b| b <= MAX_POPULATION_BYTES),
        "slice population exceeds row/byte scan budget",
    )?;
    let bytes = population.iter().try_fold(0usize, |total, row| {
        row.dense_values()
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_add(row.dense_mask().map_or(0, <[bool]>::len)))
            .and_then(|n| n.checked_add(std::mem::size_of::<Observation>()))
            .and_then(|n| total.checked_add(n))
    });
    ensure(
        bytes.is_some_and(|b| b <= MAX_POPULATION_BYTES),
        "slice population inputs/masks exceed byte scan budget",
    )
}
