//! Optional, versioned environment annotations. Attach explicitly before recording.
//!
//! These describe recorded values; they never change action bounds, observations,
//! reward ordering, clocks, or analytical support. Metadata budgets do not limit
//! ordinary RL dimensions. Missing annotations carry no semantic classification.
use super::{codec, schema::RunManifest};
use crate::{
    Result,
    core::{action::ActionSpace, ensure},
};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::HashSet};

pub const ENVIRONMENT_DESCRIPTOR_VERSION: u32 = 1;
pub const MAX_DESCRIPTOR_BYTES: usize = 64 * 1024;
pub const MAX_LABEL_ENTRIES: usize = 4096;
pub const MAX_FRAME_CHANNELS: usize = 128;
pub const MAX_GROUPS: usize = 64;
pub const MAX_CATEGORIES: usize = 256;
pub const MAX_ID_UNIT_BYTES: usize = 128;
pub const MAX_LABEL_DESCRIPTION_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentDescriptor {
    pub version: u32,
    pub observations: Vec<ObservationField>,
    pub actions: ActionDescriptor,
    pub clocks: ClockDescriptor,
    pub reward: RewardDescriptor,
    #[serde(default)]
    pub frames: Vec<FrameChannel>,
    #[serde(default)]
    pub preferences: VisualizationPreferences,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationField {
    pub id: String,
    pub label: String,
    pub unit: Option<String>,
    pub description: Option<String>,
    pub kind: ObservationType,
    pub range: Option<[f32; 2]>,
    pub coordinate: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObservationType {
    Numerical,
    /// Explicit codes in the existing f32 observation encoding.
    Categorical {
        categories: Vec<Category>,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Category {
    pub code: f32,
    pub label: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionDescriptor {
    Discrete {
        choices: Vec<DiscreteChoice>,
    },
    Continuous {
        components: Vec<ActionComponent>,
        preferred_component: Option<usize>,
    },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscreteChoice {
    pub label: String,
    pub meaning: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionComponent {
    pub label: String,
    pub unit: Option<String>,
    pub meaning: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockDescriptor {
    pub learner_revision: QuantityLabel,
    pub decision_step: QuantityLabel,
    /// A label only: absence of recorded time remains absence of time.
    pub simulation: Option<QuantityLabel>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantityLabel {
    pub label: String,
    pub unit: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardDescriptor {
    pub label: String,
    pub unit: Option<String>,
    pub description: Option<String>,
    /// References to separately recorded frame components, never inferred costs.
    #[serde(default)]
    pub components: Vec<FrameReference>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameChannel {
    pub name: String,
    pub label: String,
    pub width: usize,
    pub unit: Option<String>,
    #[serde(default)]
    pub visibility: FrameVisibility,
    pub representation: Option<FrameRepresentation>,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameVisibility {
    AgentObserved,
    DiagnosticOnly,
    Privileged,
    #[default]
    Unknown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameRepresentation {
    Scalar,
    Vector,
    Coordinates,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameReference {
    pub channel: String,
    pub index: usize,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualizationPreferences {
    pub coordinates: Vec<String>,
    pub trajectories: Vec<FrameReference>,
    pub groups: Vec<ObservationGroup>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationGroup {
    pub id: String,
    pub label: String,
    pub observations: Vec<String>,
}

fn text(value: &str, max: usize) -> Result<()> {
    ensure(
        value.len() <= max,
        "environment metadata string exceeds budget",
    )
}
fn id(value: &str) -> Result<()> {
    text(value, MAX_ID_UNIT_BYTES)?;
    ensure(
        !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control),
        "invalid environment metadata identifier",
    )
}
fn label(value: &str) -> Result<()> {
    text(value, MAX_LABEL_DESCRIPTION_BYTES)
}
fn unit(value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        text(value, MAX_ID_UNIT_BYTES)?;
    }
    Ok(())
}
fn description(value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        label(value)?;
    }
    Ok(())
}
impl EnvironmentDescriptor {
    /// Validate counts before traversing entries, then references, then bounded
    /// serialization. There is no unbounded intermediate JSON allocation.
    pub fn validate(&self, observation_dim: usize, action_space: &ActionSpace) -> Result<()> {
        ensure(
            self.version == ENVIRONMENT_DESCRIPTOR_VERSION,
            "unsupported environment descriptor version",
        )?;
        action_space.validate()?;
        let action_count = match &self.actions {
            ActionDescriptor::Discrete { choices } => choices.len(),
            ActionDescriptor::Continuous { components, .. } => components.len(),
        };
        ensure(
            self.observations.len() <= MAX_LABEL_ENTRIES
                && action_count <= MAX_LABEL_ENTRIES
                && self.observations.len().saturating_add(action_count) <= MAX_LABEL_ENTRIES,
            "environment observation/action labels exceed budget",
        )?;
        ensure(
            observation_dim > 0 && self.observations.len() == observation_dim,
            "environment observation metadata dimension mismatch",
        )?;
        ensure(
            self.frames.len() <= MAX_FRAME_CHANNELS
                && self.preferences.groups.len() <= MAX_GROUPS
                && self.preferences.coordinates.len() <= MAX_LABEL_ENTRIES
                && self.preferences.trajectories.len() <= MAX_LABEL_ENTRIES
                && self.reward.components.len() <= MAX_LABEL_ENTRIES,
            "environment metadata count exceeds budget",
        )?;
        // Nested counts are admitted before entry/string/reference work too.
        for field in &self.observations {
            if let ObservationType::Categorical { categories } = &field.kind {
                ensure(
                    !categories.is_empty() && categories.len() <= MAX_CATEGORIES,
                    "environment category count exceeds budget or is empty",
                )?;
            }
        }
        for group in &self.preferences.groups {
            ensure(
                group.observations.len() <= MAX_LABEL_ENTRIES,
                "environment group exceeds budget",
            )?;
        }
        let mut ids = HashSet::new();
        for field in &self.observations {
            id(&field.id)?;
            ensure(
                ids.insert(field.id.as_str()),
                "duplicate environment observation identifier",
            )?;
            label(&field.label)?;
            unit(field.unit.as_deref())?;
            description(field.description.as_deref())?;
            if let Some([low, high]) = field.range {
                ensure(
                    low.is_finite() && high.is_finite() && low <= high,
                    "invalid environment observation range",
                )?;
            }
            if let ObservationType::Categorical { categories } = &field.kind {
                let mut codes = HashSet::new();
                for category in categories {
                    ensure(
                        category.code.is_finite(),
                        "nonfinite environment category code",
                    )?;
                    let code = if category.code == 0. {
                        0
                    } else {
                        category.code.to_bits()
                    };
                    ensure(codes.insert(code), "duplicate environment category code")?;
                    label(&category.label)?;
                }
            }
        }
        match (&self.actions, action_space) {
            (ActionDescriptor::Discrete { choices }, ActionSpace::Discrete { count }) => {
                ensure(
                    choices.len() == *count,
                    "environment discrete choice count mismatch",
                )?;
                for choice in choices {
                    label(&choice.label)?;
                    description(choice.meaning.as_deref())?;
                }
            }
            (
                ActionDescriptor::Continuous {
                    components,
                    preferred_component,
                },
                ActionSpace::Box { .. } | ActionSpace::Unbounded { .. },
            ) => {
                ensure(
                    components.len() == action_space.dimensions(),
                    "environment action component count mismatch",
                )?;
                ensure(
                    preferred_component.is_none_or(|i| i < components.len()),
                    "invalid preferred action component",
                )?;
                for component in components {
                    label(&component.label)?;
                    unit(component.unit.as_deref())?;
                    description(component.meaning.as_deref())?;
                }
            }
            _ => ensure(false, "incompatible environment action metadata type")?,
        }
        for clock in [&self.clocks.learner_revision, &self.clocks.decision_step]
            .into_iter()
            .chain(self.clocks.simulation.iter())
        {
            label(&clock.label)?;
            unit(clock.unit.as_deref())?;
        }
        label(&self.reward.label)?;
        unit(self.reward.unit.as_deref())?;
        description(self.reward.description.as_deref())?;
        let mut channels = HashSet::new();
        for channel in &self.frames {
            id(&channel.name)?;
            ensure(
                channels.insert(channel.name.as_str()),
                "duplicate environment frame channel",
            )?;
            ensure(channel.width > 0, "zero environment frame width")?;
            label(&channel.label)?;
            unit(channel.unit.as_deref())?;
        }
        let check_reference = |reference: &FrameReference| -> Result<()> {
            id(&reference.channel)?;
            ensure(
                self.frames
                    .iter()
                    .any(|f| f.name == reference.channel && reference.index < f.width),
                "unresolved environment frame component",
            )
        };
        for reference in self
            .reward
            .components
            .iter()
            .chain(&self.preferences.trajectories)
        {
            check_reference(reference)?;
        }
        let mut coordinates = HashSet::new();
        for coordinate in &self.preferences.coordinates {
            id(coordinate)?;
            ensure(
                coordinates.insert(coordinate.as_str())
                    && self
                        .observations
                        .iter()
                        .any(|f| f.id == *coordinate && f.coordinate),
                "unresolved, duplicate or unsuitable environment coordinate",
            )?;
        }
        let mut groups = HashSet::new();
        for group in &self.preferences.groups {
            id(&group.id)?;
            label(&group.label)?;
            ensure(
                groups.insert(group.id.as_str()),
                "duplicate environment observation group",
            )?;
            let mut members = HashSet::new();
            for member in &group.observations {
                id(member)?;
                ensure(
                    ids.contains(member.as_str()) && members.insert(member.as_str()),
                    "unresolved or duplicate environment group member",
                )?;
            }
        }
        codec::bounded_json(self, MAX_DESCRIPTOR_BYTES)?;
        Ok(())
    }
}

/// One field view; generic labels allocate only for this requested field.
pub struct ObservationView<'a> {
    pub id: Cow<'a, str>,
    pub label: Cow<'a, str>,
    pub unit: Option<&'a str>,
    pub description: Option<&'a str>,
    pub kind: Option<&'a ObservationType>,
    pub range: Option<[f32; 2]>,
    pub coordinate: bool,
}
pub struct ActionView<'a> {
    pub label: Cow<'a, str>,
    pub unit: Option<&'a str>,
    pub meaning: Option<&'a str>,
}
pub struct RewardView<'a> {
    pub label: &'a str,
    pub unit: Option<&'a str>,
    pub description: Option<&'a str>,
    pub components: &'a [FrameReference],
}

/// Borrow a validated manifest, preserving its authoritative dimensions/space.
/// No allocation occurs on construction, including legacy extreme dimensions.
#[derive(Clone, Copy)]
pub struct EnvironmentSemantics<'a> {
    manifest: &'a RunManifest,
}
impl<'a> EnvironmentSemantics<'a> {
    pub fn new(manifest: &'a RunManifest) -> Self {
        Self { manifest }
    }
    pub fn descriptor(&self) -> Option<&'a EnvironmentDescriptor> {
        self.manifest.environment_descriptor.as_ref()
    }
    pub fn observation_dim(&self) -> usize {
        self.manifest.observation_dim
    }
    pub fn action_space(&self) -> &'a ActionSpace {
        &self.manifest.action_space
    }
    pub fn observation(&self, index: usize) -> Option<ObservationView<'a>> {
        if index >= self.observation_dim() {
            return None;
        }
        if let Some(field) = self.descriptor().and_then(|d| d.observations.get(index)) {
            return Some(ObservationView {
                id: Cow::Borrowed(&field.id),
                label: Cow::Borrowed(&field.label),
                unit: field.unit.as_deref(),
                description: field.description.as_deref(),
                kind: Some(&field.kind),
                range: field.range,
                coordinate: field.coordinate,
            });
        }
        let id = format!("obs[{index}]");
        let label = self
            .manifest
            .observation_labels
            .get(index)
            .filter(|label| !label.trim().is_empty())
            .map(|label| Cow::Borrowed(label.as_str()))
            .unwrap_or_else(|| Cow::Owned(id.clone()));
        Some(ObservationView {
            id: Cow::Owned(id),
            label,
            unit: None,
            description: None,
            kind: None,
            range: None,
            coordinate: false,
        })
    }
    pub fn observation_by_id(&self, id: &str) -> Option<ObservationView<'a>> {
        if let Some(d) = self.descriptor() {
            return self.observation(d.observations.iter().position(|f| f.id == id)?);
        }
        let index = id
            .strip_prefix("obs[")?
            .strip_suffix(']')?
            .parse::<usize>()
            .ok()?;
        let view = self.observation(index)?;
        (view.id == id).then_some(view)
    }
    pub fn action(&self, index: usize) -> Option<ActionView<'a>> {
        let count = match self.action_space() {
            ActionSpace::Discrete { count } => *count,
            space => space.dimensions(),
        };
        if index >= count {
            return None;
        }
        if let Some(d) = self.descriptor() {
            match &d.actions {
                ActionDescriptor::Discrete { choices } => {
                    let c = choices.get(index)?;
                    return Some(ActionView {
                        label: Cow::Borrowed(&c.label),
                        unit: None,
                        meaning: c.meaning.as_deref(),
                    });
                }
                ActionDescriptor::Continuous { components, .. } => {
                    let c = components.get(index)?;
                    return Some(ActionView {
                        label: Cow::Borrowed(&c.label),
                        unit: c.unit.as_deref(),
                        meaning: c.meaning.as_deref(),
                    });
                }
            }
        }
        Some(ActionView {
            label: Cow::Owned(match self.action_space() {
                ActionSpace::Discrete { .. } => index.to_string(),
                _ => format!("action[{index}]"),
            }),
            unit: None,
            meaning: None,
        })
    }
    pub fn clocks(&self) -> Option<&'a ClockDescriptor> {
        self.descriptor().map(|d| &d.clocks)
    }
    pub fn reward(&self) -> RewardView<'a> {
        match self.descriptor() {
            Some(d) => RewardView {
                label: &d.reward.label,
                unit: d.reward.unit.as_deref(),
                description: d.reward.description.as_deref(),
                components: &d.reward.components,
            },
            None => RewardView {
                label: &self.manifest.reward_convention,
                unit: None,
                description: None,
                components: &[],
            },
        }
    }
    pub fn frame(&self, name: &str) -> Option<&'a FrameChannel> {
        self.descriptor()?.frames.iter().find(|f| f.name == name)
    }
    pub fn preferences(&self) -> Option<&'a VisualizationPreferences> {
        self.descriptor().map(|d| &d.preferences)
    }
}
