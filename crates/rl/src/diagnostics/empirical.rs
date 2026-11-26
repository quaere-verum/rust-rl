//! Optional scientific provenance for empirical samples. Declarations establish
//! metadata compatibility; they cannot authenticate an imported sampler's claims.
//! Full physical-state futures remain descriptive, even for a Markov observation.
use super::{
    codec,
    reference::{ConfigValue, ReferenceContext, compatible_contexts, manifest_context},
    schema::*,
};
use crate::{
    Error, Result,
    core::{action::ActionSpace, ensure},
};
use serde::{Deserialize, Serialize};

pub const EMPIRICAL_METADATA_KEY: &str = "metadata/empirical.v1";
pub const EMPIRICAL_DECLARATION_VERSION: u32 = 1;
pub const MAX_EMPIRICAL_METADATA_BYTES: usize = 16 * 1024;
pub const MAX_EMPIRICAL_SAMPLES: usize = 8192;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmpiricalDeclaration {
    pub version: u32,
    pub run_id: String,
    pub revision: u64,
    pub model_role: String,
    pub checkpoint_id: Option<String>,
    pub context: ReferenceContext,
    pub conditioning: Conditioning,
}

fn text(value: &str, max: usize) -> Result<()> {
    ensure(
        !value.trim().is_empty() && value.len() <= max && !value.chars().any(char::is_control),
        "invalid empirical declaration string",
    )
}

impl EmpiricalDeclaration {
    fn validate(&self) -> Result<()> {
        ensure(
            self.version == EMPIRICAL_DECLARATION_VERSION,
            "unsupported empirical declaration version",
        )?;
        text(&self.run_id, 1024)?;
        text(&self.model_role, 1024)?;
        if let Some(id) = &self.checkpoint_id {
            text(id, 1024)?;
        }
        self.context.validate()?;
        match &self.conditioning {
            Conditioning::Observation { methodology } | Conditioning::History { methodology } => {
                text(methodology, 4096)?
            }
            Conditioning::FullSimulatorState { description } => text(description, 4096)?,
            Conditioning::InitialScenario => {}
        }
        codec::bounded_json(self, MAX_EMPIRICAL_METADATA_BYTES)?;
        Ok(())
    }
    /// Conservative owned metadata allowance, separate from raw array leases.
    pub fn retained_charge(&self) -> usize {
        let c = &self.context;
        let mut bytes = std::mem::size_of::<Self>().saturating_add(2048);
        for value in [
            &self.run_id,
            &self.model_role,
            &c.environment,
            &c.conventions.dynamics,
            &c.conventions.observation,
            &c.conventions.action,
            &c.conventions.reward_units,
            &c.conventions.reward_convention,
            &c.conventions.horizon,
            &c.conventions.terminal_treatment,
            &c.conventions.action_constraints,
        ] {
            bytes = bytes.saturating_add(value.capacity());
        }
        if let Some(id) = &self.checkpoint_id {
            bytes = bytes.saturating_add(id.capacity());
        }
        match &self.conditioning {
            Conditioning::Observation { methodology } | Conditioning::History { methodology } => {
                bytes = bytes.saturating_add(methodology.capacity())
            }
            Conditioning::FullSimulatorState { description } => {
                bytes = bytes.saturating_add(description.capacity())
            }
            Conditioning::InitialScenario => {}
        }
        if let ActionSpace::Box { low, high } = &c.action_space {
            bytes = bytes.saturating_add(
                low.capacity()
                    .saturating_add(high.capacity())
                    .saturating_mul(4),
            );
        }
        bytes = bytes.saturating_add(c.environment_config.len().saturating_mul(192));
        for (key, value) in &c.environment_config {
            bytes = bytes.saturating_add(key.capacity());
            if let ConfigValue::Text(value) = value {
                bytes = bytes.saturating_add(value.capacity());
            }
        }
        bytes
    }
}

fn bind(chunk: &Chunk, d: &EmpiricalDeclaration) -> Result<()> {
    d.validate()?;
    let identity = &chunk.context.identity;
    ensure(
        d.run_id == identity.run_id
            && d.revision == identity.learner_update_revision
            && d.model_role == identity.model_role
            && d.checkpoint_id == identity.checkpoint_id,
        "empirical declaration source identity mismatch",
    )?;
    let Record::Empirical(e) = &chunk.record else {
        return Err(Error::Config(
            "empirical declaration requires empirical samples".into(),
        ));
    };
    let returns = chunk
        .arrays
        .get(&e.returns)
        .ok_or_else(|| Error::Config("empirical returns missing".into()))?;
    ensure(
        returns.shape() == [returns.len()] && (1..=MAX_EMPIRICAL_SAMPLES).contains(&returns.len()),
        "empirical sample count exceeds admission",
    )?;
    ensure(
        e.evaluation.complete_returns && e.evaluation.independent_future_streams,
        "empirical declaration requires complete independent future returns",
    )?;
    ensure(
        e.evaluation.conditioning == d.conditioning
            && e.evaluation.discount == d.context.conventions.discount,
        "empirical declaration conditioning or discount mismatch",
    )?;
    chunk.validate()?;
    let conditional = !matches!(d.conditioning, Conditioning::InitialScenario);
    if conditional {
        ensure(
            chunk.context.probe_set.is_some() && e.probe_row.is_some(),
            "conditional empirical sample needs a bound probe row",
        )?;
        ensure(
            e.conditioning_observation.is_some(),
            "conditional empirical observation missing",
        )?;
    }
    if let Some(key) = &e.conditioning_observation {
        let observation = &chunk.arrays[key];
        ensure(
            observation.shape() == [1, d.context.observation_dim],
            "empirical observation width differs from context",
        )?;
    }
    match (&d.context.action_space, &e.conditioning_mask) {
        (ActionSpace::Discrete { count }, Some(key)) => {
            ensure(
                chunk.arrays[key].shape() == [1, *count],
                "empirical legal mask width differs from context",
            )?;
        }
        (_, Some(_)) => {
            return Err(Error::Config(
                "continuous empirical inputs cannot have legal masks".into(),
            ));
        }
        _ => {}
    }
    match (&e.initial_action, e.evaluation.initial_action) {
        (Some(key), ActionCondition::Fixed) => {
            let action = &chunk.arrays[key];
            match &d.context.action_space {
                ActionSpace::Discrete { count } => {
                    ensure(
                        action.shape() == [1] && action.dtype() == DType::U64,
                        "empirical discrete first action must be u64[1]",
                    )?;
                    let index = usize::try_from(action.as_u64()?[0])
                        .map_err(|_| Error::Shape("empirical action overflow".into()))?;
                    ensure(
                        index < *count
                            && e.conditioning_mask.as_ref().is_none_or(|key| {
                                chunk.arrays[key].as_u8().is_ok_and(|mask| mask[index] == 1)
                            }),
                        "empirical first action is illegal",
                    )?;
                }
                space => {
                    ensure(
                        action.shape() == [1, space.dimensions()]
                            && matches!(action.dtype(), DType::F32 | DType::F64),
                        "empirical continuous first action width differs",
                    )?;
                    if let ActionSpace::Box { low, high } = space {
                        ensure(
                            action
                                .to_f64()
                                .iter()
                                .zip(low)
                                .zip(high)
                                .all(|((&x, &lo), &hi)| x >= f64::from(lo) && x <= f64::from(hi)),
                            "empirical first action exceeds constraints",
                        )?;
                    }
                }
            }
        }
        (None, ActionCondition::CurrentPolicy) => {}
        _ => {
            return Err(Error::Config(
                "empirical first action declaration mismatch".into(),
            ));
        }
    }
    if matches!(d.conditioning, Conditioning::FullSimulatorState { .. }) {
        ensure(
            e.simulator_state
                .as_ref()
                .is_some_and(|k| chunk.arrays.contains_key(k)),
            "full-state empirical sample lacks physical simulator payload",
        )?;
    }
    Ok(())
}

/// Old records without this optional declaration remain readable and unknown.
pub fn declaration(chunk: &Chunk) -> Result<Option<EmpiricalDeclaration>> {
    let Some(a) = chunk.arrays.get(EMPIRICAL_METADATA_KEY) else {
        return Ok(None);
    };
    ensure(
        a.shape().len() == 1 && a.byte_len() <= MAX_EMPIRICAL_METADATA_BYTES,
        "empirical metadata exceeds admission",
    )?;
    let d: EmpiricalDeclaration = serde_json::from_slice(a.as_u8()?)?;
    bind(chunk, &d)?;
    Ok(Some(d))
}

pub fn attach_declaration(chunk: &mut Chunk, d: &EmpiricalDeclaration) -> Result<()> {
    bind(chunk, d)?;
    if let Some(existing) = declaration(chunk)? {
        return ensure(
            existing == *d,
            "immutable empirical declaration already attached",
        );
    }
    let bytes = codec::bounded_json(d, MAX_EMPIRICAL_METADATA_BYTES)?;
    chunk.arrays.insert(
        EMPIRICAL_METADATA_KEY.into(),
        Array::u8(vec![bytes.len()], bytes)?,
    );
    Ok(())
}

/// Validate the actual empirical source and authoritative run facts. This admits
/// descriptive full-state evidence; certification is a separate target check.
pub fn validate_for(chunk: &Chunk, manifest: &RunManifest) -> Result<EmpiricalDeclaration> {
    let d = declaration(chunk)?
        .ok_or_else(|| Error::Config("empirical scientific conventions unknown".into()))?;
    manifest.validate()?;
    let c = &d.context;
    ensure(
        d.run_id == manifest.run_id
            && manifest
                .starting_revision
                .checked_add(chunk.context.counters.updates)
                == Some(d.revision)
            && c.environment == manifest.environment
            && c.observation_dim == manifest.observation_dim
            && c.action_space == manifest.action_space
            && c.conventions.discount == manifest.discount
            && c.conventions.reward_convention == manifest.reward_convention,
        "empirical context differs from authoritative manifest",
    )?;
    if let Some(anchor) = manifest_context(manifest)? {
        compatible_contexts(c, &anchor)?;
    }
    Ok(d)
}

/// Admit an authored observation-conditional experiment for the exact target.
/// Neither a matching context nor a Markov flag promotes full-state evidence.
pub fn compatible_target(d: &EmpiricalDeclaration, target: &ReferenceContext) -> Result<()> {
    d.validate()?;
    ensure(
        matches!(&d.conditioning, Conditioning::Observation { methodology } if !methodology.trim().is_empty()),
        "certified empirical reference requires an authored observation-conditional sampler; full-state/history/scenario evidence is descriptive",
    )?;
    compatible_contexts(&d.context, target)
}
