//! Checked input construction and declared input equality, independent of network names.
#[cfg(feature = "neural")]
use super::probes::{self, ProbeSet};
use super::{representations::equal_row, schema::*, typed::TypedPayload};
#[cfg(feature = "neural")]
use crate::nn::{
    self,
    network::{Mlp, Network},
};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
    },
};

#[cfg(feature = "neural")]
pub enum ActivationProbeInput<'a> {
    Observations(&'a ProbeSet),
    FixedActions(&'a ProbeSet),
    PolicyActions {
        probes: &'a ProbeSet,
        actions: &'a [Action],
        identity: ProbeIdentity,
        model: String,
        stochastic: bool,
    },
}
/// Constructs the complete built-in MLP input. The generic probe API stays Unknown.
#[cfg(feature = "neural")]
pub fn mlp_activations(
    context: RecordContext,
    network: &str,
    model: &Mlp,
    input: ActivationProbeInput<'_>,
    names: &[&str],
) -> Result<Chunk> {
    let (set, actions, source) = match input {
        ActivationProbeInput::Observations(set) => (set, None, None),
        ActivationProbeInput::FixedActions(set) => {
            let actions = set.fixed_actions().ok_or_else(|| {
                crate::Error::Config("fixed activation input lacks probe actions".into())
            })?;
            (set, Some(actions), Some(ActivationActionSource::FixedProbe))
        }
        ActivationProbeInput::PolicyActions {
            probes,
            actions,
            identity,
            model,
            stochastic,
        } => (
            probes,
            Some(actions),
            Some(ActivationActionSource::Policy {
                identity,
                model,
                stochastic,
            }),
        ),
    };
    ensure(
        context.probe_set.as_deref() == Some(set.id()),
        "activation context and probe identity differ",
    )?;
    let width = set.observations()[0].dense_values().len();
    let observations = nn::observations(set.observations().into(), width)?;
    let mut action_array = None;
    let tensor =
        if let Some(actions) = actions {
            ensure(
                actions.len() == set.observations().len(),
                "activation action batch size differs",
            )?;
            let mut values = Vec::new();
            for (action, observation) in actions.iter().zip(set.observations()) {
                set.action_space().check(action, observation.dense_mask())?;
                match action {
                    Action::Continuous(v) => values.extend_from_slice(v),
                    Action::Discrete(_) | Action::Conditional(_) => return Err(crate::Error::Config(
                        "built-in concatenated MLP activation inputs require continuous actions"
                            .into(),
                    )),
                }
            }
            let array = Array::f32(
                vec![actions.len(), set.action_space().dimensions()],
                values.clone(),
            )?;
            action_array = Some(array);
            burn::tensor::Tensor::cat(
                vec![
                    observations,
                    nn::matrix(values, actions.len(), set.action_space().dimensions()),
                ],
                1,
            )
        } else {
            observations
        };
    ensure(
        tensor.dims()[1] == model.input_dim(),
        "activation complete input width differs from MLP",
    )?;
    let mut chunk = probes::activations(context.clone(), network, model, tensor, names)?;
    let Record::Activations(meta) = &mut chunk.record else {
        unreachable!()
    };
    meta.inputs = if let Some(array) = action_array {
        let key = "activation_input/actions".to_string();
        ensure(
            !chunk.arrays.contains_key(&key),
            "reserved activation action key collides",
        )?;
        chunk.arrays.insert(key.clone(), array);
        ActivationInputs::ObservationsAndActions {
            actions: key,
            source: source.unwrap(),
        }
    } else {
        ActivationInputs::Observations
    };
    validate_activation_inputs(&chunk, &set.record(context)?, width, set.action_space())?;
    Ok(chunk)
}

/// Cross-record gate against the actual immutable probe definition and manifest.
pub fn validate_activation_inputs(
    chunk: &Chunk,
    probes: &Chunk,
    observation_dim: usize,
    space: &ActionSpace,
) -> Result<()> {
    chunk.validate()?;
    probes.validate()?;
    space.validate()?;
    if let Record::Activations(meta) = &chunk.record
        && matches!(probes.record, Record::Typed(_))
    {
        let TypedPayload::Probes(set) = TypedPayload::read(probes)? else {
            return Err(crate::Error::Config(
                "expected typed activation probes".into(),
            ));
        };
        ensure(
            observation_dim == 0
                && chunk.context.identity.run_id == probes.context.identity.run_id
                && chunk.context.probe_set.is_some()
                && chunk.context.probe_set == probes.context.probe_set
                && matches!(
                    meta.inputs,
                    ActivationInputs::Observations | ActivationInputs::Unknown
                ),
            "typed activation input binding differs from its probe",
        )?;
        return ensure(
            meta.layers
                .values()
                .all(|key| chunk.arrays[key].shape()[0] == set.inputs.len()),
            "typed activation/probe row counts differ",
        );
    }
    let (Record::Activations(meta), Record::ProbeSet(set)) = (&chunk.record, &probes.record) else {
        return Err(crate::Error::Config(
            "activation input validation requires activation and probe records".into(),
        ));
    };
    ensure(
        chunk.context.identity.run_id == probes.context.identity.run_id
            && chunk.context.probe_set.as_deref() == Some(&set.id),
        "activation input probe binding differs",
    )?;
    let obs = &probes.arrays[&set.observations];
    ensure(
        obs.shape().len() == 2 && obs.shape()[1] == observation_dim,
        "activation observation width differs from manifest",
    )?;
    let n = obs.shape()[0];
    ensure(
        meta.layers
            .values()
            .all(|key| chunk.arrays[key].shape()[0] == n),
        "activation/probe row counts differ",
    )?;
    if let ActivationInputs::ObservationsAndActions { actions, source } = &meta.inputs {
        let values = &chunk.arrays[actions];
        ensure(
            values.shape()[0] == n,
            "activation input actions/probe row counts differ",
        )?;
        for row in 0..n {
            let mask = set
                .action_masks
                .as_ref()
                .map(|key| {
                    let a = &probes.arrays[key];
                    a.as_u8().map(|v| {
                        v[row * a.shape()[1]..(row + 1) * a.shape()[1]]
                            .iter()
                            .map(|&v| v != 0)
                            .collect::<Vec<_>>()
                    })
                })
                .transpose()?;
            if let Values::F64(v) = &values.values {
                let row_values = &v[row * values.shape()[1]..(row + 1) * values.shape()[1]];
                let valid = match space {
                    ActionSpace::Box { low, high } => {
                        row_values.len() == low.len()
                            && row_values
                                .iter()
                                .zip(low.iter().zip(high))
                                .all(|(&v, (&l, &h))| {
                                    v.is_finite() && v >= f64::from(l) && v <= f64::from(h)
                                })
                    }
                    ActionSpace::Unbounded { dimensions } => {
                        row_values.len() == *dimensions && row_values.iter().all(|v| v.is_finite())
                    }
                    _ => false,
                };
                ensure(valid, "activation input actions violate manifest space")?;
            } else {
                let action = match &values.values {
                    Values::U64(v) => Action::Discrete(
                        usize::try_from(v[row])
                            .map_err(|_| crate::Error::Config("action index overflow".into()))?,
                    ),
                    Values::F32(v) => Action::Continuous(
                        v[row * values.shape()[1]..(row + 1) * values.shape()[1]].to_vec(),
                    ),
                    _ => {
                        return Err(crate::Error::Config(
                            "activation action inputs require exact f32 or u64 actions".into(),
                        ));
                    }
                };
                space.check(&action, mask.as_deref())?;
            }
            if matches!(source, ActivationActionSource::FixedProbe) {
                let key = set.fixed_actions.as_ref().ok_or_else(|| {
                    crate::Error::Config("fixed activation input lacks fixed probe actions".into())
                })?;
                ensure(
                    equal_row(values, &probes.arrays[key], row, row),
                    "fixed activation actions differ from bound probe",
                )?;
            }
        }
    }
    Ok(())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputEquality {
    Equal,
    ChangedPolicy,
    Changed,
    Unknown,
}
pub fn input_source(inputs: &ActivationInputs) -> (&'static str, Option<u64>) {
    match inputs {
        ActivationInputs::Unknown => ("unknown", None),
        ActivationInputs::Observations => ("observations", None),
        ActivationInputs::ObservationsAndActions {
            source: ActivationActionSource::FixedProbe,
            ..
        } => ("observations + fixed-probe actions", None),
        ActivationInputs::ObservationsAndActions {
            source: ActivationActionSource::Policy { identity, .. },
            ..
        } => (
            "observations + policy actions",
            Some(identity.learner_update_revision),
        ),
    }
}
impl InputEquality {
    pub fn label(self) -> &'static str {
        match self {
            Self::Equal => "fixed-input activation change",
            Self::ChangedPolicy => "policy-conditioned computation change",
            Self::Changed => "changed-input computation change",
            Self::Unknown => "input lineage unknown",
        }
    }
}
/// Call after fixed probe alignment. Equality checks the recorded contract;
/// it does not authenticate imported provenance or certify weight-only change.
pub fn compare_activation_inputs(
    current: &Chunk,
    reference: &Chunk,
    row_map: &[usize],
) -> Result<InputEquality> {
    current.validate()?;
    reference.validate()?;
    let (Record::Activations(a), Record::Activations(b)) = (&current.record, &reference.record)
    else {
        return Err(crate::Error::Config("expected activation pair".into()));
    };
    Ok(match (&a.inputs, &b.inputs) {
        (ActivationInputs::Observations, ActivationInputs::Observations) => InputEquality::Equal,
        (
            ActivationInputs::ObservationsAndActions {
                actions: a,
                source: sa,
            },
            ActivationInputs::ObservationsAndActions {
                actions: b,
                source: sb,
            },
        ) => {
            if row_map
                .iter()
                .enumerate()
                .all(|(r, &s)| equal_row(&current.arrays[a], &reference.arrays[b], r, s))
            {
                InputEquality::Equal
            } else if matches!(sa, ActivationActionSource::Policy { .. })
                || matches!(sb, ActivationActionSource::Policy { .. })
            {
                InputEquality::ChangedPolicy
            } else {
                InputEquality::Changed
            }
        }
        _ => InputEquality::Unknown,
    })
}
