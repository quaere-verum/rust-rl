//! Inference for the normalized, action-conditioned HL-Gauss critic.
use crate::{
    Error, Result,
    algorithms::reppo::action_tensor_on_device,
    core::{action::Action, ensure},
    diagnostics::{
        probes::{self, ProbeSet},
        schema::*,
        value::ContinuationIdentity,
    },
    nn::{
        critics::{
            categorical::{CategoricalQ, ReppoCritic},
            scalar::ActionValue,
        },
        to_vec,
    },
    objectives::hl_gauss::HlGaussConfig,
};
use burn::tensor::{Tensor, activation::softmax};
use std::collections::BTreeMap;

use super::super::categorical::{FEATURES, NEXT_FEATURES};

fn model_input(
    context: &RecordContext,
    model: &CategoricalQ,
    set: &ProbeSet,
    actions: &[Action],
) -> Result<Tensor<2>> {
    ensure(
        actions.len() == set.observations().len() && context.probe_set.as_deref() == Some(set.id()),
        "categorical critic actions or probe identity differ",
    )?;
    for (action, observation) in actions.iter().zip(set.observations()) {
        set.action_space().check(action, observation.dense_mask())?;
    }
    action_tensor_on_device(actions, set.action_space(), &model.diagnostic_device())
}

fn matrix_array(tensor: Tensor<2>) -> Result<Array> {
    let shape = tensor.dims().to_vec();
    Array::f32(shape, to_vec(tensor)?)
}

pub(crate) fn categorical_prediction_bytes(rows: usize, bins: usize) -> Result<usize> {
    bins.checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| n.checked_mul(rows))
        .and_then(|n| n.checked_add(bins))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(|| Error::Shape("categorical output exceeds size admission".into()))
}

pub(crate) fn categorical_prediction(
    context: RecordContext,
    model: &CategoricalQ,
    support: &HlGaussConfig,
    set: &ProbeSet,
    actions: &[Action],
    condition: ActionCondition,
    continuation: Option<&ContinuationIdentity>,
) -> Result<Chunk> {
    ensure(
        matches!(
            condition,
            ActionCondition::Fixed | ActionCondition::CurrentPolicy
        ) && (!matches!(condition, ActionCondition::Fixed) || set.fixed_actions() == Some(actions)),
        "categorical critic requires complete bound actions",
    )?;
    let input = model_input(&context, model, set, actions)?;
    let logits = model.values(set.observations().into(), input)?;
    let values = to_vec(support.decode(logits.clone())?)?;
    let weights = softmax(logits.clone(), 1);
    let centers = support.centers()?;
    let chunk = Chunk {
        context,
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::CategoricalRegression {
                logits: "logits".into(),
                weights: "weights".into(),
                centers: "centers".into(),
                values: "values".into(),
            },
            action_condition: condition,
            actions: Some("actions".into()),
            continuation_policy: continuation.map(|c| c.identity.clone()),
            continuation_stochastic: continuation.map(|c| c.stochastic),
        }),
        arrays: BTreeMap::from([
            ("actions".into(), probes::actions_array(actions)?),
            ("logits".into(), matrix_array(logits)?),
            ("weights".into(), matrix_array(weights)?),
            ("centers".into(), Array::f32(vec![centers.len()], centers)?),
            ("values".into(), Array::f32(vec![values.len()], values)?),
        ]),
    };
    chunk.validate()?;
    Ok(chunk)
}

#[expect(
    clippy::too_many_arguments,
    reason = "activation identity, lineage and budget are explicit"
)]
pub(crate) fn categorical_activations(
    context: RecordContext,
    name: &str,
    model: &CategoricalQ,
    set: &ProbeSet,
    actions: &[Action],
    source: ActivationActionSource,
    auxiliary: bool,
    budget: &mut usize,
) -> Result<Option<Chunk>> {
    let (_, widths) = model.diagnostic_layout();
    let elements = set
        .observations()
        .len()
        .checked_mul(
            *widths
                .last()
                .expect("standard encoder retains a hidden block"),
        )
        .ok_or_else(|| Error::Shape("categorical activation size overflow".into()))?;
    if elements > *budget {
        return Ok(None);
    }
    let mut layers = vec![FEATURES];
    *budget -= elements;
    if auxiliary && elements <= *budget {
        *budget -= elements;
        layers.push(NEXT_FEATURES);
    }
    activation_chunk(context, name, model, set, actions, source, &layers).map(Some)
}

pub(super) fn activation_chunk(
    context: RecordContext,
    name: &str,
    model: &CategoricalQ,
    set: &ProbeSet,
    actions: &[Action],
    source: ActivationActionSource,
    layers: &[&str],
) -> Result<Chunk> {
    ensure(
        !matches!(source, ActivationActionSource::FixedProbe)
            || set.fixed_actions() == Some(actions),
        "fixed categorical activation actions differ from the probes",
    )?;
    let input = model_input(&context, model, set, actions)?;
    let features = model.representation(set.observations().into(), input)?;
    let predictor = layers
        .contains(&NEXT_FEATURES)
        .then(|| model.diagnostic_predictor(features.clone()));
    let mut meta = ActivationRecord {
        network: name.into(),
        inputs: ActivationInputs::ObservationsAndActions {
            actions: "activation_input/actions".into(),
            source,
        },
        ..Default::default()
    };
    let mut arrays = BTreeMap::from([(
        "activation_input/actions".into(),
        probes::actions_array(actions)?,
    )]);
    for (layer, tensor) in [(FEATURES, Some(features)), (NEXT_FEATURES, predictor)] {
        if layers.contains(&layer) {
            let tensor = tensor.expect("requested representation calculated");
            let array = matrix_array(tensor)?;
            meta.summaries
                .insert(layer.into(), probes::activation_summary(array.as_f32()?));
            meta.layers.insert(layer.into(), layer.into());
            arrays.insert(layer.into(), array);
        }
    }
    let chunk = Chunk {
        context,
        record: Record::Activations(meta),
        arrays,
    };
    chunk.validate()?;
    Ok(chunk)
}
