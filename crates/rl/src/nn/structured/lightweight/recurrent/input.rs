//! Borrow factual decision inputs and project only acquisition-time event rows.
use super::{
    super::pack::{LightweightRows, Regions},
    RecurrentLightweightConfig,
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionArgument, ConditionalAction},
        ensure,
        env::Observation,
        stateful::{HistoryInput, PreviousInput},
        structured::{PublicEvent, StructuredObservation},
    },
    nn::structured::{Argument, ArgumentKind, Candidate, Event, InputSchema},
};
use std::collections::{HashMap, HashSet};

pub(super) struct Frame<'a> {
    pub observation: &'a StructuredObservation,
    pub events: &'a [PublicEvent],
    pub previous: Option<PreviousAction<'a>>,
}

pub(super) struct PreviousAction<'a> {
    pub operation: usize,
    pub kind: ArgumentKind,
    pub target: Option<&'a [f32]>,
    pub location: Option<[f32; 2]>,
    pub candidate_features: &'a [f32],
    pub reward: f32,
    pub duration: u64,
}

pub(super) struct ProjectedFrame<'a> {
    size: [usize; 2],
    observation: &'a StructuredObservation,
    entities: Vec<Vec<f32>>,
    scalars: Vec<f32>,
    events: Vec<Event>,
    candidates: Vec<Candidate<ConditionalAction>>,
}

pub(super) fn structured(observation: &Observation) -> Result<&StructuredObservation> {
    match observation {
        Observation::Structured(observation) => Ok(observation),
        Observation::Dense(_) | Observation::Discrete(_) | Observation::Graph(_) => Err(
            Error::Shape("recurrent lightweight model requires structured observations".into()),
        ),
    }
}

fn map_size(observation: &StructuredObservation, schema: &InputSchema) -> Result<[usize; 2]> {
    observation.validate_spatial(schema.spatial_channels)?;
    Ok(observation
        .spatial
        .as_ref()
        .map_or([0, 0], |map| [map.width, map.height]))
}

fn candidate_location(
    observation: &StructuredObservation,
    region: u32,
    x: usize,
    y: usize,
) -> Result<[usize; 2]> {
    observation
        .region_map(region)
        .filter(|map| x < map.width && y < map.height)
        .map(|map| [map.width, map.height])
        .ok_or_else(|| Error::Shape("recurrent candidate location outside real map".into()))
}

fn feature_row(values: &[f32], width: usize) -> Result<()> {
    ensure(
        values.len() == width && values.iter().all(|value| value.is_finite()),
        "invalid recurrent feature row",
    )
}

fn grammar(schema: &InputSchema, action: ConditionalAction) -> Result<ArgumentKind> {
    let kind = match action.argument {
        None => ArgumentKind::None,
        Some(ActionArgument::Entity(_)) => ArgumentKind::Entity,
        Some(ActionArgument::Location { .. } | ActionArgument::RegionalLocation { .. }) => {
            ArgumentKind::Location
        }
    };
    ensure(
        schema.operations.get(action.operation) == Some(&kind),
        "recurrent candidate grammar mismatch",
    )?;
    Ok(kind)
}

impl<'a> PreviousAction<'a> {
    fn capture(
        input: PreviousInput<'a>,
        schema: &InputSchema,
    ) -> Result<(&'a StructuredObservation, Self)> {
        let factual = input.factual.ok_or_else(|| Error::State("structured recurrent input lacks the selection observation or primitive reward trace".into()))?;
        let selection = structured(factual.observation)?;
        let Action::Conditional(action) = input.action else {
            return Err(Error::Action(
                "structured recurrent input requires a conditional previous action".into(),
            ));
        };
        let kind = grammar(schema, **action)?;
        let mut available = selection.candidates.iter().filter(|c| c.action == **action);
        let candidate = available
            .next()
            .filter(|candidate| candidate.enabled)
            .ok_or_else(|| Error::Action("previous action was unavailable at selection".into()))?;
        ensure(
            available.next().is_none(),
            "duplicate previous action at selection",
        )?;
        feature_row(&candidate.features, schema.candidate_features)?;
        factual.reward.validate()?;
        ensure(
            input.reward.to_bits() == factual.reward.total().to_bits(),
            "recurrent previous reward differs from the primitive trace",
        )?;
        let mut target = None;
        let mut location = None;
        match action.argument {
            None => {}
            Some(ActionArgument::Entity(id)) => {
                let mut rows = selection.entities.iter().filter(|row| row.id == id);
                let row = rows.next().ok_or_else(|| {
                    Error::Shape("previous action target was absent at selection".into())
                })?;
                ensure(
                    rows.next().is_none(),
                    "duplicate previous action target at selection",
                )?;
                feature_row(&row.features, schema.entity_features)?;
                target = Some(row.features.as_slice());
            }
            Some(
                argument @ (ActionArgument::Location { .. }
                | ActionArgument::RegionalLocation { .. }),
            ) => {
                let (region, x, y) = match argument {
                    ActionArgument::Location { x, y } => (selection.primary_region, x, y),
                    ActionArgument::RegionalLocation { region, x, y } => (region, x, y),
                    ActionArgument::Entity(_) => unreachable!(),
                };
                let [width, height] = candidate_location(selection, region, x, y)?;
                ensure(
                    selection
                        .region_map(region)
                        .is_some_and(|map| map.valid(schema.spatial_channels)),
                    "invalid map for previous action",
                )?;
                location = Some([
                    x as f32 / width.saturating_sub(1).max(1) as f32,
                    y as f32 / height.saturating_sub(1).max(1) as f32,
                ]);
            }
        }
        Ok((
            selection,
            Self {
                operation: action.operation,
                kind,
                target,
                location,
                candidate_features: &candidate.features,
                reward: input.reward,
                duration: factual.reward.duration(),
            },
        ))
    }
}

impl<'a> Frame<'a> {
    pub fn from_input(input: HistoryInput<'a>, schema: &InputSchema) -> Result<Self> {
        let observation = structured(input.observation)?;
        let (selection, previous) = match input.previous {
            Some(previous) => {
                let (selection, features) = PreviousAction::capture(previous, schema)?;
                (Some(selection), Some(features))
            }
            None => (None, None),
        };
        let events = observation.introduced_events(selection)?;
        Ok(Self {
            observation,
            events,
            previous,
        })
    }

    pub fn event_start(&self) -> usize {
        self.observation.events.len() - self.events.len()
    }

    pub fn project(
        &self,
        config: &RecurrentLightweightConfig,
        include_candidates: bool,
    ) -> Result<ProjectedFrame<'a>> {
        let observation = self.observation;
        let schema = &config.model.input;
        let size = map_size(observation, schema)?;
        feature_row(&observation.scalars, schema.scalar_features)?;
        let mut scalars = observation.scalars.clone();
        for &column in &config.suppressed_scalar_features {
            scalars[column] = 0.;
        }
        let mut rows = HashMap::with_capacity(observation.entities.len());
        let entities = observation
            .entities
            .iter()
            .enumerate()
            .map(|(i, row)| {
                feature_row(&row.features, schema.entity_features)?;
                ensure(
                    rows.insert(row.id, i).is_none(),
                    "duplicate recurrent entity identity",
                )?;
                let mut features = row.features.clone();
                for &column in &config.suppressed_entity_features {
                    features[column] = 0.;
                }
                Ok(features)
            })
            .collect::<Result<_>>()?;
        let resolve = |id| {
            rows.get(&id)
                .copied()
                .ok_or_else(|| Error::Shape("recurrent reference to absent entity".into()))
        };
        let events = self
            .events
            .iter()
            .map(|event| {
                feature_row(&event.features, schema.event_features)?;
                let context = event
                    .context
                    .iter()
                    .copied()
                    .map(resolve)
                    .collect::<Result<Vec<_>>>()?;
                let mut seen = HashSet::with_capacity(context.len());
                ensure(
                    context.iter().all(|row| seen.insert(*row)),
                    "invalid recurrent event context set",
                )?;
                Ok(Event {
                    features: event.features.to_vec(),
                    entity: event.entity.map(resolve).transpose()?,
                    context,
                })
            })
            .collect::<Result<_>>()?;
        let mut keys = HashSet::new();
        let mut candidates = Vec::new();
        if include_candidates {
            candidates.reserve_exact(observation.candidates.len());
            for candidate in &observation.candidates {
                let key = candidate.action;
                grammar(schema, key)?;
                feature_row(&candidate.features, schema.candidate_features)?;
                ensure(keys.insert(key), "duplicate recurrent candidate")?;
                let argument = match key.argument {
                    None => Argument::None,
                    Some(ActionArgument::Entity(id)) => Argument::Entity(resolve(id)?),
                    Some(ActionArgument::Location { x, y }) => {
                        candidate_location(observation, observation.primary_region, x, y)?;
                        Argument::Location { x, y }
                    }
                    Some(ActionArgument::RegionalLocation { region, x, y }) => {
                        candidate_location(observation, region, x, y)?;
                        Argument::RegionalLocation { region, x, y }
                    }
                };
                candidates.push(Candidate {
                    key,
                    operation: key.operation,
                    argument,
                    features: candidate.features.clone(),
                    enabled: candidate.enabled,
                });
            }
        }
        Ok(ProjectedFrame {
            size,
            observation,
            entities,
            scalars,
            events,
            candidates,
        })
    }
}

impl ProjectedFrame<'_> {
    pub fn rows(&self) -> LightweightRows<'_, ConditionalAction> {
        LightweightRows {
            width: self.size[0],
            height: self.size[1],
            primary_region: self.observation.primary_region,
            regions: Regions::Public(&self.observation.regions),
            entities: &self.entities,
            scalars: &self.scalars,
            events: &self.events,
            candidates: &self.candidates,
        }
    }
}
