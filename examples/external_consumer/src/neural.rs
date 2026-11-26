use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    algorithms::candidate_ddqn::{CandidateDdqn, CandidateDdqnConfig},
    core::{action::ActionSpace, env::ObservationSpace, stateful::StatefulLearner},
    data::temporal::SequenceBatch,
    nn::{
        device,
        structured::{
            GraphStructuredQ, GraphStructuredQConfig, InputSchema, SemanticInput,
            StructuredModelConfig,
        },
    },
};

use crate::hall::{GAMMA, spaces};

pub fn update(batch: &SequenceBatch) -> Result<u64> {
    let spaces = spaces();
    let (ObservationSpace::Graph(schema), ActionSpace::Conditional(grammar)) =
        (&spaces.observations, &spaces.actions)
    else {
        return Err(Error::Config(
            "expected graph observations and conditional actions".into(),
        ));
    };
    let base = &schema.base;
    let input = InputSchema {
        version: 2,
        spatial_channels: base.spatial.len(),
        entity_features: base.entities.len(),
        event_features: base.events.len(),
        scalar_features: base.scalars.len(),
        candidate_features: grammar.candidate_features.len(),
        operations: grammar.operations.iter().map(|op| op.argument).collect(),
        semantics: SemanticInput {
            version: 1,
            namespace: base.namespace.clone(),
            observation_version: base.version,
            action_version: grammar.version,
            spatial: base.spatial.clone(),
            entities: base.entities.clone(),
            events: base.events.clone(),
            scalars: base.scalars.clone(),
            candidates: grammar.candidate_features.clone(),
            operations: grammar
                .operations
                .iter()
                .map(|op| op.name.clone())
                .collect(),
            vocabularies: base.vocabularies.clone(),
            conventions: base.conventions.clone(),
        },
    };
    let model = GraphStructuredQ::new(
        GraphStructuredQConfig {
            version: 1,
            base: StructuredModelConfig::small(input),
            graph: schema.graph.clone(),
            graph_width: 8,
        },
        &mut ChaCha8Rng::seed_from_u64(5),
        &device(),
    )?;
    let config = CandidateDdqnConfig {
        gamma: GAMMA as f32,
        ..CandidateDdqnConfig::default()
    };
    let mut learner = CandidateDdqn::new(model, config, spaces.observations, spaces.actions)?;
    let report = StatefulLearner::update(&mut learner, batch)?;
    Ok(report.updates)
}
