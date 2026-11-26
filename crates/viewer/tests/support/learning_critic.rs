use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*},
};
use std::collections::BTreeMap;

pub fn probes(space: &ActionSpace, reverse: bool) -> Chunk {
    let observations = if reverse {
        vec![Observation::new(vec![1.]), Observation::new(vec![0.])]
    } else {
        vec![Observation::new(vec![0.]), Observation::new(vec![1.])]
    };
    let mut context = RecordContext::new("learning", 0);
    context.probe_set = Some("fixed".into());
    let mut chunk = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "unchanged complete population",
        observations,
        None,
        space.clone(),
    )
    .unwrap()
    .record(context)
    .unwrap();
    if reverse {
        let Record::ProbeSet(p) = &chunk.record else {
            unreachable!()
        };
        chunk
            .arrays
            .insert(p.row_ids.clone(), Array::u64(vec![2], vec![1, 0]).unwrap());
    }
    chunk
}
pub fn empirical(revision: u64, samples: Vec<f64>) -> Chunk {
    let n = samples.len();
    Chunk {
        context: RecordContext::new("learning", revision),
        record: Record::Empirical(EmpiricalRecord {
            returns: "returns".into(),
            seeds: "seeds".into(),
            probe_row: None,
            initial_action: None,
            conditioning_observation: None,
            conditioning_mask: None,
            simulator_state: None,
            evaluation: EvaluationMetadata {
                conditioning: Conditioning::InitialScenario,
                initial_action: ActionCondition::CurrentPolicy,
                continuation_policy: "evaluated learner".into(),
                stochastic_policy: true,
                discount: 0.9,
                independent_future_streams: true,
                seed_methodology: "distinct explicit futures".into(),
                truncation_handling: "true terminal, no bootstrap".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            ("returns".into(), Array::f64(vec![n], samples).unwrap()),
            (
                "seeds".into(),
                Array::u64(vec![n], (0..n as u64).collect()).unwrap(),
            ),
        ]),
    }
}

pub fn critic_fixture() -> (Chunk, Chunk, Chunk, RunManifest) {
    use rl::diagnostics::{
        empirical::{self, EmpiricalDeclaration},
        reference::{self, ConfigValue, ReferenceContext, ReferenceHorizon, ReferenceObjective},
        value::{self, CriticDeclaration, ValueConventions, ValueCriticKind, ValueDeclaration},
    };
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let mut manifest =
        RunManifest::new("learning", "learning", "fixture", 1, space.clone(), 0.9).unwrap();
    let context = ReferenceContext {
        environment: manifest.environment.clone(),
        observation_dim: 1,
        typed_observation_space: None,
        action_space: space.clone(),
        conventions: ValueConventions {
            dynamics: "exact observed state".into(),
            observation: "exact f32 state".into(),
            action: "one control".into(),
            reward_units: "reward".into(),
            reward_convention: manifest.reward_convention.clone(),
            discount: 0.9,
            horizon: "two decisions".into(),
            terminal_treatment: "true terminal; no bootstrap".into(),
            action_constraints: "bounded control".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 2 },
        environment_config: BTreeMap::from([("horizon".into(), ConfigValue::Integer(2))]),
        markov_observation: true,
        decision_time_coordinate: None,
    };
    reference::attach_manifest_context(&mut manifest, &context).unwrap();
    let p = probes(&space, false);
    let mut sample = empirical(3, vec![1., 3., 5.]);
    sample.context.probe_set = Some("fixed".into());
    let Record::Empirical(e) = &mut sample.record else {
        unreachable!()
    };
    e.probe_row = Some(0);
    e.conditioning_observation = Some("observation".into());
    e.evaluation.conditioning = Conditioning::Observation {
        methodology: "authored exact observation conditional futures with fresh RNG".into(),
    };
    sample.arrays.insert(
        "observation".into(),
        Array::f32(vec![1, 1], vec![0.]).unwrap(),
    );
    let conditioning = e.evaluation.conditioning.clone();
    empirical::attach_declaration(
        &mut sample,
        &EmpiricalDeclaration {
            version: 1,
            run_id: "learning".into(),
            revision: 3,
            model_role: "online".into(),
            checkpoint_id: None,
            context: context.clone(),
            conditioning,
        },
    )
    .unwrap();
    let mut c = RecordContext::new("learning", 3);
    c.probe_set = Some("fixed".into());
    let mut critic = Chunk {
        context: c,
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Scalar {
                values: "values".into(),
            },
            action_condition: ActionCondition::StateValue,
            actions: None,
            continuation_policy: Some("evaluated learner".into()),
            continuation_stochastic: Some(true),
        }),
        arrays: BTreeMap::from([("values".into(), Array::f64(vec![2], vec![2., 4.]).unwrap())]),
    };
    value::attach_declaration(
        &mut critic,
        &ValueDeclaration {
            version: 1,
            run_id: "learning".into(),
            revision: 3,
            model_role: "online".into(),
            conventions: Some(context.conventions),
            critics: BTreeMap::from([(
                "critic".into(),
                CriticDeclaration {
                    kind: ValueCriticKind::StateValue,
                    learned_baseline: Some(true),
                    actor: None,
                    continuation: Some(value::ContinuationIdentity {
                        identity: "evaluated learner".into(),
                        stochastic: true,
                    }),
                    twin: None,
                    utility: None,
                },
            )]),
        },
    )
    .unwrap();
    (critic, p, sample, manifest)
}
