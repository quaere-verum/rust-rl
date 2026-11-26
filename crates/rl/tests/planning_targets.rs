use rl::models::predictive::{PlanningReturns, ReturnObjective};
use rl::planning::{ActionEstimate, PlanStatus, PlanningBudget, PlanningWork};
use rl::{
    Result,
    core::{
        action::{Action, ActionSpace},
        env::{EpisodeStart, Observation},
        reward::Reward,
    },
    data::temporal::{BehaviorMetadata, EpisodeStore, TemporalSink, TemporalTransition},
    planning::targets::{
        FactualBinding, PolicyTarget, SearchProvenance, SearchRevision, SearchTarget,
        factual_return,
    },
};
use std::num::{NonZeroU16, NonZeroU32};

#[test]
fn visit_temperature_and_value_improvement_preserve_complete_support() -> Result<()> {
    let support = [10, 20, 30];
    let estimates = [
        ActionEstimate {
            action: 20,
            value: 3f64.ln(),
            visits: Some(3),
            upper_bound: None,
        },
        ActionEstimate {
            action: 10,
            value: 0.,
            visits: Some(1),
            upper_bound: None,
        },
    ];
    assert_eq!(
        PolicyTarget::from_visits(&support, &estimates, 1.)?,
        PolicyTarget::Probabilities(vec![0.25, 0.75, 0.])
    );
    assert_eq!(
        PolicyTarget::from_visits(&support, &estimates, 0.)?,
        PolicyTarget::Probabilities(vec![0., 1., 0.])
    );
    let PolicyTarget::Probabilities(q) = PolicyTarget::from_visits(&support, &estimates, 2.)?
    else {
        panic!()
    };
    assert!((q[0] - 1. / (1. + 3f64.sqrt())).abs() < 1e-12);
    for tau in [-1., f64::NAN, f64::INFINITY] {
        assert!(PolicyTarget::from_visits(&support, &estimates, tau).is_err());
    }
    assert!(PolicyTarget::from_visits(&support, &[], 1.).is_err());
    assert!(
        PolicyTarget::from_action_values(
            &support,
            &estimates,
            &[0.5, 0.5, 0.],
            0.1,
            NonZeroU32::MIN
        )
        .is_err()
    );
    let epsilon = 0.25 * 0.5f64.ln() + 0.75 * 1.5f64.ln();
    let PolicyTarget::Probabilities(q) = PolicyTarget::from_action_values(
        &support[..2],
        &estimates,
        &[0.5, 0.5],
        epsilon,
        NonZeroU32::MIN,
    )?
    else {
        panic!()
    };
    assert!((q[0] - 0.25).abs() < 1e-10);
    assert!((q[1] - 0.75).abs() < 1e-10);
    assert!(
        PolicyTarget::from_action_values(
            &support[..2],
            &estimates,
            &[0.5, 0.5],
            epsilon,
            NonZeroU32::new(2).unwrap()
        )
        .is_err()
    );
    for visits in [None, Some(0)] {
        let bad = [ActionEstimate {
            action: 10,
            value: 0.,
            visits,
            upper_bound: None,
        }];
        assert!(PolicyTarget::from_visits(&support, &bad, 1.).is_err());
    }
    assert!(
        PolicyTarget::from_visits(&support, &[estimates[0].clone(), estimates[0].clone()], 1.)
            .is_err()
    );
    assert!(PolicyTarget::from_visits(&[10], &estimates, 1.).is_err());
    Ok(())
}
fn observation() -> Observation {
    Observation::dense(vec![0.], Some(vec![true, true]))
}
fn store(terminated: bool) -> Result<EpisodeStore> {
    let mut store = EpisodeStore::new(8, 1, ActionSpace::Discrete { count: 2 })?;
    store.begin_episode(0, &EpisodeStart::fresh(observation()))?;
    for timestep in 0..2 {
        store.push(&TemporalTransition {
            episode_id: 0,
            timestep,
            observation: observation(),
            action: Action::Discrete(timestep),
            reward: Reward::from_primitives(vec![1., 2.])?,
            next_observation: observation(),
            terminated: terminated && timestep == 1,
            truncated: !terminated && timestep == 1,
            behavior: BehaviorMetadata {
                revision: 0,
                log_prob: None,
            },
        })?;
    }
    Ok(store)
}
fn provenance() -> SearchProvenance {
    SearchProvenance {
        model: SearchRevision {
            identity: [1; 32],
            revision: 2,
        },
        planner: SearchRevision {
            identity: [2; 32],
            revision: 3,
        },
        policy: Some(SearchRevision {
            identity: [3; 32],
            revision: 4,
        }),
    }
}
fn target(store: &EpisodeStore) -> Result<SearchTarget<u64, u8, bool>> {
    Ok(SearchTarget {
        version: 1,
        public: 7,
        binding: FactualBinding::capture([9; 32], store, 0, 0)?,
        feasible: vec![0, 1],
        policy: Some(PolicyTarget::Probabilities(vec![0.75, 0.25])),
        value: Some(0.5),
        returns: PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 0.5,
        },
        horizon: NonZeroU16::new(2).unwrap(),
        factual_return: None,
        provenance: provenance(),
        budget: PlanningBudget::Expansions(NonZeroU32::new(2).unwrap()),
        work: PlanningWork {
            expansions: 2,
            ..Default::default()
        },
        status: PlanStatus::Exact,
        continuations: Vec::new(),
    })
}
#[test]
fn target_joins_exact_causal_history_and_roundtrips_without_copying_replay() -> Result<()> {
    let store = store(true)?;
    let target = target(&store)?;
    let encoded = serde_json::to_vec(&target)?;
    let restored: SearchTarget<u64, u8, bool> = serde_json::from_slice(&encoded)?;
    let row = restored.join([9; 32], &store, &provenance(), |public| {
        assert_eq!(public.steps().len(), 0);
        assert_eq!(public.observations().len(), 1);
        assert_eq!(public.observation(), &observation());
        Ok((7, vec![0, 1]))
    })?;
    assert!(std::ptr::eq(row, &store.episode(0).unwrap().steps[0]));
    assert_eq!(row.action, Action::Discrete(0));
    // Semantic action identities need not be the primitive action type in replay.
    assert_eq!(restored.feasible, vec![0_u8, 1]);
    Ok(())
}
#[test]
fn wrong_tape_forged_history_support_and_stale_revisions_cannot_join() -> Result<()> {
    let store = store(true)?;
    let target = target(&store)?;
    assert!(
        target
            .join([8; 32], &store, &provenance(), |_| Ok((7, vec![0, 1])))
            .is_err()
    );
    let mut forged = target.clone();
    forged.binding.history[0] ^= 1;
    assert!(
        forged
            .join([9; 32], &store, &provenance(), |_| Ok((7, vec![0, 1])))
            .is_err()
    );
    let mut stale = provenance();
    stale.policy.as_mut().unwrap().revision += 1;
    assert!(
        target
            .join([9; 32], &store, &stale, |_| Ok((7, vec![0, 1])))
            .is_err()
    );
    assert!(
        target
            .join([9; 32], &store, &provenance(), |_| Ok((8, vec![0, 1])))
            .is_err()
    );
    assert!(
        target
            .join([9; 32], &store, &provenance(), |_| Ok((7, vec![1, 0])))
            .is_err()
    );
    let later = FactualBinding::capture([9; 32], &store, 0, 1)?;
    assert_ne!(later.history, target.binding.history);
    assert_ne!(later.observation_index, target.binding.observation_index);
    Ok(())
}
#[test]
fn physical_factual_span_distinguishes_terminal_truncated_and_search_utility() -> Result<()> {
    for terminal in [false, true] {
        let store = store(terminal)?;
        let mut target = target(&store)?;
        let observed = factual_return(&target.binding, [9; 32], &store, 2, 0.5)?;
        assert_eq!(observed.reward, 2.5); // (1 + .5*2) + .5^2*(1 + .5*2)
        assert_eq!(observed.physical_ticks, 4);
        assert_eq!(
            observed.bootstrap_discount,
            if terminal { 0. } else { 0.0625 }
        );
        assert_eq!(observed.terminated, terminal);
        assert_eq!(observed.truncated, !terminal);
        target.factual_return = Some(observed);
        target.join([9; 32], &store, &provenance(), |_| Ok((7, vec![0, 1])))?;
        target.factual_return.as_mut().unwrap().reward += 1.;
        assert!(
            target
                .join([9; 32], &store, &provenance(), |_| Ok((7, vec![0, 1])))
                .is_err()
        );
        // A search utility may accompany a separately labeled factual physical
        // return, but cannot silently reuse it as a physical value target.
        target.returns.objective = ReturnObjective::Surrogate([4; 32]);
        target.factual_return = Some(factual_return(&target.binding, [9; 32], &store, 1, 0.9)?);
        target.join([9; 32], &store, &provenance(), |_| Ok((7, vec![0, 1])))?;
    }
    Ok(())
}
#[test]
fn unvisited_actions_remain_missing_and_invalid_training_targets_are_rejected() -> Result<()> {
    let policy = PolicyTarget::from_estimates(
        &[0, 1],
        &[ActionEstimate {
            action: 1,
            value: -2.,
            upper_bound: None,
            visits: Some(1),
        }],
    )?;
    assert_eq!(policy, PolicyTarget::ActionValues(vec![None, Some(-2.)]));
    let store = store(true)?;
    let target = target(&store)?;
    for invalid in [
        PolicyTarget::ActionValues(vec![None, None]),
        PolicyTarget::ActionValues(vec![Some(f64::NAN), None]),
        PolicyTarget::Probabilities(vec![0.2, 0.2]),
    ] {
        let mut bad = target.clone();
        bad.policy = Some(invalid);
        assert!(bad.validate().is_err());
    }
    let mut bad = target.clone();
    bad.work.expansions = 3;
    assert!(bad.validate().is_err());
    let mut bad = target.clone();
    bad.feasible = vec![0, 0];
    assert!(bad.validate().is_err());
    Ok(())
}

#[test]
fn a_live_plan_can_bind_before_execution_but_join_requires_the_factual_step() -> Result<()> {
    let mut store = EpisodeStore::new(8, 1, ActionSpace::Discrete { count: 2 })?;
    store.begin_episode(0, &EpisodeStart::fresh(observation()))?;
    let target = target(&store)?;
    assert!(
        target
            .join([9; 32], &store, &provenance(), |_| Ok((7, vec![0, 1])))
            .is_err()
    );
    store.push(&TemporalTransition {
        episode_id: 0,
        timestep: 0,
        observation: observation(),
        action: Action::Discrete(0),
        reward: 1_f32.into(),
        next_observation: observation(),
        terminated: true,
        truncated: true,
        behavior: BehaviorMetadata::default(),
    })?;
    let mut target = target;
    target.factual_return = Some(factual_return(&target.binding, [9; 32], &store, 1, 0.5)?);
    assert_eq!(
        target.factual_return.as_ref().unwrap().bootstrap_discount,
        0.
    );
    target.join([9; 32], &store, &provenance(), |_| Ok((7, vec![0, 1])))?;
    assert!(FactualBinding::capture([9; 32], &store, 0, 1).is_err());
    Ok(())
}

#[test]
fn hypothetical_continuations_keep_their_own_ordered_public_support() -> Result<()> {
    use rl::planning::targets::ConditionalTarget;
    let store = store(true)?;
    let mut target = target(&store)?;
    target.continuations.push(ConditionalTarget {
        after: 0,
        outcome: true,
        public: 8,
        feasible: vec![3, 4],
        policy: Some(PolicyTarget::Probabilities(vec![0.25, 0.75])),
        value: Some(0.8),
    });
    target.validate_continuations(|action, outcome, public| {
        assert_eq!((*action, *outcome, *public), (0, true, 8));
        Ok(vec![3, 4])
    })?;
    assert!(
        target
            .validate_continuations(|_, _, _| Ok(vec![4, 3]))
            .is_err()
    );
    let encoded = serde_json::to_vec(&target)?;
    assert_eq!(target, serde_json::from_slice(&encoded)?);
    target.continuations.push(target.continuations[0].clone());
    assert!(target.validate().is_err());
    Ok(())
}

#[test]
fn compacted_suffix_cannot_impersonate_the_original_public_history() -> Result<()> {
    use rl::{
        core::anchors::*,
        data::temporal::{ResourceLimits, SegmentedConfig, TemporalStorage},
    };
    let key = RepresentationId(1);
    let mut store = EpisodeStore::with_storage(
        16,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::Segmented(SegmentedConfig {
            segment_steps: 2,
            minimum_context: 4,
            required_representations: vec![key],
            limits: ResourceLimits {
                max_transitions: 16,
                max_resident_bytes: 1_000_000,
                max_anchor_bytes: 100_000,
                max_anchors: 128,
                max_anchor_payload_bytes: 64,
                max_dimension_rank: 4,
            },
        }),
    )?;
    store.begin_episode(0, &EpisodeStart::fresh(observation()))?;
    let binding = FactualBinding::capture([9; 32], &store, 0, 0)?;
    for timestep in 0..64 {
        let index = timestep + 1;
        let anchors = (index % 2 == 0).then(|| AnchorSet {
            version: 2,
            anchors: [(
                key,
                StateAnchor {
                    version: 2,
                    episode_id: 0,
                    observation_index: index,
                    phase: AnchorPhase::BeforeObservation,
                    representation_id: key,
                    model_identity: [1; 32],
                    generation: 0,
                    schema: AnchorSchema {
                        codec: 1,
                        codec_version: 1,
                        representation_identity: [2; 32],
                        dimensions: vec![1],
                    },
                    payload: (index as f32).to_le_bytes().to_vec(),
                    exact_ancestry: true,
                    source: AnchorSource::ResetReconstruction,
                },
            )]
            .into_iter()
            .collect(),
        });
        store.push_with_anchors(
            &TemporalTransition {
                episode_id: 0,
                timestep,
                observation: observation(),
                action: Action::Discrete(0),
                reward: 0_f32.into(),
                next_observation: observation(),
                terminated: false,
                truncated: false,
                behavior: BehaviorMetadata::default(),
            },
            anchors,
        )?;
    }
    store.validate()?;
    assert!(store.episode(0).unwrap().base_timestep > 0);
    assert!(binding.verify([9; 32], &store).is_err());
    assert!(FactualBinding::capture([9; 32], &store, 0, 64).is_err());
    Ok(())
}

#[test]
fn direct_conditional_join_uses_enabled_complete_candidate_keys() -> Result<()> {
    use rl::core::{
        action::{ActionArgument, ActionGrammar, ConditionalAction, Operation},
        env::ObservationSpace,
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, ObservationSchema,
            StructuredObservation,
        },
    };
    use std::sync::Arc;
    let first = EntityId { kind: 1, id: 10 };
    let second = EntityId { kind: 1, id: 20 };
    let key = |entity| ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(entity)),
    };
    let actions = ActionSpace::Conditional(Arc::new(ActionGrammar {
        namespace: "target-fixture".into(),
        version: 1,
        operations: vec![Operation {
            name: "inspect".into(),
            argument: ArgumentKind::Entity,
        }],
        candidate_features: vec![],
    }));
    let schema = ObservationSpace::Structured(Arc::new(ObservationSchema {
        namespace: "target-fixture".into(),
        version: 1,
        spatial: vec![],
        entities: vec![rl::core::structured::FeatureIdentity::new(
            "energy", "joules",
        )],
        events: vec![],
        scalars: vec![rl::core::structured::FeatureIdentity::new("public", "unit")],
        vocabularies: vec![],
        conventions: vec![],
    }));
    let frame = Observation::Structured(Arc::new(StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        entities: vec![
            EntityRow {
                id: first,
                features: vec![0.],
            },
            EntityRow {
                id: second,
                features: vec![0.],
            },
        ],
        events: vec![],
        scalars: vec![0.],
        candidates: vec![
            ActionCandidate {
                action: key(second),
                features: vec![],
                enabled: false,
            },
            ActionCandidate {
                action: key(first),
                features: vec![],
                enabled: true,
            },
        ],
    }));
    let mut store = EpisodeStore::new(8, schema, actions)?;
    store.begin_episode(0, &EpisodeStart::fresh(frame.clone()))?;
    let binding = FactualBinding::capture([9; 32], &store, 0, 0)?;
    store.push(&TemporalTransition {
        episode_id: 0,
        timestep: 0,
        observation: frame.clone(),
        action: Action::Conditional(Arc::new(key(first))),
        reward: 1_f32.into(),
        next_observation: frame,
        terminated: true,
        truncated: false,
        behavior: BehaviorMetadata::default(),
    })?;
    let mut target = SearchTarget::<u64, ConditionalAction, bool> {
        version: 1,
        public: 7,
        binding,
        feasible: vec![key(first)],
        policy: Some(PolicyTarget::Probabilities(vec![1.])),
        value: None,
        returns: PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 1.,
        },
        horizon: NonZeroU16::new(1).unwrap(),
        factual_return: None,
        provenance: provenance(),
        budget: PlanningBudget::Expansions(NonZeroU32::new(1).unwrap()),
        work: PlanningWork::default(),
        status: PlanStatus::Bounded,
        continuations: vec![],
    };
    target.join_conditional([9; 32], &store, &provenance(), |_| Ok(7))?;
    target.feasible[0] = key(second);
    assert!(
        target
            .join_conditional([9; 32], &store, &provenance(), |_| Ok(7))
            .is_err()
    );
    Ok(())
}
