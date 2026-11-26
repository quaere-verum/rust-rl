#![cfg(feature = "neural")]

#[path = "support/search_guided_doors/domain.rs"]
mod domain;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    algorithms::{
        OptimizerConfig,
        search_guided::{
            PolicyValueBatch, PolicyValuePublication, PolicyValueRow, SearchGuidedConfig,
            SearchGuidedPolicyValue,
        },
    },
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ConditionalAction},
        actor::ActMode,
        checkpoint::Checkpoint,
        env::{Env, Step},
        stateful::{StatefulActor, StatefulDecision, StatefulLearner},
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
    data::temporal::{EpisodeStore, SequenceConfig, sample_at},
    models::predictive::{PlanningModel, PlanningReturns, PredictiveModel, Transition},
    nn::structured::{LightweightPolicy, LightweightValue, NeuralGuidance, PublicValueInput},
    planning::{
        PlanStatus, PlanningBudget,
        guidance::{SearchOptions, TreeSelection, ValueCutoff, search_with_options},
        pomcp::{self, sample_weighted},
        targets::{FactualBinding, PolicyTarget, SearchProvenance, SearchRevision, SearchTarget},
    },
};
use std::{
    num::{NonZeroU16, NonZeroU32},
    sync::Arc,
};

fn learner(seed: u64) -> Result<SearchGuidedPolicyValue> {
    let mut config = domain::network_config();
    config.fused_width = 8;
    config.head_width = 8;
    let device = burn::tensor::Device::flex().autodiff();
    SearchGuidedPolicyValue::new(
        LightweightPolicy::new(
            config.clone(),
            &mut ChaCha8Rng::seed_from_u64(seed),
            &device,
        )?,
        LightweightValue::new(config, &mut ChaCha8Rng::seed_from_u64(seed + 1), &device)?,
        SearchGuidedConfig {
            gamma: 0.9,
            policy_optimizer: OptimizerConfig::default(),
            value_optimizer: OptimizerConfig::default(),
            max_pack_elements: 1024 * 1024,
        },
    )
}

fn budget() -> PlanningBudget {
    PlanningBudget::Simulations {
        trajectories: NonZeroU32::new(64).unwrap(),
        transitions: NonZeroU32::new(128).unwrap(),
    }
}

fn search<
    M: PredictiveModel<Context = domain::Public, Action = ConditionalAction, Observation = bool>,
>(
    model: &M,
    publication: &PolicyValuePublication,
    context: &domain::Public,
    feasible: &[ConditionalAction],
    rng: &mut ChaCha8Rng,
) -> Result<pomcp::Search<ConditionalAction, bool>> {
    let guide = NeuralGuidance::new(
        publication.policy(),
        publication.value(),
        publication.returns(),
        publication.limits(),
        |context: &domain::Public| {
            Ok(PublicValueInput {
                observation: domain::frame(*context),
                terminal_within: NonZeroU16::new(if context.reading.is_some() { 1 } else { 2 })
                    .unwrap(),
            })
        },
    )?;
    let mut options = SearchOptions::new(publication.returns());
    options.selection = TreeSelection::Puct { exploration: 1. };
    options.policy = Some(&guide);
    options.value = Some(ValueCutoff {
        depth: NonZeroU16::MIN,
        guide: &guide,
    });
    search_with_options(
        model,
        context,
        feasible,
        pomcp::Config {
            max_depth: NonZeroU16::new(2).unwrap(),
            gamma: publication.returns().gamma,
            exploration: 1.,
        },
        budget(),
        rng,
        options,
    )
}

struct FailedModel;

impl PlanningModel for FailedModel {
    type Context = domain::Public;
    type Action = ConditionalAction;
    type Observation = bool;

    fn actions(&self, context: &Self::Context, actions: &mut Vec<Self::Action>) {
        domain::DoorsModel.actions(context, actions);
    }

    fn planning_returns(&self) -> PlanningReturns {
        domain::DoorsModel.planning_returns()
    }
}

impl PredictiveModel for FailedModel {
    type Hidden = bool;

    fn sample_root(&self, context: &Self::Context, rng: &mut ChaCha8Rng) -> Result<bool> {
        domain::DoorsModel.sample_root(context, rng)
    }

    fn rollout(&self, context: &Self::Context) -> Option<Self::Action> {
        domain::DoorsModel.rollout(context)
    }

    fn simulate(
        &self,
        _: &Self::Context,
        _: &mut Self::Hidden,
        _: &Self::Action,
        _: &mut ChaCha8Rng,
    ) -> Result<Transition<Self::Context, Self::Observation>> {
        Err(Error::State("injected model transition failure".into()))
    }
}

#[test]
fn search_targets_train_an_executable_policy_from_factual_rows() -> Result<()> {
    let mut learner = learner(19)?;
    let publication = learner.publish()?;
    let mut env = domain::DoorsEnv::new(vec![domain::Task::new(500, 50)?])?;
    let start = env.reset(Some(11))?;
    let mut store = EpisodeStore::new(8, env.observation_space(), env.action_space().clone())?;
    store.begin_episode(0, &start)?;
    let tape = [42; 32];
    let binding = FactualBinding::capture(tape, &store, 0, 0)?;
    let public = domain::decode(binding.verify(tape, &store)?.observation())?;
    let mut feasible = Vec::new();
    domain::DoorsModel.actions(&public, &mut feasible);

    let mut planning_rng = ChaCha8Rng::seed_from_u64(7);
    let mut action_rng = ChaCha8Rng::seed_from_u64(5);
    let before_search = (
        serde_json::to_value(&env)?,
        serde_json::to_value(&store)?,
        action_rng.clone(),
    );
    let planning_before = planning_rng.clone();
    let failed = search(
        &FailedModel,
        &publication,
        &public,
        &feasible,
        &mut planning_rng,
    )?;
    assert_eq!(failed.status, PlanStatus::Failed);
    assert!(failed.action.is_none());
    assert!(failed.work.transitions > 0);
    assert_eq!(planning_rng, planning_before);
    assert_eq!(
        (
            serde_json::to_value(&env)?,
            serde_json::to_value(&store)?,
            action_rng.clone()
        ),
        before_search,
    );

    let planned = search(
        &domain::DoorsModel,
        &publication,
        &public,
        &feasible,
        &mut planning_rng,
    )?;
    assert!(planned.action.is_some());
    assert!(planned.work.transitions > 2);
    assert_ne!(planning_rng, planning_before);
    assert_eq!(
        (
            serde_json::to_value(&env)?,
            serde_json::to_value(&store)?,
            action_rng.clone()
        ),
        before_search,
    );

    let policy = PolicyTarget::from_visits(&feasible, &planned.estimates, 0.5)?;
    let PolicyTarget::Probabilities(law) = &policy else {
        panic!("expected a visit law")
    };
    let index = sample_weighted(law.iter().copied(), &mut action_rng)?;
    let mut selected = StatefulDecision {
        action: Action::Conditional(Arc::new(feasible[index])),
        log_prob: Some(law[index].ln().min(0.) as f32),
    };
    let provenance = SearchProvenance {
        model: SearchRevision {
            identity: [1; 32],
            revision: 1,
        },
        planner: SearchRevision {
            identity: [2; 32],
            revision: 1,
        },
        policy: Some(SearchRevision {
            identity: publication.identity(),
            revision: publication.revision(),
        }),
    };
    let target: SearchTarget<_, _, bool> = SearchTarget {
        version: 1,
        public,
        binding,
        feasible,
        policy: Some(policy),
        value: None,
        returns: publication.returns(),
        horizon: NonZeroU16::new(2).unwrap(),
        factual_return: None,
        provenance: provenance.clone(),
        budget: budget(),
        work: planned.work,
        status: planned.status,
        continuations: vec![],
    };
    let mut observation = start.observation;
    for timestep in 0..2 {
        let Step {
            observation: next,
            reward,
            terminated,
            truncated,
        } = env.step(&selected.action)?;
        store.push(&TemporalTransition {
            episode_id: 0,
            timestep,
            observation,
            action: selected.action,
            reward,
            next_observation: next.clone(),
            terminated,
            truncated,
            behavior: BehaviorMetadata {
                revision: publication.revision(),
                log_prob: selected.log_prob,
            },
        })?;
        if terminated {
            break;
        }
        selected = publication.decide(&(), &next, ActMode::Sample, &mut action_rng)?;
        observation = next;
    }
    let episode = store.episode(0).unwrap();
    assert!(episode.steps.last().unwrap().terminated);
    assert_eq!(episode.steps.len(), store.len());
    let PolicyTarget::Probabilities(law) = target.policy.as_ref().unwrap() else {
        unreachable!()
    };
    let actual = &episode.steps[0];
    assert_eq!(
        actual.behavior.log_prob,
        Some(law[index].ln().min(0.) as f32),
    );
    assert_eq!(
        actual.action,
        Action::Conditional(Arc::new(target.feasible[index]))
    );

    let sample = sample_at(
        &store,
        0,
        0,
        &SequenceConfig {
            burn_in: 0,
            learn_len: 2,
            n_step: 1,
        },
    )?;
    let row = target.join_supervision(tape, &store, &provenance, &sample, |decision| {
        domain::decode(decision.observation())
    })?;
    let actions = row.actions().to_vec();
    let probabilities = row.probabilities().to_vec();
    let batch = PolicyValueBatch {
        rows: vec![PolicyValueRow {
            sample,
            step: 0,
            actions,
            probabilities,
        }],
    };
    assert_eq!(learner.update(&batch)?.updates, 1);

    let mut restored = self::learner(37)?;
    let mut record = learner.checkpoint()?;
    assert!(record.metadata.get("rng").is_none());
    record.metadata["rng"] = serde_json::to_value(ChaCha8Rng::seed_from_u64(71))?;
    restored.restore(&record)?;
    assert_eq!(
        learner.update(&batch)?.metrics,
        restored.update(&batch)?.metrics
    );
    let improved = learner.publish()?;
    assert_eq!(improved.identity(), restored.publish()?.identity());
    assert_ne!(improved.identity(), publication.identity());
    let revision = improved.revision();
    let mut improved_store =
        EpisodeStore::new(8, env.observation_space(), env.action_space().clone())?;
    let mut collector = StatefulCollector::new(SyncVectorEnv::new(vec![env])?, improved, 67)?;
    let delta = collector.collect(1, ActMode::Sample, &mut improved_store)?;
    assert_eq!(delta.env_steps, 1);
    let latest = improved_store.highest_episode_id().unwrap();
    assert_eq!(
        improved_store.episode(latest).unwrap().steps[0]
            .behavior
            .revision,
        revision
    );
    Ok(())
}
