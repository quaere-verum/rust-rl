#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::data::temporal_on_policy::TemporalOnPolicyLearner;
use rl::{
    algorithms::{
        ppo::PpoConfig, recurrent_ppo::RecurrentPpo, recurrent_reppo::RecurrentReppo,
        reppo::ReppoConfig,
    },
    core::{
        action::ActionSpace,
        actor::ActMode,
        checkpoint::{Checkpoint, Snapshot},
        env::Observation,
        stateful::{BehaviorPublisher, HistoryInput, HistorySpec, PreviousInput},
        stateful_on_policy::StatefulOnPolicyActor,
    },
    data::temporal::SequenceConfig,
    envs::delayed_bit::DelayedBit,
    nn::{
        temporal_actor::TemporalActor, temporal_pathwise_actor::TemporalPathwiseActor,
        temporal_policy::TemporalPolicy, temporal_q::TemporalQ, temporal_value::TemporalValue,
    },
    objectives::hl_gauss::HlGaussConfig,
    prelude::{OnPolicyConfig, SyncVectorEnv},
    train::temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
};

fn ppo(seed: u64) -> RecurrentPpo {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    RecurrentPpo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[4], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(5), &[4], &mut rng).unwrap(),
        PpoConfig {
            epochs: 1,
            minibatch_size: 32,
            target_kl: None,
            ..Default::default()
        },
        seed + 1,
    )
    .unwrap()
}
fn reppo(seed: u64) -> RecurrentReppo {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    RecurrentReppo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[4], &mut rng).unwrap(),
        TemporalQ::new(spec, Some(5), &[4], 5, &mut rng).unwrap(),
        ReppoConfig {
            epochs: 1,
            minibatch_size: 32,
            hl_gauss: HlGaussConfig {
                min: -3.,
                max: 3.,
                bins: 5,
                ..Default::default()
            },
            ..Default::default()
        },
        seed + 1,
    )
    .unwrap()
}
fn run_config(pathwise: bool) -> TemporalOnPolicyConfig {
    TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 4,
            gae_lambda: if pathwise { None } else { Some(0.95) },
            normalize_advantages: false,
            ..Default::default()
        },
        sequence: SequenceConfig {
            burn_in: 0,
            learn_len: 4,
            n_step: 1,
        },
        ..Default::default()
    }
}
fn streams_match<A: StatefulOnPolicyActor>(left: &A, right: &A)
where
    A::State: PartialEq + std::fmt::Debug,
{
    let mut l = left.initial_state().unwrap();
    let mut r = right.initial_state().unwrap();
    let mut lr = ChaCha8Rng::seed_from_u64(701);
    let mut rr = lr.clone();
    let mut previous = None;
    for time in 0..4 {
        let observation = Observation::new(vec![0.1 * time as f32, -0.3, 0.4, 0.8, -0.1]);
        let input = HistoryInput {
            observation: &observation,
            previous: previous.as_ref().map(|action| PreviousInput {
                factual: None,
                action,
                reward: 0.2 * time as f32,
            }),
        };
        l = left.assimilate(&l, input).unwrap();
        r = right.assimilate(&r, input).unwrap();
        assert_eq!(l, r);
        let a = left
            .sample_on_policy(&l, &observation, ActMode::Sample, &mut lr)
            .unwrap();
        let b = right
            .sample_on_policy(&r, &observation, ActMode::Sample, &mut rr)
            .unwrap();
        assert_eq!(a.decision.action, b.decision.action);
        assert_eq!(a.decision.log_prob, b.decision.log_prob);
        assert_eq!(a.latent, b.latent);
        assert_eq!(a.distribution, b.distribution);
        assert_eq!(a.value, b.value);
        assert_eq!(lr, rr);
        previous = Some(a.decision.action);
    }
}
fn cache_not_recorded(learner: &impl Checkpoint, models: &[&str]) {
    let record = learner.checkpoint().unwrap();
    assert_eq!(
        record.models.keys().map(String::as_str).collect::<Vec<_>>(),
        models
    );
    assert!(
        record
            .metadata
            .as_object()
            .unwrap()
            .keys()
            .all(|name| !name.contains("cache"))
    );
}

#[test]
fn shared_publication_reuses_models_then_refreshes_after_update() {
    let mut learner = ppo(702);
    let old = learner.behavior_actor(&(), 0).unwrap();
    let again = learner.behavior_actor(&(), 0).unwrap();
    assert!(
        std::ptr::eq(old.policy(), again.policy()),
        "unchanged shared policy was frozen again"
    );
    assert!(std::ptr::eq(old.value().unwrap(), again.value().unwrap()));
    let held = serde_json::to_value(&old).unwrap();
    let record = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    for _ in 0..4 {
        learner.behavior_actor(&(), 0).unwrap();
    }
    assert!(
        record == serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
        "publication changed persisted learner state"
    );
    cache_not_recorded(&learner, &["policy", "value"]);
    let mut trainer = TemporalOnPolicyTrainer::new(
        SyncVectorEnv::new(vec![DelayedBit::new(2, None, false, 703).unwrap()]).unwrap(),
        &learner,
        run_config(false),
        704,
    )
    .unwrap();
    trainer.cycle(&mut learner).unwrap();
    let current = learner.behavior_actor(&(), learner.revision()).unwrap();
    let independent =
        TemporalActor::new(learner.policy(), Some(learner.value()), learner.revision()).unwrap();
    assert!(!std::ptr::eq(old.policy(), current.policy()));
    assert_ne!(old.model_identity(), current.model_identity());
    assert_eq!(current.model_identity(), independent.model_identity());
    streams_match(&current, &independent);
    let repeated = learner.behavior_actor(&(), learner.revision()).unwrap();
    assert!(std::ptr::eq(current.policy(), repeated.policy()));
    assert!(
        held == serde_json::to_value(&old).unwrap(),
        "update changed held frozen behavior"
    );
    old.validate_snapshot().unwrap();
    current.validate_snapshot().unwrap();
}

#[test]
fn shared_same_counter_restore_replaces_warm_cache_and_failed_restore_preserves_it() {
    let mut learner = ppo(705);
    let other = ppo(706);
    let old = learner.behavior_actor(&(), 0).unwrap();
    let held = serde_json::to_value(&old).unwrap();
    assert_eq!(learner.updates(), other.updates());
    learner.restore(&other.checkpoint().unwrap()).unwrap();
    let current = learner.behavior_actor(&(), 0).unwrap();
    let independent = TemporalActor::new(learner.policy(), Some(learner.value()), 0).unwrap();
    assert_ne!(old.model_identity(), current.model_identity());
    assert_eq!(current.model_identity(), independent.model_identity());
    assert!(!std::ptr::eq(old.policy(), current.policy()));
    streams_match(&current, &independent);
    let mut invalid = learner.checkpoint().unwrap();
    invalid.metadata["version"] = u32::MAX.into();
    assert!(learner.restore(&invalid).is_err());
    let retained = learner.behavior_actor(&(), 0).unwrap();
    assert!(
        std::ptr::eq(current.policy(), retained.policy()),
        "failed restore discarded live cache"
    );
    assert!(held == serde_json::to_value(&old).unwrap());
    cache_not_recorded(&learner, &["policy", "value"]);
}

#[test]
fn pathwise_revision_only_publication_reuses_models_then_refreshes_after_update() {
    let mut learner = reppo(707);
    let old = learner.behavior_actor(&(), 0).unwrap();
    let held = serde_json::to_value(&old).unwrap();
    learner.begin_collection().unwrap();
    let revised = learner.behavior_actor(&(), learner.revision()).unwrap();
    assert_eq!(learner.revision(), 1);
    assert_eq!(old.model_identity(), revised.model_identity());
    assert!(
        std::ptr::eq(old.policy(), revised.policy()),
        "revision-only publication refroze policy"
    );
    assert!(
        std::ptr::eq(old.critic(), revised.critic()),
        "revision-only publication refroze critic"
    );
    streams_match(&old, &revised);
    let record = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    learner.behavior_actor(&(), learner.revision()).unwrap();
    assert!(record == serde_json::to_value(learner.checkpoint().unwrap()).unwrap());
    let mut trainer = TemporalOnPolicyTrainer::new(
        SyncVectorEnv::new(vec![DelayedBit::new(2, None, false, 708).unwrap()]).unwrap(),
        &learner,
        run_config(true),
        709,
    )
    .unwrap();
    trainer.cycle(&mut learner).unwrap();
    let current = learner.behavior_actor(&(), learner.revision()).unwrap();
    let independent =
        TemporalPathwiseActor::new(learner.policy(), learner.critic(), learner.revision()).unwrap();
    assert!(!std::ptr::eq(old.policy(), current.policy()));
    assert!(!std::ptr::eq(old.critic(), current.critic()));
    assert_ne!(old.model_identity(), current.model_identity());
    assert_eq!(current.model_identity(), independent.model_identity());
    streams_match(&current, &independent);
    let again = learner.behavior_actor(&(), learner.revision()).unwrap();
    assert!(std::ptr::eq(current.policy(), again.policy()));
    assert!(std::ptr::eq(current.critic(), again.critic()));
    assert!(held == serde_json::to_value(&old).unwrap());
    cache_not_recorded(&learner, &["alpha", "beta", "critic", "policy"]);
}

#[test]
fn pathwise_same_counter_restore_replaces_warm_cache_and_failed_restore_preserves_it() {
    let mut learner = reppo(710);
    let other = reppo(711);
    let old = learner.behavior_actor(&(), 0).unwrap();
    let held = serde_json::to_value(&old).unwrap();
    assert_eq!(learner.updates(), other.updates());
    learner.restore(&other.checkpoint().unwrap()).unwrap();
    let current = learner.behavior_actor(&(), 0).unwrap();
    let independent = TemporalPathwiseActor::new(learner.policy(), learner.critic(), 0).unwrap();
    assert_ne!(old.model_identity(), current.model_identity());
    assert_eq!(current.model_identity(), independent.model_identity());
    streams_match(&current, &independent);
    let mut invalid = learner.checkpoint().unwrap();
    invalid.metadata["version"] = u32::MAX.into();
    assert!(learner.restore(&invalid).is_err());
    let retained = learner.behavior_actor(&(), 0).unwrap();
    assert!(std::ptr::eq(current.policy(), retained.policy()));
    assert!(std::ptr::eq(current.critic(), retained.critic()));
    assert!(held == serde_json::to_value(&old).unwrap());
    cache_not_recorded(&learner, &["alpha", "beta", "critic", "policy"]);
}
