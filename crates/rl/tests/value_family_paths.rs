#![cfg(all(feature = "diagnostics", feature = "neural"))]
//! Genuine small learner updates verify family-to-typed-adapter wiring. These
//! tests are not retained native recordings or evidence of policy convergence.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{models::*, schema::*, value::*},
    envs::lqg::{Lqg, LqgConfig},
    prelude::*,
};
use std::collections::BTreeMap;
fn environment() -> Lqg {
    Lqg::new(
        LqgConfig::scalar(4, 0.95, 0.5, 1., 0.2, 1., 1., 0.15, 2.).unwrap(),
        11,
    )
    .unwrap()
}
fn actor_critic(env: &Lqg) -> ActorCritic {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    ActorCritic {
        policy: Policy::new(2, env.action_space().clone(), &[8], &mut rng).unwrap(),
        value: Value::new(2, &[8], &mut rng).unwrap(),
    }
}
fn trainer(env: Lqg, episodic: bool) -> OnPolicyTrainer<SyncVectorEnv<Lqg>> {
    OnPolicyTrainer::new(
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env; 2]).unwrap(), 43).unwrap(),
        OnPolicyConfig {
            rollout_steps: 4,
            episode_guard: episodic.then_some(4),
            gae_lambda: if episodic { None } else { Some(0.95) },
            ..Default::default()
        },
    )
    .unwrap()
}
fn inspect_state(value: &Value, space: ActionSpace, enabled: bool) {
    let mut b = SnapshotBuilder::new("typed", RecordContext::new("typed", 1), 2, space).unwrap();
    b.value("baseline", value).unwrap();
    let mut c = b.finish().unwrap();
    attach_declaration(
        &mut c,
        &ValueDeclaration {
            version: 1,
            run_id: "typed".into(),
            revision: 1,
            model_role: "online".into(),
            conventions: None,
            critics: BTreeMap::from([(
                "baseline".into(),
                CriticDeclaration {
                    kind: ValueCriticKind::StateValue,
                    learned_baseline: Some(enabled),
                    actor: None,
                    continuation: None,
                    twin: None,
                    utility: None,
                },
            )]),
        },
    )
    .unwrap();
    let model = ModelBundle::load(&c, ModelLimits::default()).unwrap();
    let output = model.inspect_critics(&CriticInspectionRequest {
        critics: vec!["baseline".into()],
        observations: vec![Observation::new(vec![0.5, 0.5])],
        fixed_actions: None,
    });
    if enabled {
        let result = output.unwrap();
        assert!(
            ValuePrediction::from_critic(&result.critics["baseline"], 0, None)
                .unwrap()
                .scalar(&ValueOutput::ExpectedReturn, None)
                .unwrap()
                .is_finite()
        );
    } else {
        assert!(
            output.is_err(),
            "disabled REINFORCE baseline must never be exposed as trained V"
        );
    }
}
#[test]
fn reinforce_enabled_and_disabled_baselines_use_actual_configuration() {
    for baseline in [false, true] {
        let env = environment();
        let mut l = Reinforce::new(
            actor_critic(&env),
            ReinforceConfig {
                baseline,
                ..Default::default()
            },
            44,
        )
        .unwrap();
        let space = env.action_space().clone();
        trainer(env, true)
            .train(&mut l, Budget::vector_steps(8), None)
            .unwrap();
        assert!(l.updates() > 0);
        inspect_state(&l.value(), space, l.config().baseline);
    }
}
#[test]
fn vmpo_actual_updated_value_exports_state_value() {
    let env = environment();
    let space = env.action_space().clone();
    let mut l = Vmpo::new(
        actor_critic(&env),
        VmpoConfig {
            epochs: 1,
            minibatch_size: 8,
            ..Default::default()
        },
        44,
    )
    .unwrap();
    trainer(env, false)
        .train(&mut l, Budget::vector_steps(8), None)
        .unwrap();
    assert!(l.updates() > 0);
    inspect_state(&l.value(), space, true);
}
#[test]
fn ddpg_actual_updated_critic_exports_scalar_action_value() {
    let env = environment();
    let space = env.action_space().clone();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut l = Ddpg::new(
        Deterministic::new(2, space.clone(), &[8], &mut rng).unwrap(),
        QValue::new(2, 1, 1, &[8], &mut rng).unwrap(),
        Default::default(),
        43,
    )
    .unwrap();
    let mut trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(vec![env; 2]).unwrap(), 44).unwrap(),
        OffPolicyConfig {
            replay_capacity: 128,
            batch_size: 8,
            warmup_transitions: 8,
            updates_per_transition: 0.25,
            ..Default::default()
        },
        45,
    )
    .unwrap();
    trainer
        .train(&mut l, Budget::vector_steps(16), None)
        .unwrap();
    assert!(l.updates() > 0);
    let mut b =
        SnapshotBuilder::new("typed", RecordContext::new("ddpg", l.updates()), 2, space).unwrap();
    b.deterministic("policy", &l.policy()).unwrap();
    b.scalar("critic", &l.critic()).unwrap();
    let model = ModelBundle::load(&b.finish().unwrap(), ModelLimits::default()).unwrap();
    let result = model
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["critic".into()],
            observations: vec![Observation::new(vec![0.5, 0.5])],
            fixed_actions: Some(vec![Action::Continuous(vec![0.2])]),
        })
        .unwrap();
    assert!(
        matches!(ValuePrediction::from_critic(&result.critics["critic"],0,None).unwrap(),ValuePrediction::Scalar(v)if v.is_finite())
    );
}
