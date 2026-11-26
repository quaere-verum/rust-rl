#![cfg(feature = "neural")]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        stateful::{HistoryInput, HistorySpec, PreviousInput},
        stateful_on_policy::{OnPolicySample, StatefulOnPolicyActor},
    },
    nn::{
        temporal_actor::TemporalActor, temporal_pathwise_actor::TemporalPathwiseActor,
        temporal_policy::TemporalPolicy, temporal_q::TemporalQ, temporal_value::TemporalValue,
    },
    prelude::*,
};
use serde::Serialize;

fn exact<T: Serialize>(left: &T, right: &T) {
    assert_eq!(
        serde_json::to_vec(left).unwrap(),
        serde_json::to_vec(right).unwrap()
    );
}
fn sample(s: &OnPolicySample) -> serde_json::Value {
    serde_json::json!({"action":s.decision.action,"log_prob":s.decision.log_prob,"latent":s.latent,"distribution":s.distribution,"value":s.value})
}

fn exercise<A: StatefulOnPolicyActor>(actor: &A)
where
    A::State: Serialize,
{
    let mut rng = ChaCha8Rng::seed_from_u64(791);
    let mut states = (0..16)
        .map(|_| actor.initial_state().unwrap())
        .collect::<Vec<_>>();
    let mut previous: Vec<Option<Action>> = vec![None; 16];
    let mut streams = (0..16)
        .map(|i| ChaCha8Rng::seed_from_u64(101 + i))
        .collect::<Vec<_>>();
    for tick in 0..8 {
        let observations = (0..16)
            .map(|lane| {
                let mut observation =
                    Observation::new((0..2504).map(|_| rng.random_range(-0.8..0.8)).collect());
                if let ActionSpace::Discrete { count } = actor.action_space() {
                    observation.as_dense_mut().unwrap().action_mask = Some(
                        (0..*count)
                            .map(|a| a == 0 || (a + lane + tick) % 3 != 0)
                            .collect(),
                    );
                }
                observation
            })
            .collect::<Vec<_>>();
        for lane in 0..16 {
            if (tick + lane) % 7 == 0 {
                states[lane] = actor.initial_state().unwrap();
                previous[lane] = None;
            }
        }
        let rewards = (0..16)
            .map(|i| (i as f32 - tick as f32) / 19.0)
            .collect::<Vec<_>>();
        let inputs = (0..16)
            .map(|i| HistoryInput {
                observation: &observations[i],
                previous: previous[i].as_ref().map(|action| PreviousInput {
                    factual: None,
                    action,
                    reward: rewards[i],
                }),
            })
            .collect::<Vec<_>>();
        let scalar = (0..16)
            .map(|i| actor.assimilate(&states[i], inputs[i]).unwrap())
            .collect::<Vec<_>>();
        // Different peers, order and inactive subsets at every physical step.
        for batch in [1, 2, 4, 8, 16] {
            let selected = (0..16)
                .rev()
                .filter(|&i| i == 0 || (i + tick) % 4 != 0)
                .collect::<Vec<_>>();
            for group in selected.chunks(batch) {
                let priors = group.iter().map(|&i| &states[i]).collect::<Vec<_>>();
                let history = group.iter().map(|&i| inputs[i]).collect::<Vec<_>>();
                let actual = actor.assimilate_canonical_batch(&priors, &history).unwrap();
                let post = actual.iter().collect::<Vec<_>>();
                let obs = group.iter().map(|&i| &observations[i]).collect::<Vec<_>>();
                let mut group_rng = group
                    .iter()
                    .map(|&i| streams[i].clone())
                    .collect::<Vec<_>>();
                let sampled = actor
                    .sample_on_policy_canonical_batch(&post, &obs, ActMode::Sample, &mut group_rng)
                    .unwrap();
                let values = actor.state_values_canonical_batch(&post, &obs).unwrap();
                for (row, &lane) in group.iter().enumerate() {
                    exact(&actual[row], &scalar[lane]);
                    let mut expected_rng = streams[lane].clone();
                    let expected = actor
                        .sample_on_policy(
                            &scalar[lane],
                            &observations[lane],
                            ActMode::Sample,
                            &mut expected_rng,
                        )
                        .unwrap();
                    exact(&sample(&sampled[row]), &sample(&expected));
                    exact(&group_rng[row], &expected_rng);
                    exact(
                        &values[row],
                        &actor
                            .state_value(&scalar[lane], &observations[lane])
                            .unwrap(),
                    );
                }
            }
        }
        for lane in 0..16 {
            previous[lane] = Some(
                actor
                    .decide(
                        &scalar[lane],
                        &observations[lane],
                        ActMode::Sample,
                        &mut streams[lane],
                    )
                    .unwrap()
                    .action,
            );
        }
        states = scalar;
    }
}

#[test]
fn wide_policy_and_independent_value_are_exact_under_regrouping() {
    for hidden in [4, 16, 64] {
        for space in [
            ActionSpace::Discrete { count: 14 },
            ActionSpace::Box {
                low: vec![-2.0, -1.0],
                high: vec![2.0, 3.0],
            },
        ] {
            let spec = HistorySpec::new(2504, space).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            let actor = TemporalActor::new(
                TemporalPolicy::new(spec.clone(), Some(hidden), &[16], &mut rng).unwrap(),
                Some(TemporalValue::new(spec, Some(hidden), &[16], &mut rng).unwrap()),
                3,
            )
            .unwrap();
            exercise(&actor);
        }
    }
}

#[test]
fn pathwise_actor_preserves_independent_q_memory_and_distribution_bits() {
    for space in [
        ActionSpace::Discrete { count: 14 },
        ActionSpace::Box {
            low: vec![-2.0, -1.0],
            high: vec![2.0, 3.0],
        },
    ] {
        let spec = HistorySpec::new(2504, space).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let actor = TemporalPathwiseActor::new(
            TemporalPolicy::new(spec.clone(), Some(4), &[8], &mut rng).unwrap(),
            TemporalQ::new(spec, Some(4), &[8], 7, &mut rng).unwrap(),
            3,
        )
        .unwrap();
        exercise(&actor);
    }
}

#[test]
fn canonical_policy_only_decisions_do_not_evaluate_an_unused_critic() {
    use burn::{module::Param, tensor::Tensor};
    use rl::{core::stateful::StatefulActor, nn::temporal_actor::TemporalActorState};
    let spec = HistorySpec::new(2504, ActionSpace::Discrete { count: 14 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let policy = TemporalPolicy::new(spec.clone(), Some(4), &[], &mut rng).unwrap();
    let mut value = TemporalValue::new(spec, Some(4), &[], &mut rng).unwrap();
    value.head.layers[0].weight =
        Param::from_tensor(Tensor::full([4, 1], f32::MAX, &rl::nn::device()));
    let actor = TemporalActor::new(policy, Some(value), 3).unwrap();
    let state = TemporalActorState {
        policy: vec![0.2; 4],
        value: Some(vec![1.; 4]),
    };
    let observation = Observation::new(vec![0.1; 2504]);
    let mut expected_rng = rng.clone();
    let expected = actor
        .decide(&state, &observation, ActMode::Sample, &mut expected_rng)
        .unwrap();
    let mut streams = [rng.clone(), rng];
    let decisions = actor
        .decide_canonical_batch(
            &[&state, &state],
            &[&observation, &observation],
            ActMode::Sample,
            &mut streams,
        )
        .expect("a policy-only decision must not query the unrelated overflowing critic");
    assert_eq!(decisions[0].action, expected.action);
    assert_eq!(
        decisions[0].log_prob.unwrap().to_bits(),
        expected.log_prob.unwrap().to_bits()
    );
    exact(&streams[0], &expected_rng);
}
