#![cfg(feature = "neural")]

#[path = "support/planning_doors.rs"]
mod doors;

use doors::{Public, SCAN, frame, guess, model_config};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    approx::{FeatureMap, FittedQ, StatisticalActor, regression::LinearModel},
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        checkpoint::Checkpoint,
        env::{Observation, ObservationSpace, PublicHistory},
        reward::{BatchObjective, Reward},
        stateful::StatefulActor,
        structured::{EntityId, EventStamp, PublicEvent},
        temporal::TemporalSpaces,
    },
    nn::structured::{FrozenRecurrentFeatures, PackLimits, RecurrentLightweightQ},
};
use std::{hint::black_box, num::NonZeroUsize, sync::Arc, time::Instant};

fn spaces() -> TemporalSpaces {
    let input = model_config().model.input;
    TemporalSpaces {
        observations: ObservationSpace::Structured(Arc::new(input.observation_schema())),
        actions: ActionSpace::Conditional(Arc::new(input.action_grammar())),
    }
}

fn network(seed: u64) -> Result<RecurrentLightweightQ> {
    RecurrentLightweightQ::new(
        model_config(),
        &mut ChaCha8Rng::seed_from_u64(seed),
        &Default::default(),
    )
}

fn conditional(action: ConditionalAction) -> Action {
    Action::Conditional(Arc::new(action))
}

fn lane(length: usize, source: u32) -> Result<Vec<PublicHistory>> {
    let mut history = PublicHistory::new(frame(Public { reading: None }));
    let mut result = vec![history.clone()];
    let mut notebook = Vec::new();
    for step in 1..length {
        let mut observation = frame(Public {
            reading: (!step.is_multiple_of(4))
                .then_some((step + source as usize).is_multiple_of(2)),
        });
        let Observation::Structured(observation_frame) = &mut observation else {
            unreachable!()
        };
        let observation_frame = Arc::make_mut(observation_frame);
        let action = if step % 4 == 1 {
            conditional(SCAN)
        } else {
            conditional(guess(step.is_multiple_of(2)))
        };
        let reward = if step.is_multiple_of(3) {
            Reward::from_primitives(vec![0.25, -0.5, 0.125])?
        } else {
            Reward::One(0.0)
        };
        if !step.is_multiple_of(3) {
            let entity = EntityId {
                kind: 1,
                id: u64::from(step.is_multiple_of(2)),
            };
            notebook.push(PublicEvent {
                features: Arc::from([if entity.id == 1 { 1.0 } else { -1.0 }]),
                entity: Some(entity),
                context: Arc::from([entity]),
                stamp: Some(EventStamp {
                    source,
                    ordinal: notebook.len() as u64,
                    tick: history.physical_timestep() + reward.duration(),
                }),
            });
        }
        observation_frame.events.clone_from(&notebook);
        let count = observation_frame.candidates.len();
        observation_frame.candidates.rotate_left(step % count);
        history = history.append(action, reward, observation)?;
        result.push(history.clone());
    }
    Ok(result)
}

#[test]
fn incremental_frozen_features_match_exact_prefixes_in_independent_lanes() -> Result<()> {
    let exact = FrozenRecurrentFeatures::new(network(7)?, PackLimits::default())?;
    let incremental = exact
        .clone()
        .with_incremental(NonZeroUsize::new(3).unwrap());
    assert_eq!(exact.spec(), incremental.spec());
    let lanes = [lane(19, 1)?, lane(12, 2)?];
    let actions = spaces().actions;
    for step in 0..lanes[0].len() {
        for history in lanes.iter().rev().filter_map(|lane| lane.get(step)) {
            let legal = history.observation().finite_actions(&actions)?;
            let expected = exact.encode(history, &legal)?;
            assert_eq!(
                incremental.encode(history, &legal)?,
                expected,
                "history {step}"
            );
            let last = legal.len() - 1;
            assert_eq!(
                incremental.encode(history, &legal[last..])?,
                expected[last..]
            );
        }
    }
    // The early frame has been evicted and must be reconstructed from its reset.
    let evicted = &lanes[1][3];
    let legal = evicted.observation().finite_actions(&actions)?;
    assert_eq!(
        incremental.encode(evicted, &legal)?,
        exact.encode(evicted, &legal)?
    );
    Ok(())
}

#[test]
fn incremental_actor_restores_with_a_fresh_history_and_encoder_revision() -> Result<()> {
    let model = network(17)?;
    let encoder_record = model.checkpoint()?;
    let exact = FrozenRecurrentFeatures::new(model, PackLimits::default())?;
    let actor = StatisticalActor::new(
        exact
            .clone()
            .with_incremental(NonZeroUsize::new(2).unwrap()),
        FittedQ::new(
            exact.spec().clone(),
            spaces(),
            BatchObjective::Physical { gamma: 0.9 },
            LinearModel::new(vec![1.0; exact.spec().dimension()])?,
        )?,
        8,
        0.2,
    )?;
    let histories = lane(7, 3)?;
    for history in &histories {
        actor.action_values(history)?;
    }
    let record = actor.checkpoint()?;
    let mut model = network(81)?;
    model.restore(&encoder_record)?;
    let restored_features = FrozenRecurrentFeatures::new(model, PackLimits::default())?
        .with_incremental(NonZeroUsize::new(2).unwrap());
    let restored =
        StatisticalActor::<_, LinearModel>::restore_with_features(restored_features, &record)?;
    let original = &histories[5];
    let history: PublicHistory = serde_json::from_slice(&serde_json::to_vec(original)?)?;
    assert_eq!(restored.revision(), 8);
    assert_eq!(
        actor.action_values(original)?,
        restored.action_values(&history)?
    );
    let continued = history.append(
        conditional(guess(false)),
        0.5,
        history.observation().clone(),
    )?;
    let newer = StatisticalActor::new(
        restored.feature_map().clone(),
        restored.model().clone(),
        9,
        0.2,
    )?;
    assert_eq!(newer.revision(), 9);
    assert_eq!(
        newer.action_values(&continued)?,
        restored.action_values(&continued)?
    );

    let changed = FrozenRecurrentFeatures::new(network(19)?, PackLimits::default())?;
    assert_ne!(changed.spec(), exact.spec());
    let changed_incremental = changed
        .clone()
        .with_incremental(NonZeroUsize::new(2).unwrap());
    assert!(
        StatisticalActor::<_, LinearModel>::restore_with_features(
            changed_incremental.clone(),
            &record
        )
        .is_err()
    );
    let actions = spaces().actions;
    for history in &histories {
        let legal = history.observation().finite_actions(&actions)?;
        assert_eq!(
            changed_incremental.encode(history, &legal)?,
            changed.encode(history, &legal)?
        );
    }
    Ok(())
}

#[test]
fn cold_incremental_replay_recovers_a_history_beyond_the_packed_prefix_budget() -> Result<()> {
    let mut config = model_config();
    config.max_tokens = 4;
    let model = RecurrentLightweightQ::new(
        config,
        &mut ChaCha8Rng::seed_from_u64(7),
        &Default::default(),
    )?;
    let reference = FrozenRecurrentFeatures::new(model, PackLimits::default())?;
    let capacity = NonZeroUsize::new(2).unwrap();
    let warm = reference.clone().with_incremental(capacity);
    let encoder_bytes = serde_json::to_vec(&warm)?;
    let histories = lane(9, 4)?;
    let action_space = spaces().actions;
    for history in &histories {
        warm.encode(
            history,
            &history.observation().finite_actions(&action_space)?,
        )?;
    }
    let history = histories.last().unwrap();
    let actions = history.observation().finite_actions(&action_space)?;
    assert!(reference.encode(history, &actions).is_err());
    assert_eq!(encoder_bytes, serde_json::to_vec(&warm)?);
    let restored: PublicHistory = serde_json::from_slice(&serde_json::to_vec(history)?)?;
    let cold: FrozenRecurrentFeatures = serde_json::from_slice(&encoder_bytes)?;
    assert_eq!(cold.spec(), warm.spec());
    assert_eq!(
        warm.encode(history, &actions)?,
        cold.encode(&restored, &actions)?
    );
    let mut corrupted: serde_json::Value = serde_json::from_slice(&encoder_bytes)?;
    let first = warm.spec().representation.unwrap()[0];
    corrupted["representation"][0] = serde_json::json!(first ^ 1);
    assert!(serde_json::from_value::<FrozenRecurrentFeatures>(corrupted).is_err());
    Ok(())
}

#[test]
#[ignore = "bounded CPU timing for chronological recurrent feature queries"]
fn recurrent_prefix_and_incremental_cpu_timing() -> Result<()> {
    let config = model_config();
    let exact = FrozenRecurrentFeatures::new(network(7)?, PackLimits::default())?;
    let capacity = NonZeroUsize::new(4).unwrap();
    let actions = spaces().actions;
    for length in [8, 32, 96] {
        let lanes = [lane(length, 1)?, lane(length * 2 / 3, 2)?];
        let mut queries = Vec::new();
        for step in 0..length {
            for history in lanes.iter().filter_map(|lane| lane.get(step)) {
                queries.push((history, history.observation().finite_actions(&actions)?));
            }
        }
        let mut reference = Vec::new();
        let mut cached = Vec::new();
        for _ in 0..3 {
            let incremental = exact.clone().with_incremental(capacity);
            let start = Instant::now();
            for (history, actions) in &queries {
                black_box(exact.encode(history, actions)?);
            }
            reference.push(start.elapsed());
            let start = Instant::now();
            for (history, actions) in &queries {
                black_box(incremental.encode(history, actions)?);
            }
            cached.push(start.elapsed());
        }
        reference.sort_unstable();
        cached.sort_unstable();
        let state_payload = capacity.get()
            * (config.hidden_width + 3 * exact.spec().dimension())
            * size_of::<f32>();
        println!(
            "length {length}, {} queries, exact {:.2} ms, incremental {:.2} ms, {:.2}x, cache payload at most {state_payload} B",
            queries.len(),
            reference[1].as_secs_f64() * 1000.0,
            cached[1].as_secs_f64() * 1000.0,
            reference[1].as_secs_f64() / cached[1].as_secs_f64()
        );
    }
    Ok(())
}
