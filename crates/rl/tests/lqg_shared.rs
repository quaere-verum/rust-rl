use rl::{
    collect::vector_env::{SyncVectorEnv, VectorEnv},
    core::{action::Action, env::Env},
    envs::lqg::{BatchedLqg, Lqg, LqgConfig},
};
use std::sync::Arc;

fn noisy_lqg() -> Lqg {
    Lqg::new(
        LqgConfig::scalar(128, 0.9, 0.5, 1., 0.2, 2., 1., 0.1, 2.).unwrap(),
        37,
    )
    .unwrap()
}

#[test]
fn clones_share_immutable_parameters() {
    let first = noisy_lqg();
    let second = first.clone();
    assert!(std::ptr::eq(first.config(), second.config()));
}

#[test]
fn cloning_preserves_independent_state_and_rng_progress() {
    let mut first = noisy_lqg();
    first.reset(None).unwrap();
    let mut second = first.clone();
    let untouched = serde_json::to_vec(&second).unwrap();
    let action = Action::Continuous(vec![0.25]);
    let expected = first.step(&action).unwrap();
    first.reset(Some(999)).unwrap();
    first.step(&action).unwrap();
    assert_eq!(untouched, serde_json::to_vec(&second).unwrap());
    let actual = second.step(&action).unwrap();
    assert_eq!(actual.observation, expected.observation);
    assert_eq!(actual.reward, expected.reward);
}

#[test]
fn serialized_lqg_preserves_the_next_noise_and_terminal_reward() {
    let mut original = noisy_lqg();
    original.reset(None).unwrap();
    let action = Action::Continuous(vec![-0.125]);
    for _ in 0..17 {
        original.step(&action).unwrap();
    }
    let mut restored: Lqg =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    for _ in 17..128 {
        let expected = original.step(&action).unwrap();
        let actual = restored.step(&action).unwrap();
        assert_eq!(actual.observation, expected.observation);
        assert_eq!(actual.reward, expected.reward);
        assert_eq!(actual.terminated, expected.terminated);
    }
    assert!(restored.step(&action).is_err());
}

#[test]
fn deserialization_rejects_invalid_configuration_or_time() {
    let record = serde_json::to_value(noisy_lqg()).unwrap();
    let mut invalid_time = record.clone();
    invalid_time["time"] = serde_json::json!(129);
    assert!(serde_json::from_value::<Lqg>(invalid_time).is_err());
    let mut invalid_config = record;
    invalid_config["config"]["initial"] = serde_json::json!([]);
    assert!(serde_json::from_value::<Lqg>(invalid_config).is_err());
    let mut invalid_state = serde_json::to_value(noisy_lqg()).unwrap();
    invalid_state["state"] =
        serde_json::to_value(nalgebra::DVector::from_vec(vec![0., 1.])).unwrap();
    assert!(serde_json::from_value::<Lqg>(invalid_state).is_err());
}

#[test]
fn explicitly_shared_parameters_retain_identity_and_are_validated() {
    let config = Arc::new(noisy_lqg().config().clone());
    let env = Lqg::from_shared(Arc::clone(&config), 1).unwrap();
    assert!(std::ptr::eq(config.as_ref(), env.config()));
    let mut invalid = config.as_ref().clone();
    invalid.initial.clear();
    assert!(Lqg::from_shared(Arc::new(invalid), 1).is_err());
}

fn coupled_config(dimensions: usize) -> LqgConfig {
    use nalgebra::DMatrix;
    use rl::core::action::ActionSpace;
    let matrix = |scale| DMatrix::identity(dimensions, dimensions) * scale;
    let controls = dimensions.saturating_sub(1).max(1);
    let mut dynamics = matrix(0.9);
    if dimensions > 1 {
        dynamics[(0, 1)] = 0.125;
        dynamics[(1, 0)] = -0.075;
    }
    LqgConfig {
        a: vec![dynamics; 128],
        b: vec![DMatrix::identity(dimensions, controls) * 0.5; 128],
        q: vec![matrix(1.); 128],
        r: vec![DMatrix::identity(controls, controls) * 0.2; 128],
        terminal: matrix(2.),
        noise: vec![matrix(0.1); 128],
        initial: vec![1.; dimensions],
        action_space: ActionSpace::Box {
            low: vec![-2.; controls],
            high: vec![2.; controls],
        },
    }
}

#[test]
fn batched_lanes_match_sync_across_1024_steps_resets_and_restore() {
    for dimensions in [1, 3] {
        let config = Arc::new(coupled_config(dimensions));
        let controls = config.action_space.dimensions();
        let seeds: Vec<_> = (100..107).collect();
        let mut batch = BatchedLqg::from_shared(Arc::clone(&config), &seeds).unwrap();
        let mut sync = SyncVectorEnv::new(
            seeds
                .iter()
                .map(|seed| Lqg::from_shared(Arc::clone(&config), *seed).unwrap())
                .collect(),
        )
        .unwrap();
        let mut ended = vec![true; seeds.len()];
        for time in 0..1024 {
            let resets: Vec<_> = ended
                .iter()
                .enumerate()
                .map(|(lane, ended)| *ended || (time % 71 == 0 && lane % 2 == 0))
                .collect();
            let reseed = (time % 71 == 0).then_some(seeds.as_slice());
            let batched = batch.reset(&resets, reseed).unwrap();
            let scalar = sync.reset(&resets, reseed).unwrap();
            assert!(
                batched
                    .iter()
                    .chain(&scalar)
                    .flatten()
                    .all(|start| start.context.is_none())
            );
            assert!(
                batched
                    .iter()
                    .map(|start| start.as_ref().map(|s| &s.observation))
                    .eq(scalar
                        .iter()
                        .map(|start| start.as_ref().map(|s| &s.observation)))
            );
            let actions: Vec<_> = (0..seeds.len())
                .map(|lane| {
                    ((time + lane) % 5 != 0)
                        .then(|| Action::Continuous(vec![(lane as f32 - 3.) * 0.125; controls]))
                })
                .collect();
            let actual = batch.step(&actions).unwrap();
            let expected = sync.step(&actions).unwrap();
            for (lane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                match (actual, expected) {
                    (Some(actual), Some(expected)) => {
                        assert_eq!(actual.observation, expected.observation, "time={time}");
                        assert_eq!(actual.reward, expected.reward, "time={time}");
                        assert_eq!(actual.terminated, expected.terminated);
                        assert_eq!(actual.truncated, expected.truncated);
                        ended[lane] = actual.ended();
                    }
                    (None, None) => ended[lane] = false,
                    _ => panic!("active lane mismatch at time={time}, lane={lane}"),
                }
            }
            if time == 511 {
                let bytes = serde_json::to_vec(&batch).unwrap();
                batch = serde_json::from_slice(&bytes).unwrap();
            }
        }
    }
}

#[test]
fn batch_validates_shapes_and_actions_before_advancing_any_lane() {
    let mut batch = BatchedLqg::new(coupled_config(1), &[3, 4]).unwrap();
    let actions = vec![Some(Action::Continuous(vec![0.])); 2];
    assert!(batch.step(&actions).is_err());
    assert!(batch.reset(&[true], None).is_err());
    assert!(batch.reset(&[true, true], Some(&[3])).is_err());
    batch.reset(&[true, true], None).unwrap();
    let before = serde_json::to_vec(&batch).unwrap();
    assert!(batch.step(&actions[..1]).is_err());
    assert!(
        batch
            .step(&[actions[0].clone(), Some(Action::Continuous(vec![99.]))])
            .is_err()
    );
    assert_eq!(before, serde_json::to_vec(&batch).unwrap());
    assert!(BatchedLqg::new(coupled_config(1), &[]).is_err());
}

#[test]
fn batch_snapshot_stores_configuration_once_and_rejects_corrupt_lanes() {
    let config = Arc::new(coupled_config(3));
    let seeds: Vec<_> = (0..8).collect();
    let batch = BatchedLqg::from_shared(Arc::clone(&config), &seeds).unwrap();
    let sync = SyncVectorEnv::new(
        seeds
            .iter()
            .map(|seed| Lqg::from_shared(Arc::clone(&config), *seed).unwrap())
            .collect(),
    )
    .unwrap();
    assert!(
        serde_json::to_vec(&batch).unwrap().len() < serde_json::to_vec(&sync).unwrap().len() / 4
    );
    let mut record = serde_json::to_value(batch).unwrap();
    record["lanes"][0]["time"] = serde_json::json!(129);
    assert!(serde_json::from_value::<BatchedLqg>(record).is_err());
}

#[test]
fn batch_step_into_retains_result_storage() {
    let mut batch = BatchedLqg::new(coupled_config(1), &[3, 4]).unwrap();
    batch.reset(&[true, true], None).unwrap();
    let actions = vec![Some(Action::Continuous(vec![0.])); 2];
    let mut result = Vec::with_capacity(2);
    batch.step_into(&actions, &mut result).unwrap();
    let allocation = result.as_ptr();
    batch.step_into(&actions, &mut result).unwrap();
    assert_eq!(result.as_ptr(), allocation);
    assert!(result.iter().all(Option::is_some));
}
