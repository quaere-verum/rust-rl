use rl::{core::action::FiniteActionKey, data::temporal::sample_at};

use super::*;
use hall::{SAFE_EXIT, use_door};

#[test]
fn tabular_policy_uses_the_past_cue_to_act_at_an_aliased_fork() -> Result<()> {
    let replay = exploration_replay()?;
    let mut forks = [None, None];
    for episode in replay.episodes().filter(|episode| episode.steps.len() == 2) {
        let phase = episode.observations[0]
            .as_graph()
            .unwrap()
            .observation
            .scalars[0];
        forks[usize::from(phase > 0.)] = Some(&episode.observations[1]);
    }
    assert!(forks[0].is_some());
    assert_eq!(forks[0], forks[1]);
    let batch = SequenceSampler::new(71).sample(&replay, &WINDOW, 64)?;
    assert_eq!(tabular_choices(&batch)?, [20, 30]);
    Ok(())
}

#[test]
fn legal_action_keys_survive_row_reordering() -> Result<()> {
    let mut env = DoorHall::new(Side::Left);
    let observation = env.reset(None)?.observation;
    let mut graph = observation.as_graph().unwrap().clone();
    graph.observation.entities.reverse();
    graph.observation.candidates.reverse();
    graph.nodes.reverse();
    let reordered = rl::core::env::Observation::Graph(graph.into());
    reordered.validate_in(&env.observation_space(), env.action_space(), true)?;
    let original = observation.finite_actions(env.action_space())?;
    assert_eq!(reordered.finite_actions(env.action_space())?, original);
    for action in &original {
        assert_eq!(
            observation.finite_action_key(action)?,
            reordered.finite_action_key(action)?
        );
    }
    assert!(env.step(&Action::Conditional(use_door(20).into())).is_err());
    Ok(())
}

#[test]
fn custom_td_clips_the_update_from_a_real_terminal_reward() -> Result<()> {
    let replay = exploration_replay()?;
    let action = Action::Conditional(use_door(SAFE_EXIT).into());
    let episode = replay
        .episodes()
        .find(|episode| episode.steps[0].action == action)
        .expect("exploration reached the safe exit");
    assert_eq!(episode.steps[0].reward.total(), 2.);
    let state = RememberCue.key(&RememberCue.observe(
        &None,
        HistoryInput {
            observation: &episode.observations[0],
            previous: None,
        },
    )?)?;
    let batch = SequenceBatch {
        config: WINDOW,
        samples: vec![sample_at(&replay, episode.id, 0, &WINDOW)?],
    };
    let mut learner = BoundedTd::new(spaces())?;
    assert_eq!(learner.update(&batch)?.updates, 1);
    assert_eq!(
        learner
            .table
            .value(&state, &FiniteActionKey::Conditional(use_door(SAFE_EXIT))),
        1.
    );
    Ok(())
}

#[test]
fn exact_and_sampled_planning_choose_the_passage() -> Result<()> {
    let (exact, value, sampled) = predictive::plan(State::Entrance(Side::Left))?;
    assert_eq!(exact, 10);
    assert_eq!(sampled, 10);
    assert!((value - 6.375).abs() < 1e-12);
    Ok(())
}

#[cfg(feature = "neural")]
#[test]
fn candidate_ddqn_updates_from_factual_graph_decisions() -> Result<()> {
    let batch = SequenceSampler::new(71).sample(&exploration_replay()?, &WINDOW, 8)?;
    assert_eq!(neural::update(&batch)?, 1);
    Ok(())
}
