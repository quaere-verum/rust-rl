mod hall;
mod learner;
#[cfg(feature = "neural")]
mod neural;
mod predictive;
#[cfg(test)]
mod tests;

use rl::{
    Error, Result,
    algorithms::tabular::TabularActor,
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{
        action::Action,
        actor::ActMode,
        env::Env,
        state_key::StateKeyBuilder,
        stateful::{BehaviorPublisher, HistoryInput, StatefulLearner},
    },
    data::temporal::{EpisodeStore, SequenceBatch, SequenceConfig, SequenceSampler},
};

use hall::{DoorHall, RememberCue, Side, State, spaces, target};
use learner::BoundedTd;

const WINDOW: SequenceConfig = SequenceConfig {
    n_step: 1,
    learn_len: 1,
    burn_in: 0,
};

fn main() -> Result<()> {
    let replay = exploration_replay()?;
    let batch = SequenceSampler::new(71).sample(&replay, &WINDOW, 64)?;
    let doors = tabular_choices(&batch)?;
    println!(
        "The tabular policy uses doors {} and {} after the two cues.",
        doors[0], doors[1]
    );
    let entrance = DoorHall::new(Side::Left).reset(None)?.observation;
    let root = RememberCue.key(&RememberCue.observe(
        &None,
        HistoryInput {
            observation: &entrance,
            previous: None,
        },
    )?)?;
    let (exact, value, sampled) = predictive::plan(root)?;
    println!(
        "Exact planning chooses door {exact} with value {value:.3}. Sampled planning chooses door {sampled}."
    );
    #[cfg(feature = "neural")]
    println!(
        "Candidate DDQN completed {} update.",
        neural::update(&batch)?
    );
    Ok(())
}

fn collect(
    actor: TabularActor<RememberCue>,
    mode: ActMode,
    vector_steps: usize,
) -> Result<EpisodeStore> {
    let env = SyncVectorEnv::new(vec![DoorHall::new(Side::Left), DoorHall::new(Side::Right)])?;
    let mut collector = StatefulCollector::new(env, actor, 53)?;
    let spaces = spaces();
    let mut replay = EpisodeStore::new(256, spaces.observations, spaces.actions)?;
    collector.collect(vector_steps, mode, &mut replay)?;
    Ok(replay)
}

fn exploration_replay() -> Result<EpisodeStore> {
    let learner = BoundedTd::new(spaces())?;
    collect(learner.behavior_actor(&1., 0)?, ActMode::Sample, 64)
}

fn tabular_choices(batch: &SequenceBatch) -> Result<[u64; 2]> {
    let mut learner = BoundedTd::new(spaces())?;
    for _ in 0..6 {
        learner.update(batch)?;
    }
    let replay = collect(learner.behavior_actor(&0., 1)?, ActMode::Deterministic, 24)?;
    let mut doors = [None, None];
    for episode in replay.episodes() {
        let [_, decision] = episode.steps.as_slice() else {
            continue;
        };
        let State::Entrance(side) = RememberCue.key(&RememberCue.observe(
            &None,
            HistoryInput {
                observation: &episode.observations[0],
                previous: None,
            },
        )?)?
        else {
            return Err(Error::State("expected to begin at the entrance".into()));
        };
        let Action::Conditional(action) = &decision.action else {
            return Err(Error::Action("expected a door action".into()));
        };
        let slot = match side {
            Side::Left => 0,
            Side::Right => 1,
        };
        doors[slot] = Some(target(**action)?);
    }
    match doors {
        [Some(left), Some(right)] => Ok([left, right]),
        _ => Err(Error::State("policy did not reach both forks".into())),
    }
}
