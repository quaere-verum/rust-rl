//! Two ordinary controllers for one partially observed graph environment.
#[cfg(all(feature = "diagnostics", feature = "neural"))]
#[path = "support/structured_control/diagnostics.rs"]
mod diagnostics;
#[path = "support/structured_control.rs"]
mod workflow;

use rl::{
    Result,
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{actor::ActMode, env::Env},
    data::temporal::EpisodeStore,
};
use workflow::{SignalGraph, exact_plan, train_tabular};

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    if arguments.next().as_deref() == Some("--record") {
        #[cfg(all(feature = "diagnostics", feature = "neural"))]
        return diagnostics::cli(arguments);
        #[cfg(not(all(feature = "diagnostics", feature = "neural")))]
        return Err(rl::Error::Config(
            "--record requires the neural and diagnostics features".into(),
        ));
    }
    let first = SignalGraph::new(31);
    let second = SignalGraph::new(37);
    let mut replay =
        EpisodeStore::new(256, first.observation_space(), first.action_space().clone())?;
    let envs = SyncVectorEnv::new(vec![first.clone(), second.clone()])?;
    let mut tabular = StatefulCollector::new(envs, train_tabular(768)?, 41)?;
    tabular.collect(64, ActMode::Deterministic, &mut replay)?;
    let successes = replay
        .episodes()
        .filter(|episode| {
            episode
                .steps
                .last()
                .is_some_and(|step| step.reward.total() == 1.0)
        })
        .count();
    println!(
        "The tabular policy routed {successes} of {} packets correctly.",
        replay.episodes().len(),
    );
    println!(
        "The exact optimal expected delivery reward is {:.2}.",
        exact_plan()?.states[0].value,
    );

    #[cfg(feature = "neural")]
    {
        use workflow::neural::{build_learner, build_trainer};

        let mut learner = build_learner(&first, 43)?;
        let mut trainer = build_trainer(vec![first, second], &learner, 47)?;
        for _ in 0..8 {
            trainer.step(&mut learner)?;
        }
        println!(
            "Graph Candidate DDQN performed {} updates from {} real decisions.",
            learner.updates(),
            trainer.counters().env_steps,
        );
    }
    Ok(())
}
