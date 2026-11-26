use super::*;
use crate::planning::guidance::search_with_strategy;

#[derive(Clone, Copy)]
enum Public {
    Root,
    Live,
}

struct Chance {
    terminal: Cell<u32>,
}

impl PlanningModel for Chance {
    type Context = Public;
    type Action = u8;
    type Observation = u8;

    fn actions(&self, context: &Public, actions: &mut Vec<u8>) {
        actions.extend(match context {
            Public::Root => [0, 1],
            Public::Live => [2, 3],
        });
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 0.5,
        }
    }
}

impl PredictiveModel for Chance {
    type Hidden = ();

    fn sample_root(&self, _: &Public, _: &mut ChaCha8Rng) -> Result<()> {
        Ok(())
    }

    fn rollout(&self, _: &Public) -> Option<u8> {
        Some(2)
    }

    fn simulate(
        &self,
        _: &Public,
        _: &mut (),
        action: &u8,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Public, u8>> {
        let (observation, reward, duration, next) = match action {
            0 if rng.random_bool(0.25) => {
                self.terminal.set(self.terminal.get() + 1);
                (0, 4., 1, None)
            }
            0 => (1, -1., 2, Some(Public::Live)),
            1 => (2, -0.25, 1, None),
            2 => (3, -10., 1, None),
            3 => (4, -4., 1, None),
            _ => unreachable!(),
        };
        Ok(Transition {
            observation,
            reward,
            duration: NonZeroU16::new(duration).unwrap(),
            next,
        })
    }
}

#[test]
fn bellman_backup_maximizes_child_actions_and_averages_chance() {
    let model = Chance {
        terminal: Cell::new(0),
    };
    let result = search_with_strategy(
        &model,
        &Public::Root,
        &[0, 1],
        Config {
            max_depth: NonZeroU16::new(2).unwrap(),
            gamma: 0.5,
            exploration: 2.,
        },
        budget(512, 1024),
        &mut ChaCha8Rng::seed_from_u64(7),
        SearchOptions::new(model.planning_returns()).with_strategy(SearchStrategy {
            backup: Backup::Bellman,
            ..SearchStrategy::default()
        }),
    )
    .unwrap();
    let probe = &result.estimates[0];
    let frequency = f64::from(model.terminal.get()) / f64::from(probe.visits.unwrap());
    assert!(frequency > 0. && frequency < 1.);
    // The live branch is -1 + 0.5^2 * max(-10, -4) = -2.
    let expected = frequency * 4. + (1. - frequency) * -2.;
    assert!((probe.value - expected).abs() < 1e-12, "{result:?}");
    assert_eq!(result.action, Some(1));
    assert_eq!(result.work.simulations, 512);
}

#[test]
fn halving_allocates_completed_rounds_and_keeps_stable_ties() {
    struct TiedArms;
    impl PlanningModel for TiedArms {
        type Context = ();
        type Action = u8;
        type Observation = ();

        fn actions(&self, _: &(), actions: &mut Vec<u8>) {
            actions.extend(0..5);
        }

        fn planning_returns(&self) -> PlanningReturns {
            TwoDoors.planning_returns()
        }
    }
    impl PredictiveModel for TiedArms {
        type Hidden = ();

        fn sample_root(&self, _: &(), _: &mut ChaCha8Rng) -> Result<()> {
            Ok(())
        }

        fn rollout(&self, _: &()) -> Option<u8> {
            None
        }

        fn simulate(
            &self,
            _: &(),
            _: &mut (),
            _: &u8,
            _: &mut ChaCha8Rng,
        ) -> Result<Transition<(), ()>> {
            Ok(Transition {
                observation: (),
                reward: 0.,
                duration: NonZeroU16::MIN,
                next: None,
            })
        }
    }

    let result = search_with_strategy(
        &TiedArms,
        &(),
        &[4, 2, 1, 0, 3],
        config(),
        budget(22, 22),
        &mut ChaCha8Rng::seed_from_u64(4),
        SearchOptions::new(TiedArms.planning_returns()).with_strategy(SearchStrategy {
            root_selection: RootSelection::SequentialHalving,
            ..SearchStrategy::default()
        }),
    )
    .unwrap();
    let visits = result
        .estimates
        .iter()
        .map(|row| row.visits)
        .collect::<Vec<_>>();
    assert_eq!(visits, [Some(9), Some(8), Some(3), Some(1), Some(1)]);
    assert_eq!(result.action, Some(4));
    assert_eq!(result.work.simulations, 22);
    assert_eq!(result.work.transitions, 22);
}

#[test]
fn bellman_cutoff_only_bootstraps_live_outcomes_in_physical_time() {
    struct Leaf;
    impl ValueGuide<Public> for Leaf {
        fn value(&self, context: &Public, remaining: u16, _: &PlanningReturns) -> Result<f64> {
            assert!(matches!(context, Public::Live));
            assert_eq!(remaining, 1);
            Ok(-7.)
        }
    }

    let model = Chance {
        terminal: Cell::new(0),
    };
    let mut configured = SearchOptions::new(model.planning_returns());
    configured.value = Some(ValueCutoff {
        depth: NonZeroU16::MIN,
        guide: &Leaf,
    });
    let result = search_with_strategy(
        &model,
        &Public::Root,
        &[0],
        Config {
            max_depth: NonZeroU16::new(2).unwrap(),
            gamma: 0.5,
            exploration: 0.,
        },
        budget(64, 64),
        &mut ChaCha8Rng::seed_from_u64(7),
        configured.with_strategy(SearchStrategy {
            backup: Backup::Bellman,
            ..SearchStrategy::default()
        }),
    )
    .unwrap();
    let frequency = f64::from(model.terminal.get()) / 64.;
    assert!(frequency > 0. && frequency < 1.);
    let expected = frequency * 4. + (1. - frequency) * -2.75;
    assert!((result.estimates[0].value - expected).abs() < 1e-12);
    assert_eq!(result.work.simulations, 64);
    assert_eq!(result.work.transitions, 64);
    assert!(result.continuations.is_empty());
}

#[test]
fn halving_does_not_count_interrupted_returns() {
    let result = search_with_strategy(
        &TwoDoors,
        &Context::Root,
        &[Action::Guess(true), Action::Scan, Action::Guess(false)],
        config(),
        budget(8, 2),
        &mut ChaCha8Rng::seed_from_u64(4),
        options().with_strategy(SearchStrategy {
            backup: Backup::Bellman,
            root_selection: RootSelection::SequentialHalving,
        }),
    )
    .unwrap();
    assert_eq!(result.status, PlanStatus::Exhausted);
    assert_eq!(result.work.simulations, 1);
    assert_eq!(result.work.transitions, 2);
    assert_eq!(result.action, Some(Action::Guess(true)));
    assert_eq!(result.estimates.len(), 1);
}
