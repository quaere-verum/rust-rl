use rl::{
    Error, Result,
    planning::execution::{Continuation, ReplanReason, resolve_continuation},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    Inspect,
    Wait,
}

#[derive(Debug, PartialEq, Eq)]
struct Route {
    destination: u8,
    energy: u16,
}

fn branches() -> Vec<(bool, Route)> {
    vec![
        (
            false,
            Route {
                destination: 1,
                energy: 2,
            },
        ),
        (
            true,
            Route {
                destination: 2,
                energy: 3,
            },
        ),
    ]
}

#[test]
fn factual_outcome_selects_only_a_currently_feasible_public_continuation() -> Result<()> {
    for (outcome, destination, energy) in [(false, 1, 2), (true, 2, 3)] {
        let next = resolve_continuation(
            &Action::Inspect,
            &Action::Inspect,
            &outcome,
            branches(),
            |route| Ok(route.energy <= 3),
        )?;
        assert_eq!(
            next,
            Continuation::Continue(Route {
                destination,
                energy
            })
        );
    }
    // The same reading no longer authorizes the route when public resources
    // change. The caller must replan at this ordinary actor decision boundary.
    assert_eq!(
        resolve_continuation(
            &Action::Inspect,
            &Action::Inspect,
            &true,
            branches(),
            |route| { Ok(route.energy <= 2) }
        )?,
        Continuation::Replan(ReplanReason::Infeasible),
    );
    Ok(())
}

#[test]
fn substituted_action_or_missing_branch_never_uses_an_imagined_continuation() -> Result<()> {
    assert_eq!(
        resolve_continuation(&Action::Inspect, &Action::Wait, &true, branches(), |_| {
            panic!("different factual action cannot authorize a continuation")
        })?,
        Continuation::Replan(ReplanReason::DifferentAction),
    );
    assert_eq!(
        resolve_continuation(
            &Action::Inspect,
            &Action::Inspect,
            &true,
            vec![(
                false,
                Route {
                    destination: 1,
                    energy: 2
                }
            )],
            |_| { panic!("unobserved branch cannot authorize a continuation") }
        )?,
        Continuation::Replan(ReplanReason::UnmatchedOutcome),
    );
    Ok(())
}

#[test]
fn public_feasibility_error_is_not_silently_converted_to_a_valid_plan() {
    let result = resolve_continuation(
        &Action::Inspect,
        &Action::Inspect,
        &true,
        branches(),
        |_| Err(Error::State("invalid public resource state".into())),
    );
    assert!(
        matches!(result, Err(Error::State(message)) if message == "invalid public resource state")
    );
}
