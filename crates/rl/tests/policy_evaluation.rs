use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
use rl::{
    Result,
    models::predictive::{PredictiveModel, Transition},
    planning::{
        guidance::PolicyGuide,
        policy_evaluation::{
            PolicyEvaluationConfig, PolicyEvaluationIneligibility, evaluate_reference_policy,
            improve_reference_policy,
        },
        targets::PolicyTarget,
    },
};
use std::{
    cell::Cell,
    num::{NonZeroU16, NonZeroU32},
};

const GAMMA: f64 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Scan,
    Guess(bool),
}

const ACTIONS: [Action; 3] = [Action::Scan, Action::Guess(false), Action::Guess(true)];

#[derive(Default)]
struct Doors {
    particles: Cell<u32>,
    attempts: Cell<u32>,
    fail_on_attempt: Option<u32>,
}

impl PlanningModel for Doors {
    type Context = Option<bool>;
    type Action = Action;
    type Observation = bool;

    fn actions(&self, public: &Option<bool>, actions: &mut Vec<Action>) {
        actions.extend(if public.is_none() {
            &ACTIONS
        } else {
            &ACTIONS[1..]
        });
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: GAMMA,
        }
    }
}

impl PredictiveModel for Doors {
    type Hidden = bool;

    fn sample_root(&self, public: &Option<bool>, rng: &mut ChaCha8Rng) -> Result<bool> {
        assert_eq!(*public, None, "a continuation must keep the root particle");
        self.particles.set(self.particles.get() + 1);
        Ok(rng.random())
    }

    fn rollout(&self, _: &Option<bool>) -> Option<Action> {
        panic!("use the designated frozen public guide")
    }

    fn simulate(
        &self,
        public: &Option<bool>,
        hidden: &mut bool,
        action: &Action,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Option<bool>, bool>> {
        self.attempts.set(self.attempts.get() + 1);
        if self.fail_on_attempt == Some(self.attempts.get()) {
            return Err(rl::Error::State("predictive transition failed".into()));
        }
        Ok(match action {
            Action::Scan => {
                assert!(public.is_none());
                let reading = if rng.random_bool(0.85) {
                    *hidden
                } else {
                    !*hidden
                };
                Transition {
                    observation: reading,
                    reward: 0.,
                    duration: NonZeroU16::new(2).unwrap(),
                    next: Some(Some(reading)),
                }
            }
            Action::Guess(guess) => Transition {
                observation: *guess == *hidden,
                reward: if *guess == *hidden { GAMMA } else { 0. },
                duration: NonZeroU16::new(2).unwrap(),
                next: None,
            },
        })
    }
}

#[derive(Default)]
struct Reference {
    root_calls: Cell<u32>,
    continuation_calls: Cell<u32>,
}

impl PolicyGuide<Option<bool>, Action> for Reference {
    fn prior(&self, public: &Option<bool>, actions: &[Action]) -> Result<Vec<f64>> {
        if let Some(reading) = public {
            assert_eq!(actions, &ACTIONS[1..]);
            self.continuation_calls
                .set(self.continuation_calls.get() + 1);
            Ok(if *reading {
                vec![0.1, 0.9]
            } else {
                vec![0.9, 0.1]
            })
        } else {
            assert_eq!(actions, ACTIONS);
            self.root_calls.set(self.root_calls.get() + 1);
            Ok(vec![0.5, 0.5, 0.])
        }
    }
}

fn config(samples: u32, max_transitions: u32, termination_bound: u16) -> PolicyEvaluationConfig {
    PolicyEvaluationConfig {
        samples_per_action: NonZeroU32::new(samples).unwrap(),
        max_transitions: NonZeroU32::new(max_transitions).unwrap(),
        termination_bound: NonZeroU16::new(termination_bound).unwrap(),
        epsilon: 0.08,
    }
}

#[test]
fn forced_actions_follow_public_stochastic_continuation_and_keep_each_particle() -> Result<()> {
    let samples = 16_384;
    let model = Doors::default();
    let reference = Reference::default();
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    let evaluated = evaluate_reference_policy(
        &model,
        &None,
        &ACTIONS,
        &reference,
        config(samples, 4 * samples, 2),
        &mut rng,
    )?;
    assert_eq!(evaluated.returns, model.planning_returns());
    assert_eq!(evaluated.work.simulations, 3 * samples);
    assert_eq!(evaluated.completed_trajectories, 3 * samples);
    assert_eq!(evaluated.work.transitions, 4 * samples);
    assert_eq!(model.particles.get(), 3 * samples);
    assert_eq!(model.attempts.get(), 4 * samples);
    assert_eq!(reference.root_calls.get(), 1);
    assert_eq!(reference.continuation_calls.get(), samples);

    let completed = evaluated.outcome.unwrap();
    let exact = [
        GAMMA.powi(3) * (0.85 * 0.9 + 0.15 * 0.1),
        GAMMA * 0.5,
        GAMMA * 0.5,
    ];
    let payoffs = [GAMMA.powi(3), GAMMA, GAMMA];
    for ((actual, expected), payoff) in completed.actions.iter().zip(exact).zip(payoffs) {
        let standard_error = actual.standard_error.unwrap();
        let success = actual.mean / payoff;
        let expected_standard_error =
            payoff * (success * (1. - success) / f64::from(samples - 1)).sqrt();
        assert_eq!(actual.samples, samples);
        assert!((actual.mean - expected).abs() < 3.5 * standard_error);
        assert!((standard_error - expected_standard_error).abs() < 1e-12);
    }
    assert_eq!(
        completed
            .actions
            .iter()
            .map(|row| row.action)
            .collect::<Vec<_>>(),
        ACTIONS
    );
    assert_eq!(completed.reference, [0.5, 0.5, 0.]);
    assert!(completed.improvement.predicted_gain > 0.);
    assert!(completed.improvement.reverse_kl <= 0.08 + 1e-12);
    let PolicyTarget::Probabilities(target) = &completed.improvement.target else {
        unreachable!()
    };
    assert_eq!(target[2], 0.);
    let exact_gain = target
        .iter()
        .zip(completed.reference.iter())
        .zip(exact)
        .map(|((&q, &p), v)| (q - p) * v)
        .sum::<f64>();
    assert!(exact_gain > 0.);
    assert!(completed.gain_standard_error.unwrap() > 0.);
    assert!(completed.gain_lower_bound(0., 1., 0.05)? > 0.);
    Ok(())
}

#[test]
fn real_termination_requires_no_policy_call_even_when_reward_is_zero() -> Result<()> {
    let model = Doors::default();
    let reference = Reference::default();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let evaluated = evaluate_reference_policy(
        &model,
        &None,
        &ACTIONS,
        &reference,
        config(1, 4, 2),
        &mut rng,
    )?;
    let completed = evaluated.outcome.unwrap();
    assert!(completed.actions.iter().any(|row| row.mean == 0.));
    assert_eq!(reference.continuation_calls.get(), 1);
    assert_eq!(evaluated.work.transitions, 4);
    assert!(
        completed
            .actions
            .iter()
            .all(|row| row.standard_error.is_none())
    );
    assert!(completed.gain_standard_error.is_none());
    assert!(completed.gain_lower_bound(0., 1., 0.05)? <= 0.);
    Ok(())
}

#[test]
fn uncompleted_rollouts_and_missing_ordered_support_cannot_create_a_target() -> Result<()> {
    let model = Doors::default();
    let reference = Reference::default();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    for (feasible, config, status, work) in [
        (
            &ACTIONS[..],
            config(2, 1, 2),
            PolicyEvaluationIneligibility::TransitionCap,
            1,
        ),
        (
            &ACTIONS[..],
            config(2, 8, 1),
            PolicyEvaluationIneligibility::TerminationBound,
            1,
        ),
        (
            &[ACTIONS[1], ACTIONS[0], ACTIONS[2]],
            config(2, 8, 2),
            PolicyEvaluationIneligibility::RootSupport,
            0,
        ),
        (
            &ACTIONS[..2],
            config(2, 8, 2),
            PolicyEvaluationIneligibility::RootSupport,
            0,
        ),
    ] {
        let evaluated =
            evaluate_reference_policy(&model, &None, feasible, &reference, config, &mut rng)?;
        assert_eq!(evaluated.outcome.unwrap_err(), status);
        assert_eq!(evaluated.work.transitions, work);
    }
    assert_eq!(model.attempts.get(), 2);
    assert_eq!(model.particles.get(), 2);
    assert_eq!(reference.continuation_calls.get(), 0);
    Ok(())
}

#[test]
fn exact_improvement_retains_small_advantages_and_zero_prior_support() -> Result<()> {
    let budget = 0.25 * 0.5f64.ln() + 0.75 * 1.5f64.ln();
    let prior = [0.5, 0.5, 0.];
    for scale in [1., 1e-50, 1e-300] {
        let improved = improve_reference_policy(&ACTIONS, &prior, &[0., scale, 1e100], budget)?;
        let PolicyTarget::Probabilities(target) = improved.target else {
            unreachable!()
        };
        assert!((target[0] - 0.25).abs() < 1e-10);
        assert!((target[1] - 0.75).abs() < 1e-10);
        assert_eq!(target[2], 0.);
        assert!((improved.reverse_kl - budget).abs() < 1e-10);
        assert!((improved.predicted_gain / scale - 0.25).abs() < 1e-10);
    }
    let identity = improve_reference_policy(&ACTIONS, &prior, &[0., 1., 2.], 0.)?;
    assert_eq!(identity.target, PolicyTarget::Probabilities(prior.to_vec()));
    assert_eq!(identity.predicted_gain, 0.);
    assert!(improve_reference_policy(&ACTIONS, &prior, &[0., 1., f64::NAN], 0.08).is_err());
    Ok(())
}

#[test]
fn failed_model_calls_are_charged_and_discard_the_root() -> Result<()> {
    let model = Doors {
        fail_on_attempt: Some(3),
        ..Doors::default()
    };
    let reference = Reference::default();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let evaluated = evaluate_reference_policy(
        &model,
        &None,
        &ACTIONS,
        &reference,
        config(1, 4, 2),
        &mut rng,
    )?;
    assert_eq!(evaluated.work.transitions, 3);
    assert_eq!(evaluated.work.simulations, 2);
    assert_eq!(evaluated.completed_trajectories, 1);
    assert!(
        matches!(evaluated.outcome, Err(PolicyEvaluationIneligibility::Model(error)) if error.contains("predictive transition failed"))
    );
    Ok(())
}

#[test]
fn planner_revision_covers_the_budget_termination_and_kl_configuration() -> Result<()> {
    let baseline = config(2, 12, 3);
    let revision = baseline.revision()?;
    assert_eq!(revision.revision, 1);
    assert_eq!(revision, baseline.revision()?);
    for alternative in [
        config(3, 12, 3),
        config(2, 16, 3),
        config(2, 12, 4),
        PolicyEvaluationConfig {
            epsilon: 0.12,
            ..baseline
        },
    ] {
        assert_ne!(alternative.revision()?, revision);
    }
    Ok(())
}
