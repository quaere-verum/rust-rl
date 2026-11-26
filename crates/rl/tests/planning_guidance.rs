use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
use rl::planning::{PlanStatus, PlanningBudget};
use rl::{
    Result,
    core::reward::Reward,
    models::predictive::{PredictiveModel, Transition},
    planning::{
        guidance::{
            PolicyGuide, SearchOptions, ValueCutoff, ValueGuide, search_with_options,
            validate_prior,
        },
        pomcp,
    },
};
use std::{
    cell::Cell,
    num::{NonZeroU16, NonZeroU32},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Scan,
    Guess(bool),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Public {
    reading: Option<bool>,
}
struct TwoDoors;
fn returns() -> PlanningReturns {
    PlanningReturns {
        objective: ReturnObjective::Physical,
        gamma: 0.9,
    }
}
impl PlanningModel for TwoDoors {
    type Context = Public;
    type Action = Action;
    type Observation = bool;

    fn actions(&self, context: &Public, out: &mut Vec<Action>) {
        if context.reading.is_none() {
            out.push(Action::Scan);
        }
        out.extend([Action::Guess(false), Action::Guess(true)]);
    }

    fn planning_returns(&self) -> PlanningReturns {
        returns()
    }
}

impl PredictiveModel for TwoDoors {
    type Hidden = bool;
    fn sample_root(&self, _: &Public, rng: &mut ChaCha8Rng) -> Result<bool> {
        Ok(rng.random())
    }
    fn rollout(&self, _: &Public) -> Option<Action> {
        panic!("guided adapter must use the supplied public guide")
    }
    fn simulate(
        &self,
        context: &Public,
        hidden: &mut bool,
        action: &Action,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Public, bool>> {
        Ok(match action {
            Action::Scan => {
                assert!(context.reading.is_none());
                let reading = if rng.random_bool(0.85) {
                    *hidden
                } else {
                    !*hidden
                };
                Transition {
                    observation: reading,
                    reward: 0.,
                    duration: NonZeroU16::new(2).unwrap(),
                    next: Some(Public {
                        reading: Some(reading),
                    }),
                }
            }
            Action::Guess(guess) => Transition {
                observation: *guess == *hidden,
                reward: f64::from(*guess == *hidden),
                duration: NonZeroU16::new(1).unwrap(),
                next: None,
            },
        })
    }
}
struct Guide {
    policy_calls: Cell<usize>,
    value_calls: Cell<usize>,
}
impl Guide {
    fn new() -> Self {
        Self {
            policy_calls: Cell::new(0),
            value_calls: Cell::new(0),
        }
    }
}
impl PolicyGuide<Public, Action> for Guide {
    fn prior(&self, context: &Public, feasible: &[Action]) -> Result<Vec<f64>> {
        self.policy_calls.set(self.policy_calls.get() + 1);
        let selected = context.reading.map_or(Action::Scan, Action::Guess);
        Ok(feasible
            .iter()
            .map(|action| f64::from(*action == selected))
            .collect())
    }
}
impl ValueGuide<Public> for Guide {
    fn value(&self, context: &Public, remaining: u16, contract: &PlanningReturns) -> Result<f64> {
        self.value_calls.set(self.value_calls.get() + 1);
        assert_eq!(*contract, returns());
        assert!(context.reading.is_some());
        assert_eq!(remaining, 1);
        Ok(0.85)
    }
}
fn config() -> pomcp::Config {
    pomcp::Config {
        max_depth: NonZeroU16::new(2).unwrap(),
        gamma: 0.9,
        exploration: 1.,
    }
}
fn budget() -> PlanningBudget {
    PlanningBudget::Simulations {
        trajectories: NonZeroU32::new(1).unwrap(),
        transitions: NonZeroU32::new(3).unwrap(),
    }
}
fn options(
    guide: &(impl PolicyGuide<Public, Action> + ValueGuide<Public>),
    rollout_policy: bool,
) -> SearchOptions<'_, Public, Action> {
    SearchOptions {
        policy: Some(guide),
        rollout_policy,
        value: (!rollout_policy).then_some(ValueCutoff {
            depth: NonZeroU16::new(1).unwrap(),
            guide,
        }),
        ..SearchOptions::new(returns())
    }
}
#[test]
fn finite_public_leaf_value_has_the_independently_enumerated_physical_discount() -> Result<()> {
    // Uniform hidden door, 85% accurate reading: summing both hidden worlds and
    // both public readings gives .85 success, two physical ticks after the scan.
    let exact: f64 = [false, true]
        .into_iter()
        .flat_map(|hidden| {
            [false, true].map(move |reading| {
                0.5 * if hidden == reading { 0.85 } else { 0.15 } * f64::from(hidden == reading)
            })
        })
        .sum();
    assert!((exact - 0.85).abs() < 1e-15);
    for seed in 0..8 {
        let guide = Guide::new();
        let result = search_with_options(
            &TwoDoors,
            &Public { reading: None },
            &[Action::Scan],
            config(),
            budget(),
            &mut ChaCha8Rng::seed_from_u64(seed),
            options(&guide, false),
        )?;
        assert_eq!(result.status, PlanStatus::Approximate);
        assert_eq!(result.action, Some(Action::Scan));
        assert!((result.estimates[0].value - exact * 0.9_f64.powi(2)).abs() < 1e-15);
        assert_eq!(result.work.transitions, 1);
        assert_eq!(guide.value_calls.get(), 1);
    }
    Ok(())
}
#[test]
fn rollout_policy_uses_only_public_readings_and_terminal_predictions_have_no_leaf_value()
-> Result<()> {
    let root = Public { reading: None };
    for seed in 0..16 {
        let mut reference_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut hidden = TwoDoors.sample_root(&root, &mut reference_rng)?;
        let reading = TwoDoors
            .simulate(&root, &mut hidden, &Action::Scan, &mut reference_rng)?
            .next
            .unwrap();
        let guide = Guide::new();
        // Identical public readings induce identical guidance regardless of the
        // hidden door; the guide has no hidden-state input.
        let support = [Action::Guess(false), Action::Guess(true)];
        let prior = guide.prior(&reading, &support)?;
        assert_eq!(prior[usize::from(reading.reading.unwrap())], 1.);
        let result = search_with_options(
            &TwoDoors,
            &root,
            &[Action::Scan],
            config(),
            budget(),
            &mut ChaCha8Rng::seed_from_u64(seed),
            options(&guide, true),
        )?;
        assert_eq!(
            result.estimates[0].value,
            f64::from(reading.reading.unwrap() == hidden) * 0.9_f64.powi(2)
        );
        assert_eq!(result.work.transitions, 2);
        assert_eq!(guide.value_calls.get(), 0);
    }
    Ok(())
}
#[test]
fn contracts_and_action_priors_are_rejected_before_consuming_the_planner_stream() -> Result<()> {
    let guide = Guide::new();
    for mismatch in 0..5 {
        let mut advice = options(&guide, false);
        let mut search_config = config();
        match mismatch {
            0 => advice.returns.gamma = 1.,
            1 => advice.returns.objective = ReturnObjective::Surrogate([7; 32]),
            2 => advice.value.as_mut().unwrap().depth = search_config.max_depth,
            3 => {
                advice.rollout_policy = true;
                advice.policy = None;
            }
            _ => search_config.gamma = 0.5,
        }
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let before = serde_json::to_value(&rng)?;
        assert!(
            search_with_options(
                &TwoDoors,
                &Public { reading: None },
                &[Action::Scan],
                search_config,
                budget(),
                &mut rng,
                advice
            )
            .is_err()
        );
        assert_eq!(before, serde_json::to_value(&rng)?);
    }
    let support = [Action::Guess(false), Action::Guess(true)];
    for bad in [
        vec![1.],
        vec![0.4, 0.4],
        vec![1.1, -0.1],
        vec![f64::NAN, 0.],
        vec![f64::INFINITY, 0.],
    ] {
        assert!(validate_prior(&support, &bad).is_err());
    }
    assert!(validate_prior(&[Action::Scan, Action::Scan], &[0.5, 0.5]).is_err());
    validate_prior(&support, &[0.5, 0.5])?;
    Ok(())
}
#[test]
fn physical_reward_precision_preserves_existing_f32_returns() -> Result<()> {
    let reward = Reward::from_primitives(vec![1., 2., 3.])?;
    let (value, discount) = reward.discounted_f64(0.5)?;
    assert_eq!((value, discount), (2.75, 0.125));
    for gamma in [0., 0.5, 0.9, 1.] {
        let (wide, wide_discount) = reward.discounted_f64(f64::from(gamma))?;
        assert_eq!(
            reward.discounted(gamma)?,
            (wide as f32, wide_discount as f32)
        );
    }
    assert!(reward.discounted_f64(f64::NAN).is_err());
    Ok(())
}

struct ConstantGuide {
    invalid_prior: bool,
    value: f64,
}
impl PolicyGuide<Public, Action> for ConstantGuide {
    fn prior(&self, context: &Public, feasible: &[Action]) -> Result<Vec<f64>> {
        if self.invalid_prior && context.reading.is_some() {
            Ok(vec![1.; feasible.len()])
        } else {
            Ok(vec![1. / feasible.len() as f64; feasible.len()])
        }
    }
}
impl ValueGuide<Public> for ConstantGuide {
    fn value(&self, _: &Public, _: u16, _: &PlanningReturns) -> Result<f64> {
        Ok(self.value)
    }
}
#[test]
fn invalid_leaf_guides_preserve_failed_search_work_and_rng() -> Result<()> {
    for rollout_policy in [false, true] {
        let guide = ConstantGuide {
            invalid_prior: true,
            value: f64::NAN,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        let before = serde_json::to_value(&rng)?;
        let result = search_with_options(
            &TwoDoors,
            &Public { reading: None },
            &[Action::Scan],
            config(),
            budget(),
            &mut rng,
            options(&guide, rollout_policy),
        )?;
        assert_eq!(result.status, PlanStatus::Failed);
        assert_eq!(result.work.transitions, 1);
        assert_eq!(result.work.simulations, 0);
        assert!(result.action.is_none());
        assert_eq!(before, serde_json::to_value(&rng)?);
    }
    Ok(())
}
#[test]
fn maximum_prior_ties_keep_model_action_order_without_a_second_random_stream() -> Result<()> {
    for seed in 0..8 {
        let mut reference = ChaCha8Rng::seed_from_u64(seed);
        let mut hidden = TwoDoors.sample_root(&Public { reading: None }, &mut reference)?;
        TwoDoors.simulate(
            &Public { reading: None },
            &mut hidden,
            &Action::Scan,
            &mut reference,
        )?;
        let guide = ConstantGuide {
            invalid_prior: false,
            value: 0.,
        };
        let mut actual = ChaCha8Rng::seed_from_u64(seed);
        let result = search_with_options(
            &TwoDoors,
            &Public { reading: None },
            &[Action::Scan],
            config(),
            budget(),
            &mut actual,
            options(&guide, true),
        )?;
        // Guess(false) occurs first in the model's feasible order.
        assert_eq!(
            result.estimates[0].value,
            f64::from(!hidden) * 0.9_f64.powi(2)
        );
        assert_eq!(
            serde_json::to_value(actual)?,
            serde_json::to_value(reference)?
        );
    }
    Ok(())
}

#[test]
fn finite_pomdp_search_supervision_joins_real_duration_rewards_without_anticipation() -> Result<()>
{
    use rl::{
        core::{
            action::{Action as PhysicalAction, ActionSpace},
            env::{EpisodeStart, Observation},
        },
        data::temporal::{BehaviorMetadata, EpisodeStore, TemporalSink, TemporalTransition},
        planning::targets::{
            FactualBinding, PolicyTarget, SearchProvenance, SearchRevision, SearchTarget,
            factual_return,
        },
    };
    let root = Public { reading: None };
    let initial = Observation::dense(vec![-1.], Some(vec![true, true, true]));
    let after = Observation::dense(vec![1.], Some(vec![false, true, true]));
    let mut store = EpisodeStore::new(8, 1, ActionSpace::Discrete { count: 3 })?;
    store.begin_episode(0, &EpisodeStart::fresh(initial.clone()))?;
    let binding = FactualBinding::capture([17; 32], &store, 0, 0)?;
    let guide = Guide::new();
    let search = search_with_options(
        &TwoDoors,
        &root,
        &[Action::Scan],
        config(),
        budget(),
        &mut ChaCha8Rng::seed_from_u64(5),
        options(&guide, false),
    )?;
    let provenance = SearchProvenance {
        model: SearchRevision {
            identity: [1; 32],
            revision: 1,
        },
        planner: SearchRevision {
            identity: [2; 32],
            revision: 1,
        },
        policy: Some(SearchRevision {
            identity: [3; 32],
            revision: 1,
        }),
    };
    let mut target = SearchTarget::<Public, Action, bool> {
        version: 1,
        public: root,
        binding,
        feasible: vec![Action::Scan],
        policy: Some(PolicyTarget::from_estimates(
            &[Action::Scan],
            &search.estimates,
        )?),
        value: Some(search.estimates[0].value),
        returns: returns(),
        horizon: config().max_depth,
        factual_return: None,
        provenance: provenance.clone(),
        budget: budget(),
        work: search.work,
        status: search.status,
        continuations: vec![],
    };
    // Only now does the real trajectory reveal its reading and terminal reward.
    store.push(&TemporalTransition {
        episode_id: 0,
        timestep: 0,
        observation: initial,
        action: PhysicalAction::Discrete(0),
        reward: Reward::from_primitives(vec![0., 0.])?,
        next_observation: after.clone(),
        terminated: false,
        truncated: false,
        behavior: BehaviorMetadata::default(),
    })?;
    store.push(&TemporalTransition {
        episode_id: 0,
        timestep: 1,
        observation: after.clone(),
        action: PhysicalAction::Discrete(2),
        reward: 1_f32.into(),
        next_observation: after,
        terminated: true,
        truncated: false,
        behavior: BehaviorMetadata::default(),
    })?;
    target.factual_return = Some(factual_return(&target.binding, [17; 32], &store, 2, 0.9)?);
    let row = target.join([17; 32], &store, &provenance, |prefix| {
        assert!(prefix.steps().is_empty());
        assert_eq!(prefix.observation().dense_values(), &[-1.]);
        Ok((Public { reading: None }, vec![Action::Scan]))
    })?;
    assert_eq!(row.reward.duration(), 2);
    assert_eq!(store.len(), 2); // Only the two actual transitions, no search rollouts.
    assert!((target.value.unwrap() - 0.85 * 0.9_f64.powi(2)).abs() < 1e-15);
    assert_eq!(
        target.factual_return.as_ref().unwrap().reward,
        0.9_f64.powi(2)
    );
    assert_eq!(target.factual_return.as_ref().unwrap().physical_ticks, 3);
    Ok(())
}
