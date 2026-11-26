//! Solve an explicit bridge model, then search it and an empirical model.
//! Run with `cargo run -p rust-rl --no-default-features --example model_planning`.

use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
use rl::planning::PlanningBudget;
use std::{
    collections::BTreeMap,
    num::{NonZeroU16, NonZeroU32},
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        env::{DiscreteObservationSpace, Env, EpisodeStart, Observation, ObservationSpace, Step},
        reward::Reward,
        transition::Transition as FactualTransition,
    },
    models::predictive::{
        EnumerableModel, OutcomeDistribution, PredictiveModel, Transition, WeightedTransition,
    },
    planning::{finite::FiniteModel, pomcp},
};

const HOME: u64 = 0;
const FAR_BANK: u64 = 1;
const DONE: u64 = 2;
const STOP: usize = 0;
const CROSS: usize = 1;
const CLAIM: usize = 2;
const ACTION_NAMES: [&str; 3] = ["stop", "cross", "claim"];
const ACTION_SPACE: ActionSpace = ActionSpace::Discrete { count: 3 };
const GAMMA: f64 = 0.9;

fn menu(symbol: u64) -> &'static [usize] {
    match symbol {
        HOME => &[STOP, CROSS],
        FAR_BANK => &[CLAIM],
        _ => &[],
    }
}

fn observe(symbol: u64) -> Observation {
    let mask = (0..ACTION_NAMES.len())
        .map(|action| menu(symbol).contains(&action))
        .collect();
    Observation::discrete(symbol, Some(mask))
}

fn public_symbol(frame: &Observation) -> Result<u64> {
    frame
        .as_discrete()
        .map(|observation| observation.symbol)
        .ok_or_else(|| Error::Shape("expected a bridge observation symbol".into()))
}

fn crossing_reward() -> Result<Reward> {
    Reward::from_primitives(vec![-1., 0.])
}

struct Bridge {
    symbol: u64,
    rng: ChaCha8Rng,
    ticks: u64,
}

impl Env for Bridge {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Discrete(DiscreteObservationSpace { symbols: 3 })
    }

    fn action_space(&self) -> &ActionSpace {
        &ACTION_SPACE
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.symbol = HOME;
        Ok(EpisodeStart::fresh(observe(HOME)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        observe(self.symbol).check_action(self.action_space(), action)?;
        let (symbol, reward) = match action {
            Action::Discrete(STOP) => (DONE, Reward::from(2.)),
            Action::Discrete(CROSS) => {
                let next = if self.rng.random_bool(0.8) {
                    FAR_BANK
                } else {
                    DONE
                };
                (next, crossing_reward()?)
            }
            Action::Discrete(CLAIM) => (DONE, Reward::from(10.)),
            _ => return Err(Error::Action("unknown bridge action".into())),
        };
        self.ticks += reward.duration();
        self.symbol = symbol;
        Ok(Step {
            observation: observe(symbol),
            reward,
            terminated: symbol == DONE,
            truncated: false,
        })
    }
}

#[derive(Clone, PartialEq)]
struct Outcome {
    symbol: u64,
    reward: Reward,
    terminated: bool,
}

impl Outcome {
    fn new(symbol: u64, reward: Reward) -> Self {
        Self {
            symbol,
            reward,
            terminated: symbol == DONE,
        }
    }

    fn predict(&self) -> Result<Transition<u64, u64>> {
        let (reward, _) = self.reward.discounted_f64(GAMMA)?;
        let duration = u16::try_from(self.reward.duration())
            .ok()
            .and_then(NonZeroU16::new)
            .ok_or_else(|| Error::State("invalid bridge action duration".into()))?;
        Ok(Transition {
            observation: self.symbol,
            reward,
            duration,
            next: (!self.terminated).then_some(self.symbol),
        })
    }
}

// Each row has an outcome and either its explicit weight or an observed count.
struct BridgeModel(BTreeMap<(u64, usize), Vec<(f64, Outcome)>>);

impl BridgeModel {
    fn explicit() -> Result<Self> {
        let crossing = crossing_reward()?;
        Ok(Self(BTreeMap::from([
            ((HOME, STOP), vec![(1., Outcome::new(DONE, 2.0.into()))]),
            (
                (HOME, CROSS),
                vec![
                    (0.8, Outcome::new(FAR_BANK, crossing.clone())),
                    (0.2, Outcome::new(DONE, crossing)),
                ],
            ),
            (
                (FAR_BANK, CLAIM),
                vec![(1., Outcome::new(DONE, 10.0.into()))],
            ),
        ])))
    }

    fn fit(factual: &[FactualTransition]) -> Result<Self> {
        let mut rows = BTreeMap::<_, Vec<(f64, Outcome)>>::new();
        for record in factual {
            let Action::Discrete(action) = record.action else {
                return Err(Error::Action("expected a discrete bridge action".into()));
            };
            if record.truncated {
                return Err(Error::State(
                    "this example fits only uncensored steps".into(),
                ));
            }
            let context = public_symbol(&record.observation)?;
            let outcome = Outcome {
                symbol: public_symbol(&record.next_observation)?,
                reward: record.reward.clone(),
                terminated: record.terminated,
            };
            let outcomes = rows.entry((context, action)).or_default();
            if let Some((count, _)) = outcomes.iter_mut().find(|(_, seen)| *seen == outcome) {
                *count += 1.;
            } else {
                outcomes.push((1., outcome));
            }
        }
        let fitted = Self(rows);
        for context in [HOME, FAR_BANK] {
            for &action in menu(context) {
                fitted.law(context, action)?;
            }
        }
        Ok(fitted)
    }

    fn law(&self, context: u64, action: usize) -> Result<&[(f64, Outcome)]> {
        self.0
            .get(&(context, action))
            .map(Vec::as_slice)
            .ok_or_else(|| Error::State("no observations for this bridge state and action".into()))
    }
}

impl PlanningModel for BridgeModel {
    type Context = u64;
    type Action = usize;
    type Observation = u64;

    fn actions(&self, context: &u64, actions: &mut Vec<usize>) {
        actions.extend_from_slice(menu(*context));
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: GAMMA,
        }
    }
}

impl PredictiveModel for BridgeModel {
    type Hidden = ();

    fn sample_root(&self, _context: &u64, _rng: &mut ChaCha8Rng) -> Result<()> {
        Ok(())
    }

    fn rollout(&self, context: &u64) -> Option<usize> {
        menu(*context).first().copied()
    }

    fn simulate(
        &self,
        context: &u64,
        _hidden: &mut (),
        action: &usize,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<u64, u64>> {
        let row = self.law(*context, *action)?;
        let index = pomcp::sample_weighted(row.iter().map(|(mass, _)| *mass), rng)?;
        row[index].1.predict()
    }
}

impl EnumerableModel for BridgeModel {
    fn enumerate(
        &self,
        context: &u64,
        action: &usize,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<u64, u64>> {
        let row = self.law(*context, *action)?;
        let total = row.iter().map(|(mass, _)| mass).sum::<f64>();
        let outcomes = row
            .iter()
            .take(cap.get() as usize)
            .map(|(mass, outcome)| {
                Ok(WeightedTransition {
                    probability: mass / total,
                    transition: outcome.predict()?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(OutcomeDistribution {
            outcomes,
            complete: row.len() <= cap.get() as usize,
        })
    }
}

fn factual_step(
    env: &mut Bridge,
    factual: &mut Vec<FactualTransition>,
    observation: Observation,
    action: usize,
) -> Result<Step> {
    let action = Action::Discrete(action);
    let step = env.step(&action)?;
    factual.push(FactualTransition {
        observation,
        action,
        reward: step.reward.clone(),
        next_observation: step.observation.clone(),
        terminated: step.terminated,
        truncated: step.truncated,
    });
    Ok(step)
}

fn search(
    model: &impl PredictiveModel<Context = u64, Action = usize, Observation = u64>,
    root: u64,
) -> Result<pomcp::Search<usize, u64>> {
    pomcp::search(
        model,
        &root,
        menu(root),
        pomcp::Config {
            max_depth: NonZeroU16::new(2).unwrap(),
            gamma: GAMMA,
            exploration: 2.,
        },
        PlanningBudget::Simulations {
            trajectories: NonZeroU32::new(1024).unwrap(),
            transitions: NonZeroU32::new(2048).unwrap(),
        },
        &mut ChaCha8Rng::seed_from_u64(29),
    )
}

#[derive(Debug, PartialEq, Eq)]
struct FactualCounts {
    rows: usize,
    ticks: u64,
    rng_words: u128,
}

impl FactualCounts {
    fn read(env: &Bridge, factual: &[FactualTransition]) -> Self {
        Self {
            rows: factual.len(),
            ticks: env.ticks,
            rng_words: env.rng.get_word_pos(),
        }
    }
}

struct Demo {
    optimal_action: usize,
    optimal_value: f64,
    safe_value: f64,
    learned_cross_probability: f64,
    explicit_search: pomcp::Search<usize, u64>,
    learned_search: pomcp::Search<usize, u64>,
    before_search: FactualCounts,
    after_search: FactualCounts,
    after_execution: FactualCounts,
}

fn run_demo() -> Result<Demo> {
    let mut env = Bridge {
        symbol: DONE,
        rng: ChaCha8Rng::seed_from_u64(7),
        ticks: 0,
    };
    let mut factual = Vec::new();
    for _ in 0..64 {
        for action in [STOP, CROSS] {
            let root = env.reset(None)?.observation;
            let step = factual_step(&mut env, &mut factual, root, action)?;
            if !step.ended() {
                factual_step(&mut env, &mut factual, step.observation, CLAIM)?;
            }
        }
    }

    let explicit = BridgeModel::explicit()?;
    let learned = BridgeModel::fit(&factual)?;
    let cap = NonZeroU32::new(2).unwrap();
    let optimum = FiniteModel::from_model(&explicit, &[HOME, FAR_BANK], cap)?
        .value_iteration(NonZeroU32::new(32).unwrap(), 1e-10)?;
    let learned_cross_probability = learned
        .enumerate(&HOME, &CROSS, cap)?
        .outcomes
        .iter()
        .filter(|row| row.transition.observation == FAR_BANK)
        .map(|row| row.probability)
        .sum();

    let observation = env.reset(None)?.observation;
    let root = public_symbol(&observation)?;
    let before_search = FactualCounts::read(&env, &factual);
    let explicit_search = search(&explicit, root)?;
    let learned_search = search(&learned, root)?;
    let after_search = FactualCounts::read(&env, &factual);

    let selected = learned_search.action.ok_or_else(|| {
        Error::State("the learned bridge model did not recommend an action".into())
    })?;
    factual_step(&mut env, &mut factual, observation, selected)?;
    Ok(Demo {
        optimal_action: *optimum.states[0].greedy_action(),
        optimal_value: optimum.states[0].value,
        safe_value: optimum.states[0].action_values()[0].value,
        learned_cross_probability,
        explicit_search,
        learned_search,
        before_search,
        after_search,
        after_execution: FactualCounts::read(&env, &factual),
    })
}

fn main() -> Result<()> {
    let demo = run_demo()?;
    println!(
        "Exact finite DP chooses {} at {:.2}. Stopping would pay {:.2}.",
        ACTION_NAMES[demo.optimal_action], demo.optimal_value, demo.safe_value
    );
    println!(
        "Factual experience estimates a {:.1}% crossing chance.",
        100. * demo.learned_cross_probability
    );
    for (name, report) in [
        ("Explicit", &demo.explicit_search),
        ("Fitted", &demo.learned_search),
    ] {
        let action = report
            .action
            .map_or("no action", |index| ACTION_NAMES[index]);
        println!(
            "{name} model search chooses {action} after {} imagined transitions.",
            report.work.transitions
        );
    }
    for (phase, counts) in [
        ("Before search", demo.before_search),
        ("After search", demo.after_search),
        ("After one real action", demo.after_execution),
    ] {
        println!(
            "{phase}, there are {} factual rows and {} physical ticks. The environment RNG is at word {}.",
            counts.rows, counts.ticks, counts.rng_words
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_and_fitted_models_plan_through_the_same_sampler() -> rl::Result<()> {
        let demo = run_demo()?;
        assert!((demo.optimal_value - 5.48).abs() < 1e-12);
        assert_eq!(demo.safe_value, 2.);
        assert_eq!(demo.optimal_action, CROSS);
        assert_eq!(demo.explicit_search.action, Some(CROSS));
        assert_eq!(demo.learned_search.action, Some(CROSS));
        assert_eq!(demo.learned_cross_probability, 53. / 64.);
        Ok(())
    }

    #[test]
    fn search_leaves_factual_records_ticks_and_rng_untouched() -> rl::Result<()> {
        let demo = run_demo()?;
        assert!(demo.explicit_search.work.transitions > 0);
        assert!(demo.learned_search.work.transitions > 0);
        assert_eq!(demo.before_search, demo.after_search);
        assert_eq!(demo.after_execution.rows, demo.before_search.rows + 1);
        assert_eq!(demo.after_execution.ticks, demo.before_search.ticks + 2);
        assert!(demo.after_execution.rng_words > demo.before_search.rng_words);
        Ok(())
    }
}
