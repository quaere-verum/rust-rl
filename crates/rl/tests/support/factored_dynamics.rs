use std::num::{NonZeroU16, NonZeroU32};

use rl::{
    Result,
    models::{
        factored::{FactoredModel, FitConfig, FitLimits},
        predictive::{
            EnumerableModel, OutcomeDistribution, PlanningModel, PlanningReturns, ReturnObjective,
            Transition, WeightedTransition,
        },
    },
    planning::finite::FiniteModel,
};

pub const VARIABLES: usize = 8;
pub const STATES: usize = 1 << VARIABLES;
pub const ACTIONS: usize = 4;
pub const HORIZON: NonZeroU16 = NonZeroU16::new(4).unwrap();
pub const LIMITS: FitLimits = FitLimits {
    max_parents: 2,
    max_candidates: 296,
    max_total_cells: 8_256,
    max_work: 2_636_800,
};

pub const fn config(pseudocount: f64) -> FitConfig {
    FitConfig {
        pseudocount,
        limits: LIMITS,
    }
}

pub fn bits(state: usize) -> [usize; VARIABLES] {
    std::array::from_fn(|factor| (state >> factor) & 1)
}

pub fn encode(state: &[usize; VARIABLES]) -> usize {
    state
        .iter()
        .enumerate()
        .map(|(factor, &value)| value << factor)
        .sum()
}

pub struct DenseLaw(pub Vec<[f64; STATES]>);

impl DenseLaw {
    pub fn from_rows(mut law: impl FnMut(usize, usize) -> Result<[f64; STATES]>) -> Result<Self> {
        let mut rows = Vec::with_capacity(STATES * ACTIONS);
        for state in 0..STATES {
            for action in 0..ACTIONS {
                rows.push(law(state, action)?);
            }
        }
        Ok(Self(rows))
    }

    pub fn from_factored(model: &FactoredModel) -> Result<Self> {
        let assignments: [_; STATES] = std::array::from_fn(bits);
        Self::from_rows(|state, action| {
            let marginals = model.predict_factors(&assignments[state], action)?;
            Ok(std::array::from_fn(|next| {
                marginals
                    .iter()
                    .zip(assignments[next])
                    .map(|(row, value)| row[value])
                    .product()
            }))
        })
    }

    pub fn prepare(&self) -> Result<FiniteModel<usize, usize>> {
        let contexts: [_; STATES] = std::array::from_fn(|index| index);
        FiniteModel::for_horizon(self, &contexts, NonZeroU32::new(STATES as u32).unwrap())
    }
}

impl PlanningModel for DenseLaw {
    type Context = usize;
    type Action = usize;
    type Observation = usize;

    fn actions(&self, _: &usize, actions: &mut Vec<usize>) {
        actions.extend(0..ACTIONS);
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 1.,
        }
    }
}

impl EnumerableModel for DenseLaw {
    fn enumerate(
        &self,
        state: &usize,
        action: &usize,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<usize, usize>> {
        let mut positive = self.0[state * ACTIONS + action]
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, p)| p > 0.);
        let outcomes = positive
            .by_ref()
            .take(cap.get() as usize)
            .map(|(next, probability)| WeightedTransition {
                probability,
                transition: Transition {
                    observation: next,
                    reward: f64::from(next.count_ones()) / VARIABLES as f64,
                    duration: NonZeroU16::MIN,
                    next: Some(next),
                },
            })
            .collect();
        Ok(OutcomeDistribution {
            outcomes,
            complete: positive.next().is_none(),
        })
    }
}
