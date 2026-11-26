use crate::{Result, core::reward::Reward};

/// Physical reward accumulation shared by factual search joins and Monte Carlo
/// supervision. Preserve planner precision until conversion to a neural target.
pub(crate) fn accumulate_rewards<'a>(
    traces: impl IntoIterator<Item = &'a Reward>,
    gamma: f64,
) -> Result<(f64, f64, u64)> {
    crate::core::ensure(
        gamma.is_finite() && (0.0..=1.0).contains(&gamma),
        "invalid factual physical-time discount",
    )?;
    let mut reward = 0.;
    let mut discount = 1.;
    let mut ticks = 0_u64;
    for trace in traces {
        let (immediate, continuation) = trace.discounted_f64(gamma)?;
        reward += discount * immediate;
        discount *= continuation;
        ticks = ticks
            .checked_add(trace.duration())
            .ok_or_else(|| crate::Error::State("factual physical-time overflow".into()))?;
    }
    crate::core::ensure(
        reward.is_finite() && discount.is_finite(),
        "nonfinite factual return",
    )?;
    Ok((reward, discount, ticks))
}

/// Discounted returns with explicit per-transition next-state estimates.
pub fn discounted_returns(
    rewards: &[f32],
    next_values: &[f32],
    terminated: &[bool],
    truncated: &[bool],
    gamma: f32,
) -> Result<Vec<f32>> {
    discounted_returns_with(
        rewards,
        next_values,
        terminated,
        truncated,
        Discounts::Constant(gamma),
    )
}

#[derive(Clone, Copy)]
pub(crate) enum Discounts<'a> {
    Constant(f32),
    PerDecision(&'a [f32]),
}

impl Discounts<'_> {
    #[inline]
    pub(crate) fn at(self, decision: usize) -> f32 {
        match self {
            Self::Constant(gamma) => gamma,
            Self::PerDecision(discounts) => discounts[decision],
        }
    }

    fn validate(self, length: usize) -> Result<()> {
        let valid = |gamma: f32| gamma.is_finite() && (0.0..=1.0).contains(&gamma);
        crate::core::ensure(
            match self {
                Self::Constant(gamma) => valid(gamma),
                Self::PerDecision(discounts) => {
                    discounts.len() == length && discounts.iter().copied().all(valid)
                }
            },
            "invalid trajectory continuation discounts",
        )
    }
}

pub(crate) struct DiscountedRewards {
    pub(crate) rewards: Vec<f32>,
    gamma: f32,
    continuation: Option<Vec<f32>>,
}

impl DiscountedRewards {
    pub(crate) fn new<'a>(
        traces: impl ExactSizeIterator<Item = &'a Reward> + Clone,
        gamma: f32,
    ) -> Result<Self> {
        if traces.clone().all(|trace| matches!(trace, Reward::One(_))) {
            return Ok(Self {
                rewards: traces.map(Reward::total).collect(),
                gamma,
                continuation: None,
            });
        }
        let mut rewards = Vec::with_capacity(traces.len());
        let mut continuation = Vec::with_capacity(traces.len());
        for trace in traces {
            let (reward, discount) = trace.discounted(gamma)?;
            rewards.push(reward);
            continuation.push(discount);
        }
        Ok(Self {
            rewards,
            gamma,
            continuation: Some(continuation),
        })
    }

    pub(crate) fn discounts(&self) -> Discounts<'_> {
        self.continuation
            .as_deref()
            .map_or(Discounts::Constant(self.gamma), Discounts::PerDecision)
    }
}

pub(crate) fn discounted_returns_with(
    rewards: &[f32],
    next_values: &[f32],
    terminated: &[bool],
    truncated: &[bool],
    discounts: Discounts<'_>,
) -> Result<Vec<f32>> {
    validate(rewards, next_values, terminated, truncated, discounts)?;
    let mut result = vec![0.0; rewards.len()];
    let mut tail = 0.0;
    for t in (0..rewards.len()).rev() {
        let continuation = if terminated[t] {
            0.0
        } else if truncated[t] || t + 1 == rewards.len() {
            next_values[t]
        } else {
            tail
        };
        tail = rewards[t] + discounts.at(t) * continuation;
        result[t] = tail;
    }
    Ok(result)
}

pub(crate) fn validate(
    rewards: &[f32],
    next_values: &[f32],
    terminated: &[bool],
    truncated: &[bool],
    discounts: Discounts<'_>,
) -> Result<()> {
    crate::core::ensure(
        [next_values.len(), terminated.len(), truncated.len()]
            .iter()
            .all(|&n| n == rewards.len()),
        "trajectory lengths differ",
    )?;
    discounts.validate(rewards.len())?;
    crate::core::ensure(
        rewards.iter().chain(next_values).all(|x| x.is_finite()),
        "nonfinite trajectory values",
    )
}
