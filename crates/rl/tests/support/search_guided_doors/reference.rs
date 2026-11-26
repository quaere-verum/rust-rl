//! Evaluation-only finite-belief dynamic programming. Neither the environment,
//! predictive model nor learner imports this module. Enumerates both hidden
//! worlds and sensor outcomes independently of the model's integer update.
use super::domain::{GAMMA, Public, SCAN, guess};
use rl::{Result, core::action::ConditionalAction};

pub fn posterior_right(public: Public) -> f64 {
    let prior = f64::from(public.task.prior_milli) / 1000.;
    match public.reading {
        None => prior,
        Some(reading) => {
            let right = prior * if reading { 0.85 } else { 0.15 };
            let left = (1. - prior) * if reading { 0.15 } else { 0.85 };
            right / (right + left)
        }
    }
}
pub fn sensor_probability(public: Public, reading: bool) -> f64 {
    let prior = posterior_right(public);
    prior * if reading { 0.85 } else { 0.15 } + (1. - prior) * if reading { 0.15 } else { 0.85 }
}
pub fn action_values(public: Public) -> Vec<(ConditionalAction, f64)> {
    let right = posterior_right(public);
    let mut values = Vec::new();
    if public.reading.is_none() {
        let mut continuation = 0.;
        for reading in [false, true] {
            let after = Public {
                reading: Some(reading),
                ..public
            };
            continuation += sensor_probability(public, reading) * best_value(after);
        }
        values.push((
            SCAN,
            -f64::from(public.task.cost()) + GAMMA.powi(2) * continuation,
        ));
    }
    values.extend([
        (guess(false), 1. - 2. * right),
        (guess(true), 2. * right - 1.),
    ]);
    values
}
pub fn best_value(public: Public) -> f64 {
    action_values(public)
        .into_iter()
        .map(|(_, v)| v)
        .fold(f64::NEG_INFINITY, f64::max)
}
pub fn optimal_action(public: Public) -> ConditionalAction {
    action_values(public)
        .into_iter()
        .reduce(|best, next| if next.1 > best.1 + 1e-10 { next } else { best })
        .unwrap()
        .0
}
pub fn deployed_value(
    public: Public,
    mut policy: impl FnMut(Public) -> Result<ConditionalAction>,
) -> Result<f64> {
    fn immediate(public: Public, action: ConditionalAction) -> Result<f64> {
        let right = posterior_right(public);
        if action == guess(true) {
            Ok(2. * right - 1.)
        } else if action == guess(false) {
            Ok(1. - 2. * right)
        } else {
            Err(rl::Error::Action(
                "evaluation policy returned an unavailable guess".into(),
            ))
        }
    }
    let action = policy(public)?;
    if action != SCAN {
        return immediate(public, action);
    }
    if public.reading.is_some() {
        return Err(rl::Error::Action("evaluation policy repeated scan".into()));
    }
    let mut value = -f64::from(public.task.cost());
    for reading in [false, true] {
        let after = Public {
            reading: Some(reading),
            ..public
        };
        value +=
            GAMMA.powi(2) * sensor_probability(public, reading) * immediate(after, policy(after)?)?;
    }
    Ok(value)
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Outcome {
    pub expected_return: f64,
    pub success: f64,
    pub success_ticks: f64,
    pub scan_probability: f64,
}
impl Outcome {
    pub fn add_weighted(&mut self, weight: f64, other: Self) {
        self.expected_return += weight * other.expected_return;
        self.success += weight * other.success;
        self.success_ticks += weight * other.success_ticks;
        self.scan_probability += weight * other.scan_probability;
    }
}

pub struct FrozenEvaluation {
    pub actions: Vec<(ConditionalAction, Outcome)>,
    pub deployed: Outcome,
}

/// Exact first-action intervention with the supplied complete-action law on
/// subsequent observations. Only the finite public sensor law is integrated.
pub fn evaluate_policy(
    public: Public,
    mut law: impl FnMut(Public) -> Result<Vec<(ConditionalAction, f64)>>,
) -> Result<FrozenEvaluation> {
    fn guesses(public: Public) -> [(ConditionalAction, Outcome); 2] {
        let right = posterior_right(public);
        [(guess(false), 1. - right), (guess(true), right)].map(|(action, success)| {
            (
                action,
                Outcome {
                    expected_return: 2. * success - 1.,
                    success,
                    success_ticks: success,
                    scan_probability: 0.,
                },
            )
        })
    }

    fn mix(
        support: &[(ConditionalAction, Outcome)],
        law: Vec<(ConditionalAction, f64)>,
    ) -> Result<Outcome> {
        if !support
            .iter()
            .map(|row| row.0)
            .eq(law.iter().map(|row| row.0))
        {
            return Err(rl::Error::Action(
                "door reference law differs from ordered legal actions".into(),
            ));
        }
        let actions: Vec<_> = support.iter().map(|row| row.0).collect();
        let probabilities: Vec<_> = law.iter().map(|row| row.1).collect();
        rl::planning::guidance::validate_prior(&actions, &probabilities)?;
        let mut result = Outcome::default();
        for ((_, outcome), probability) in support.iter().zip(probabilities) {
            result.add_weighted(probability, *outcome);
        }
        Ok(result)
    }

    let mut actions = Vec::with_capacity(3);
    if public.reading.is_none() {
        let mut continuation = Outcome::default();
        for reading in [false, true] {
            let after = Public {
                reading: Some(reading),
                ..public
            };
            let value = mix(&guesses(after), law(after)?)?;
            continuation.add_weighted(sensor_probability(public, reading), value);
        }
        actions.push((
            SCAN,
            Outcome {
                expected_return: -f64::from(public.task.cost())
                    + GAMMA.powi(2) * continuation.expected_return,
                success: continuation.success,
                success_ticks: continuation.success_ticks + 2. * continuation.success,
                scan_probability: 1.,
            },
        ));
    }
    actions.extend(guesses(public));
    let deployed = mix(&actions, law(public)?)?;
    Ok(FrozenEvaluation { actions, deployed })
}
