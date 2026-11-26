//! Independent finite-horizon numerical benchmark for alpha execution.
use super::Config;
use nalgebra::{DMatrix, SymmetricEigen};
use rand_chacha::ChaCha8Rng;
use rl::{Error, Result};
use rl::{
    diagnostics::{
        reference::{
            ConfigValue, ReferenceCheckpoint, ReferenceContext, ReferenceHorizon,
            ReferenceInformation, ReferenceKind, ReferenceLifetime, ReferenceObjective,
            ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
        },
        value::ValueConventions,
    },
    prelude::{ActMode, Action, ActionSpace, Actor, Observation, ObservationBatch},
};
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub(super) struct Grid {
    pub(super) inventory: usize,
    pub(super) alpha: usize,
    pub(super) actions: usize,
    pub(super) quadrature: usize,
    pub(super) alpha_limit: f64,
}

/// The fixed numerical policy used for both recordings and episode evaluation.
pub(super) fn demonstration_grid() -> Grid {
    Grid {
        inventory: 31,
        alpha: 161,
        actions: 61,
        quadrature: 9,
        alpha_limit: 2.,
    }
}

fn gaussian_quadrature(count: usize) -> Result<Vec<(f64, f64)>> {
    if !(1..=15).contains(&count) {
        return Err(Error::Config(
            "quadrature order must be 1 through 15".into(),
        ));
    }
    let mut jacobi = DMatrix::zeros(count, count);
    for i in 1..count {
        let coefficient = (i as f64).sqrt();
        jacobi[(i - 1, i)] = coefficient;
        jacobi[(i, i - 1)] = coefficient;
    }
    let eigen = SymmetricEigen::new(jacobi);
    let mut result: Vec<_> = (0..count)
        .map(|i| (eigen.eigenvalues[i], eigen.eigenvectors[(0, i)].powi(2)))
        .collect();
    result.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(result)
}

struct NumericalReference {
    config: Config,
    grid: Grid,
    quadrature: Vec<(f64, f64)>,
    values: Vec<f64>,
    /// Mean clamped quadrature mass under uniform enumerated grid states/actions;
    /// this is a numerical boundary diagnostic, not a policy visitation probability.
    clamped_mass_mean: f64,
}

pub(super) struct NumericalProvider {
    reference: NumericalReference,
    specification: ReferenceSpec,
    space: ActionSpace,
}

impl NumericalProvider {
    pub(super) fn new(config: Config, grid: Grid) -> Result<Self> {
        let space = ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        };
        let specification = ReferenceSpec {
            id: "alpha-grid-dp".into(),
            context: context(config, space.clone()),
            quality: ReferenceQuality {
                kind: ReferenceKind::ApproximateNumerical,
                method: "Finite-horizon grid dynamic programming with Gaussian quadrature, linear interpolation and discrete target search".into(),
                assumptions: vec![
                    "Values and decisions are approximate; they are not certified continuous optima.".into(),
                    "Signal innovations outside the grid use the nearest boundary in numerical continuation; the deployed reference clips signal inputs to that domain.".into(),
                    "Predicted values describe the grid-DP benchmark continuation, not the historical learner continuation.".into(),
                ],
            },
            target: ReferenceTarget::BenchmarkPolicy,
            lifetime: ReferenceLifetime::Fixed,
            information: ReferenceInformation::LearnerObservation,
            uncertainty: Some("Grid, action, quadrature and boundary approximation; no certified error bound".into()),
        };
        specification.validate()?;
        Ok(Self {
            reference: NumericalReference::solve(config, grid, false)?,
            specification,
            space,
        })
    }

    fn coordinates(&self, observation: &Observation) -> Result<(usize, f64, f64)> {
        if let &[q, alpha, remaining] = observation.dense_values()
            && q.is_finite()
            && alpha.is_finite()
            && remaining.is_finite()
            && (-1. ..=1.).contains(&q)
            && (0. ..=1.).contains(&remaining)
        {
            let horizon = self.reference.config.horizon;
            let remaining_steps = f64::from(remaining) * horizon as f64;
            let rounded = remaining_steps.round();
            if (remaining_steps - rounded).abs() <= 1e-4 {
                return Ok((horizon - rounded as usize, f64::from(q), f64::from(alpha)));
            }
        }
        Err(Error::State(
            "invalid alpha-execution reference observation".into(),
        ))
    }
}

pub(super) fn context(config: Config, space: ActionSpace) -> ReferenceContext {
    ReferenceContext {
        environment: "alpha_execution_v1".into(), observation_dim: 3, typed_observation_space: None, action_space: space,
        conventions: ValueConventions {
            dynamics: "Independent Gaussian alpha and price innovations; execution depletes alpha before the price move".into(),
            observation: "Signed inventory, available alpha and remaining decision fraction".into(),
            action: "Target inventory in [-1, 1]; signed executed trade is target minus inventory".into(),
            reward_units: "normalized P&L".into(),
            reward_convention: "Post-trade inventory earns the next price change, less execution costs and final forced liquidation cost".into(),
            discount: 1., horizon: format!("{} decisions", config.horizon),
            terminal_treatment: "At the final decision, the post-trade position earns the final price move and is then liquidated exactly once".into(),
            action_constraints: "Target inventory in [-1, 1]".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: config.horizon as u32 },
        environment_config: BTreeMap::from([
            ("version".into(), ConfigValue::Integer(1)), ("horizon".into(), ConfigValue::Integer(config.horizon as i64)),
            ("dt".into(), ConfigValue::Real(config.dt)), ("eta".into(), ConfigValue::Real(config.eta)),
            ("kappa".into(), ConfigValue::Real(config.kappa)), ("sigma_alpha".into(), ConfigValue::Real(config.sigma_alpha)),
            ("sigma_price".into(), ConfigValue::Real(config.sigma_price)), ("spread".into(), ConfigValue::Real(config.spread)),
            ("impact".into(), ConfigValue::Real(config.impact)),
        ]),
        markov_observation: true,
        decision_time_coordinate: Some(2),
    }
}

impl Actor for NumericalProvider {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.clear();
        for observation in observations.iter() {
            let (period, q, alpha) = self.coordinates(observation)?;
            let alpha = alpha.clamp(
                -self.reference.grid.alpha_limit,
                self.reference.grid.alpha_limit,
            );
            actions.push(Action::Continuous(vec![
                self.reference.action(period, q, alpha)? as f32,
            ]));
        }
        Ok(())
    }
}

impl ReferenceProvider for NumericalProvider {
    fn specification(&self) -> &ReferenceSpec {
        &self.specification
    }
    fn actor(&self) -> Option<&dyn Actor> {
        Some(self)
    }
    fn evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }
    fn state_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
    ) -> Result<Option<f64>> {
        let (period, q, alpha) = self.coordinates(observation)?;
        if period >= self.reference.config.horizon || alpha.abs() > self.reference.grid.alpha_limit
        {
            return Ok(None);
        }
        self.reference.value(period, q, alpha).map(Some)
    }
    fn action_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<f64>> {
        let (period, q, alpha) = self.coordinates(observation)?;
        if period >= self.reference.config.horizon || alpha.abs() > self.reference.grid.alpha_limit
        {
            return Ok(None);
        }
        self.space.check(action, None)?;
        let Action::Continuous(values) = action else {
            unreachable!()
        };
        self.reference
            .q_value(period, q, alpha, f64::from(values[0]))
            .map(Some)
    }
}

impl NumericalReference {
    fn solve(config: Config, grid: Grid, progress: bool) -> Result<Self> {
        config.validate()?;
        if config.horizon > 128
            || !(2..=201).contains(&grid.inventory)
            || !(2..=401).contains(&grid.alpha)
            || !(2..=201).contains(&grid.actions)
            || !grid.alpha_limit.is_finite()
            || grid.alpha_limit <= 0.
        {
            return Err(Error::Config(
                "invalid or excessive numerical reference grid".into(),
            ));
        }
        config
            .horizon
            .checked_mul(grid.inventory)
            .and_then(|n| n.checked_mul(grid.alpha))
            .and_then(|n| n.checked_mul(grid.actions))
            .and_then(|n| n.checked_mul(grid.quadrature))
            .filter(|n| *n <= 80_000_000)
            .ok_or_else(|| Error::Config("numerical reference exceeds work budget".into()))?;
        let length = (config.horizon + 1)
            .checked_mul(grid.inventory)
            .and_then(|n| n.checked_mul(grid.alpha))
            .filter(|n| *n <= 10_000_000)
            .ok_or_else(|| Error::Config("numerical reference grid exceeds memory limit".into()))?;
        let mut reference = Self {
            config,
            grid,
            quadrature: gaussian_quadrature(grid.quadrature)?,
            values: vec![0.; length],
            clamped_mass_mean: 0.,
        };
        let start = Instant::now();
        let mut boundary_sum = 0.;
        let mut boundary_cases = 0usize;
        for period in (0..config.horizon).rev() {
            for iq in 0..grid.inventory {
                let q = coordinate(iq, grid.inventory, 1.);
                for ia in 0..grid.alpha {
                    let alpha = coordinate(ia, grid.alpha, grid.alpha_limit);
                    let mut best = f64::NEG_INFINITY;
                    for action_index in 0..grid.actions {
                        let action = coordinate(action_index, grid.actions, 1.);
                        let (value, boundary) = reference.q_unchecked(period, q, alpha, action);
                        best = best.max(value);
                        if period + 1 < config.horizon {
                            boundary_sum += boundary;
                            boundary_cases += 1;
                        }
                    }
                    let index = reference.index(period, iq, ia);
                    reference.values[index] = best;
                }
            }
            if progress {
                let done = config.horizon - period;
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!(
                    "alpha DP: {done}/{} horizons, {:.2}s elapsed, estimated {:.2}s remaining",
                    config.horizon,
                    elapsed,
                    elapsed * (config.horizon - done) as f64 / done as f64
                );
            }
        }
        if boundary_cases > 0 {
            reference.clamped_mass_mean = boundary_sum / boundary_cases as f64;
        }
        Ok(reference)
    }

    fn index(&self, period: usize, iq: usize, ia: usize) -> usize {
        (period * self.grid.inventory + iq) * self.grid.alpha + ia
    }

    fn interpolate(&self, period: usize, q: f64, alpha: f64) -> f64 {
        let (qi, qf) = position(q, self.grid.inventory, 1.);
        let (ai, af) = position(alpha, self.grid.alpha, self.grid.alpha_limit);
        let v00 = self.values[self.index(period, qi, ai)];
        let v10 = self.values[self.index(period, qi + 1, ai)];
        let v01 = self.values[self.index(period, qi, ai + 1)];
        let v11 = self.values[self.index(period, qi + 1, ai + 1)];
        (1. - af) * ((1. - qf) * v00 + qf * v10) + af * ((1. - qf) * v01 + qf * v11)
    }

    fn q_unchecked(&self, period: usize, q: f64, alpha: f64, action: f64) -> (f64, f64) {
        let trade = action - q;
        let depleted = alpha * (-self.config.eta * trade.abs()).exp();
        let mut value = action * depleted * self.config.dt - self.config.cost(trade);
        if period + 1 == self.config.horizon {
            return (value - self.config.cost(action), 0.);
        }
        let mean = (-self.config.kappa * self.config.dt).exp() * depleted;
        let scale = self.config.alpha_innovation_scale();
        let mut outside = 0.;
        for &(node, weight) in &self.quadrature {
            let next_alpha = mean + scale * node;
            if next_alpha.abs() > self.grid.alpha_limit {
                outside += weight;
            }
            value += weight * self.interpolate(period + 1, action, next_alpha);
        }
        (value, outside)
    }

    fn validate_state(&self, period: usize, q: f64, alpha: f64) -> Result<()> {
        if period >= self.config.horizon
            || !(-1. ..=1.).contains(&q)
            || !alpha.is_finite()
            || alpha.abs() > self.grid.alpha_limit
        {
            return Err(Error::State(
                "state outside numerical reference domain".into(),
            ));
        }
        Ok(())
    }

    fn value(&self, period: usize, q: f64, alpha: f64) -> Result<f64> {
        self.validate_state(period, q, alpha)?;
        Ok(self.interpolate(period, q, alpha))
    }

    fn q_value(&self, period: usize, q: f64, alpha: f64, action: f64) -> Result<f64> {
        self.validate_state(period, q, alpha)?;
        if !(-1. ..=1.).contains(&action) {
            return Err(Error::State(
                "action outside numerical reference domain".into(),
            ));
        }
        Ok(self.q_unchecked(period, q, alpha, action).0)
    }

    fn action(&self, period: usize, q: f64, alpha: f64) -> Result<f64> {
        self.validate_state(period, q, alpha)?;
        let mut best = f64::NEG_INFINITY;
        let mut selected = 0.;
        for i in 0..self.grid.actions {
            let action = coordinate(i, self.grid.actions, 1.);
            let value = self.q_unchecked(period, q, alpha, action).0;
            if value > best {
                best = value;
                selected = action;
            }
        }
        Ok(selected)
    }
}

fn coordinate(index: usize, count: usize, limit: f64) -> f64 {
    -limit + 2. * limit * index as f64 / (count - 1) as f64
}

fn position(value: f64, count: usize, limit: f64) -> (usize, f64) {
    let scaled = (value.clamp(-limit, limit) + limit) * (count - 1) as f64 / (2. * limit);
    let index = (scaled.floor() as usize).min(count - 2);
    (index, (scaled - index as f64).clamp(0., 1.))
}

pub(super) fn convergence_report() -> Result<()> {
    let config = Config::default();
    let base = Grid {
        inventory: 41,
        alpha: 81,
        actions: 81,
        quadrature: 5,
        alpha_limit: 2.,
    };
    let grids = [
        ("base", base),
        (
            "inventory_refined",
            Grid {
                inventory: 81,
                ..base
            },
        ),
        ("alpha_refined", Grid { alpha: 161, ..base }),
        (
            "actions_refined",
            Grid {
                actions: 161,
                ..base
            },
        ),
        (
            "quadrature_refined",
            Grid {
                quadrature: 9,
                ..base
            },
        ),
        (
            "wider_domain",
            Grid {
                alpha: 121,
                alpha_limit: 3.,
                ..base
            },
        ),
        (
            "fine_base",
            Grid {
                inventory: 31,
                alpha: 161,
                actions: 61,
                quadrature: 9,
                alpha_limit: 2.,
            },
        ),
        (
            "fine_alpha_refined",
            Grid {
                inventory: 31,
                alpha: 321,
                actions: 61,
                quadrature: 9,
                alpha_limit: 2.,
            },
        ),
        (
            "fine_quadrature_refined",
            Grid {
                inventory: 31,
                alpha: 161,
                actions: 61,
                quadrature: 15,
                alpha_limit: 2.,
            },
        ),
        (
            "fine_inventory_refined",
            Grid {
                inventory: 61,
                alpha: 161,
                actions: 61,
                quadrature: 9,
                alpha_limit: 2.,
            },
        ),
        (
            "fine_actions_refined",
            Grid {
                inventory: 31,
                alpha: 161,
                actions: 121,
                quadrature: 9,
                alpha_limit: 2.,
            },
        ),
    ];
    println!(
        "grid,period,inventory,alpha,numerical_value,numerical_action,uniform_grid_action_clamped_mass,solve_seconds"
    );
    for (name, grid) in grids {
        eprintln!(
            "alpha DP: starting {name} grid {}x{}, {} actions, {} quadrature nodes",
            grid.inventory, grid.alpha, grid.actions, grid.quadrature
        );
        let start = Instant::now();
        let reference = NumericalReference::solve(config, grid, true)?;
        let seconds = start.elapsed().as_secs_f64();
        for period in [0, config.horizon / 2, config.horizon - 1] {
            for q in [-0.5, 0., 0.5] {
                for alpha in [-1.9, -1.5, -0.5, -0.2, 0., 0.2, 0.5, 1.5, 1.9] {
                    println!(
                        "{name},{period},{q:.3},{alpha:.3},{:.9},{:.9},{:.9},{seconds:.3}",
                        reference.value(period, q, alpha)?,
                        reference.action(period, q, alpha)?,
                        reference.clamped_mass_mean
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_quadrature_integrates_standard_normal_moments() {
        let quadrature = gaussian_quadrature(5).unwrap();
        for (power, expected) in [(0, 1.), (1, 0.), (2, 1.), (4, 3.), (6, 15.)] {
            let actual: f64 = quadrature.iter().map(|(x, w)| w * x.powi(power)).sum();
            assert!(
                (actual - expected).abs() < 1e-11,
                "moment {power}: {actual}"
            );
        }
    }

    #[test]
    fn one_period_reference_agrees_with_analytic_values_and_liquidation() {
        let model = Config {
            horizon: 1,
            eta: 0.,
            sigma_price: 0.,
            sigma_alpha: 0.,
            spread: 0.1,
            impact: 0.05,
            ..Config::default()
        };
        let grid = Grid {
            inventory: 9,
            alpha: 9,
            actions: 9,
            quadrature: 3,
            alpha_limit: 1.,
        };
        let reference = NumericalReference::solve(model, grid, false).unwrap();
        assert!((reference.q_value(0, 0.2, 0.4, 0.6).unwrap() - 0.114).abs() < 1e-12);
        let free = Config {
            spread: 0.,
            impact: 0.,
            ..model
        };
        let reference = NumericalReference::solve(free, grid, false).unwrap();
        assert!((reference.value(0, 0., 0.5).unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(reference.action(0, 0., 0.5).unwrap(), 1.);
        assert!((reference.q_value(0, 0., 0.5, -1.).unwrap() + 0.5).abs() < 1e-12);
    }

    #[test]
    fn numerical_value_respects_sign_symmetry_and_rejects_invalid_states() {
        let model = Config {
            horizon: 3,
            ..Config::default()
        };
        let grid = Grid {
            inventory: 9,
            alpha: 11,
            actions: 9,
            quadrature: 5,
            alpha_limit: 1.5,
        };
        let reference = NumericalReference::solve(model, grid, false).unwrap();
        let positive = reference.q_value(0, 0.25, 0.3, 0.5).unwrap();
        let negative = reference.q_value(0, -0.25, -0.3, -0.5).unwrap();
        assert!((positive - negative).abs() < 1e-10);
        assert!(reference.q_value(0, 1.1, 0.3, 0.5).is_err());
        assert!(reference.q_value(3, 0., 0.3, 0.5).is_err());
    }

    #[test]
    fn costs_create_a_no_trade_region_for_small_signals() {
        let free = Config {
            horizon: 1,
            eta: 0.,
            sigma_alpha: 0.,
            spread: 0.,
            impact: 0.,
            ..Config::default()
        };
        let costly = Config {
            spread: 0.1,
            impact: 0.05,
            ..free
        };
        let grid = Grid {
            inventory: 11,
            alpha: 11,
            actions: 11,
            quadrature: 1,
            alpha_limit: 1.,
        };
        let costless = NumericalReference::solve(free, grid, false).unwrap();
        let with_costs = NumericalReference::solve(costly, grid, false).unwrap();
        assert_eq!(costless.action(0, 0., 0.1).unwrap(), 1.);
        assert!(with_costs.action(0, 0., 0.1).unwrap().abs() < 1e-12);
    }

    #[test]
    fn baseline_does_not_preposition_when_current_signal_is_zero() {
        let grid = Grid {
            inventory: 21,
            alpha: 41,
            actions: 21,
            quadrature: 5,
            alpha_limit: 2.,
        };
        let reference = NumericalReference::solve(Config::default(), grid, false).unwrap();
        assert!(reference.action(0, 0., 0.).unwrap().abs() < 1e-12);
        assert!(
            reference.q_value(0, 0., 0., 0.).unwrap() > reference.q_value(0, 0., 0., 1.).unwrap()
        );
    }

    #[test]
    fn provider_is_approximate_and_does_not_claim_values_outside_its_domain() -> Result<()> {
        use rl::diagnostics::reference::{
            ReferenceCheckpoint, ReferenceKind, ReferenceProvider, ReferenceTarget,
        };
        use rl::prelude::{ActMode, Action, Observation};
        let config = Config {
            horizon: 1,
            eta: 0.,
            sigma_alpha: 0.,
            sigma_price: 0.,
            spread: 0.1,
            impact: 0.05,
            ..Config::default()
        };
        let grid = Grid {
            inventory: 9,
            alpha: 9,
            actions: 9,
            quadrature: 3,
            alpha_limit: 1.,
        };
        let provider = NumericalProvider::new(config, grid)?;
        assert_eq!(
            provider.specification().quality.kind,
            ReferenceKind::ApproximateNumerical
        );
        assert_eq!(
            provider.specification().target,
            ReferenceTarget::BenchmarkPolicy
        );
        assert!(provider.optimal_decision_domain().is_none());
        let checkpoint = ReferenceCheckpoint {
            revision: 0,
            policy_id: "test",
            policy: &provider,
            mode: ActMode::Deterministic,
        };
        let value = provider
            .action_value(
                &checkpoint,
                &Observation::new(vec![0.2, 0.4, 1.]),
                &Action::Continuous(vec![0.6]),
            )?
            .unwrap();
        assert!((value - 0.114).abs() < 1e-7);
        assert!(
            provider
                .state_value(&checkpoint, &Observation::new(vec![0., 1.1, 1.]))?
                .is_none()
        );
        Ok(())
    }
}
