//! Pure actual-action Gaussian marginal mathematics. No sampling or learner state.
use crate::{Result, core::ensure};

/// One latent normal, optionally transformed by affine(tanh(x)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianMarginal {
    pub mean: f64,
    pub log_std: f64,
    pub bounds: Option<[f64; 2]>,
}
/// Hard work bound per expectation, including subdivision probes.
pub const MAX_QUADRATURE_EVALUATIONS: usize = 32_768;
/// Actual-action statistics, not latent-normal parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianActionStatistics {
    pub mean: f64,
    pub standard_deviation: f64,
}
impl GaussianMarginal {
    /// Selected-only central integration, at most 65,536 total evaluations,
    /// each pass capped at 32,768. Propagate integration targets, omitted
    /// tails and roundoff allowances through sqrt and affine scale. Accept only
    /// if the entire estimated std interval exceeds 1e-6 of the half-range and
    /// its radius is <=1e-8 relative (also <=2.5e-7 action units: a quarter of
    /// the last displayed fixed-decimal unit). These are estimates, not certified
    /// intervals. Unresolved precision is unavailable, never a zero spread.
    /// Bounded-action spread that scales to zero or a subnormal is unavailable;
    /// its relative-error thresholds cannot safely use normal-number arithmetic.
    pub fn action_statistics(&self) -> Result<GaussianActionStatistics> {
        let (sigma, center, scale) = self.parameters()?;
        if self.bounds.is_none() {
            return Ok(GaussianActionStatistics {
                mean: self.mean,
                standard_deviation: sigma,
            });
        }
        ensure(
            sigma <= 1e6 && self.mean.abs() <= 1e12,
            "bounded Gaussian statistics outside supported quadrature range",
        )?;
        // Keep saturated residuals before centering: tanh(x)-offset computed
        // directly never rounds a small residual to zero by subtracting from 1.
        let offset = if self.mean >= 0. { 1. } else { -1. };
        let (mean, mean_evaluations) = self.normal_integral_counted(
            sigma,
            None,
            1e-13,
            MAX_QUADRATURE_EVALUATIONS,
            12,
            offset,
        )?;
        let mut remaining = 2 * MAX_QUADRATURE_EVALUATIONS - mean_evaluations;
        ensure(
            mean.is_finite() && (offset + mean).abs() <= 1. + 1e-12,
            "Gaussian action mean precision unavailable",
        )?;
        // The log derivative of the logistic residual has magnitude <=2.
        // Argument/exp/division roundoff therefore scales with the residual,
        // unlike an absolute tanh error that swamps a saturated variance.
        let roundoff = 16. * f64::EPSILON * (self.mean.abs() + 12. * sigma + 1.);
        let mean_error = 1e-13 + 1e-32 + roundoff * mean.abs();
        let mut tolerance = 1e-13;
        for _ in 0..2 {
            let (variance, evaluations) = self.normal_integral_counted(
                sigma,
                Some(mean),
                tolerance,
                remaining.min(MAX_QUADRATURE_EVALUATIONS),
                12,
                offset,
            )?;
            remaining -= evaluations;
            ensure(
                variance.is_finite() && variance > 0. && variance <= 1. + 1e-12,
                "Gaussian action spread precision unavailable",
            )?;
            let std = variance.sqrt();
            // E[(X-m)^2] differs from true central variance by mean_error^2.
            // Bounded squared central tails outside +/-12 are <2e-32.
            // Cauchy-Schwarz propagates residual/centering arithmetic in RMS:
            // |E[(x+e)^2-x^2]| <=2*sqrt(E[x^2])*sqrt(E[e^2])+E[e^2].
            // Adding central variance and mean^2 here only estimates a scale;
            // the computed variance itself never subtracts raw second moments.
            let residual_rms =
                (variance + tolerance + 2e-32 + (mean.abs() + mean_error).powi(2)).sqrt();
            let arithmetic_rms = roundoff * (residual_rms + mean.abs());
            let variance_error = tolerance
                + 2e-32
                + mean_error.powi(2)
                + 2. * std * arithmetic_rms
                + arithmetic_rms.powi(2)
                + 64. * f64::EPSILON * variance;
            let lower = (variance - variance_error).max(0.).sqrt();
            let upper = (variance + variance_error).sqrt();
            let standard_deviation = scale * std;
            // Reject affine underflow before error scaling/comparison: both
            // error thresholds can otherwise round to zero and falsely pass.
            ensure(
                standard_deviation > 0. && standard_deviation.is_normal(),
                "Gaussian action statistics affine precision unavailable",
            )?;
            let action_error =
                scale * (std - lower).max(upper - std) + 4. * f64::EPSILON * standard_deviation;
            let permitted_error = if (1e-4..1e6).contains(&standard_deviation) {
                (standard_deviation * 1e-8).min(2.5e-7)
            } else {
                standard_deviation * 1e-8
            };
            if lower > 1e-6 && action_error.is_finite() && action_error <= permitted_error {
                let mean = center + scale * (offset + mean).clamp(-1., 1.);
                ensure(
                    mean.is_finite() && standard_deviation.is_finite(),
                    "Gaussian action statistics affine precision unavailable",
                )?;
                return Ok(GaussianActionStatistics {
                    mean,
                    standard_deviation,
                });
            }
            // Adapt only the quadrature contribution, never the cutoff. Tail
            // and roundoff uncertainty can still make the second pass fail.
            let allowed_std_error = permitted_error / scale - 4. * f64::EPSILON * std;
            let variance_target = allowed_std_error * (2. * std - allowed_std_error);
            let non_quadrature_error = variance_error - tolerance;
            tolerance = tolerance.min((variance_target - non_quadrature_error) * 0.5);
            ensure(
                tolerance.is_finite() && tolerance > 0.,
                "Gaussian action statistics affine precision unavailable",
            )?;
        }
        Err(crate::Error::Numerical(
            "Gaussian action spread precision unavailable".into(),
        ))
    }
    fn parameters(&self) -> Result<(f64, f64, f64)> {
        let sigma = self.log_std.exp();
        ensure(
            self.mean.is_finite() && self.log_std.is_finite() && sigma.is_finite() && sigma > 0.,
            "invalid/unrepresentable Gaussian parameters",
        )?;
        let (center, scale) = match self.bounds {
            None => (0., 1.),
            Some([l, h]) => {
                let (center, scale) = (l / 2. + h / 2., h / 2. - l / 2.);
                ensure(
                    l.is_finite()
                        && h.is_finite()
                        && l < h
                        && center.is_finite()
                        && scale.is_finite()
                        && scale > 0.,
                    "invalid/unrepresentable Gaussian bounds",
                )?;
                (center, scale)
            }
        };
        Ok((sigma, center, scale))
    }
    /// Expected actual action, never the transformed latent mean/median.
    /// Adaptive Simpson integration in standard-normal coordinates, split at
    /// latent transition points as well as unit normal intervals. Normal tails
    /// outside +/-9 contribute <3e-19. The summed estimated absolute integration
    /// error is <=1e-11 before the affine scale (target <=1e-8).
    /// Work/precision exhaustion and unsupported extreme ranges are explicit.
    pub fn expected_action(&self) -> Result<f64> {
        self.expectation(MAX_QUADRATURE_EVALUATIONS)
    }
    fn expectation(&self, budget: usize) -> Result<f64> {
        let (sigma, center, scale) = self.parameters()?;
        if self.bounds.is_none() {
            return Ok(self.mean);
        }
        ensure(
            sigma <= 1e6 && self.mean.abs() <= 1e12,
            "bounded Gaussian expectation outside supported quadrature range",
        )?;
        if self.mean == 0. {
            return Ok(center);
        }
        let value = self.normal_integral(sigma, None, 1e-11, budget)?;
        ensure(
            value.is_finite() && value.abs() <= 1. + 1e-10,
            "Gaussian expectation integration failed",
        )?;
        let value = center + scale * value.clamp(-1., 1.);
        ensure(value.is_finite(), "Gaussian expectation affine overflow")?;
        Ok(value)
    }
    fn normal_integral(
        &self,
        sigma: f64,
        central: Option<f64>,
        tolerance: f64,
        budget: usize,
    ) -> Result<f64> {
        self.normal_integral_counted(sigma, central, tolerance, budget, 9, 0.)
            .map(|(value, _)| value)
    }
    fn normal_integral_counted(
        &self,
        sigma: f64,
        central: Option<f64>,
        tolerance: f64,
        budget: usize,
        extent: i32,
        offset: f64,
    ) -> Result<(f64, usize)> {
        // Both callers use extent 9 or 12: at most 25 unit-grid points and
        // nine latent transition points. Keep this bounded scratch on-stack.
        debug_assert!((0..=12).contains(&extent));
        let mut points = [0.; 34];
        let mut count = 0;
        for z in -extent..=extent {
            points[count] = f64::from(z);
            count += 1;
        }
        for t in [-20., -10., -5., -2., 0., 2., 5., 10., 20.] {
            let z = (t - self.mean) / sigma;
            if (-(extent as f64)..extent as f64).contains(&z) {
                points[count] = z;
                count += 1;
            }
        }
        // Equal total_cmp keys have identical bits, so unstable ordering keeps
        // the same values. Dedup still uses float equality, including +/-0.
        points[..count].sort_unstable_by(f64::total_cmp);
        let mut unique = 0;
        for i in 0..count {
            if unique == 0 || points[i] != points[unique - 1] {
                points[unique] = points[i];
                unique += 1;
            }
        }
        let mut integral = Quadrature {
            mean: self.mean,
            sigma,
            evaluations: 0,
            budget,
            central,
            offset,
        };
        let mut value = 0.;
        let mut previous_endpoint = None;
        for w in points[..unique].windows(2) {
            let (a, b) = (w[0], w[1]);
            let m = a + (b - a) / 2.;
            let left = if let Some(endpoint) = previous_endpoint {
                // Preserve the old logical allowance and failure position even
                // though the identical endpoint no longer needs transcendents.
                integral.charge_evaluation()?;
                endpoint
            } else {
                integral.eval(a)?
            };
            let values = [left, integral.eval(m)?, integral.eval(b)?];
            previous_endpoint = Some(values[2]);
            let whole = (b - a) / 6. * (values[0] + 4. * values[1] + values[2]);
            value += integral.refine(
                [a, b],
                values,
                whole,
                tolerance * (b - a) / (2. * extent as f64),
                0,
            )?;
        }
        Ok((value, integral.evaluations))
    }
    /// Actual-action density with both affine and tanh Jacobians. Density at
    /// finite bounded support endpoints is defined as zero; use CDF for masses.
    pub fn density(&self, action: f64) -> Result<f64> {
        let (sigma, center, scale) = self.parameters()?;
        ensure(!action.is_nan(), "NaN Gaussian action")?;
        if !action.is_finite() {
            return Ok(0.);
        }
        let (latent, jacobian) = if let Some([l, h]) = self.bounds {
            if action <= l || action >= h {
                return Ok(0.);
            }
            let y = (action - center) / scale;
            ensure(
                y.abs() < 1.,
                "action transform exceeds floating-point resolution",
            )?;
            // log(1-y^2) as a sum avoids cancellation at support boundaries.
            (y.atanh(), scale.ln() + (1. - y).ln() + (1. + y).ln())
        } else {
            (action, 0.)
        };
        let z = standardize(latent, self.mean, sigma);
        let log_density =
            -0.5 * z * z - self.log_std - 0.5 * (2. * std::f64::consts::PI).ln() - jacobian;
        let value = log_density.exp();
        ensure(
            value.is_finite(),
            "Gaussian density exceeds floating-point range; use CDF",
        )?;
        Ok(value)
    }
    /// Cumulative mass including bounded endpoints and +/- infinity.
    pub fn cdf(&self, action: f64) -> Result<f64> {
        let (sigma, center, scale) = self.parameters()?;
        ensure(!action.is_nan(), "NaN Gaussian action")?;
        let latent = if let Some([l, h]) = self.bounds {
            if action <= l {
                return Ok(0.);
            }
            if action >= h {
                return Ok(1.);
            }
            let y = (action - center) / scale;
            ensure(
                y.abs() < 1.,
                "action transform exceeds floating-point resolution",
            )?;
            y.atanh()
        } else {
            action
        };
        Ok(0.5 * libm::erfc(-standardize(latent, self.mean, sigma) / std::f64::consts::SQRT_2))
    }
}
fn standardize(value: f64, mean: f64, sigma: f64) -> f64 {
    let delta = value - mean;
    if delta.is_infinite() && value.is_finite() {
        value / sigma - mean / sigma
    } else {
        delta / sigma
    }
}
struct Quadrature {
    mean: f64,
    sigma: f64,
    evaluations: usize,
    budget: usize,
    central: Option<f64>,
    offset: f64,
}
impl Quadrature {
    fn charge_evaluation(&mut self) -> Result<()> {
        ensure(
            self.evaluations < self.budget,
            "Gaussian quadrature evaluation budget exhausted",
        )?;
        self.evaluations += 1;
        Ok(())
    }
    fn eval(&mut self, z: f64) -> Result<f64> {
        self.charge_evaluation()?;
        #[cfg(test)]
        tests::EVALUATION_WORK.with(|work| work.set(work.get() + 1));
        let latent = self.mean + self.sigma * z;
        let value = if self.offset == 0. {
            latent.tanh()
        } else {
            -2. * self.offset / (1. + (2. * self.offset * latent).exp())
        };
        let value = self.central.map_or(value, |mean| (value - mean).powi(2));
        Ok(value * (-z * z / 2.).exp() / (2. * std::f64::consts::PI).sqrt())
    }
    fn refine(
        &mut self,
        [a, b]: [f64; 2],
        v: [f64; 3],
        whole: f64,
        tolerance: f64,
        depth: usize,
    ) -> Result<f64> {
        let m = a + (b - a) / 2.;
        let l = a + (m - a) / 2.;
        let r = m + (b - m) / 2.;
        ensure(
            depth < 48 && a < l && l < m && m < r && r < b,
            "Gaussian quadrature precision/subdivision budget exhausted",
        )?;
        let (fl, fr) = (self.eval(l)?, self.eval(r)?);
        let left = (m - a) / 6. * (v[0] + 4. * fl + v[1]);
        let right = (b - m) / 6. * (v[1] + 4. * fr + v[2]);
        let delta = left + right - whole;
        if delta.abs() <= 15. * tolerance {
            return Ok(left + right + delta / 15.);
        }
        Ok(
            self.refine([a, m], [v[0], fl, v[1]], left, tolerance / 2., depth + 1)?
                + self.refine([m, b], [v[1], fr, v[2]], right, tolerance / 2., depth + 1)?,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    thread_local! {
        pub(super) static EVALUATION_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    // Frozen pre-reuse outer integration from 0c18044. Keep Vec, stable sort,
    // dedup and all three independent endpoint/midpoint evaluations here so the
    // optimized breakpoint/reuse path has a same-platform bit-exact oracle.
    // Integrand and recursive Simpson arithmetic were not changed by this work;
    // share that kernel rather than copying unrelated numerical implementation.
    fn baseline_normal_integral_counted(
        g: &GaussianMarginal,
        sigma: f64,
        central: Option<f64>,
        tolerance: f64,
        budget: usize,
        extent: i32,
        offset: f64,
    ) -> Result<(f64, usize)> {
        let mut points = Vec::with_capacity(30);
        points.extend((-extent..=extent).map(f64::from));
        for t in [-20., -10., -5., -2., 0., 2., 5., 10., 20.] {
            let z = (t - g.mean) / sigma;
            if (-(extent as f64)..extent as f64).contains(&z) {
                points.push(z);
            }
        }
        points.sort_by(f64::total_cmp);
        points.dedup();
        let mut integral = Quadrature {
            mean: g.mean,
            sigma,
            evaluations: 0,
            budget,
            central,
            offset,
        };
        let mut value = 0.;
        for w in points.windows(2) {
            let (a, b) = (w[0], w[1]);
            let m = a + (b - a) / 2.;
            let values = [integral.eval(a)?, integral.eval(m)?, integral.eval(b)?];
            let whole = (b - a) / 6. * (values[0] + 4. * values[1] + values[2]);
            value += integral.refine(
                [a, b],
                values,
                whole,
                tolerance * (b - a) / (2. * extent as f64),
                0,
            )?;
        }
        Ok((value, integral.evaluations))
    }
    #[test]
    fn quadrature_preserves_bits_and_logical_budget() {
        // Windows captures remain numeric references with native-math tolerance;
        // exact bits and logical budgets are checked against the frozen oracle.
        for (mean, sigma, extent, tolerance, offset, central, reference_bits) in [
            (0.7, 0.8, 9, 1e-11, 0., None, 0x3fdcd6a1434c81fe),
            (35.5, 5., 12, 1e-13, 1., None, 0xbd8ceac43d33d95d),
            (-35.5, 5., 12, 1e-13, -1., None, 0x3d8ceac43d33d95c),
            (
                2.,
                100.,
                12,
                1e-13,
                1.,
                Some(-0.9840440286120649),
                0x3fefbc9142f716be,
            ),
            (-0., 1., 12, 1e-13, 1., None, 0xbff0000000000004),
        ] {
            let g = GaussianMarginal {
                mean,
                log_std: f64::ln(sigma),
                bounds: Some([-1., 1.]),
            };
            let sigma = g.log_std.exp();
            let (value, logical) = g
                .normal_integral_counted(
                    sigma,
                    central,
                    tolerance,
                    MAX_QUADRATURE_EVALUATIONS,
                    extent,
                    offset,
                )
                .unwrap();
            let (baseline, baseline_logical) = baseline_normal_integral_counted(
                &g,
                sigma,
                central,
                tolerance,
                MAX_QUADRATURE_EVALUATIONS,
                extent,
                offset,
            )
            .unwrap();
            assert_eq!(
                (value.to_bits(), logical),
                (baseline.to_bits(), baseline_logical)
            );
            let reference = f64::from_bits(reference_bits);
            assert!((value - reference).abs() <= 1e-12 * reference.abs());
            let exact = g
                .normal_integral_counted(sigma, central, tolerance, logical, extent, offset)
                .unwrap();
            assert_eq!(exact, (value, logical));
            for budget in [0, 1, 2, 3, 4, 5, logical - 1] {
                let baseline_error = baseline_normal_integral_counted(
                    &g, sigma, central, tolerance, budget, extent, offset,
                )
                .unwrap_err()
                .to_string();
                assert_eq!(
                    baseline_error,
                    "invalid configuration: Gaussian quadrature evaluation budget exhausted"
                );
                assert_eq!(
                    g.normal_integral_counted(sigma, central, tolerance, budget, extent, offset)
                        .unwrap_err()
                        .to_string(),
                    baseline_error
                );
            }
        }
    }
    #[test]
    fn quadrature_reuses_shared_endpoint_work() {
        let g = GaussianMarginal {
            mean: 0.7,
            log_std: 0.,
            bounds: Some([-1., 1.]),
        };
        EVALUATION_WORK.with(|work| work.set(0));
        let (_, logical) = g
            .normal_integral_counted(1., None, 1e-11, MAX_QUADRATURE_EVALUATIONS, 9, 0.)
            .unwrap();
        let physical = EVALUATION_WORK.with(|work| work.get());
        assert_eq!(
            physical,
            logical - 22,
            "22 adjacent intervals must reuse their shared endpoint"
        );
    }
    #[test]
    fn selected_action_statistics_are_transformed_and_offset_stable() {
        for (mean, sigma, bounds, expected_mean, expected_sd) in [
            (0., 1., [-1., 1.], 0., 0.6279287303491068),
            (0.7, 0.8, [-3., 7.], 4.252999857528296, 2.4028991300362614),
            (2., 5., [-3., 7.], 3.5307420245323082, 4.361608484131246),
            (2., 100., [-1., 1.], 0.01595597138793514, 0.9958757376652606),
            (
                0.7,
                0.001,
                [-3., 7.],
                5.021836967506777,
                0.0031736994131100053,
            ),
            (
                0.7,
                0.8,
                [1e12 - 1., 1e12 + 1.],
                1e12 + 0.45059997150565934,
                0.48057982600725224,
            ),
        ] {
            let g = GaussianMarginal {
                mean,
                log_std: f64::ln(sigma),
                bounds: Some(bounds),
            };
            let stats = g.action_statistics().unwrap();
            assert!((stats.mean - expected_mean).abs() <= 5e-10, "{stats:?}");
            assert!(
                (stats.standard_deviation - expected_sd).abs() <= 5e-10,
                "{stats:?}"
            );
        }
        for (mean, sigma) in [(0.7, 1e-10_f64), (15., 0.2), (20., 0.2)] {
            let g = GaussianMarginal {
                mean,
                log_std: sigma.ln(),
                bounds: Some([-1., 1.]),
            };
            assert!(
                g.action_statistics().is_err(),
                "Unresolved positive spread must never become zero"
            );
        }
        let exact = GaussianMarginal {
            mean: 4.,
            log_std: 3_f64.ln(),
            bounds: None,
        }
        .action_statistics()
        .unwrap();
        assert_eq!(exact.mean, 4.);
        assert!((exact.standard_deviation - 3.).abs() < 1e-14);
    }
    #[test]
    fn exhausted_integration_budget_is_an_error() {
        let g = GaussianMarginal {
            mean: 0.7,
            log_std: 0.,
            bounds: Some([-1., 1.]),
        };
        assert!(g.expectation(4).is_err());
        assert!(g.expected_action().unwrap().is_finite());
    }
}
