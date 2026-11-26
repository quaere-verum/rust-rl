//! Discrete ACER's conditional estimator, Retrace and probability-space trust region.
use crate::{Result, core::ensure};

#[derive(Debug, Clone)]
pub struct AcerGradient {
    pub sampled: Vec<f64>,
    pub correction: Vec<f64>,
    pub probability_gradient: Vec<f64>,
    pub logit_gradient: Vec<f64>,
    pub log_importance: f64,
    pub truncated: bool,
}
#[allow(clippy::too_many_arguments)]
pub fn actor_gradient(
    p: &[f64],
    mu: &[f64],
    q: &[f64],
    mask: &[bool],
    action: usize,
    ret: f64,
    c: f64,
) -> Result<AcerGradient> {
    distribution(p, mask, true)?;
    distribution(mu, mask, false)?;
    ensure(
        q.len() == p.len()
            && q.iter().all(|q| q.is_finite())
            && ret.is_finite()
            && c.is_finite()
            && c >= 0.
            && action < p.len()
            && mask[action]
            && mu[action] > 0.,
        "invalid ACER action, values or truncation",
    )?;
    let value = p.iter().zip(q).map(|(p, q)| p * q).sum::<f64>();
    let log_importance = p[action].ln() - mu[action].ln();
    let mut sampled = vec![0.; p.len()];
    // A quotient overflow is harmless here: min(c,+infinity)=c. No ratio is
    // differentiated, serialized as infinity, or formed at zero behavior support.
    sampled[action] = (p[action] / mu[action]).min(c) * (ret - value);
    let correction = p
        .iter()
        .zip(mu)
        .zip(q)
        .zip(mask)
        .map(|(((&p, &mu), &q), &legal)| {
            if legal {
                (p - c * mu).max(0.) * (q - value)
            } else {
                0.
            }
        })
        .collect::<Vec<_>>();
    let coefficients = sampled
        .iter()
        .zip(&correction)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let sum = coefficients.iter().sum::<f64>();
    let logit_gradient = coefficients
        .iter()
        .zip(p)
        .map(|(b, p)| b - p * sum)
        .collect::<Vec<_>>();
    let probability_gradient = coefficients
        .iter()
        .zip(p)
        .zip(mask)
        .map(|((&b, &p), &legal)| if legal { b / p } else { 0. })
        .collect::<Vec<_>>();
    ensure(
        probability_gradient
            .iter()
            .chain(&logit_gradient)
            .all(|g| g.is_finite()),
        "nonfinite ACER conditional gradient",
    )?;
    Ok(AcerGradient {
        sampled,
        correction,
        probability_gradient,
        logit_gradient,
        log_importance,
        truncated: c == 0. || log_importance > c.ln(),
    })
}
#[derive(Debug, Clone)]
pub struct TrustRegion {
    pub probability_gradient: Vec<f64>,
    pub logit_gradient: Vec<f64>,
    pub multiplier: f64,
    pub dot_before: f64,
    pub dot_after: f64,
}
pub fn trust_region(
    p: &[f64],
    average: &[f64],
    gradient: &[f64],
    mask: &[bool],
    delta: f64,
) -> Result<TrustRegion> {
    distribution(p, mask, true)?;
    distribution(average, mask, false)?;
    ensure(
        gradient.len() == p.len()
            && gradient
                .iter()
                .zip(mask)
                .all(|(&g, &m)| g.is_finite() && (m || g == 0.))
            && delta.is_finite()
            && delta >= 0.,
        "invalid ACER probability-space trust input",
    )?;
    let k = p
        .iter()
        .zip(average)
        .zip(mask)
        .map(|((&p, &old), &legal)| if legal { -old / p } else { 0. })
        .collect::<Vec<_>>();
    let dot_before = k.iter().zip(gradient).map(|(k, g)| k * g).sum::<f64>();
    let norm = k.iter().map(|k| k * k).sum::<f64>();
    ensure(
        dot_before.is_finite() && norm.is_finite() && norm > 0.,
        "nonfinite ACER trust-region normal",
    )?;
    let multiplier = ((dot_before - delta) / norm).max(0.);
    let probability_gradient = gradient
        .iter()
        .zip(&k)
        .map(|(g, k)| g - multiplier * k)
        .collect::<Vec<_>>();
    let mean = p
        .iter()
        .zip(&probability_gradient)
        .map(|(p, g)| p * g)
        .sum::<f64>();
    let logit_gradient = p
        .iter()
        .zip(&probability_gradient)
        .map(|(p, g)| p * (g - mean))
        .collect::<Vec<_>>();
    let dot_after = k
        .iter()
        .zip(&probability_gradient)
        .map(|(k, g)| k * g)
        .sum::<f64>();
    ensure(
        probability_gradient
            .iter()
            .chain(&logit_gradient)
            .all(|g| g.is_finite())
            && dot_after.is_finite(),
        "nonfinite projected ACER gradient",
    )?;
    Ok(TrustRegion {
        probability_gradient,
        logit_gradient,
        multiplier,
        dot_before,
        dot_after,
    })
}
#[derive(Debug, Clone)]
pub struct RetraceStep {
    pub policy: Vec<f64>,
    pub behavior: Vec<f64>,
    pub q: Vec<f64>,
    pub mask: Vec<bool>,
    pub action: usize,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}
pub fn retrace(
    steps: &[RetraceStep],
    bootstrap: Option<f64>,
    gamma: f64,
    lambda: f64,
) -> Result<Vec<f64>> {
    ensure(
        !steps.is_empty()
            && gamma.is_finite()
            && (0. ..=1.).contains(&gamma)
            && lambda.is_finite()
            && (0. ..=1.).contains(&lambda),
        "invalid ACER Retrace sequence/discount",
    )?;
    let last = steps.last().unwrap();
    ensure(
        last.terminated == bootstrap.is_none() && bootstrap.is_none_or(|v| v.is_finite()),
        "ACER Retrace terminal/bootstrap contract differs",
    )?;
    for (i, s) in steps.iter().enumerate() {
        distribution(&s.policy, &s.mask, false)?;
        distribution(&s.behavior, &s.mask, false)?;
        ensure(
            s.q.len() == s.policy.len()
                && s.q.iter().all(|q| q.is_finite())
                && s.reward.is_finite()
                && s.action < s.q.len()
                && s.mask[s.action]
                && s.behavior[s.action] > 0.
                && !(s.terminated && s.truncated)
                && (i + 1 == steps.len() || !(s.terminated || s.truncated)),
            "invalid ACER Retrace action/value or internal episode boundary",
        )?;
    }
    let mut targets = vec![0.; steps.len()];
    let mut ret = bootstrap.unwrap_or(0.);
    for (i, s) in steps.iter().enumerate().rev() {
        let target = s.reward + gamma * ret;
        targets[i] = target;
        let trace = lambda * (s.policy[s.action] / s.behavior[s.action]).min(1.);
        let value = s.policy.iter().zip(&s.q).map(|(p, q)| p * q).sum::<f64>();
        ret = value + trace * (target - s.q[s.action]);
        ensure(
            target.is_finite() && ret.is_finite(),
            "nonfinite ACER Retrace value",
        )?;
    }
    Ok(targets)
}

fn distribution(p: &[f64], mask: &[bool], positive: bool) -> Result<()> {
    ensure(
        !p.is_empty()
            && p.len() == mask.len()
            && mask.iter().any(|m| *m)
            && p.iter().zip(mask).all(|(&p, &m)| {
                p.is_finite() && p >= 0. && if m { !positive || p > 0. } else { p == 0. }
            })
            && (p.iter().sum::<f64>() - 1.).abs() < 1e-10,
        "invalid ACER masked distribution/support",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-11, "{a} != {b}");
    }
    #[test]
    fn truncated_estimator_plus_bias_correction_has_the_literal_conditional_expectation() {
        let mu = [0.9, 0.1];
        let p = [0.2, 0.8];
        let q = [-0.5, 1.];
        let mut sampled = [0.; 2];
        let mut corrected = [0.; 2];
        for a in 0..2 {
            let r = actor_gradient(&p, &mu, &q, &[true; 2], a, q[a], 2.).unwrap();
            for j in 0..2 {
                let baseline = r.sampled.iter().sum::<f64>();
                sampled[j] += mu[a] * (r.sampled[j] - p[j] * baseline);
                corrected[j] += mu[a] * r.logit_gradient[j];
            }
        }
        close(sampled[0], -0.204);
        close(sampled[1], 0.204);
        close(corrected[0], -0.24);
        close(corrected[1], 0.24);
        close(corrected[0] - sampled[0], -0.036);
    }
    #[test]
    fn probability_space_projection_and_its_logit_jacobian_match_literal_calculus() {
        let r = trust_region(&[0.25, 0.75], &[0.5, 0.5], &[-3., 1.], &[true; 2], 0.1).unwrap();
        close(r.multiplier, 1.1775);
        close(r.dot_after, 0.1);
        close(r.probability_gradient[0], -0.645);
        close(r.probability_gradient[1], 1.785);
        close(r.logit_gradient[0], -0.455625);
        close(r.logit_gradient[1], 0.455625);
        let objective = |z: f64| {
            let p = z.exp() / (z.exp() + 1.);
            p * (-0.645) + (1. - p) * 1.785
        };
        let z = (1f64 / 3.).ln();
        let fd = (objective(z + 1e-5) - objective(z - 1e-5)) / 2e-5;
        assert!((fd - r.logit_gradient[0]).abs() < 1e-10);
        let inactive =
            trust_region(&[0.25, 0.75], &[0.5, 0.5], &[3., -1.], &[true; 2], 0.1).unwrap();
        assert_eq!(inactive.multiplier, 0.);
        assert_eq!(inactive.probability_gradient, [3., -1.]);
    }
    fn steps(terminal: bool) -> Vec<RetraceStep> {
        vec![
            RetraceStep {
                policy: vec![0.5, 0.5],
                behavior: vec![0.5, 0.5],
                q: vec![0., 0.],
                mask: vec![true; 2],
                action: 0,
                reward: 0.25,
                terminated: false,
                truncated: false,
            },
            RetraceStep {
                policy: vec![0.25, 0.75],
                behavior: vec![0.5, 0.5],
                q: vec![2., 4.],
                mask: vec![true; 2],
                action: 1,
                reward: -1.,
                terminated: terminal,
                truncated: !terminal,
            },
        ]
    }
    #[test]
    fn retrace_uses_next_action_ratio_cap_one_and_preserves_truncation_bootstrap() {
        let r = retrace(&steps(true), None, 0.9, 1.).unwrap();
        close(r[0], -1.1);
        close(r[1], -1.);
        let r = retrace(&steps(false), Some(2.), 0.9, 1.).unwrap();
        close(r[0], 0.52);
        close(r[1], 0.8);
        let r = retrace(&steps(true), None, 0.9, 0.5).unwrap();
        close(r[0], 1.15);
        let r = retrace(&steps(true), None, 0.9, 0.).unwrap();
        close(r[0], 3.4);
        assert!(retrace(&steps(true), Some(2.), 0.9, 1.).is_err());
        assert!(retrace(&steps(false), None, 0.9, 1.).is_err());
        let mut invalid = steps(true);
        invalid[0].truncated = true;
        assert!(retrace(&invalid, None, 0.9, 1.).is_err());
    }
    #[test]
    fn masks_and_unsampled_zero_behavior_support_have_correct_all_action_correction() {
        let r = actor_gradient(
            &[0.2, 0.8, 0.],
            &[1., 0., 0.],
            &[-0.5, 1., 999.],
            &[true, true, false],
            0,
            -0.5,
            2.,
        )
        .unwrap();
        close(r.logit_gradient[0], -0.24);
        close(r.logit_gradient[1], 0.24);
        assert_eq!(r.probability_gradient[2], 0.);
        assert_eq!(r.logit_gradient[2], 0.);
        assert!(
            actor_gradient(&[0.2, 0.8], &[1., 0.], &[-0.5, 1.], &[true; 2], 1, 1., 2.).is_err()
        );
        assert!(
            actor_gradient(
                &[0.2, 0.8, 0.],
                &[1., 0., 0.],
                &[-0.5, 1., 999.],
                &[true, true, false],
                2,
                0.,
                2.
            )
            .is_err()
        );
    }
    #[test]
    fn very_rare_behavior_action_uses_finite_bounded_coefficients_and_log_ratio() {
        let r = actor_gradient(
            &[0.5, 0.5],
            &[1e-320, 1.],
            &[1., -1.],
            &[true; 2],
            0,
            1.,
            10.,
        )
        .unwrap();
        close(r.logit_gradient[0], 5.25);
        close(r.logit_gradient[1], -5.25);
        assert!(r.log_importance > 700. && r.log_importance.is_finite() && r.truncated);
        let c0 = actor_gradient(
            &[0.2, 0.8],
            &[0.9, 0.1],
            &[-0.5, 1.],
            &[true; 2],
            0,
            99.,
            0.,
        )
        .unwrap();
        close(c0.logit_gradient[0], -0.24);
        close(c0.logit_gradient[1], 0.24);
    }
}
