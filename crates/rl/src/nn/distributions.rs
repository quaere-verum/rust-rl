use super::{matrix_on_device, to_vec};
use crate::{
    Error, Result,
    core::{
        action::Action,
        actor::{ActMode, EpisodeBehavior, PathwiseBehavior, RolloutBehavior},
        ensure,
    },
};
use burn::tensor::{
    Bool, Int, Tensor, TensorData,
    activation::{log_softmax, softplus},
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
// Keep short-lived tensor handles inline; boxing adds an allocation to every policy forward.
#[allow(clippy::large_enum_variant)]
pub enum Distribution {
    Categorical {
        logits: Tensor<2>,
        masks: Vec<bool>,
    },
    Gaussian {
        mean: Tensor<2>,
        log_std: Tensor<2>,
        bounds: Option<(Vec<f32>, Vec<f32>)>,
    },
}

/// A current-policy Gaussian sample that retains gradients through its action.
#[derive(Debug, Clone)]
pub struct PathwiseSample {
    pub actions: Tensor<2>,
    pub log_probs: Tensor<1>,
    pub latents: Tensor<2>,
}
/// Exact categorical terms, with zero probability and finite zero log term on masks.
#[derive(Debug, Clone)]
pub struct CategoricalTerms {
    pub probabilities: Tensor<2>,
    pub log_probs: Tensor<2>,
}
impl Distribution {
    /// Mix each categorical row with the uniform law over that row's legal actions.
    /// The returned distribution can be used for both sampling and policy scores.
    pub fn with_uniform_mixture(self, epsilon: &[f32]) -> Result<Self> {
        self.validate()?;
        let Self::Categorical { logits, masks } = self else {
            return Err(Error::Action(
                "uniform exploration mixture requires categorical actions".into(),
            ));
        };
        let [rows, actions] = logits.dims();
        ensure(
            epsilon.len() == rows && epsilon.iter().all(|p| (0. ..=1.).contains(p)),
            "one exploration probability in [0,1] is required per categorical row",
        )?;
        if epsilon.iter().all(|&p| p == 0.) {
            return Ok(Self::Categorical { logits, masks });
        }
        let device = logits.device();
        let log_policy_weight = matrix_on_device(
            epsilon.iter().map(|&p| (-p).ln_1p()).collect(),
            rows,
            1,
            &device,
        );
        let log_uniform_mass = matrix_on_device(
            masks
                .chunks_exact(actions)
                .zip(epsilon)
                .map(|(mask, &p)| {
                    let legal = mask.iter().filter(|&&valid| valid).count();
                    // Zero-mixture rows retain their original logits below.
                    f64::from(if p > 0. { p } else { 1. }).ln() as f32 - (legal as f32).ln()
                })
                .collect(),
            rows,
            1,
            &device,
        )
        .repeat_dim(1, actions);
        let log_policy_mass = Self::categorical_log_probs(&logits, &masks) + log_policy_weight;
        let maximum = log_policy_mass.clone().max_pair(log_uniform_mass.clone());
        let mixed = maximum.clone()
            + ((log_policy_mass - maximum.clone()).exp() + (log_uniform_mass - maximum).exp())
                .log();
        let original = Tensor::<2, Bool>::from_data(
            TensorData::new(
                epsilon.iter().map(|&p| p == 0.).collect::<Vec<_>>(),
                [rows, 1],
            ),
            &device,
        )
        .repeat_dim(1, actions);
        Ok(Self::Categorical {
            logits: mixed.mask_where(original, logits),
            masks,
        })
    }

    pub fn categorical_terms(&self) -> Result<CategoricalTerms> {
        self.validate()?;
        let Self::Categorical { logits, masks } = self else {
            return Err(Error::Action(
                "categorical terms require a categorical distribution".into(),
            ));
        };
        let lp = Self::categorical_log_probs(logits, masks);
        Ok(CategoricalTerms {
            probabilities: lp.clone().exp(),
            log_probs: lp.mask_fill(Self::invalid_mask(logits, masks), 0.),
        })
    }
    /// Gaussian reparameterization using only caller-owned noise.
    pub fn reparameterize(&self, rng: &mut ChaCha8Rng) -> Result<PathwiseSample> {
        self.reparameterize_with_noise(self.gaussian_noise(rng)?)
    }
    /// Score the known base variate rather than recover it from a rounded latent.
    /// Opt-in for SAC: established REPPO/entropy/rollout scoring stays unchanged.
    pub(crate) fn reparameterize_with_sampled_score(
        &self,
        rng: &mut ChaCha8Rng,
    ) -> Result<PathwiseSample> {
        self.reparameterize_with_noise_and_sampled_score(self.gaussian_noise(rng)?)
    }
    fn gaussian_noise(&self, rng: &mut ChaCha8Rng) -> Result<Tensor<2>> {
        self.validate()?;
        let Self::Gaussian { mean, .. } = self else {
            return Err(Error::Action(
                "pathwise sampling requires a Gaussian; enumerate categorical actions".into(),
            ));
        };
        let [n, d] = mean.dims();
        let noise = matrix_on_device(
            (0..n * d)
                .map(|_| rng.sample::<f32, _>(StandardNormal))
                .collect(),
            n,
            d,
            &mean.device(),
        );
        Ok(noise)
    }
    /// Fixed-noise form for deterministic gradient checks. Noise is detached.
    pub fn reparameterize_with_noise(&self, noise: Tensor<2>) -> Result<PathwiseSample> {
        self.reparameterize_noise(noise, false)
    }
    pub(crate) fn reparameterize_with_noise_and_sampled_score(
        &self,
        noise: Tensor<2>,
    ) -> Result<PathwiseSample> {
        self.reparameterize_noise(noise, true)
    }
    fn reparameterize_noise(
        &self,
        noise: Tensor<2>,
        sampled_score: bool,
    ) -> Result<PathwiseSample> {
        self.validate()?;
        let Self::Gaussian {
            mean,
            log_std,
            bounds,
        } = self
        else {
            return Err(Error::Action(
                "pathwise sampling requires a Gaussian".into(),
            ));
        };
        ensure(
            noise.dims() == mean.dims() && to_vec(noise.clone())?.iter().all(|x| x.is_finite()),
            "invalid reparameterization noise",
        )?;
        let noise = noise.without_autodiff().to_device(&mean.device());
        let latents = mean.clone() + log_std.clone().exp() * noise.clone();
        let actions = if let Some((low, high)) = bounds {
            let [n, d] = mean.dims();
            let scale = matrix_on_device(
                low.iter().zip(high).map(|(l, h)| (h - l) * 0.5).collect(),
                1,
                d,
                &mean.device(),
            )
            .repeat_dim(0, n);
            let center = matrix_on_device(
                low.iter().zip(high).map(|(l, h)| (h + l) * 0.5).collect(),
                1,
                d,
                &mean.device(),
            )
            .repeat_dim(0, n);
            let actions = latents.clone().tanh() * scale + center;
            if sampled_score {
                // Affine f32 roundoff can leave an asymmetric Box at saturation.
                // Correct only the representable endpoints; the score is the
                // continuous transformed density, as elsewhere in this API.
                let low = matrix_on_device(low.clone(), 1, d, &mean.device()).repeat_dim(0, n);
                let high = matrix_on_device(high.clone(), 1, d, &mean.device()).repeat_dim(0, n);
                actions.max_pair(low).min_pair(high)
            } else {
                actions
            }
        } else {
            latents.clone()
        };
        let log_probs = if sampled_score {
            Self::gaussian_score(log_std.clone(), noise, latents.clone(), bounds)
        } else {
            Self::gaussian_log_prob(mean.clone(), log_std.clone(), latents.clone(), bounds)
        };
        if sampled_score {
            ensure(
                to_vec(actions.clone())?
                    .iter()
                    .chain(to_vec(log_probs.clone())?.iter())
                    .all(|x| x.is_finite()),
                "nonfinite pathwise action or sampled score",
            )?;
        }
        Ok(PathwiseSample {
            actions,
            log_probs,
            latents,
        })
    }
    fn validate(&self) -> Result<()> {
        match self {
            Self::Categorical { logits, masks } => {
                let [n, k] = logits.dims();
                ensure(
                    n > 0
                        && k > 0
                        && masks.len() == n * k
                        && masks.chunks(k).all(|row| row.iter().any(|x| *x)),
                    "invalid categorical shape or all-invalid live mask",
                )
            }
            Self::Gaussian {
                mean,
                log_std,
                bounds,
            } => {
                let [n, d] = mean.dims();
                ensure(
                    n > 0 && d > 0 && log_std.dims() == [n, d],
                    "invalid Gaussian tensor shapes",
                )?;
                if let Some((lo, hi)) = bounds {
                    ensure(
                        lo.len() == d
                            && hi.len() == d
                            && lo.iter().zip(hi).all(|(l, h)| {
                                l.is_finite() && h.is_finite() && h > l && (h - l).is_finite()
                            }),
                        "invalid Gaussian bounds",
                    )?;
                }
                Ok(())
            }
        }
    }
    fn invalid_mask(logits: &Tensor<2>, masks: &[bool]) -> Tensor<2, Bool> {
        Tensor::from_data(
            TensorData::new(masks.iter().map(|x| !*x).collect::<Vec<_>>(), logits.dims()),
            &logits.device(),
        )
    }
    fn categorical_log_probs(logits: &Tensor<2>, masks: &[bool]) -> Tensor<2> {
        log_softmax(
            logits
                .clone()
                .mask_fill(Self::invalid_mask(logits, masks), f32::NEG_INFINITY),
            1,
        )
    }
    fn categorical_indices(
        logits: &Tensor<2>,
        masks: &[bool],
        actions: &[Action],
    ) -> Result<Tensor<2, Int>> {
        let [n, k] = logits.dims();
        ensure(actions.len() == n, "action batch mismatch")?;
        let mut indices = Vec::with_capacity(n);
        for (i, a) in actions.iter().enumerate() {
            if let Action::Discrete(a) = a {
                ensure(*a < k && masks[i * k + *a], "invalid categorical action")?;
                indices.push(*a as i64);
            } else {
                return Err(Error::Action(
                    "categorical policy requires discrete actions".into(),
                ));
            }
        }
        Ok(Tensor::<2, Int>::from_data(
            TensorData::new(indices, [n, 1]),
            &logits.device(),
        ))
    }
    fn gaussian_log_prob(
        mean: Tensor<2>,
        log_std: Tensor<2>,
        latent: Tensor<2>,
        bounds: &Option<(Vec<f32>, Vec<f32>)>,
    ) -> Tensor<1> {
        let z = (latent.clone() - mean) / log_std.clone().exp();
        Self::gaussian_score(log_std, z, latent, bounds)
    }
    fn gaussian_score(
        log_std: Tensor<2>,
        z: Tensor<2>,
        latent: Tensor<2>,
        bounds: &Option<(Vec<f32>, Vec<f32>)>,
    ) -> Tensor<1> {
        let mut lp = z.powf_scalar(2.0) * (-0.5) - log_std - 0.5 * std::f64::consts::TAU.ln();
        if let Some((low, high)) = bounds {
            let n = latent.dims()[0];
            let d = low.len();
            let scales: Vec<_> = low
                .iter()
                .zip(high)
                .map(|(l, h)| ((h - l) * 0.5).ln())
                .collect();
            let affine = matrix_on_device(scales, 1, d, &latent.device()).repeat_dim(0, n);
            // log(1-tanh(z)^2) without subtraction near saturation.
            let jacobian = (latent.clone() * (-1.0) - softplus(latent * (-2.0), 1.0)
                + std::f64::consts::LN_2)
                * 2.0;
            lp = lp - affine - jacobian;
        }
        lp.sum_dim(1).squeeze_dim(1)
    }
    pub fn log_prob(&self, actions: &[Action], latents: &[Option<Vec<f32>>]) -> Result<Tensor<1>> {
        self.validate()?;
        match self {
            Self::Categorical { logits, masks } => {
                let index = Self::categorical_indices(logits, masks, actions)?;
                Ok(Self::categorical_log_probs(logits, masks)
                    .gather(1, index)
                    .squeeze_dim(1))
            }
            Self::Gaussian {
                mean,
                log_std,
                bounds,
            } => {
                let [n, d] = mean.dims();
                ensure(
                    actions.len() == n && latents.len() == n,
                    "Gaussian batch mismatch",
                )?;
                let mut z = Vec::with_capacity(n * d);
                for (a, latent) in actions.iter().zip(latents) {
                    let Action::Continuous(a) = a else {
                        return Err(Error::Action(
                            "Gaussian policy requires continuous actions".into(),
                        ));
                    };
                    ensure(
                        a.len() == d && a.iter().all(|x| x.is_finite()),
                        "invalid Gaussian action",
                    )?;
                    if bounds.is_some() {
                        let latent = latent.as_ref().ok_or_else(|| {
                            Error::State(
                                "squashed likelihood requires the stored latent sample".into(),
                            )
                        })?;
                        ensure(
                            latent.len() == d && latent.iter().all(|x| x.is_finite()),
                            "invalid latent",
                        )?;
                        let (low, high) = bounds.as_ref().expect("bounded");
                        ensure(
                            (0..d).all(|i| {
                                let scale = (high[i] - low[i]) * 0.5;
                                let transformed =
                                    (high[i] + low[i]) * 0.5 + scale * latent[i].tanh();
                                a[i] >= low[i]
                                    && a[i] <= high[i]
                                    && (a[i] - transformed).abs() <= 1e-5 * scale.max(1.)
                            }),
                            "bounded action does not match its latent sample",
                        )?;
                        z.extend_from_slice(latent);
                    } else {
                        z.extend_from_slice(a);
                    }
                }
                Ok(Self::gaussian_log_prob(
                    mean.clone(),
                    log_std.clone(),
                    matrix_on_device(z, n, d, &mean.device()),
                    bounds,
                ))
            }
        }
    }
    /// PPO retains the categorical normalization until its KL early-stop decision.
    pub(crate) fn log_prob_with_categorical_cache(
        &self,
        actions: &[Action],
        latents: &[Option<Vec<f32>>],
    ) -> Result<(Tensor<1>, Option<Tensor<2>>)> {
        if let Self::Categorical { logits, masks } = self {
            self.validate()?;
            let index = Self::categorical_indices(logits, masks, actions)?;
            let normalized = Self::categorical_log_probs(logits, masks);
            let selected = normalized.clone().gather(1, index).squeeze_dim(1);
            Ok((selected, Some(normalized)))
        } else {
            Ok((self.log_prob(actions, latents)?, None))
        }
    }
    pub(crate) fn entropy_with_categorical_cache(
        &self,
        cache: Option<Tensor<2>>,
        rng: &mut ChaCha8Rng,
    ) -> Result<Tensor<1>> {
        if let (Self::Categorical { logits, masks }, Some(lp)) = (self, cache) {
            Ok(
                (lp.clone().exp() * lp.mask_fill(Self::invalid_mask(logits, masks), 0.))
                    .sum_dim(1)
                    .squeeze_dim(1)
                    .neg(),
            )
        } else {
            self.entropy(rng)
        }
    }
    /// Sample actions without likelihoods or persistent latent samples.
    pub fn sample_actions_into(
        &self,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        self.sample_into::<false, false>(mode, rng, actions, &mut Vec::new(), &mut Vec::new())
    }
    /// Fill action, likelihood and required latent buffers; leave values to the value model.
    pub fn sample_behavior_into(
        &self,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> Result<()> {
        self.sample_into::<true, true>(
            mode,
            rng,
            &mut out.actions,
            &mut out.log_probs,
            &mut out.latents,
        )
    }
    /// Q-based rollout sampling without a fabricated state value.
    pub fn sample_pathwise_into(
        &self,
        rng: &mut ChaCha8Rng,
        out: &mut PathwiseBehavior,
    ) -> Result<()> {
        self.sample_into::<true, true>(
            ActMode::Sample,
            rng,
            &mut out.actions,
            &mut out.log_probs,
            &mut out.latents,
        )
    }
    /// Episode training needs only actions and bounded-policy latents, never behavior likelihoods.
    /// Baseline values, when requested, are filled independently by the episode actor.
    pub fn sample_episode_into(
        &self,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut EpisodeBehavior,
    ) -> Result<()> {
        self.sample_into::<false, true>(
            mode,
            rng,
            &mut out.actions,
            &mut Vec::new(),
            &mut out.latents,
        )
    }
    /// Allocating convenience for action-only inference.
    pub fn sample(&self, mode: ActMode, rng: &mut ChaCha8Rng) -> Result<Vec<Action>> {
        let mut actions = Vec::new();
        self.sample_actions_into(mode, rng, &mut actions)?;
        Ok(actions)
    }
    fn sample_into<const LIKELIHOODS: bool, const LATENTS: bool>(
        &self,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
        log_probs: &mut Vec<f32>,
        latents: &mut Vec<Option<Vec<f32>>>,
    ) -> Result<()> {
        self.validate()?;
        match self {
            Self::Categorical { logits, masks } => {
                let [n, k] = logits.dims();
                let mut weights = to_vec(logits.clone())?;
                ensure(
                    weights
                        .iter()
                        .zip(masks)
                        .all(|(x, valid)| !valid || x.is_finite()),
                    "nonfinite categorical logits",
                )?;
                actions.resize(n, Action::Discrete(0));
                if LIKELIHOODS {
                    log_probs.resize(n, 0.);
                }
                if LATENTS {
                    latents.resize_with(n, || None);
                }
                for (i, row) in weights.chunks_mut(k).enumerate() {
                    let mask = &masks[i * k..(i + 1) * k];
                    let mut max = f32::NEG_INFINITY;
                    let mut chosen = 0;
                    for (j, (&logit, &valid)) in row.iter().zip(mask).enumerate() {
                        // Preserve the existing last-index tie behavior.
                        if valid && logit >= max {
                            max = logit;
                            chosen = j;
                        }
                    }
                    let mut sum = 0.;
                    if mode == ActMode::Sample || LIKELIHOODS {
                        for (weight, &valid) in row.iter_mut().zip(mask) {
                            *weight = if valid { (*weight - max).exp() } else { 0. };
                            sum += *weight;
                        }
                    }
                    if mode == ActMode::Sample {
                        // Exactly one owned uniform per row, including one-valid-action rows.
                        let threshold = rng.random::<f32>() * sum;
                        let mut cumulative = 0.;
                        for (j, (&weight, &valid)) in row.iter().zip(mask).enumerate() {
                            if valid && weight > 0. {
                                chosen = j;
                                cumulative += weight;
                                if threshold < cumulative {
                                    break;
                                }
                            }
                        }
                    }
                    actions[i] = Action::Discrete(chosen);
                    if LIKELIHOODS {
                        log_probs[i] = row[chosen].ln() - sum.ln();
                    }
                    if LATENTS {
                        latents[i] = None;
                    }
                }
            }
            Self::Gaussian {
                mean,
                log_std,
                bounds,
            } => {
                let [n, d] = mean.dims();
                let means = to_vec(mean.clone())?;
                ensure(
                    means.iter().all(|x| x.is_finite()),
                    "nonfinite Gaussian means",
                )?;
                // Deterministic actions require neither standard deviations nor densities.
                let log_stds = if mode == ActMode::Sample || LIKELIHOODS {
                    let values = to_vec(log_std.clone())?;
                    ensure(
                        values.iter().all(|x| x.is_finite()),
                        "nonfinite Gaussian log deviations",
                    )?;
                    values
                } else {
                    Vec::new()
                };
                actions.resize_with(n, || Action::Continuous(Vec::new()));
                if LIKELIHOODS {
                    log_probs.resize(n, 0.);
                }
                if LATENTS {
                    latents.resize_with(n, || None);
                }
                for i in 0..n {
                    if !matches!(actions[i], Action::Continuous(_)) {
                        actions[i] = Action::Continuous(Vec::new());
                    }
                    let Action::Continuous(action) = &mut actions[i] else {
                        unreachable!()
                    };
                    action.resize(d, 0.);
                    let stored_latent = if LATENTS && bounds.is_some() {
                        let z = latents[i].get_or_insert_with(Vec::new);
                        z.resize(d, 0.);
                        Some(z)
                    } else {
                        if LATENTS {
                            latents[i] = None;
                        }
                        None
                    };
                    let mut stored_latent = stored_latent;
                    let mut lp = 0.;
                    for j in 0..d {
                        let index = i * d + j;
                        let std = if mode == ActMode::Sample || LIKELIHOODS {
                            log_stds[index].exp()
                        } else {
                            0.
                        };
                        // Keep row-major StandardNormal draws identical across all paths.
                        let z = means[index]
                            + if mode == ActMode::Sample {
                                std * rng.sample::<f32, _>(StandardNormal)
                            } else {
                                0.
                            };
                        ensure(z.is_finite(), "nonfinite Gaussian sample")?;
                        action[j] = if let Some((low, high)) = bounds {
                            (low[j] + high[j]) * 0.5 + (high[j] - low[j]) * 0.5 * z.tanh()
                        } else {
                            z
                        };
                        ensure(action[j].is_finite(), "nonfinite Gaussian action")?;
                        if let Some(latent) = stored_latent.as_mut() {
                            latent[j] = z;
                        }
                        if LIKELIHOODS {
                            let normalized = (z - means[index]) / std;
                            lp += -0.5 * normalized * normalized
                                - log_stds[index]
                                - 0.5 * std::f32::consts::TAU.ln();
                            if let Some((low, high)) = bounds {
                                // Absolute-value form avoids overflow and cancellation at either tail.
                                let jacobian = 2.
                                    * (std::f32::consts::LN_2
                                        - z.abs()
                                        - (-2. * z.abs()).exp().ln_1p());
                                lp -= ((high[j] - low[j]) * 0.5).ln() + jacobian;
                            }
                        }
                    }
                    if LIKELIHOODS {
                        log_probs[i] = lp;
                    }
                }
            }
        }
        if LIKELIHOODS {
            ensure(
                log_probs.iter().all(|x| x.is_finite()),
                "nonfinite distribution log-probability",
            )?;
        }
        Ok(())
    }
    /// Analytic entropy except for tanh Gaussians, which use a reparameterized sample.
    pub fn entropy(&self, rng: &mut ChaCha8Rng) -> Result<Tensor<1>> {
        self.validate()?;
        Ok(match self {
            Self::Categorical { logits, masks } => {
                let lp = Self::categorical_log_probs(logits, masks);
                (lp.clone().exp() * lp.mask_fill(Self::invalid_mask(logits, masks), 0.))
                    .sum_dim(1)
                    .squeeze_dim(1)
                    .neg()
            }
            Self::Gaussian {
                mean,
                log_std,
                bounds: None,
            } => {
                let _ = mean;
                (log_std.clone() + 0.5 * (1. + std::f64::consts::TAU.ln()))
                    .sum_dim(1)
                    .squeeze_dim(1)
            }
            Self::Gaussian {
                mean,
                log_std,
                bounds,
            } => {
                let [n, d] = mean.dims();
                let eps = matrix_on_device(
                    (0..n * d)
                        .map(|_| rng.sample::<f32, _>(StandardNormal))
                        .collect(),
                    n,
                    d,
                    &mean.device(),
                );
                let z = mean.clone() + log_std.clone().exp() * eps;
                Self::gaussian_log_prob(mean.clone(), log_std.clone(), z, bounds).neg()
            }
        })
    }
    /// KL(self || other). A shared invertible action transform leaves Gaussian KL unchanged.
    pub fn kl(&self, other: &Self) -> Result<Tensor<1>> {
        self.validate()?;
        other.validate()?;
        match (self, other) {
            (
                Self::Categorical {
                    logits: a,
                    masks: ma,
                },
                Self::Categorical {
                    logits: b,
                    masks: mb,
                },
            ) => {
                ensure(
                    a.dims() == b.dims() && ma == mb,
                    "KL requires matching categorical supports",
                )?;
                let la = Self::categorical_log_probs(a, ma);
                let lb = Self::categorical_log_probs(b, mb);
                Ok(
                    (la.clone().exp() * (la - lb).mask_fill(Self::invalid_mask(a, ma), 0.))
                        .sum_dim(1)
                        .squeeze_dim(1),
                )
            }
            (
                Self::Gaussian {
                    mean: a,
                    log_std: sa,
                    bounds: ba,
                },
                Self::Gaussian {
                    mean: b,
                    log_std: sb,
                    bounds: bb,
                },
            ) => {
                ensure(
                    a.dims() == b.dims() && ba == bb,
                    "KL requires matching Gaussian transforms",
                )?;
                Ok((sb.clone() - sa.clone()
                    + ((sa.clone() * 2.).exp() + (a.clone() - b.clone()).powf_scalar(2.))
                        / ((sb.clone() * 2.).exp() * 2.)
                    - 0.5)
                    .sum_dim(1)
                    .squeeze_dim(1))
            }
            _ => Err(Error::Config("incompatible distributions for KL".into())),
        }
    }
}

#[cfg(test)]
mod categorical_reuse_tests {
    use super::*;
    use crate::nn::{matrix, to_vec};
    use rand::SeedableRng;

    #[test]
    fn cached_masked_likelihood_and_entropy_match_public_values_and_gradients() {
        let values = vec![2., 100., -1., -20., 0.5, 1.];
        let masks = vec![true, false, true, false, true, true];
        let actions = [Action::Discrete(0), Action::Discrete(2)];
        let latents = [None, None];
        let public_logits = matrix(values.clone(), 2, 3).require_grad();
        let cached_logits = matrix(values, 2, 3).require_grad();
        let public = Distribution::Categorical {
            logits: public_logits.clone(),
            masks: masks.clone(),
        };
        let cached = Distribution::Categorical {
            logits: cached_logits.clone(),
            masks,
        };
        let public_lp = public.log_prob(&actions, &latents).unwrap();
        let public_entropy = public.entropy(&mut ChaCha8Rng::seed_from_u64(9)).unwrap();
        let (cached_lp, cache) = cached
            .log_prob_with_categorical_cache(&actions, &latents)
            .unwrap();
        let cached_entropy = cached
            .entropy_with_categorical_cache(cache, &mut ChaCha8Rng::seed_from_u64(9))
            .unwrap();
        for (a, b) in to_vec(public_lp.clone())
            .unwrap()
            .iter()
            .zip(to_vec(cached_lp.clone()).unwrap())
        {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in to_vec(public_entropy.clone())
            .unwrap()
            .iter()
            .zip(to_vec(cached_entropy.clone()).unwrap())
        {
            assert!((a - b).abs() < 1e-6);
        }
        let public_grads = (public_lp.sum() + public_entropy.sum()).backward();
        let cached_grads = (cached_lp.sum() + cached_entropy.sum()).backward();
        let a = to_vec(public_logits.grad(&public_grads).unwrap()).unwrap();
        let b = to_vec(cached_logits.grad(&cached_grads).unwrap()).unwrap();
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
#[cfg(test)]
mod sampled_gaussian_score_tests {
    use super::*;
    use crate::nn::{matrix, to_vec};

    #[test]
    fn sampled_score_retains_the_known_noise_at_tiny_variance() {
        let distribution = Distribution::Gaussian {
            mean: matrix(vec![1.], 1, 1),
            log_std: matrix(vec![-20.], 1, 1),
            bounds: None,
        };
        let sample = distribution
            .reparameterize_with_noise_and_sampled_score(matrix(vec![1.], 1, 1))
            .unwrap();
        assert_eq!(to_vec(sample.latents).unwrap(), vec![1.]);
        let expected = 20. - 0.5 - 0.5 * std::f64::consts::TAU.ln();
        assert!((to_vec(sample.log_probs).unwrap()[0] as f64 - expected).abs() < 2e-6);
        // Preserve the historical explicit-latent score for existing callers.
        let legacy = distribution
            .reparameterize_with_noise(matrix(vec![1.], 1, 1))
            .unwrap();
        assert!((to_vec(legacy.log_probs).unwrap()[0] as f64 - expected - 0.5).abs() < 2e-6);
    }

    #[test]
    fn sampled_score_has_affine_tanh_jacobian_and_detached_noise() {
        let mean = matrix(vec![0.3, -0.4], 1, 2).require_grad();
        let logs = matrix(vec![-0.2, 0.1], 1, 2).require_grad();
        let noise = matrix(vec![0.7, -0.6], 1, 2).require_grad();
        let distribution = Distribution::Gaussian {
            mean: mean.clone(),
            log_std: logs.clone(),
            bounds: Some((vec![-2., 1.], vec![4., 5.])),
        };
        let sample = distribution
            .reparameterize_with_noise_and_sampled_score(noise.clone())
            .unwrap();
        let z = [
            0.3f64 + (-0.2f64).exp() * 0.7,
            -0.4f64 + 0.1f64.exp() * -0.6,
        ];
        let expected: f64 = (0..2)
            .map(|i| {
                -0.5 * [0.7f64, -0.6][i].powi(2)
                    - [-0.2f64, 0.1][i]
                    - 0.5 * std::f64::consts::TAU.ln()
                    - [3f64, 2.][i].ln()
                    - (1. - z[i].tanh().powi(2)).ln()
            })
            .sum();
        assert!((to_vec(sample.log_probs.clone()).unwrap()[0] as f64 - expected).abs() < 2e-6);
        let grads = sample.log_probs.sum().backward();
        assert!(noise.grad(&grads).is_none());
        let mean_grad = to_vec(mean.grad(&grads).unwrap()).unwrap();
        let log_grad = to_vec(logs.grad(&grads).unwrap()).unwrap();
        for i in 0..2 {
            assert!((mean_grad[i] as f64 - 2. * z[i].tanh()).abs() < 2e-6);
            let expected = -1. + 2. * z[i].tanh() * [-0.2f64, 0.1][i].exp() * [0.7f64, -0.6][i];
            assert!((log_grad[i] as f64 - expected).abs() < 2e-6);
        }
    }
}
