pub use super::common::OptimizerConfig;
use super::{
    common::{OnPolicyRecord, OnPolicyState, optimize},
    ppo_joint,
};
use crate::{
    Result,
    core::{
        actor::{Actor, OnPolicyActorRef},
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        learner::{Learner, OnPolicyLearner, UpdateReport},
    },
    data::{
        minibatch::shuffled_indices_into,
        rollout::{OnPolicyBatch, OnPolicyMetadata},
    },
    nn::{
        critics::scalar::{StateValue, Value},
        finite_loss,
        policy::{ActorCritic, Policy, StochasticPolicy},
        to_vec, vector,
    },
    objectives::on_policy::{ppo_clipped_surrogate, ppo_ratio_and_kl},
};
use burn::optim::GradientsParams;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    pub policy: OptimizerConfig,
    pub value: OptimizerConfig,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub clip: f32,
    pub entropy: f32,
    pub target_kl: Option<f32>,
}
impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            policy: Default::default(),
            value: Default::default(),
            epochs: 4,
            minibatch_size: 64,
            clip: 0.2,
            entropy: 0.,
            target_kl: Some(0.03),
        }
    }
}
impl PpoConfig {
    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.value.validate()?;
        ensure(
            self.epochs > 0
                && self.minibatch_size > 0
                && self.clip.is_finite()
                && self.clip > 0.
                && self.clip < 1.
                && self.entropy.is_finite()
                && self.entropy >= 0.
                && self.target_kl.is_none_or(|k| k.is_finite() && k > 0.),
            "invalid PPO configuration",
        )
    }
}

pub struct Ppo<P: StochasticPolicy = Policy, V: StateValue = Value> {
    pub(crate) state: OnPolicyState<P, V>,
    pub(crate) config: PpoConfig,
    joint: Option<JointOptimization>,
    normalize_minibatch_advantages: bool,
    update_settings: Option<PpoUpdateSettings>,
}

/// Effective values for PPO updates, optionally supplied by an external schedule.
///
/// Learning rates must be finite and nonnegative; `clip` must be in `[0, 1)`.
/// These values do not define schedule progression or change the base configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PpoUpdateSettings {
    pub policy_learning_rate: f64,
    pub value_learning_rate: f64,
    pub clip: f32,
}
impl PpoUpdateSettings {
    fn validate(&self) -> Result<()> {
        ensure(self.is_valid(), "invalid PPO update settings")
    }

    pub(super) fn is_valid(&self) -> bool {
        self.policy_learning_rate.is_finite()
            && self.policy_learning_rate >= 0.
            && self.value_learning_rate.is_finite()
            && self.value_learning_rate >= 0.
            && self.clip.is_finite()
            && (0. ..1.).contains(&self.clip)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JointOptimization {
    value_loss_coefficient: f32,
    max_grad_norm: f32,
}

pub(crate) fn normalize_advantages(values: &mut [f32]) {
    if values.len() <= 1 {
        return;
    }
    // Use the backend's f32 mean, like SB3, but accumulate squared deviations
    // in f64 so large finite values do not overflow during variance reduction.
    let mean_f32 = vector(values).mean().into_scalar::<f32>();
    let mean = if mean_f32.is_finite() {
        mean_f32 as f64
    } else {
        values.iter().map(|&x| x as f64).sum::<f64>() / values.len() as f64
    };
    let variance = values
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let scale_f64 = variance.sqrt();
    let scale = (scale_f64 as f32) + 1e-8;
    let mean_f32 = mean as f32;
    for x in values {
        let centered = *x - mean_f32;
        // Keep SB3-compatible f32 arithmetic unless centering or the scale
        // overflows for finite inputs; only those cases need wider arithmetic.
        *x = if centered.is_finite() && scale.is_finite() {
            centered / scale
        } else {
            ((*x as f64 - mean) / (scale_f64 + 1e-8)) as f32
        };
    }
}

impl JointOptimization {
    fn validate(&self, config: &PpoConfig) -> Result<()> {
        ensure(
            ppo_joint::valid_settings(config, self.value_loss_coefficient, self.max_grad_norm),
            "joint PPO requires finite settings and disabled per-tensor clipping",
        )
    }
}
impl<P: StochasticPolicy, V: StateValue> Ppo<P, V> {
    pub fn new(models: ActorCritic<P, V>, config: PpoConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: OnPolicyState::new(models, &config.policy, &config.value, seed)?,
            config,
            joint: None,
            normalize_minibatch_advantages: false,
            update_settings: None,
        })
    }
    /// Opt into one norm across actor and weighted critic gradients.
    ///
    /// Both optimizer configurations must disable their per-tensor clipping.
    /// Parameter identities must be unique within and between the two models.
    /// This does not change advantage normalization or other PPO settings.
    pub fn new_with_joint_gradient_clipping(
        models: ActorCritic<P, V>,
        config: PpoConfig,
        seed: u64,
        value_loss_coefficient: f32,
        max_grad_norm: f32,
    ) -> Result<Self> {
        let joint = JointOptimization {
            value_loss_coefficient,
            max_grad_norm,
        };
        config.validate()?;
        joint.validate(&config)?;
        crate::nn::optimization::validate_disjoint(&models.policy, &models.value)?;
        let mut learner = Self::new(models, config, seed)?;
        learner.joint = Some(joint);
        Ok(learner)
    }
    pub fn config(&self) -> &PpoConfig {
        &self.config
    }
    /// Current effective update values. `config()` continues to return the base configuration.
    pub fn update_settings(&self) -> PpoUpdateSettings {
        self.update_settings.unwrap_or(PpoUpdateSettings {
            policy_learning_rate: self.config.policy.learning_rate,
            value_learning_rate: self.config.value.learning_rate,
            clip: self.config.clip,
        })
    }
    /// Atomically set values for subsequent updates without resetting optimizer state.
    ///
    /// Call between updates; the caller owns and persists any schedule progression.
    /// Zero values are allowed. A zero learning rate leaves parameters unchanged but
    /// still advances Adam moments and update counters; it does not pause the optimizer.
    /// Negative learning rates and negative clipping are rejected.
    pub fn set_update_settings(&mut self, settings: PpoUpdateSettings) -> Result<()> {
        settings.validate()?;
        self.update_settings = Some(settings);
        Ok(())
    }
    /// Resume using the base configuration's learning rates and clip width.
    pub fn clear_update_settings(&mut self) {
        self.update_settings = None;
    }
    /// Normalize each selected minibatch with sample standard deviation.
    ///
    /// Supply raw advantages: with `OnPolicyTrainer`, set
    /// `OnPolicyConfig::normalize_advantages` to false. Singleton minibatches
    /// are left unchanged. The setting is preserved in checkpoints.
    pub fn with_minibatch_advantage_normalization(mut self, enabled: bool) -> Self {
        self.normalize_minibatch_advantages = enabled;
        self
    }
    pub fn updates(&self) -> u64 {
        self.state.updates
    }
    /// Independent inference snapshot, including running-state buffers.
    pub fn policy(&self) -> P {
        crate::nn::target::frozen_copy(&self.state.models.policy)
    }
    pub fn value(&self) -> V {
        crate::nn::target::frozen_copy(&self.state.models.value)
    }
}
impl<P: StochasticPolicy, V: StateValue> OnPolicyLearner for Ppo<P, V> {
    fn collection_actor(&self) -> OnPolicyActorRef<'_> {
        OnPolicyActorRef::Bootstrapped(&self.state.models)
    }
    fn normalizes_advantages_in_update(&self) -> bool {
        self.normalize_minibatch_advantages
    }
}
impl<P: StochasticPolicy, V: StateValue> Checkpoint for Ppo<P, V> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.state.record("ppo", &self.config)?;
        if let Some(joint) = &self.joint {
            record.metadata["joint_optimization"] = serde_json::to_value(joint)?;
        }
        if self.normalize_minibatch_advantages {
            record.metadata["normalize_minibatch_advantages"] = true.into();
        }
        if let Some(settings) = self.update_settings {
            record.metadata["update_settings"] = serde_json::to_value(settings)?;
        }
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("ppo")?;
        let metadata: OnPolicyRecord<PpoConfig> = serde_json::from_value(record.metadata.clone())?;
        metadata.config.validate()?;
        let joint: Option<JointOptimization> = record
            .metadata
            .get("joint_optimization")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?;
        let normalize_minibatch_advantages: bool = record
            .metadata
            .get("normalize_minibatch_advantages")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?
            .unwrap_or(false);
        let update_settings: Option<PpoUpdateSettings> = record
            .metadata
            .get("update_settings")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?;
        if let Some(settings) = &update_settings {
            settings.validate()?;
        }
        if let Some(joint) = &joint {
            joint.validate(&metadata.config)?;
        }
        let state = self.state.restored(
            record,
            &metadata,
            &metadata.config.policy,
            &metadata.config.value,
        )?;
        if joint.is_some() {
            crate::nn::optimization::validate_disjoint(&state.models.policy, &state.models.value)?;
        }
        *self = Self {
            state,
            config: metadata.config,
            joint,
            normalize_minibatch_advantages,
            update_settings,
        };
        Ok(())
    }
}
impl<P: StochasticPolicy, V: StateValue> Learner for Ppo<P, V> {
    type Batch = OnPolicyBatch;
    fn actor(&self) -> &dyn Actor {
        &self.state.models
    }
    fn update(&mut self, batch: &OnPolicyBatch) -> Result<UpdateReport> {
        batch.validate()?;
        batch.bootstrapped()?;
        self.config.validate()?;
        let settings = self.update_settings();
        settings.validate()?;
        let mut value_config = self.config.value.clone();
        value_config.learning_rate = settings.value_learning_rate;
        let mut report = UpdateReport::default();
        let mut indices = Vec::new();
        let mut b = OnPolicyBatch::default();
        'epochs: for _ in 0..self.config.epochs {
            shuffled_indices_into(
                batch.len(),
                self.config.minibatch_size,
                &mut self.state.rng,
                &mut indices,
            )?;
            for selection in indices.chunks(self.config.minibatch_size) {
                batch.select_validated_into(selection, &mut b);
                if self.normalize_minibatch_advantages
                    && let OnPolicyMetadata::Bootstrapped(metadata) = &mut b.metadata
                {
                    normalize_advantages(&mut metadata.advantages);
                }
                let metadata = b.bootstrapped()?;
                let dist = self
                    .state
                    .models
                    .policy
                    .distribution((&b.observations).into())?;
                let (logp, categorical_cache) = if self.config.entropy > 0. {
                    dist.log_prob_with_categorical_cache(&b.actions, &b.latents)?
                } else {
                    (dist.log_prob(&b.actions, &b.latents)?, None)
                };
                let (ratio, kl) = ppo_ratio_and_kl(logp, vector(&metadata.log_probs));
                let kl = to_vec(kl)?[0];
                if self.config.target_kl.is_some_and(|limit| kl > limit) {
                    report.early_stopped = true;
                    break 'epochs;
                }
                let adv = vector(&metadata.advantages);
                let mut loss = ppo_clipped_surrogate(ratio, adv, settings.clip);
                if self.config.entropy > 0. {
                    loss = loss
                        - dist
                            .entropy_with_categorical_cache(categorical_cache, &mut self.state.rng)?
                            .mean()
                            * self.config.entropy;
                }
                let (p, v) = if let Some(joint) = &self.joint {
                    let value_loss = (self.state.models.value.values((&b.observations).into())?
                        - vector(&b.returns))
                    .powf_scalar(2.)
                    .mean();
                    let p = finite_loss(&loss)?;
                    let v = finite_loss(&value_loss)?;
                    let mut gradients =
                        (loss + value_loss * joint.value_loss_coefficient).backward();
                    let mut pg =
                        GradientsParams::from_module(&mut gradients, &self.state.models.policy);
                    let mut vg =
                        GradientsParams::from_module(&mut gradients, &self.state.models.value);
                    let norm = ppo_joint::clip(
                        &self.state.models.policy,
                        &self.state.models.value,
                        &mut pg,
                        &mut vg,
                        joint.max_grad_norm,
                    )?;
                    self.state.models.policy = self.state.policy_optimizer.step(
                        settings.policy_learning_rate,
                        self.state.models.policy.clone(),
                        pg,
                    );
                    self.state.models.value = self.state.value_optimizer.step(
                        settings.value_learning_rate,
                        self.state.models.value.clone(),
                        vg,
                    );
                    report = report.metric("joint_gradient_norm", norm as f64);
                    (p, v)
                } else {
                    let p = optimize(
                        &mut self.state.models.policy,
                        &mut self.state.policy_optimizer,
                        settings.policy_learning_rate,
                        loss,
                    )?;
                    let v = self.state.fit_value(&b, &value_config)?;
                    (p, v)
                };
                self.state.updates += 1;
                report.updates += 1;
                report = report
                    .metric("policy_loss", p)
                    .metric("value_loss", v)
                    .metric("approx_kl", kl as f64);
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod normalization_parity_tests {
    use super::normalize_advantages;
    use sha2::{Digest, Sha256};

    #[test]
    fn lqg_saved_minibatch_normalization_matches_torch() {
        let raw = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sb3_lqg_advantage_normalization.json"
        ))
        .expect("missing LQG normalization fixture");
        let digest = format!("{:x}", Sha256::digest(&raw));
        assert_eq!(
            digest,
            "87f4a48099da0d0410e13801a45d73a9dfa1f5099b443563eef5d0b421c69908"
        );
        let fixture: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(
            fixture["source_sha256"],
            "d782b0f84844f0649ab08f132afb50d271ad431fa2ca1caae3cd828283fba7b2"
        );
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 256);
        let mut maximum = 0_f32;
        let mut different = 0_usize;
        let mut count = 0_usize;
        let mut exceeded = 0_usize;
        for case in cases {
            let mut actual: Vec<f32> = case["advantages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let expected: Vec<f32> = case["normalized"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(actual.len(), 64);
            assert_eq!(expected.len(), actual.len());
            normalize_advantages(&mut actual);
            for (a, e) in actual.into_iter().zip(expected) {
                assert!(a.is_finite() && e.is_finite());
                let error = (a - e).abs();
                maximum = maximum.max(error);
                exceeded += usize::from(error > 2e-6);
                different += usize::from(a.to_bits() != e.to_bits());
                count += 1;
            }
        }
        println!(
            "LQG advantage normalization: max absolute error {maximum:e}; bitwise differences {different}/{count}; above tolerance {exceeded}"
        );
        assert_eq!(
            exceeded, 0,
            "normalized advantages exceeded absolute tolerance"
        );
    }

    #[test]
    fn normalization_preserves_singletons_and_zeroes_constant_batches() {
        let mut singleton = [2.5];
        normalize_advantages(&mut singleton);
        assert_eq!(singleton, [2.5]);
        let mut constant = [3.; 4];
        normalize_advantages(&mut constant);
        assert_eq!(constant, [0.; 4]);
    }

    #[test]
    fn normalization_handles_finite_extreme_scale_without_overflow() {
        let mut scale_overflow = [f32::MAX, -f32::MAX, -f32::MAX];
        normalize_advantages(&mut scale_overflow);
        assert!(scale_overflow.iter().all(|x| x.is_finite()));
        assert!((scale_overflow[0] - 1.1547005).abs() < 1e-6);
        assert!((scale_overflow[1] + 0.57735026).abs() < 1e-6);
    }

    #[test]
    fn normalization_handles_finite_extreme_centering_without_overflow() {
        let mut subtraction_overflow = vec![f32::MAX; 100];
        subtraction_overflow.push(-f32::MAX);
        normalize_advantages(&mut subtraction_overflow);
        assert!(subtraction_overflow.iter().all(|x| x.is_finite()));
        let expected = -100_f32 / 101_f32.sqrt();
        assert!((subtraction_overflow[100] - expected).abs() < 1e-5);
    }
}
