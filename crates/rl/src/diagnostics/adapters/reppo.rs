//! REPPO uses the common recording path with a normalized HL-Gauss critic.
use super::*;
use crate::{
    algorithms::reppo::Reppo,
    diagnostics::value::{EntropyTemperatureSource, EntropyTiming, LogDensityConvention},
};

impl DiagnosticLearner for Reppo {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        Ok(LearnerInfo {
            algorithm: "REPPO",
            model: "rust-rl/reppo/categorical-q/1",
            configuration: serde_json::json!({
                "training": self.config(),
                "critic_target": {
                    "continuation": "frozen rollout behavior used to prepare soft TD(lambda) targets",
                    "snapshot_alpha": "current entropy multiplier at the recorded boundary, after the last completed learner update",
                },
            }),
            discount: None,
        })
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(StochasticActor::new(&self.policy())))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        let policy = self.policy();
        let critic = self.critic();
        let config = self.config();
        let mut continuation = policy_identity(request.context, "online", true);
        continuation.identity.push_str("/rollout-behavior");
        capture.role_with_target(
            RoleCapture {
                role: "online",
                policy_name: "policy",
                policy: ActorModel::Stochastic(&policy),
                critics: &[NamedCritic::new(
                    "soft_q",
                    CriticModel::Categorical(
                        &critic,
                        &config.hl_gauss,
                        config.auxiliary_coefficient > 0.,
                    ),
                )],
                continuation,
            },
            ReturnTarget::SoftActionValue {
                // This is the prospective multiplier of the recorded learner.
                // The preceding rollout targets used the pre-update multiplier.
                alpha: self.multipliers().0,
                temperature_source: EntropyTemperatureSource::Snapshot,
                entropy_timing: EntropyTiming::TransitionSuccessor,
                log_density: LogDensityConvention::PolicySampleInActionCoordinates,
            },
        )?;
        Ok(capture.records)
    }
}
