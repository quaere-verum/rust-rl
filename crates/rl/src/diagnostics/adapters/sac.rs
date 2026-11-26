use super::{
    ActorModel, Capture, NamedCritic, RoleCapture, learner_info, policy_identity,
    scalar_action_gradient,
};
use crate::{
    Result,
    algorithms::sac::Sac,
    core::{action::Action, actor::Actor, env::Observation, learner::Learner},
    diagnostics::{
        evaluation::StochasticActor,
        reference::ActionGradient,
        schema::{Chunk, RecordContext},
        session::{CaptureRequest, DiagnosticLearner, LearnerInfo},
        value::{
            ContinuationIdentity, EntropyTemperatureSource, EntropyTiming, LogDensityConvention,
            ReturnTarget,
        },
    },
};

impl DiagnosticLearner for Sac {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        let mut info = learner_info(
            "SAC",
            serde_json::json!({
                "training": self.config(),
                "policy": "state-dependent diagonal Gaussian with optional affine tanh transform",
                "alpha_at_starting_revision": self.alpha(),
                "critic_target": "one-step replay reward plus discounted minimum target Q minus alpha times next-action log density",
            }),
        )?;
        info.discount = Some(f64::from(self.config().gamma));
        Ok(info)
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(StochasticActor::new(&self.policy())))
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        // SAC bootstraps using the online actor, even with the target critics.
        Ok(Some((
            self.diagnostic_actor()?,
            policy_identity(context, "online", true),
        )))
    }

    fn diagnostic_action_gradient(
        &self,
        name: &str,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<ActionGradient>> {
        let critic = match name {
            "q1" => self.critic(),
            "q2" => self.second_critic(),
            _ => return Ok(None),
        };
        scalar_action_gradient(&critic, observation, action, self.actor().action_space())
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        let alpha = self.alpha();
        let policy = self.policy();
        let continuation = policy_identity(request.context, "online", true);
        for (role, first, second) in [
            ("online", self.critic(), self.second_critic()),
            ("target", self.target_critic(), self.target_second_critic()),
        ] {
            capture.role_with_target(
                RoleCapture {
                    role,
                    // This is a self-contained copy of the same online actor,
                    // which supplies next actions for both critic roles.
                    policy_name: "policy",
                    policy: ActorModel::StateGaussian(&policy),
                    critics: &[
                        NamedCritic::twin("q1", &first, 0),
                        NamedCritic::twin("q2", &second, 1),
                    ],
                    continuation: continuation.clone(),
                },
                ReturnTarget::SoftActionValue {
                    alpha,
                    temperature_source: EntropyTemperatureSource::Snapshot,
                    entropy_timing: EntropyTiming::ReplayBootstrap { n_step: 1 },
                    log_density: LogDensityConvention::PolicySampleInActionCoordinates,
                },
            )?;
        }
        Ok(capture.records)
    }
}
