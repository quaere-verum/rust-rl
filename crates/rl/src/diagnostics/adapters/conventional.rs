//! Integrations for the standard stochastic state-value and scalar DDPG models.
use super::*;
use crate::algorithms::{ddpg::Ddpg, mdpo::Mdpo, reinforce::Reinforce, vmpo::Vmpo};

const BATCH_VALUE_TARGET: &str = "Unregularized trainer-produced returns under frozen pre-update rollout behavior. The trainer selects GAE and bootstrapping. Exact calibration requires that rollout policy.";

fn on_policy_records(
    request: &CaptureRequest<'_>,
    policy: &Policy,
    value: Option<Value>,
) -> Result<Vec<Chunk>> {
    let critic = value
        .as_ref()
        .map(|value| NamedCritic::new("value", CriticModel::State(value)));
    let mut continuation = policy_identity(request.context, "online", true);
    // The critic fits targets collected before the update. The deployed
    // snapshot describes the separately updated actor at this boundary.
    continuation.identity.push_str("/rollout-behavior");
    let mut capture = Capture::new(request);
    capture.role(RoleCapture {
        role: "online",
        policy_name: "policy",
        policy: ActorModel::Stochastic(policy),
        critics: critic.as_slice(),
        continuation,
    })?;
    Ok(capture.records)
}

impl DiagnosticLearner for Reinforce {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info(
            "REINFORCE",
            serde_json::json!({
                "training": self.config(),
                "state_value_target": self.config().baseline.then_some(
                    "Complete-episode discounted returns under frozen pre-update rollout behavior"
                ),
                "baseline_trained_at_start": self.config().baseline && self.updates() > 0,
            }),
        )
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(StochasticActor::new(&self.policy())))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let value = (self.config().baseline && self.updates() > 0).then(|| self.value());
        on_policy_records(request, &self.policy(), value)
    }
}

impl DiagnosticLearner for Vmpo {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        let (eta, alpha) = self.multipliers();
        learner_info(
            "V-MPO",
            serde_json::json!({
                "training": self.config(),
                "multipliers_at_start": { "eta": eta, "alpha": alpha },
                "state_value_target": BATCH_VALUE_TARGET,
            }),
        )
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(StochasticActor::new(&self.policy())))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        on_policy_records(request, &self.policy(), Some(self.value()))
    }
}

impl DiagnosticLearner for Mdpo {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info(
            "MDPO",
            serde_json::json!({
                "training": self.config(),
                "state_value_target": BATCH_VALUE_TARGET,
            }),
        )
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(StochasticActor::new(&self.policy())))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        on_policy_records(request, &self.policy(), Some(self.value()))
    }
}

impl<Q: DiagnosticContinuousQ> DiagnosticLearner for Ddpg<Deterministic, Q> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        learner_info("DDPG", self.training_config())
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(DeterministicActor::new(&self.policy())))
    }

    fn diagnostic_evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn diagnostic_action_gradient(
        &self,
        name: &str,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<ActionGradient>> {
        if name != "q" {
            return Ok(None);
        }
        self.critic()
            .diagnostic_gradient(observation, action, self.actor().action_space())
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        Ok(Some((
            Box::new(DeterministicActor::new(&self.target_policy())),
            policy_identity(context, "target", false),
        )))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut capture = Capture::new(request);
        let continuation = policy_identity(request.context, "target", false);
        for (role, name, policy, critic) in [
            ("online", "policy", self.policy(), self.critic()),
            (
                "target",
                "target_policy",
                self.target_policy(),
                self.target_critic(),
            ),
        ] {
            capture.role(RoleCapture {
                role,
                policy_name: name,
                policy: ActorModel::Deterministic(&policy),
                critics: &[NamedCritic::new("q", critic.diagnostic_scalar())],
                continuation: continuation.clone(),
            })?;
        }
        Ok(capture.records)
    }
}
