use super::{
    ActorObjective, Batch, PreparedTargets, RecurrentReppo, RecurrentReppoPolicy, ReppoWindow,
    action_tensor,
};
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::data::temporal_on_policy::TemporalLearnerBinding;
use crate::{
    Result,
    algorithms::{recurrent_on_policy::minibatch::WindowBatch, reppo::ReppoConfig},
    core::{
        action::ActionSpace, checkpoint::LearnerRecord, ensure, env::Observation,
        temporal::TemporalExecutionMode,
    },
    data::temporal_on_policy::TemporalBatchKind,
    nn::{
        checkpoint::RecordBurnExt,
        temporal::TemporalEncoder,
        temporal_actor::TemporalActorState,
        temporal_models::{self, TemporalModelArchitecture},
        temporal_pathwise_actor::{TemporalPathwiseActor, TemporalPathwiseActorState},
        temporal_policy::{TemporalPolicy, distribution_from_snapshots},
        temporal_q::{TemporalQ, TemporalQArchitecture},
        to_vec,
    },
};
use burn::tensor::Tensor;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

impl RecurrentReppoPolicy for TemporalPolicy {
    type Critic = TemporalQ;
    type Actor = TemporalPathwiseActor;
    type PolicyArchitecture = TemporalModelArchitecture;
    type CriticArchitecture = TemporalQArchitecture;
    type CriticInput = (Tensor<2>, Tensor<2>);

    const RECORD_VERSION: u32 = 1;

    fn architectures(
        &self,
        critic: &TemporalQ,
    ) -> Result<(Self::PolicyArchitecture, Self::CriticArchitecture)> {
        Ok((self.architecture()?, critic.architecture()?))
    }

    fn validate_pair(&self, critic: &TemporalQ, config: &ReppoConfig) -> Result<()> {
        self.architectures(critic)?;
        ensure(
            self.spec() == critic.spec()
                && critic.output_dim() == config.hl_gauss.bins
                && critic.action_dim() == self.spec().action_width(),
            "recurrent REPPO policy/Q schema or HL-Gauss bins differ",
        )?;
        ensure(
            config.auxiliary_coefficient == 0. || critic.supports_auxiliary(),
            "recurrent REPPO critic lacks auxiliary capability",
        )
    }

    fn validate_binding(
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
        binding: &TemporalLearnerBinding,
        _: &ReppoConfig,
    ) -> Result<()> {
        policy.validate()?;
        critic.validate()?;
        ensure(
            binding.dense_spec() == Some(&policy.spec) && critic.model.spec == policy.spec,
            "recurrent REPPO policy and Q architectures differ from the collection binding",
        )
    }

    fn validate_record_models(
        record: &LearnerRecord,
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
    ) -> Result<()> {
        temporal_models::validate_record_model(
            record,
            "policy",
            policy,
            policy.spec.action_space.dimensions(),
        )?;
        TemporalQ::validate_record_model(record, "critic", critic)
    }

    fn restore_models(
        record: &LearnerRecord,
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
    ) -> Result<(Self, TemporalQ)> {
        let policy = record.load_model(
            "policy",
            &TemporalPolicy::from_architecture_with_mode(
                policy,
                &mut ChaCha8Rng::seed_from_u64(0),
                true,
            )?,
        )?;
        Ok((
            policy,
            TemporalQ::restore_from_record(record, "critic", critic)?,
        ))
    }

    fn publish(policy: Self, critic: TemporalQ, revision: u64) -> Result<TemporalPathwiseActor> {
        TemporalPathwiseActor::new(policy, critic, revision)
    }

    fn published_models(actor: &TemporalPathwiseActor) -> (&Self, &TemporalQ) {
        (actor.policy(), actor.critic())
    }

    fn with_revision(actor: &TemporalPathwiseActor, revision: u64) -> TemporalPathwiseActor {
        actor.with_revision(revision)
    }

    fn binding(actor: &TemporalPathwiseActor) -> Result<TemporalLearnerBinding> {
        TemporalLearnerBinding::new(
            TemporalBatchKind::Pathwise,
            actor.revision(),
            actor.model_identity(),
            actor.spec().clone(),
        )
    }

    fn supports_optimized_exact(&self, critic: &TemporalQ) -> bool {
        !self.observation_only() && !critic.observation_only()
    }

    fn windows<'a>(
        &self,
        critic: &TemporalQ,
        batch: &'a Batch,
        mode: TemporalExecutionMode,
    ) -> Result<WindowBatch<'a, TemporalPathwiseActor>> {
        let widths = self
            .supports_optimized_exact(critic)
            .then_some([self.encoder.output_dim(), critic.encoder.output_dim()]);
        WindowBatch::with_layout(batch, mode, self.spec(), widths)
    }

    fn audit_recordings(batch: &Batch, full: &ReppoWindow<'_>) -> Result<()> {
        let tokens = batch.tokens();
        let (policy, critic) = Self::published_models(&batch.behavior);
        let pw = policy.encoder.output_dim();
        let qw = critic.encoder.output_dim();
        let pf = to_vec(full.policy_history(policy, &tokens)?)?;
        let qf = to_vec(full.q_history(critic, &tokens)?)?;
        for (i, row) in batch.rows.iter().enumerate() {
            let sample = &batch.sequences.samples[row.sample];
            batch.behavior.validate_recording(
                &TemporalPathwiseActorState {
                    policy: TemporalActorState {
                        policy: pf[i * pw..(i + 1) * pw].to_vec(),
                        value: None,
                    },
                    critic: qf[i * qw..(i + 1) * qw].to_vec(),
                },
                &sample.observations[row.time],
                &sample.steps[row.time].action,
                row.latent.as_deref(),
                &row.distribution,
                row.log_prob,
                row.value,
            )?;
        }
        Ok(())
    }

    fn prepare_targets(
        learner: &RecurrentReppo<Self>,
        batch: &Batch,
        full: &ReppoWindow<'_>,
        rng: &mut ChaCha8Rng,
    ) -> Result<PreparedTargets> {
        learner.prepare_sampled_targets(batch, full, rng)
    }

    fn critic_input(
        critic: &TemporalQ,
        window: &ReppoWindow<'_>,
        tokens: &[(usize, usize)],
    ) -> Result<Self::CriticInput> {
        let history = window.q_history(critic, tokens)?;
        let actions = action_tensor(&window.actions(), &critic.spec().action_space)?;
        Ok((history, actions))
    }

    fn critic_logits(
        critic: &TemporalQ,
        (history, actions): &Self::CriticInput,
    ) -> Result<Tensor<2>> {
        critic.values_from_features(history.clone(), actions.clone())
    }

    fn predict_representation(
        critic: &TemporalQ,
        (history, actions): Self::CriticInput,
    ) -> Result<Tensor<2>> {
        critic.predict_representation_from_features(history, actions)
    }

    fn actor_for_window(
        learner: &RecurrentReppo<Self>,
        window: &ReppoWindow<'_>,
        tokens: &[(usize, usize)],
        observations: &[&Observation],
        rng: &mut ChaCha8Rng,
    ) -> Result<ActorObjective> {
        let snapshots = window.behavior_distributions();
        let behavior = distribution_from_snapshots(&snapshots, &learner.policy.device())?;
        learner.actor_objective_from_window(
            window.distribution(&learner.policy, tokens)?,
            observations,
            &behavior,
            rng,
            |critic| window.q_history(critic, tokens),
        )
    }

    fn entropy_support(&self, observation: &Observation) -> Option<usize> {
        match &self.spec().action_space {
            ActionSpace::Discrete { count } => {
                Some(observation.dense_mask().map_or(*count, |mask| {
                    mask.iter().filter(|&&enabled| enabled).count()
                }))
            }
            _ => None,
        }
    }
}
