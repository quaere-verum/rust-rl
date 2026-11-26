use super::{Minibatch, RecurrentOnPolicyPolicy, WindowBatch};
use crate::data::temporal_on_policy::TemporalLearnerBinding;
use crate::{
    Result,
    core::{
        checkpoint::LearnerRecord,
        ensure,
        stateful::{ReconstructionHistory, StatefulActor},
        temporal::TemporalExecutionMode,
    },
    data::temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch},
    nn::{
        checkpoint::RecordBurnExt,
        distributions::Distribution,
        temporal::TemporalEncoder,
        temporal_actor::{TemporalActor, TemporalActorState},
        temporal_models::{self, TemporalModelArchitecture},
        temporal_policy::TemporalPolicy,
        temporal_value::TemporalValue,
    },
};
use burn::tensor::{Device, Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

impl RecurrentOnPolicyPolicy for TemporalPolicy {
    type Value = TemporalValue;
    type Actor = TemporalActor;
    type Architecture = TemporalModelArchitecture;
    type Distribution = Distribution;

    const RECORD_VERSION: u32 = 2;

    fn policy_device(&self) -> Device {
        self.device()
    }

    fn architectures(
        &self,
        value: Option<&TemporalValue>,
    ) -> Result<(TemporalModelArchitecture, Option<TemporalModelArchitecture>)> {
        let policy = self.architecture()?;
        let value = value.map(TemporalValue::architecture).transpose()?;
        if let Some(value) = &value {
            ensure(
                value.spec == policy.spec,
                "temporal actor/value history specs differ",
            )?;
        }
        Ok((policy, value))
    }

    fn publish(policy: Self, value: Option<TemporalValue>, revision: u64) -> Result<TemporalActor> {
        TemporalActor::new(policy, value, revision)
    }

    fn published_models(actor: &TemporalActor) -> (&Self, Option<&TemporalValue>) {
        (actor.policy(), actor.value())
    }

    fn with_revision(actor: &TemporalActor, revision: u64) -> TemporalActor {
        actor.with_revision(revision)
    }

    fn binding(actor: &TemporalActor, kind: TemporalBatchKind) -> Result<TemporalLearnerBinding> {
        TemporalLearnerBinding::new(
            kind,
            actor.revision(),
            actor.model_identity(),
            actor.spec().clone(),
        )
    }

    fn restore_models(
        record: &LearnerRecord,
        policy: &TemporalModelArchitecture,
        value: Option<&TemporalModelArchitecture>,
        binding: &TemporalLearnerBinding,
    ) -> Result<(Self, Option<TemporalValue>)> {
        ensure(
            binding.dense_spec() == Some(&policy.spec),
            "invalid recurrent learner metadata or model capabilities",
        )?;
        temporal_models::validate_record_model(
            record,
            "policy",
            policy,
            policy.spec.action_space.dimensions(),
        )?;
        if let Some(value) = value {
            value.validate()?;
            ensure(
                value.spec == policy.spec && value.output_dim == 1,
                "recurrent checkpoint actor/value architecture mismatch",
            )?;
            temporal_models::validate_record_model(record, "value", value, 0)?;
        }
        // Audit both records before allocating either untrusted architecture.
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let policy = record.load_model(
            "policy",
            &TemporalPolicy::from_architecture_with_mode(policy, &mut rng, true)?,
        )?;
        let value = value
            .map(|architecture| {
                record.load_model(
                    "value",
                    &TemporalValue::from_architecture_with_mode(architecture, &mut rng, true)?,
                )
            })
            .transpose()?;
        Ok((policy, value))
    }

    fn supports_optimized_exact(&self, value: Option<&TemporalValue>) -> bool {
        value.is_some_and(|value| !self.observation_only() && !value.observation_only())
    }

    fn windows<'a>(
        &self,
        batch: &'a TemporalOnPolicyBatch<TemporalActor>,
        mode: TemporalExecutionMode,
        value: Option<&TemporalValue>,
    ) -> Result<WindowBatch<'a>> {
        let widths = value
            .filter(|value| self.supports_optimized_exact(Some(value)))
            .map(|value| [self.encoder.output_dim(), value.encoder.output_dim()]);
        WindowBatch::with_layout(batch, mode, self.spec(), widths)
    }

    fn distribution(
        &self,
        batch: &Minibatch<'_>,
        tokens: &[(usize, usize)],
        _: bool,
    ) -> Result<Distribution> {
        batch.distribution(self, tokens)
    }

    fn values(
        value: &TemporalValue,
        batch: &Minibatch<'_>,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<1>> {
        batch.values(value, tokens)
    }

    fn reconstruct_published_state(
        actor: &TemporalActor,
        history: ReconstructionHistory<'_>,
        mode: TemporalExecutionMode,
    ) -> Result<Option<TemporalActorState>> {
        if mode.is_reference() {
            Ok(None)
        } else {
            actor.reconstruct_encoded_history(history)
        }
    }
}
