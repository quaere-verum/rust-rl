//! GRU state estimation and a portable hidden-state codec for temporal replay.
//! The reconstruction contract itself requires no action head or policy.
use super::recurrent_actor::RecurrentActor;
use crate::{
    Error, Result,
    core::{
        anchors::{AnchorSchema, MAX_ANCHOR_PAYLOAD_BYTES},
        ensure,
        reconstruction::ReconstructionModel,
        stateful::{HistoryInput, StatefulActor},
    },
};
use burn::nn::activation::Activation;
use sha2::{Digest, Sha256};

/// One lane's finite IEEE754 binary32 values, in explicit little-endian order.
pub const GRU_F32_LE_CODEC: u32 = 0x4752_5531;
pub const GRU_F32_LE_CODEC_VERSION: u32 = 1;

/// An immutable GRU representation with a cached state schema. The schema binds
/// history feature ordering and GRU controls, not trainable parameter values or
/// head architecture. Same-schema older parameters can therefore be decoded by
/// an explicitly approximate reconstruction mode.
///
/// Model identity deliberately reuses the actor's full content identity. Thus
/// even a head-only change is conservatively declared stale, although it leaves
/// GRU state evolution unchanged. The caller owns the parameter-generation clock;
/// it is independent of the actor's behavior/exploration revision.
#[derive(Debug, Clone)]
pub struct GruRepresentation {
    actor: RecurrentActor,
    generation: u64,
    schema: AnchorSchema,
    payload_bytes: usize,
}
impl GruRepresentation {
    pub fn new(actor: RecurrentActor, generation: u64) -> Result<Self> {
        let model = actor.model();
        let hidden = model.hidden_dim();
        let payload_bytes = hidden
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::Shape("GRU anchor payload length overflow".into()))?;
        ensure(
            payload_bytes > 0 && payload_bytes <= MAX_ANCHOR_PAYLOAD_BYTES,
            "GRU anchor payload exceeds resource limit",
        )?;
        let gru = &model.encoder.gru;
        ensure(
            matches!(&gru.gate_activation, Activation::Sigmoid(_))
                && matches!(&gru.hidden_activation, Activation::Tanh(_)),
            "unsupported GRU anchor activation schema",
        )?;
        let mut identity = Sha256::new();
        identity.update(b"rust-rl-gru-state-schema-v1\0");
        identity.update(GRU_F32_LE_CODEC.to_le_bytes());
        identity.update(GRU_F32_LE_CODEC_VERSION.to_le_bytes());
        // Keep this version in step with HistorySpec's canonical feature layout.
        identity.update(b"history-order-v1:observation,previous-action-onehot,previous-reward,previous-present\0");
        let history_spec = serde_json::to_vec(actor.spec())?;
        identity.update((history_spec.len() as u64).to_le_bytes());
        identity.update(history_spec);
        identity.update((model.input_dim() as u64).to_le_bytes());
        identity.update((hidden as u64).to_le_bytes());
        identity.update([u8::from(gru.reset_after)]);
        match gru.clip {
            Some(clip) => {
                identity.update([1]);
                identity.update(clip.to_bits().to_le_bytes());
            }
            None => identity.update([0]),
        }
        identity.update(b"gate=sigmoid;candidate=tanh;state=finite-ieee754-f32-le;prior=zero\0");
        let schema = AnchorSchema {
            codec: GRU_F32_LE_CODEC,
            codec_version: GRU_F32_LE_CODEC_VERSION,
            representation_identity: identity.finalize().into(),
            dimensions: vec![hidden],
        };
        schema.validate()?;
        Ok(Self {
            actor,
            generation,
            schema,
            payload_bytes,
        })
    }

    pub fn actor(&self) -> &RecurrentActor {
        &self.actor
    }
}
impl ReconstructionModel for GruRepresentation {
    type State = Vec<f32>;

    fn model_identity(&self) -> [u8; 32] {
        self.actor.model_identity()
    }
    fn generation(&self) -> u64 {
        self.generation
    }
    fn schema(&self) -> &AnchorSchema {
        &self.schema
    }
    fn initial_state(&self) -> Result<Self::State> {
        self.actor.initial_state()
    }
    fn validate_state(&self, state: &Self::State) -> Result<()> {
        self.actor.validate_state(state)
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        // Actor assimilation evaluates only the encoder, never its Q head.
        self.actor.assimilate(prior, input)
    }
    fn encode_state(&self, state: &Self::State) -> Result<Vec<u8>> {
        self.validate_state(state)?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(self.payload_bytes)
            .map_err(|e| Error::State(format!("cannot reserve GRU anchor payload: {e}")))?;
        for value in state {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        Ok(payload)
    }
    fn decode_state(&self, payload: &[u8]) -> Result<Self::State> {
        ensure(
            payload.len() == self.payload_bytes,
            "GRU anchor payload length differs from hidden-state schema",
        )?;
        let state = payload
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        self.validate_state(&state)?;
        Ok(state)
    }
}
