use super::RecurrentLightweightState;
use crate::{
    Error, Result,
    core::{ensure, structured::StructuredObservation},
    nn::to_vec,
};
use burn::tensor::{Device, Tensor};
use serde::{Deserialize, Serialize};

/// Host-owned factual recurrent memory and the current frame's new-event boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecurrentLightweightMemory {
    hidden: Vec<f32>,
    event_start: usize,
}

impl RecurrentLightweightMemory {
    pub(crate) fn zero(width: usize) -> Self {
        Self {
            hidden: vec![0.; width],
            event_start: 0,
        }
    }

    pub(crate) fn from_state(state: RecurrentLightweightState) -> Result<Self> {
        Ok(Self {
            hidden: to_vec(state.memory)?,
            event_start: state.event_start,
        })
    }

    pub(crate) fn on_device(&self, device: &Device) -> RecurrentLightweightState {
        RecurrentLightweightState {
            memory: Tensor::<1>::from_data(self.hidden.as_slice(), device)
                .reshape([1, self.hidden.len()]),
            event_start: self.event_start,
        }
    }

    pub(crate) fn validate(&self, width: usize) -> Result<()> {
        ensure(
            self.hidden.len() == width && self.hidden.iter().all(|value| value.is_finite()),
            "invalid recurrent lightweight memory",
        )
    }

    pub(crate) fn validate_at(
        &self,
        width: usize,
        observation: &StructuredObservation,
        preceding: Option<&StructuredObservation>,
    ) -> Result<()> {
        self.validate(width)?;
        let events = observation.events.get(self.event_start..).ok_or_else(|| {
            Error::State("recurrent event boundary exceeds the saved observation".into())
        })?;
        if let Some(preceding) = preceding {
            ensure(
                self.event_start == preceding.events.len(),
                "recurrent memory disagrees with the factual event boundary",
            )?;
        }
        observation.validate_event_links(events)
    }
}
