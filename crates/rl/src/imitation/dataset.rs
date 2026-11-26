//! Immutable, validated observation/action data. No rewards or teacher at deployment.
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::Observation,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "RecordedDemonstrations")]
pub struct Demonstrations {
    pub(super) space: ActionSpace,
    pub(super) observations: Vec<Observation>,
    pub(super) actions: Vec<Action>,
    #[serde(skip)]
    pub(super) latents: Vec<Option<Vec<f32>>>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordedDemonstrations {
    space: ActionSpace,
    observations: Vec<Observation>,
    actions: Vec<Action>,
}
impl TryFrom<RecordedDemonstrations> for Demonstrations {
    type Error = crate::Error;
    fn try_from(r: RecordedDemonstrations) -> Result<Self> {
        Self::new(r.space, r.observations, r.actions)
    }
}
impl Demonstrations {
    /// Bounded demonstrations must be strictly interior: endpoints have no
    /// finite inverse tanh. This constructor never clips or perturbs expert data.
    pub fn new(
        space: ActionSpace,
        observations: Vec<Observation>,
        actions: Vec<Action>,
    ) -> Result<Self> {
        space.validate()?;
        if !matches!(space, ActionSpace::Discrete { .. }) {
            crate::nn::gaussian_policy::validate_gaussian_space(&space)?;
        }
        ensure(
            !observations.is_empty() && observations.len() == actions.len(),
            "empty or misaligned demonstration data",
        )?;
        let width = observations[0].dense_values().len();
        ensure(width > 0, "empty demonstration observation")?;
        let mut latents = Vec::with_capacity(actions.len());
        for (obs, action) in observations.iter().zip(&actions) {
            obs.validate(width, &space, true)?;
            space.check(action, obs.dense_mask())?;
            let latent = if let (ActionSpace::Box { low, high }, Action::Continuous(values)) =
                (&space, action)
            {
                let mut z = Vec::with_capacity(values.len());
                for ((&a, &l), &h) in values.iter().zip(low).zip(high) {
                    ensure(
                        a > l && a < h,
                        "bounded demonstration at endpoint has no finite inverse; provide an interior action or use an unbounded policy contract",
                    )?;
                    // Log distances avoid rounding a strict interior value to
                    // +/-1 during normalization, even near an asymmetric bound.
                    let value =
                        (0.5 * ((a as f64 - l as f64).ln() - (h as f64 - a as f64).ln())) as f32;
                    ensure(value.is_finite(), "nonfinite demonstration inverse")?;
                    z.push(value);
                }
                Some(z)
            } else {
                None
            };
            latents.push(latent);
        }
        Ok(Self {
            space,
            observations,
            actions,
            latents,
        })
    }
    pub fn len(&self) -> usize {
        self.actions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
    pub fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&Observation, &Action)> {
        self.observations.iter().zip(&self.actions)
    }
}
