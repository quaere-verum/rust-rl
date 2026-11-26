use std::num::NonZeroU16;

use burn::tensor::{Device, Tensor, TensorData};
use rand_chacha::ChaCha8Rng;

use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        ensure,
        env::ObservationBatch,
    },
    nn::{
        critics::scalar::ActionValue,
        matrix_on_device,
        policy::{StochasticPolicy, checked_distribution, sample_executed_actions},
        to_vec,
    },
};

/// Monte Carlo continuation under the supplied pre-update continuous actor.
///
/// The caller supplies live successor observations and a target critic.
/// Samples use the actor's executed action coordinates and caller-owned RNG.
/// Multiply the result by each transition's continuation discount when
/// constructing Bellman targets. No gradient reaches the actor or critic.
pub fn sampled_current_policy_continuation(
    current_policy: &impl StochasticPolicy,
    target_critic: &impl ActionValue,
    observations: ObservationBatch<'_>,
    samples: NonZeroU16,
    rng: &mut ChaCha8Rng,
) -> Result<Tensor<1>> {
    ensure(
        !observations.is_empty()
            && matches!(
                current_policy.action_space(),
                ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
            )
            && current_policy.action_space().dimensions() == target_critic.action_dim()
            && target_critic.output_dim() == 1,
        "sampled continuous policy evaluation requires live states and a matching scalar critic",
    )?;
    let current_policy = current_policy.valid();
    let distribution = checked_distribution(&current_policy, observations)?;
    let action_space = current_policy.action_space();
    let mut decisions = Vec::new();
    let target = target_critic.valid();
    let mut sums = vec![0.0_f64; observations.len()];
    for _ in 0..samples.get() {
        sample_executed_actions(
            &distribution,
            action_space,
            observations,
            ActMode::Sample,
            rng,
            &mut decisions,
        )?;
        let mut packed = Vec::new();
        for action in &decisions {
            let Action::Continuous(values) = action else {
                unreachable!("checked continuous law")
            };
            packed.extend_from_slice(values);
        }
        let actions = matrix_on_device(
            packed,
            observations.len(),
            action_space.dimensions(),
            &Device::flex(),
        );
        let values = target.values(observations, actions)?;
        ensure(
            values.dims() == [observations.len(), 1],
            "scalar target critic returned an invalid shape",
        )?;
        for (sum, value) in sums.iter_mut().zip(to_vec(values)?) {
            ensure(
                value.is_finite(),
                "nonfinite continuous policy continuation",
            )?;
            *sum += f64::from(value);
        }
    }
    let means: Vec<f32> = sums
        .into_iter()
        .map(|sum| (sum / f64::from(samples.get())) as f32)
        .collect();
    Ok(Tensor::from_data(
        TensorData::new(means, [observations.len()]),
        &Device::flex(),
    ))
}
