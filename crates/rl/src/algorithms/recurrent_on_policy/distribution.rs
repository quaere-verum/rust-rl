use crate::{
    Result,
    core::{action::Action, env::Observation, stateful_on_policy::DistributionSnapshot},
    nn::{distributions::Distribution, temporal_policy::distribution_from_snapshots},
};
use burn::tensor::{Device, Tensor};
use rand_chacha::ChaCha8Rng;

/// Row-aligned differentiable laws used by the recurrent policy objectives.
/// Each likelihood, entropy and KL tensor has one scalar per input row in the
/// same order as the requested temporal tokens.
pub trait RecurrentPolicyDistribution: Sized {
    type EntropyCache;

    fn log_prob(&self, actions: &[Action], latents: &[Option<Vec<f32>>]) -> Result<Tensor<1>>;
    fn log_prob_with_cache(
        &self,
        actions: &[Action],
        latents: &[Option<Vec<f32>>],
    ) -> Result<(Tensor<1>, Option<Self::EntropyCache>)> {
        Ok((self.log_prob(actions, latents)?, None))
    }
    fn entropy(&self, rng: &mut ChaCha8Rng) -> Result<Tensor<1>>;
    fn entropy_with_cache(
        &self,
        _cache: Option<Self::EntropyCache>,
        rng: &mut ChaCha8Rng,
    ) -> Result<Tensor<1>> {
        self.entropy(rng)
    }
    /// Row-wise KL(self || behavior) on matching supports.
    fn kl(&self, behavior: &Self) -> Result<Tensor<1>>;
    fn snapshot(&self, row: usize, observation: &Observation) -> Result<DistributionSnapshot>;
    fn from_snapshots(snapshots: &[DistributionSnapshot], device: &Device) -> Result<Self>;
}

impl RecurrentPolicyDistribution for Distribution {
    type EntropyCache = Tensor<2>;

    fn log_prob(&self, actions: &[Action], latents: &[Option<Vec<f32>>]) -> Result<Tensor<1>> {
        Distribution::log_prob(self, actions, latents)
    }

    fn log_prob_with_cache(
        &self,
        actions: &[Action],
        latents: &[Option<Vec<f32>>],
    ) -> Result<(Tensor<1>, Option<Self::EntropyCache>)> {
        self.log_prob_with_categorical_cache(actions, latents)
    }

    fn entropy(&self, rng: &mut ChaCha8Rng) -> Result<Tensor<1>> {
        Distribution::entropy(self, rng)
    }

    fn entropy_with_cache(
        &self,
        cache: Option<Self::EntropyCache>,
        rng: &mut ChaCha8Rng,
    ) -> Result<Tensor<1>> {
        self.entropy_with_categorical_cache(cache, rng)
    }

    fn kl(&self, behavior: &Self) -> Result<Tensor<1>> {
        Distribution::kl(self, behavior)
    }

    fn snapshot(&self, row: usize, _: &Observation) -> Result<DistributionSnapshot> {
        DistributionSnapshot::from_distribution(self, row)
    }

    fn from_snapshots(snapshots: &[DistributionSnapshot], device: &Device) -> Result<Self> {
        distribution_from_snapshots(snapshots, device)
    }
}
