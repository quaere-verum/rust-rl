use crate::{Result, core::ensure};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

/// A shuffled epoch without replacement; the final short minibatch is retained.
pub fn shuffled_batches(
    len: usize,
    batch_size: usize,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<Vec<usize>>> {
    ensure(
        len > 0 && batch_size > 0,
        "empty dataset or zero minibatch size",
    )?;
    let mut indices: Vec<_> = (0..len).collect();
    indices.shuffle(rng);
    Ok(indices.chunks(batch_size).map(|x| x.to_vec()).collect())
}

/// The learner owns this storage for the whole update; shuffling matches `shuffled_batches`.
#[cfg(any(test, feature = "neural"))]
pub(crate) fn shuffled_indices_into(
    len: usize,
    batch_size: usize,
    rng: &mut ChaCha8Rng,
    indices: &mut Vec<usize>,
) -> Result<()> {
    ensure(
        len > 0 && batch_size > 0,
        "empty dataset or zero minibatch size",
    )?;
    indices.clear();
    indices.extend(0..len);
    indices.shuffle(rng);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{action::Action, env::Observation},
        data::rollout::{BootstrappedBatch, OnPolicyBatch, OnPolicyMetadata},
    };
    use rand::SeedableRng;

    #[test]
    fn retained_epoch_matches_public_order_and_rng_with_short_final_batch() {
        let mut original = ChaCha8Rng::seed_from_u64(73);
        let mut retained = original.clone();
        let mut indices = Vec::new();
        for len in [11, 7, 11] {
            let expected = shuffled_batches(len, 4, &mut original).unwrap();
            shuffled_indices_into(len, 4, &mut retained, &mut indices).unwrap();
            assert_eq!(
                indices.chunks(4).map(<[usize]>::to_vec).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(original, retained);
            assert_eq!(indices.len(), len);
        }
        let pointer = indices.as_ptr();
        shuffled_indices_into(8, 4, &mut retained, &mut indices).unwrap();
        assert_eq!(indices.as_ptr(), pointer);
    }

    #[test]
    #[ignore = "bounded cost comparison for 64 minibatches per epoch"]
    fn many_minibatches_compare_public_and_reused_selection_cost() {
        use std::{hint::black_box, time::Instant};
        let len = 1025;
        let batch = OnPolicyBatch {
            episode_count: 1,
            observations: (0..len)
                .map(|i| Observation::new(vec![i as f32; 8]))
                .collect(),
            actions: (0..len)
                .map(|i| Action::Continuous(vec![i as f32; 4]))
                .collect(),
            latents: vec![None; len],
            returns: vec![1.; len],
            discount_weights: vec![1.; len],
            metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
                log_probs: vec![0.; len],
                values: vec![0.; len],
                advantages: vec![1.; len],
            }),
        };
        let mut old_rng = ChaCha8Rng::seed_from_u64(93);
        let mut new_rng = old_rng.clone();
        const ROUNDS: usize = 32;
        let old_start = Instant::now();
        for _ in 0..ROUNDS {
            for selection in shuffled_batches(len, 16, &mut old_rng).unwrap() {
                black_box(batch.select(&selection).unwrap());
            }
        }
        let old = old_start.elapsed();
        let mut indices = Vec::new();
        let mut selected = OnPolicyBatch::default();
        let new_start = Instant::now();
        for _ in 0..ROUNDS {
            shuffled_indices_into(len, 16, &mut new_rng, &mut indices).unwrap();
            for selection in indices.chunks(16) {
                batch.select_validated_into(selection, &mut selected);
                black_box(&selected);
            }
        }
        let new = new_start.elapsed();
        assert_eq!(old_rng, new_rng);
        println!(
            "many_minibatches rounds={ROUNDS} rows={len} batch=16 public_us={:.1} reused_us={:.1}",
            old.as_secs_f64() * 1e6,
            new.as_secs_f64() * 1e6
        );
    }
}
