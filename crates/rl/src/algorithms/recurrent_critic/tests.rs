use super::*;
use crate::{
    algorithms::recurrent_mpo::{RecurrentMpo, RecurrentMpoConfig},
    core::{
        action::{Action, ActionSpace},
        env::{Observation, ObservationBatch},
    },
    data::temporal::{SequenceConfig, SequenceSample, TemporalStep},
    nn::{
        distributions::Distribution,
        network::Mlp,
        temporal::GruEncoder,
        temporal_batch::EpisodeInputCache,
        temporal_models::{PreparedHistory, gather_history_features},
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_work::EpisodeInferencePlan,
    },
    objectives::mpo::{e_step, weighted_projection},
};
use burn::{grad_clipping::GradientClipping, optim::GradientsParams};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn window(episode_id: u64, start: usize, ended: bool) -> SequenceSample {
    let end = start + 3;
    SequenceSample {
        episode_id,
        trajectory_start: 0,
        history_start: 0,
        previous: None,
        origin: SequenceOrigin::Reset,
        learn_start: start,
        burn_start: start - 2,
        observations: (0..=end)
            .map(|time| {
                Observation::dense(
                    vec![((17 * time + episode_id as usize) % 23) as f32 * 0.04],
                    Some(vec![true, time % 3 != 0]),
                )
            })
            .collect(),
        steps: (0..end)
            .map(|time| TemporalStep {
                action: Action::Discrete(usize::from(time % 3 == 1)),
                reward: ((time % 11) as f32 * 0.03 - 0.1).into(),
                terminated: ended && time + 1 == end,
                truncated: false,
                behavior: Default::default(),
            })
            .collect(),
        loss_mask: (0..end).map(|time| time >= start).collect(),
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
struct GradientBits {
    present: Vec<bool>,
    raw: Vec<u32>,
    clipped: Vec<u32>,
}

fn gradient_bits(model: &impl Module, loss: Tensor<1>) -> GradientBits {
    struct Visit<'a> {
        gradients: &'a GradientsParams,
        bits: GradientBits,
    }
    impl ModuleVisitor for Visit<'_> {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            let gradient = self.gradients.get::<D>(parameter.id);
            self.bits.present.push(gradient.is_some());
            if let Some(gradient) = gradient {
                self.bits
                    .raw
                    .extend(bits(to_vec(gradient.clone()).unwrap()));
                self.bits.clipped.extend(bits(
                    to_vec(GradientClipping::Norm(1e-6).clip_gradient(gradient)).unwrap(),
                ));
            }
        }
    }
    let gradients = GradientsParams::from_grads(loss.backward(), model);
    let mut visitor = Visit {
        gradients: &gradients,
        bits: GradientBits::default(),
    };
    model.visit(&mut visitor);
    assert_ne!(visitor.bits.raw, visitor.bits.clipped, "fixture must clip");
    visitor.bits
}

fn bits(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

#[test]
fn mpo_shared_tapes_preserve_projection_and_backup_gradients() {
    let batch = SequenceBatch {
        config: SequenceConfig {
            burn_in: 2,
            learn_len: 3,
            n_step: 1,
        },
        samples: vec![
            window(7, 60, true),
            window(8, 49, false),
            window(7, 40, false),
            window(7, 40, false),
        ],
    };
    batch.validate().unwrap();
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(654);
    let policy = TemporalPolicy::new(spec.clone(), Some(4), &[6], &mut rng)
        .unwrap()
        .train();
    let mut q = || {
        GruQ::new(
            GruEncoder::new(spec.input_dim(), 5, &mut rng).unwrap(),
            Mlp::new(5, &[7], 2, &mut rng).unwrap(),
        )
        .unwrap()
    };
    let mut critic = RecurrentDiscreteCritic::new(
        q(),
        spec.clone(),
        RecurrentCriticConfig {
            n_step: 1,
            ..Default::default()
        },
    )
    .unwrap();
    critic.target = frozen_copy(&q());
    let mpo = RecurrentMpo::new(policy.clone(), critic, RecurrentMpoConfig::default()).unwrap();
    let critic = mpo.critic();
    let cache = EpisodeInputCache::try_from_validated_replay(&batch, &spec)
        .unwrap()
        .unwrap();
    let compact = EpisodeInferencePlan::try_new(&cache, &[5], &device()).unwrap();
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        assert!(compact.is_some());
    }
    let history = PreparedHistory::new(cache.pack(&[0, 1, 2, 3]).unwrap(), None, &device());
    let (input, masks) = history.input_and_masks();
    let (online, target) = critic.reconstructed_q_values(&batch).unwrap();
    let forward = critic
        .full_q_forward(&batch, input.clone(), masks, compact.as_ref())
        .unwrap();
    assert_eq!(bits(online.clone()), bits(forward.selection.clone()));
    assert_eq!(bits(target), bits(forward.evaluation));

    let tokens: Vec<_> = batch
        .samples
        .iter()
        .enumerate()
        .flat_map(|(row, s)| (s.learn_start..s.steps.len()).map(move |t| (row, t)))
        .collect();
    let observations: Vec<_> = tokens
        .iter()
        .map(|&(row, t)| &batch.samples[row].observations[t])
        .collect();
    let current = policy.distribution(&batch, &tokens).unwrap();
    let shared = policy
        .distribution_from_features(
            gather_history_features(
                history
                    .unroll_features(&policy.encoder, &batch.samples)
                    .unwrap(),
                &tokens,
            ),
            ObservationBatch::from_refs(&observations),
        )
        .unwrap();
    let (
        Distribution::Categorical {
            masks: ref_mask, ..
        },
        Distribution::Categorical {
            masks: opt_mask, ..
        },
    ) = (&current, &shared)
    else {
        unreachable!()
    };
    assert_eq!(ref_mask, opt_mask);
    let expected = current.categorical_terms().unwrap();
    let actual = shared.categorical_terms().unwrap();
    assert_eq!(
        bits(to_vec(expected.log_probs.clone()).unwrap()),
        bits(to_vec(actual.log_probs.clone()).unwrap())
    );
    assert_eq!(
        bits(to_vec(expected.probabilities.clone().without_autodiff()).unwrap()),
        bits(to_vec(actual.probabilities.clone().without_autodiff()).unwrap()),
    );
    let e_step_from = |p: Tensor<2>, q: &[f32]| {
        let probabilities = to_vec(p.without_autodiff()).unwrap();
        let old: Vec<_> = probabilities
            .chunks_exact(2)
            .flat_map(|row| {
                let sum = f64::from(row[0]) + f64::from(row[1]);
                row.iter().map(move |&p| f64::from(p) / sum)
            })
            .collect();
        let q: Vec<_> = tokens
            .iter()
            .flat_map(|&(row, t)| {
                let i = (row * masks.time + t) * 2;
                [f64::from(q[i]), f64::from(q[i + 1])]
            })
            .collect();
        e_step(&old, &q, ref_mask, 2, 0.1).unwrap()
    };
    let expected_e = e_step_from(expected.probabilities, &online);
    let actual_e = e_step_from(actual.probabilities, &forward.selection);
    assert!(
        expected_e
            .probabilities
            .iter()
            .map(|p| p.to_bits())
            .eq(actual_e.probabilities.iter().map(|p| p.to_bits()))
    );
    assert_eq!(
        expected_e.reverse_kl.to_bits(),
        actual_e.reverse_kl.to_bits()
    );
    assert_eq!(
        expected_e.temperature.map(f64::to_bits),
        actual_e.temperature.map(f64::to_bits)
    );
    assert_eq!(expected_e.dual.to_bits(), actual_e.dual.to_bits());
    assert_eq!(expected_e.active, actual_e.active);
    let posterior = Tensor::from_data(
        TensorData::new(
            expected_e
                .probabilities
                .iter()
                .map(|p| *p as f32)
                .collect::<Vec<_>>(),
            [tokens.len(), 2],
        ),
        &device(),
    );
    let expected_loss = weighted_projection(expected.log_probs, posterior.clone());
    let actual_loss = weighted_projection(actual.log_probs, posterior);
    assert_eq!(
        bits(to_vec(expected_loss.clone()).unwrap()),
        bits(to_vec(actual_loss.clone()).unwrap())
    );
    assert_eq!(
        gradient_bits(&policy, expected_loss),
        gradient_bits(&policy, actual_loss)
    );

    let probabilities = mpo.policy_probabilities(&batch).unwrap();
    for backup in [
        CriticBackup::PolicyEvaluation(&probabilities),
        CriticBackup::DoubleQOptimality,
        CriticBackup::MaximumOptimality,
    ] {
        let expected = match backup {
            CriticBackup::PolicyEvaluation(p) => {
                critic.reference_objective_for_policy(&batch, Some(p))
            }
            CriticBackup::DoubleQOptimality => critic.reference_objective(&batch),
            CriticBackup::MaximumOptimality => critic.two_pass_objective(
                &batch,
                &encode_history_batch(&batch, &spec).unwrap(),
                backup,
            ),
        }
        .unwrap();
        let forward = critic
            .full_q_forward(&batch, input.clone(), masks, compact.as_ref())
            .unwrap();
        let actual = critic.full_objective(&batch, backup, forward).unwrap();
        assert_eq!(expected.targets, actual.targets);
        assert!(expected.targets.iter().zip(&actual.targets).all(|(a, b)| {
            a.value.to_bits() == b.value.to_bits() && a.discount.to_bits() == b.discount.to_bits()
        }));
        assert_eq!(
            bits(to_vec(expected.loss.clone()).unwrap()),
            bits(to_vec(actual.loss.clone()).unwrap())
        );
        assert_eq!(
            gradient_bits(&critic.online, expected.loss),
            gradient_bits(&critic.online, actual.loss)
        );
    }
}
