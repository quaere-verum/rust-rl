#![cfg(feature = "neural")]

#[path = "support/planning_doors.rs"]
mod doors;
use burn::tensor::{Device, Tensor};
use doors::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::planning::PlanningBudget;
use rl::{
    Result,
    algorithms::OptimizerConfig,
    core::{action::ConditionalAction, env::Observation},
    data::{
        policy_supervision::PolicySupervision,
        temporal::{EpisodeStore, SequenceConfig, sample_at},
    },
    models::predictive::PlanningModel,
    nn::{
        optimization::optimize_validated,
        structured::{
            CandidatePolicy, LightweightPolicy, PackLimits, RecurrentLightweightPolicy,
            StructuredSample, candidate_supervision_loss, recurrent_supervision_loss,
        },
    },
    planning::{
        pomcp,
        targets::{
            FactualBinding, PolicyTarget, PublicDecision, SearchProvenance, SearchRevision,
            SearchTarget,
        },
    },
};
use std::num::{NonZeroU16, NonZeroU32};

fn provenance() -> SearchProvenance {
    SearchProvenance {
        model: SearchRevision {
            identity: [1; 32],
            revision: 1,
        },
        planner: SearchRevision {
            identity: [2; 32],
            revision: 1,
        },
        policy: None,
    }
}
fn public(decision: &PublicDecision<'_>) -> Result<Public> {
    let Observation::Structured(frame) = decision.observation() else {
        panic!()
    };
    Ok(Public {
        reading: if frame.scalars[0] == 0. {
            None
        } else {
            Some(frame.scalars[0] > 0.)
        },
    })
}
fn search_target(
    store: &EpisodeStore,
    index: usize,
) -> Result<SearchTarget<Public, ConditionalAction, bool>> {
    let binding = FactualBinding::capture([9; 32], store, 7, index)?;
    let public = public(&binding.verify([9; 32], store)?)?;
    let mut feasible = Vec::new();
    Doors.actions(&public, &mut feasible);
    let horizon = NonZeroU16::new(if index == 0 { 2 } else { 1 }).unwrap();
    let budget = PlanningBudget::Simulations {
        trajectories: NonZeroU32::new(1024).unwrap(),
        transitions: NonZeroU32::new(2048).unwrap(),
    };
    let search = pomcp::search(
        &Doors,
        &public,
        &feasible,
        pomcp::Config {
            max_depth: horizon,
            gamma: 0.9,
            exploration: 1.,
        },
        budget,
        &mut ChaCha8Rng::seed_from_u64(17),
    )?;
    assert_eq!(
        search.action,
        Some(if index == 0 { SCAN } else { guess(true) })
    );
    Ok(SearchTarget {
        version: 1,
        public,
        binding,
        policy: Some(PolicyTarget::from_visits(&feasible, &search.estimates, 1.)?),
        feasible,
        value: None,
        returns: returns(),
        horizon,
        factual_return: None,
        provenance: provenance(),
        budget,
        work: search.work,
        status: search.status,
        continuations: vec![],
    })
}
fn sequence(store: &EpisodeStore, index: usize) -> rl::data::temporal::SequenceSample {
    sample_at(
        store,
        7,
        index,
        &SequenceConfig {
            burn_in: 1,
            learn_len: 1,
            n_step: 1,
        },
    )
    .unwrap()
}

#[test]
fn stateless_factual_rows_support_feedforward_but_require_history_for_recurrence() -> Result<()> {
    use rl::data::temporal::{SampleLimits, SequenceBatch, SequenceOrigin, SequenceSampler};
    let store = factual_store()?;
    let config = SequenceConfig {
        burn_in: 0,
        learn_len: 1,
        n_step: 1,
    };
    let mut batch = SequenceBatch {
        config,
        samples: vec![],
    };
    SequenceSampler::new(17).sample_stateless_into(
        &store,
        &config,
        12,
        &SampleLimits::default(),
        &mut batch,
    )?;
    assert!(batch.samples.iter().any(|s| s.history_start > 0));
    let device = Device::flex().autodiff();
    let mut rng = ChaCha8Rng::seed_from_u64(37);
    let feedforward = LightweightPolicy::new(model_config().model, &mut rng, &device)?;
    let recurrent = RecurrentLightweightPolicy::new(model_config(), &mut rng, &device)?;
    for sample in &batch.samples {
        assert_eq!(sample.origin, SequenceOrigin::Stateless);
        let Observation::Structured(frame) = &sample.observations[0] else {
            panic!()
        };
        let keys = frame
            .candidates
            .iter()
            .filter(|c| c.enabled)
            .map(|c| c.action)
            .collect::<Vec<_>>();
        let teacher = vec![1. / keys.len() as f64; keys.len()];
        let row = PolicySupervision::new(sample, 0, &keys, &teacher)?;
        assert!(
            finite_loss(&candidate_supervision_loss(
                &feedforward,
                &[row],
                PackLimits::default()
            )?)?
            .is_finite()
        );
        assert!(recurrent_supervision_loss(&recurrent, &[row], PackLimits::default()).is_err());
        let target = search_target(&store, sample.history_start)?;
        assert!(
            target
                .join_supervision([9; 32], &store, &provenance(), sample, public)
                .is_err()
        );
        let mut malformed = sample.clone();
        malformed.observations.pop();
        assert!(PolicySupervision::new(&malformed, 0, &keys, &teacher).is_err());
        let mut context_only = sample.clone();
        context_only.trajectory_start = sample.history_start + 1;
        assert!(PolicySupervision::new(&context_only, 0, &keys, &teacher).is_err());
    }
    Ok(())
}

#[test]
fn search_and_mpo_teachers_update_an_existing_feedforward_policy() -> Result<()> {
    let store = factual_store()?;
    let target = search_target(&store, 0)?;
    let sample = sequence(&store, 0);
    let row = target.join_supervision([9; 32], &store, &provenance(), &sample, public)?;
    let config = model_config().model;
    let mut policy = LightweightPolicy::new(
        config,
        &mut ChaCha8Rng::seed_from_u64(37),
        &Device::flex().autodiff(),
    )?;
    let limits = PackLimits::default();
    let input = StructuredSample::from_observation(row.observation(), policy.input_schema())?;
    let before_p = policy.complete_probabilities(std::slice::from_ref(&input), limits)?[0].clone();
    let before = finite_loss(&candidate_supervision_loss(&policy, &[row], limits)?)?;
    let optimizer = OptimizerConfig {
        learning_rate: 0.005,
        ..Default::default()
    };
    let mut adam = optimizer.build()?;
    for _ in 0..12 {
        let loss = candidate_supervision_loss(&policy, &[row], limits)?;
        optimize_validated(&mut policy, &mut adam, optimizer.learning_rate, loss)?;
    }
    let after = finite_loss(&candidate_supervision_loss(&policy, &[row], limits)?)?;
    let after_p = policy.complete_probabilities(&[input], limits)?[0].clone();
    assert!(after < before, "CE {before} -> {after}");
    let distance = |p: &[f64]| {
        p.iter()
            .zip(row.probabilities())
            .map(|(p, q)| (p - q).abs())
            .sum::<f64>()
    };
    assert!(distance(&after_p) < distance(&before_p));
    let exact = exact_values(0.5, true, 2);
    assert!((exact[0] - 0.6885).abs() < 1e-12);
    let teacher = rl::objectives::mpo::e_step(&[1. / 3.; 3], &exact, &[true; 3], 3, 0.1)?;
    let mpo = PolicySupervision::new(&sample, 0, &target.feasible, &teacher.probabilities)?;
    let loss = candidate_supervision_loss(&policy, &[mpo], limits)?;
    assert!(finite_loss(&loss)?.is_finite());
    optimize_validated(&mut policy, &mut adam, optimizer.learning_rate, loss)?;
    Ok(())
}

#[test]
fn recurrent_search_supervision_uses_causal_history_and_selected_learning_rows() -> Result<()> {
    let store = factual_store()?;
    let target = search_target(&store, 1)?;
    let sample = sequence(&store, 1);
    assert_eq!(sample.loss_mask, [false, true]);
    assert_eq!(sample.learn_start, 1);
    let row = target.join_supervision([9; 32], &store, &provenance(), &sample, public)?;
    let mut policy = RecurrentLightweightPolicy::new(
        model_config(),
        &mut ChaCha8Rng::seed_from_u64(53),
        &Device::flex().autodiff(),
    )?;
    let limits = PackLimits::default();
    let before = finite_loss(&recurrent_supervision_loss(&policy, &[row], limits)?)?;
    let mut altered_future = sample.clone();
    altered_future.observations[2] = frame(Public {
        reading: Some(false),
    });
    let future_row =
        target.join_supervision([9; 32], &store, &provenance(), &altered_future, public)?;
    assert_eq!(
        before,
        finite_loss(&recurrent_supervision_loss(&policy, &[future_row], limits)?)?
    );
    let mut adam = OptimizerConfig::default().build()?;
    for _ in 0..16 {
        let loss = recurrent_supervision_loss(&policy, &[row], limits)?;
        optimize_validated(&mut policy, &mut adam, 0.005, loss)?;
    }
    let after = finite_loss(&recurrent_supervision_loss(&policy, &[row], limits)?)?;
    assert!(after < before, "recurrent CE {before} -> {after}");
    assert!(
        PolicySupervision::new(
            &sample,
            0,
            &[SCAN, guess(false), guess(true)],
            &[1., 0., 0.]
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn supervision_join_rejects_forged_sequence_prefix_and_provenance() -> Result<()> {
    let store = factual_store()?;
    let target = search_target(&store, 1)?;
    let sample = sequence(&store, 1);
    for changed in 0..4 {
        let mut forged = sample.clone();
        match changed {
            0 => forged.episode_id += 1,
            1 => forged.steps[0].reward = 3f32.into(),
            2 => {
                forged.observations[0] = frame(Public {
                    reading: Some(false),
                })
            }
            _ => forged.steps[1].reward = 3f32.into(),
        }
        assert!(
            target
                .join_supervision([9; 32], &store, &provenance(), &forged, public)
                .is_err()
        );
    }
    assert!(
        target
            .join_supervision([8; 32], &store, &provenance(), &sample, public)
            .is_err()
    );
    let mut stale = provenance();
    stale.planner.revision += 1;
    assert!(
        target
            .join_supervision([9; 32], &store, &stale, &sample, public)
            .is_err()
    );
    let mut missing = target.clone();
    missing.feasible.pop();
    missing.policy = Some(PolicyTarget::Probabilities(vec![1.]));
    assert!(
        missing
            .join_supervision([9; 32], &store, &provenance(), &sample, public)
            .is_err()
    );
    Ok(())
}

#[test]
fn recurrent_adapter_preserves_reference_detachment_and_gradients() -> Result<()> {
    use burn::{
        module::{Module, ModuleVisitor, Param},
        optim::GradientsParams,
    };
    use rl::{
        nn::{structured::StructuredHistoryInput, to_vec},
        objectives::policy_supervision::distribution_loss,
    };
    fn gradients(policy: &RecurrentLightweightPolicy, loss: Tensor<1>) -> Vec<f32> {
        struct Collect(GradientsParams, Vec<f32>);
        impl ModuleVisitor for Collect {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                if let Some(g) = self.0.get::<D>(p.id) {
                    self.1.extend(to_vec(g).unwrap());
                } else {
                    self.1.extend(vec![0.; p.val().dims().iter().product()]);
                }
            }
        }
        let mut collect = Collect(GradientsParams::from_grads(loss.backward(), policy), vec![]);
        policy.visit(&mut collect);
        collect.1
    }
    let store = factual_store()?;
    let target = search_target(&store, 1)?;
    let sample = sequence(&store, 1);
    let row = target.join_supervision([9; 32], &store, &provenance(), &sample, public)?;
    let policy = RecurrentLightweightPolicy::new(
        model_config(),
        &mut ChaCha8Rng::seed_from_u64(53),
        &Device::flex().autodiff(),
    )?;
    let limits = PackLimits::default();
    let loss = recurrent_supervision_loss(&policy, &[row], limits)?;
    let expected_loss = finite_loss(&loss)?;
    let actual = gradients(&policy, loss);
    let inputs = sample.exact_history_inputs()?.take(2).collect::<Vec<_>>();
    let reference = |learn_start| -> Result<Tensor<1>> {
        let evaluated = policy.evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start,
            }],
            limits,
        )?;
        distribution_loss(
            vec![evaluated[0].log_prob_many(1, row.actions())?],
            &[row.probabilities().iter().map(|&p| p as f32).collect()],
        )
    };
    let expected = gradients(&policy, reference(1)?);
    assert_eq!(actual, expected);
    let full = reference(0)?;
    assert_eq!(expected_loss, finite_loss(&full)?);
    let full = gradients(&policy, full);
    assert!(
        actual.iter().zip(full).any(|(a, b)| (a - b).abs() > 1e-8),
        "detachment must suppress prefix gradients"
    );
    Ok(())
}

#[test]
fn sampled_conditional_search_converges_to_independent_public_belief_dp() -> Result<()> {
    let exact = exact_values(0.5, true, 2);
    assert_eq!(&exact[1..], &[0.5, 0.5]);
    for simulations in [64, 256, 1024, 4096] {
        let mut correct = 0;
        let mut error = 0.;
        for seed in 0..16 {
            let search = pomcp::search(
                &Doors,
                &Public { reading: None },
                &[SCAN, guess(false), guess(true)],
                pomcp::Config {
                    max_depth: NonZeroU16::new(2).unwrap(),
                    gamma: 0.9,
                    exploration: 1.,
                },
                PlanningBudget::Simulations {
                    trajectories: NonZeroU32::new(simulations).unwrap(),
                    transitions: NonZeroU32::new(simulations * 2).unwrap(),
                },
                &mut ChaCha8Rng::seed_from_u64(seed),
            )?;
            correct += usize::from(search.action == Some(SCAN));
            error += search
                .estimates
                .iter()
                .zip(&exact)
                .map(|(a, q)| (a.value - q).abs())
                .sum::<f64>()
                / 3.;
        }
        eprintln!(
            "budget={simulations} optimal={correct}/16 value_mae={}",
            error / 16.
        );
        if simulations == 4096 {
            assert!(correct >= 14 && error / 16. < 0.05);
        }
    }
    Ok(())
}

fn finite_loss(loss: &Tensor<1>) -> Result<f64> {
    let value = f64::from(loss.clone().into_scalar::<f32>());
    assert!(value.is_finite());
    Ok(value)
}
