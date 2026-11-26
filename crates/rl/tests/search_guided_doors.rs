#![cfg(feature = "neural")]

#[path = "support/search_guided_doors/domain.rs"]
mod domain;
#[path = "support/search_guided_doors/reference.rs"]
mod reference;
use domain::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    core::{action::Action, env::Env},
    models::predictive::PredictiveModel,
};
use std::sync::Arc;

#[test]
fn post_scan_inputs_forget_sunk_cost_but_roots_and_factual_rewards_retain_it() -> Result<()> {
    let cheap = Public::root(Task::new(300, 50)?);
    let expensive = Public::root(Task::new(300, 450)?);
    assert_ne!(frame(cheap), frame(expensive));
    for reading in [false, true] {
        let a = Public {
            reading: Some(reading),
            ..cheap
        };
        let b = Public {
            reading: Some(reading),
            ..expensive
        };
        assert_eq!(frame(a), frame(b));
        assert_eq!(reference::action_values(a), reference::action_values(b));
        assert_eq!(frame(a).scalars[1], 0.);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(81);
    let mut hidden = false;
    let a = DoorsModel.simulate(&cheap, &mut hidden, &SCAN, &mut rng.clone())?;
    let b = DoorsModel.simulate(&expensive, &mut hidden, &SCAN, &mut rng)?;
    assert_eq!(a.reward, -f64::from(cheap.task.cost()));
    assert_eq!(b.reward, -f64::from(expensive.task.cost()));
    assert_eq!(frame(a.next.unwrap()), frame(b.next.unwrap()));
    Ok(())
}

#[test]
fn exact_reference_has_both_information_and_immediate_decisions() -> Result<()> {
    let informative = Public::root(Task::new(500, 50)?);
    let immediate = Public::root(Task::new(800, 50)?);
    assert_eq!(reference::optimal_action(informative), SCAN);
    assert_eq!(reference::optimal_action(immediate), guess(true));
    let values = reference::action_values(informative);
    assert!((values[0].1 - (0.9_f64.powi(2) * 0.7 - f64::from(0.05_f32))).abs() < 1e-12);
    assert_eq!(values[1].1, 0.);
    assert_eq!(values[2].1, 0.);
    for prior in [150, 300, 500, 700, 850] {
        for reading in [false, true] {
            let public = Public {
                task: Task::new(prior, 200)?,
                reading: Some(reading),
            };
            assert!(
                (public.probability_right() - reference::posterior_right(public)).abs() < 1e-14
            );
        }
    }
    let oracle = reference::deployed_value(informative, |p| Ok(reference::optimal_action(p)))?;
    let ignores_reading = reference::deployed_value(informative, |p| {
        Ok(if p.reading.is_none() {
            SCAN
        } else {
            guess(true)
        })
    })?;
    assert!((oracle - values[0].1).abs() < 1e-12);
    assert!((ignores_reading + f64::from(0.05_f32)).abs() < 1e-12);
    assert!(oracle - ignores_reading > 0.5);
    Ok(())
}

#[test]
fn frozen_policy_reference_forces_each_action_and_integrates_stochastic_guesses() -> Result<()> {
    let root = Public::root(Task::new(500, 100)?);
    let evaluation = reference::evaluate_policy(root, |public| {
        Ok(if let Some(reading) = public.reading {
            vec![
                (guess(false), if reading { 0.25 } else { 0.75 }),
                (guess(true), if reading { 0.75 } else { 0.25 }),
            ]
        } else {
            vec![(SCAN, 0.5), (guess(false), 0.2), (guess(true), 0.3)]
        })
    })?;
    let [(scan, after_scan), (left, after_left), (right, after_right)] =
        evaluation.actions.as_slice()
    else {
        panic!("every legal root action")
    };
    assert_eq!([*scan, *left, *right], [SCAN, guess(false), guess(true)]);
    let after_scan_return = -f64::from(root.task.cost()) + GAMMA.powi(2) * 0.35;
    assert!((after_scan.expected_return - after_scan_return).abs() < 1e-12);
    assert_eq!(after_left.expected_return, 0.);
    assert_eq!(after_right.expected_return, 0.);
    assert!((evaluation.deployed.expected_return - 0.5 * after_scan_return).abs() < 1e-12);
    assert!((evaluation.deployed.success - 0.5875).abs() < 1e-12);
    assert!((evaluation.deployed.success_ticks - 1.2625).abs() < 1e-12);
    assert_eq!(evaluation.deployed.scan_probability, 0.5);
    assert!(reference::evaluate_policy(root, |_| Ok(vec![(SCAN, 1.)])).is_err());
    Ok(())
}

#[test]
fn physical_and_predictive_sampling_agree_with_the_independent_joint_law() -> Result<()> {
    let task = Task::new(300, 150)?;
    let public = Public::root(task);
    let mut physical = DoorsEnv::new(vec![task])?;
    let mut planning = ChaCha8Rng::seed_from_u64(601);
    let mut physical_positive = 0;
    let mut imagined_positive = 0;
    let mut physical_success = 0;
    let mut imagined_success = 0;
    assert_eq!(decode(&physical.reset(Some(602))?.observation)?, public);
    for i in 0..6000 {
        if i != 0 {
            physical.reset(None)?;
        }
        let scan = physical.step(&Action::Conditional(Arc::new(SCAN)))?;
        assert_eq!(scan.reward.primitives(), &[-task.cost(), 0.]);
        assert!(!scan.ended());
        let rl::core::env::Observation::Structured(frame) = scan.observation else {
            panic!("structured frame")
        };
        let reading = frame.scalars[3] > 0.;
        physical_positive += usize::from(reading);
        let guessed = physical.step(&Action::Conditional(Arc::new(guess(reading))))?;
        assert!(guessed.terminated);
        assert_eq!(guessed.reward.duration(), 1);
        physical_success += usize::from(guessed.reward.total() > 0.);
        let mut hidden = DoorsModel.sample_root(&public, &mut planning)?;
        let scan = DoorsModel.simulate(&public, &mut hidden, &SCAN, &mut planning)?;
        assert_eq!(scan.reward, -f64::from(task.cost()));
        assert_eq!(scan.duration.get(), 2);
        imagined_positive += usize::from(scan.observation);
        let guessed = DoorsModel.simulate(
            &scan.next.unwrap(),
            &mut hidden,
            &guess(scan.observation),
            &mut planning,
        )?;
        assert!(guessed.next.is_none());
        imagined_success += usize::from(guessed.reward > 0.);
    }
    let positive = 0.3 * 0.85 + 0.7 * 0.15;
    for observed in [physical_positive, imagined_positive] {
        assert!((observed as f64 / 6000. - positive).abs() < 0.025);
    }
    for observed in [physical_success, imagined_success] {
        assert!((observed as f64 / 6000. - 0.85).abs() < 0.025);
    }
    Ok(())
}
