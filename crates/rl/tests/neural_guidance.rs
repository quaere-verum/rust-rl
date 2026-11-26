#![cfg(feature = "neural")]

#[path = "support/planning_doors.rs"]
mod doors;
use burn::{
    module::Param,
    tensor::{Device, Tensor},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::planning::PlanningBudget;
use rl::{
    Result,
    core::{action::ConditionalAction, env::Observation},
    models::predictive::{PlanningModel, PlanningReturns, PredictiveModel, ReturnObjective},
    nn::{
        structured::{
            CandidatePolicy, LightweightPolicy, LightweightValue, NeuralGuidance, PackLimits,
            PublicValueInput, StructuredSample,
        },
        vector,
    },
    planning::{
        guidance::{
            PolicyGuide, SearchOptions, TreeSelection, ValueCutoff, ValueGuide, search_with_options,
        },
        pomcp,
    },
};
use std::num::{NonZeroU16, NonZeroU32};

// Domain-owned, purely public projection. No hidden particle or physical Env
// handle is available here. Bounds cover every legal continuation.
fn project(public: &doors::Public) -> Result<PublicValueInput> {
    let Observation::Structured(observation) = doors::frame(*public) else {
        unreachable!()
    };
    Ok(PublicValueInput {
        observation: (*observation).clone(),
        terminal_within: NonZeroU16::new(if public.reading.is_none() { 2 } else { 1 }).unwrap(),
    })
}

fn models() -> Result<(LightweightPolicy, LightweightValue)> {
    let mut rng = ChaCha8Rng::seed_from_u64(39);
    let device = Device::flex().autodiff();
    Ok((
        LightweightPolicy::new(doors::model_config().model, &mut rng, &device)?,
        LightweightValue::new(doors::model_config().model, &mut rng, &device)?,
    ))
}

#[test]
fn public_projection_and_policy_probabilities_match_factual_inputs_exactly() -> Result<()> {
    let (policy, value) = models()?;
    let guide = NeuralGuidance::new(
        &policy,
        &value,
        doors::returns(),
        PackLimits::default(),
        project,
    )?;
    let store = doors::factual_store()?;
    for (index, reading) in [(0, None), (1, Some(true))] {
        let public = doors::Public { reading };
        let input = project(&public)?;
        let Observation::Structured(factual) = &store.episode(7).unwrap().observations[index]
        else {
            panic!()
        };
        assert_eq!(&input.observation, factual.as_ref());
        let mut actions = vec![];
        doors::Doors.actions(&public, &mut actions);
        let direct = policy
            .complete_probabilities(
                &[StructuredSample::from_observation(
                    factual,
                    policy.input_schema(),
                )?],
                PackLimits::default(),
            )?
            .remove(0);
        let direct = direct
            .into_iter()
            .zip(&factual.candidates)
            .filter_map(|(p, a)| a.enabled.then_some(p))
            .collect::<Vec<_>>();
        assert_eq!(guide.prior(&public, &actions)?, direct);
        assert!(guide.value(&public, 2, &doors::returns())?.is_finite());
        let mut reversed = actions.clone();
        reversed.reverse();
        for wrong in [
            reversed,
            actions[1..].to_vec(),
            vec![actions[0]; actions.len()],
        ] {
            assert!(guide.prior(&public, &wrong).is_err());
        }
    }
    // Hypothetical scans condition only the public reading. Two hidden worlds
    // yielding the same reading must have identical neural input and prediction.
    let root = doors::Public { reading: None };
    let mut observed = [None, None];
    for hidden in [false, true] {
        for seed in 0..100 {
            let mut particle = hidden;
            let transition = doors::Doors.simulate(
                &root,
                &mut particle,
                &doors::SCAN,
                &mut ChaCha8Rng::seed_from_u64(seed),
            )?;
            if transition.observation {
                let next = transition.next.unwrap();
                observed[usize::from(hidden)] = Some((
                    project(&next)?.observation,
                    guide.prior(&next, &[doors::guess(false), doors::guess(true)])?,
                    guide.value(&next, 1, &doors::returns())?,
                ));
                break;
            }
        }
    }
    assert!(observed.iter().all(Option::is_some));
    assert_eq!(observed[0], observed[1]);
    Ok(())
}

#[test]
fn full_return_values_reject_short_horizons_mismatched_objectives_and_nonfinite_inference()
-> Result<()> {
    let (policy, mut value) = models()?;
    let root = doors::Public { reading: None };
    let guide = NeuralGuidance::new(
        &policy,
        &value,
        doors::returns(),
        PackLimits::default(),
        project,
    )?;
    assert!(guide.value(&root, 1, &doors::returns()).is_err());
    assert!(guide.value(&root, 0, &doors::returns()).is_err());
    for returns in [
        PlanningReturns {
            gamma: 0.8,
            ..doors::returns()
        },
        PlanningReturns {
            objective: ReturnObjective::Surrogate([1; 32]),
            ..doors::returns()
        },
    ] {
        assert!(guide.value(&root, 2, &returns).is_err());
    }
    assert!(
        NeuralGuidance::new(
            &policy,
            &value,
            PlanningReturns {
                objective: ReturnObjective::Surrogate([1; 32]),
                ..doors::returns()
            },
            PackLimits::default(),
            project
        )
        .is_err()
    );
    let head = value.head.layers.last_mut().unwrap();
    head.bias = Some(Param::from_tensor(vector(&[f32::NAN])));
    let invalid = NeuralGuidance::new(
        &policy,
        &value,
        doors::returns(),
        PackLimits::default(),
        project,
    )?;
    assert!(invalid.value(&root, 2, &doors::returns()).is_err());
    let bad_projection = |public: &doors::Public| {
        let mut input = project(public)?;
        input.observation.scalars[0] = f32::INFINITY;
        Ok(input)
    };
    let invalid = NeuralGuidance::new(
        &policy,
        &value,
        doors::returns(),
        PackLimits::default(),
        bad_projection,
    )?;
    assert!(
        invalid
            .prior(
                &root,
                &[doors::SCAN, doors::guess(false), doors::guess(true)]
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn existing_mcts_uses_both_neural_guides_with_physical_discount_and_repeatable_work() -> Result<()>
{
    let (policy, mut value) = models()?;
    // Diagnostic constant head isolates cutoff arithmetic; it is not evidence
    // of learned value accuracy. Actual Adam fitting has a separate test.
    let head = value.head.layers.last_mut().unwrap();
    head.weight = Param::from_tensor(Tensor::zeros(
        head.weight.val().dims(),
        &head.weight.val().device(),
    ));
    head.bias = Some(Param::from_tensor(vector(&[0.85])));
    let guide = NeuralGuidance::new(
        &policy,
        &value,
        doors::returns(),
        PackLimits::default(),
        project,
    )?;
    let root = doors::Public { reading: None };
    let actions: [ConditionalAction; 3] = [doors::SCAN, doors::guess(false), doors::guess(true)];
    let run = |seed| {
        let mut options = SearchOptions::new(doors::returns());
        options.selection = TreeSelection::Puct { exploration: 1. };
        options.policy = Some(&guide);
        options.value = Some(ValueCutoff {
            depth: NonZeroU16::MIN,
            guide: &guide,
        });
        search_with_options(
            &doors::Doors,
            &root,
            &actions,
            pomcp::Config {
                max_depth: NonZeroU16::new(2).unwrap(),
                gamma: 0.9,
                exploration: 1.,
            },
            PlanningBudget::Simulations {
                trajectories: NonZeroU32::new(256).unwrap(),
                transitions: NonZeroU32::new(512).unwrap(),
            },
            &mut ChaCha8Rng::seed_from_u64(seed),
            options,
        )
    };
    for seed in [7, 17, 37] {
        let first = run(seed)?;
        let repeated = run(seed)?;
        assert_eq!(first.action, Some(doors::SCAN));
        assert_eq!(first.action, repeated.action);
        assert_eq!(first.estimates, repeated.estimates);
        assert_eq!(first.work, repeated.work);
        let scan = first
            .estimates
            .iter()
            .find(|e| e.action == doors::SCAN)
            .unwrap();
        assert!((scan.value - 0.9_f64.powi(2) * f64::from(0.85_f32)).abs() < 1e-12);
    }
    Ok(())
}

#[test]
#[ignore = "bounded release timing probe; run alone, with --nocapture"]
fn neural_guidance_latency_panel() -> Result<()> {
    use std::time::Instant;
    let (policy, value) = models()?;
    let guide = NeuralGuidance::new(
        &policy,
        &value,
        doors::returns(),
        PackLimits::default(),
        project,
    )?;
    let root = doors::Public { reading: None };
    let actions = [doors::SCAN, doors::guess(false), doors::guess(true)];
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.
    };
    for role in ["policy", "value"] {
        let mut timings = vec![];
        for index in 0..136 {
            let start = Instant::now();
            if role == "policy" {
                std::hint::black_box(guide.prior(&root, &actions)?);
            } else {
                std::hint::black_box(guide.value(&root, 2, &doors::returns())?);
            }
            if index >= 8 {
                timings.push(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        eprintln!(
            "{}",
            serde_json::json!({"case":"inference", "role":role, "median_ms":median(timings), "samples":128})
        );
    }
    for budget in [64, 256, 1024] {
        for mode in 0..4 {
            let mut times = vec![];
            let mut work = vec![];
            for seed in 0..8 {
                let mut options = SearchOptions::new(doors::returns());
                if mode & 1 != 0 {
                    options.selection = TreeSelection::Puct { exploration: 1. };
                    options.policy = Some(&guide);
                }
                if mode & 2 != 0 {
                    options.value = Some(ValueCutoff {
                        depth: NonZeroU16::MIN,
                        guide: &guide,
                    });
                }
                let start = Instant::now();
                let result = search_with_options(
                    &doors::Doors,
                    &root,
                    &actions,
                    pomcp::Config {
                        max_depth: NonZeroU16::new(2).unwrap(),
                        gamma: 0.9,
                        exploration: 1.,
                    },
                    PlanningBudget::Simulations {
                        trajectories: NonZeroU32::new(budget).unwrap(),
                        transitions: NonZeroU32::new(2 * budget).unwrap(),
                    },
                    &mut ChaCha8Rng::seed_from_u64(seed),
                    options,
                )?;
                times.push(start.elapsed().as_secs_f64() * 1000.);
                work.push(result.work);
            }
            eprintln!(
                "{}",
                serde_json::json!({"case":"search", "mode":mode, "simulations":budget,
                "median_ms":median(times), "work":work, "seeds":"0..8"})
            );
        }
    }
    Ok(())
}
