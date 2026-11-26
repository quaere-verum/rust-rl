//! Run the bounded reference comparison with `cargo test --test rare_loss_distribution -- --ignored --nocapture`.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::objectives::categorical::CategoricalSupport;
use std::time::Instant;

type Atom = (f64, f64);
const LOSS: f64 = 50.;
const TAIL: f64 = 0.05;

fn categorical_fixed_point(q: f64, gamma: f64, iterations: usize) -> (Vec<Atom>, f64) {
    let support = CategoricalSupport::uniform((-LOSS / (1. - gamma)) as f32, 0., 51).unwrap();
    let mut masses = vec![0.; support.atoms().len()];
    *masses.last_mut().unwrap() = 1.;
    let mut clipping = 0.;
    for _ in 0..iterations {
        let quiet = support.project(0., gamma, &masses).unwrap();
        let loss = support.project(-LOSS, gamma, &masses).unwrap();
        clipping = (1. - q) * (quiet.clipped_below + quiet.clipped_above)
            + q * (loss.clipped_below + loss.clipped_above);
        for (target, (quiet, loss)) in masses
            .iter_mut()
            .zip(quiet.probabilities.into_iter().zip(loss.probabilities))
        {
            *target = (1. - q) * quiet + q * loss;
        }
    }
    (
        support
            .atoms()
            .iter()
            .copied()
            .map(f64::from)
            .zip(masses)
            .collect(),
        clipping,
    )
}

fn quantile_fixed_point(q: f64, gamma: f64, count: usize, iterations: usize) -> Vec<Atom> {
    let step = 1. / count as f64;
    let mut quantiles = vec![0.; count];
    let mut mixture = Vec::with_capacity(2 * count);
    for _ in 0..iterations {
        mixture.clear();
        for &value in &quantiles {
            mixture.push((gamma * value, (1. - q) * step));
            mixture.push((gamma * value - LOSS, q * step));
        }
        mixture.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut index = 0;
        let mut cumulative = mixture[0].1;
        for (i, value) in quantiles.iter_mut().enumerate() {
            let probability = (i as f64 + 0.5) * step;
            while cumulative < probability && index + 1 < mixture.len() {
                index += 1;
                cumulative += mixture[index].1;
            }
            *value = mixture[index].0;
        }
    }
    quantiles.into_iter().map(|value| (value, step)).collect()
}

fn reference(q: f64, gamma: f64, count: usize, seed: u64) -> Vec<Atom> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Stop at a horizon with worst-case omitted absolute return below 1e-8.
    let horizon = reference_horizon(gamma);
    let elapsed_discounts: Vec<_> = (0..horizon).map(|t| gamma.powi(t as i32)).collect();
    let no_loss_log = (-q).ln_1p();
    let mass = 1. / count as f64;
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let mut elapsed = 0;
        let mut losses = 0.;
        // Geometric waiting times sample the Bernoulli loss process without
        // simulating intervening zero rewards. The first wait can be zero.
        loop {
            let waiting = ((-rng.random::<f64>()).ln_1p() / no_loss_log) as usize;
            let Some(discount) = elapsed_discounts.get(elapsed + waiting) else {
                break;
            };
            losses -= LOSS * discount;
            elapsed += waiting + 1;
        }
        samples.push((losses, mass));
    }
    samples.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    samples
}

fn reference_horizon(gamma: f64) -> usize {
    ((1e-8 * (1. - gamma) / LOSS).ln() / gamma.ln()).ceil() as usize
}

#[derive(Debug)]
struct Summary {
    mean: f64,
    variance: f64,
    cvar: f64,
    severe_mass: f64,
}

fn summarize(atoms: &[Atom]) -> Summary {
    let mean = atoms.iter().map(|(value, mass)| value * mass).sum::<f64>();
    let variance = atoms
        .iter()
        .map(|(value, mass)| (value - mean).powi(2) * mass)
        .sum::<f64>();
    let mut remaining = TAIL;
    let mut tail_total = 0.;
    for &(value, mass) in atoms {
        let take = remaining.min(mass);
        tail_total += take * value;
        remaining -= take;
    }
    let severe_mass = atoms
        .iter()
        .filter(|(value, _)| *value <= -LOSS)
        .map(|(_, mass)| mass)
        .sum();
    Summary {
        mean,
        variance,
        cvar: tail_total / TAIL,
        severe_mass,
    }
}

fn wasserstein(a: &[Atom], b: &[Atom]) -> f64 {
    let (mut i, mut j) = (0, 0);
    let (mut mass_a, mut mass_b) = (a[0].1, b[0].1);
    let mut distance = 0.;
    while i < a.len() && j < b.len() {
        let mass = mass_a.min(mass_b);
        distance += mass * (a[i].0 - b[j].0).abs();
        mass_a -= mass;
        mass_b -= mass;
        if mass_a <= 0. {
            i += 1;
            mass_a = a.get(i).map_or(0., |atom| atom.1);
        }
        if mass_b <= 0. {
            j += 1;
            mass_b = b.get(j).map_or(0., |atom| atom.1);
        }
    }
    distance
}

#[test]
fn rare_loss_categorical_fixed_points_preserve_analytical_means() {
    for q in [0.02, 0.002] {
        for gamma in [0.9, 0.99] {
            let (prediction, clipped) = categorical_fixed_point(q, gamma, reference_horizon(gamma));
            let exact = -LOSS * q / (1. - gamma);
            let summary = summarize(&prediction);
            assert!(
                (summary.mean - exact).abs() < 1e-7,
                "{q} {gamma}: {} != {exact}",
                summary.mean
            );
            assert!(summary.severe_mass > 0.);
            assert!(clipped < 1e-15);
        }
    }
}

#[test]
#[ignore = "bounded distributional benchmark with two independent sample references per setting"]
fn compare_rare_loss_fixed_points() {
    const SAMPLES: usize = 200_000;
    println!(
        "q gamma method mean variance lower_CVaR05 W1 mass_at_or_below_-50 clipped_mass elapsed_s"
    );
    for q in [0.02, 0.002] {
        for gamma in [0.9, 0.99] {
            let start = Instant::now();
            let empirical = reference(q, gamma, SAMPLES, 804);
            let replica = reference(q, gamma, SAMPLES, 1948);
            let summary = summarize(&empirical);
            let exact_mean = -LOSS * q / (1. - gamma);
            let exact_variance = LOSS.powi(2) * q * (1. - q) / (1. - gamma.powi(2));
            let mean_standard_error = (exact_variance / SAMPLES as f64).sqrt();
            for estimate in [&summary, &summarize(&replica)] {
                assert!((estimate.mean - exact_mean).abs() < 5. * mean_standard_error);
                assert!((estimate.variance / exact_variance - 1.).abs() < 0.12);
            }
            println!("{q} {gamma} analytical {exact_mean:.6} {exact_variance:.6} - - - - -");
            print_comparison(
                q,
                gamma,
                "reference",
                &summary,
                wasserstein(&empirical, &replica),
                0.,
                start,
            );

            let iterations = reference_horizon(gamma);
            let start = Instant::now();
            let (atoms, clipping) = categorical_fixed_point(q, gamma, iterations);
            print_comparison(
                q,
                gamma,
                "categorical51",
                &summarize(&atoms),
                wasserstein(&atoms, &empirical),
                clipping,
                start,
            );
            for count in [51, 200] {
                let start = Instant::now();
                let atoms = quantile_fixed_point(q, gamma, count, iterations);
                print_comparison(
                    q,
                    gamma,
                    if count == 51 {
                        "quantile51"
                    } else {
                        "quantile200"
                    },
                    &summarize(&atoms),
                    wasserstein(&atoms, &empirical),
                    0.,
                    start,
                );
            }
        }
    }
}

fn print_comparison(
    q: f64,
    gamma: f64,
    method: &str,
    summary: &Summary,
    distance: f64,
    clipping: f64,
    start: Instant,
) {
    println!(
        "{q} {gamma} {method} {:.6} {:.6} {:.6} {distance:.6} {:.6} {clipping:.3e} {:.3}",
        summary.mean,
        summary.variance,
        summary.cvar,
        summary.severe_mass,
        start.elapsed().as_secs_f64()
    );
}
