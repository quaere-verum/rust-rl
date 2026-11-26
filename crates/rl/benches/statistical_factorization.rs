//! CPU preparation, solver and prediction measurements on fixed weighted designs.
//! Run as a benchmark so the counting allocator stays isolated from the library.

#[path = "support/counting_allocator.rs"]
mod counting_allocator;

use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    approx::regression::{DiagonalPenalty, Predictor, Regressor, WeightedRidge},
};
use serde::Serialize;
use std::{hint::black_box, time::Instant};

const CUTOFF: f64 = 1e-12;
const SOLUTION_TOLERANCE: f64 = 1e-8;
const PREDICTIONS: usize = 10_000;

#[derive(Serialize)]
struct Cost {
    milliseconds: f64,
    allocations: u64,
    live_bytes_delta: i128,
}

fn measure<T>(f: impl FnOnce() -> T) -> (T, Cost) {
    let before = counting_allocator::snapshot();
    let started = Instant::now();
    let result = f();
    let elapsed = started.elapsed();
    let after = counting_allocator::snapshot();
    (
        result,
        Cost {
            milliseconds: elapsed.as_secs_f64() * 1000.,
            allocations: after.0 - before.0,
            live_bytes_delta: i128::from(after.1) - i128::from(before.1),
        },
    )
}

#[derive(Serialize)]
struct CaseReport {
    rows: usize,
    base_width: usize,
    gates: usize,
    design: Cost,
    svd_prepare: Cost,
    svd_first_target: Cost,
    svd_reused_target: Cost,
    svd_rank: usize,
    augmented_matrix: Cost,
    qr_factor: Cost,
    qr_first_target: Cost,
    qr_reused_target: Cost,
    maximum_coefficient_difference: Option<f64>,
    prediction_count: usize,
    predictions: Cost,
}

fn features(rows: usize, base_width: usize, gates: usize, singular: bool) -> DMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(9322);
    let packed = DMatrix::from_fn(rows, base_width, |_, _| rng.random_range(-1.0..1.0));
    DMatrix::from_fn(rows, base_width * gates, |row, col| {
        let part = col / base_width;
        let coordinate = if singular && col == 1 {
            0
        } else {
            col % base_width
        };
        let mass = if gates == 1 {
            1.
        } else {
            // Nonuniform but everywhere supported fixed local basis weights.
            let descriptor = row as f64 / rows as f64;
            let weight = |gate| {
                let center = (gate as f64 + 0.5) / gates as f64;
                (-10. * (descriptor - center).powi(2)).exp()
            };
            weight(part) / (0..gates).map(weight).sum::<f64>()
        };
        mass * packed[(row, coordinate)]
    })
}

fn profile(
    rows: usize,
    base_width: usize,
    gates: usize,
    penalty: f64,
    singular: bool,
) -> Result<CaseReport> {
    let (x, design) = measure(|| features(rows, base_width, gates, singular));
    let width = x.ncols();
    let weights: Vec<_> = (0..rows).map(|i| 1. + (i % 5) as f64).collect();
    let mean_weight: f64 = weights.iter().sum();
    let weight_roots: Vec<_> = weights
        .iter()
        .map(|weight| (weight / mean_weight).sqrt())
        .collect();
    let y: Vec<_> = (0..rows).map(|i| (0.013 * i as f64).sin()).collect();
    let next_y: Vec<_> = y.iter().map(|v| 0.65 * v + 0.2).collect();
    let ridge = WeightedRidge {
        penalty: DiagonalPenalty::Uniform(penalty),
        relative_cutoff: CUTOFF,
    };
    let (svd, svd_prepare) = measure(|| ridge.prepare(&x, &weights));
    let svd = svd?;
    let (first, svd_first_target) = measure(|| svd(&y));
    let (model, report) = first?;
    let (second, svd_reused_target) = measure(|| svd(&next_y));
    let (second_model, _) = second?;

    let augmented_rows = rows + if penalty == 0. { 0 } else { width };
    let (design_matrix, augmented_matrix) = measure(|| {
        let mut matrix = DMatrix::zeros(augmented_rows, width);
        for col in 0..width {
            for row in 0..rows {
                matrix[(row, col)] = weight_roots[row] * x[(row, col)];
            }
            if penalty > 0. {
                matrix[(rows + col, col)] = penalty.sqrt();
            }
        }
        matrix
    });
    let ((factor, upper), qr_factor) = measure(|| {
        let factor = design_matrix.col_piv_qr();
        let upper = factor.r();
        (factor, upper)
    });
    let diagonal_max = upper.diagonal().iter().map(|d| d.abs()).fold(0., f64::max);
    let rank_screen = upper
        .diagonal()
        .iter()
        .all(|d| d.abs() > diagonal_max * CUTOFF);
    let solve = |target: &[f64]| {
        if !rank_screen {
            return None;
        }
        let mut rhs = DVector::zeros(augmented_rows);
        for i in 0..rows {
            rhs[i] = target[i] * weight_roots[i];
        }
        factor.q_tr_mul(&mut rhs);
        let mut coefficients = upper.solve_upper_triangular(&rhs.rows(0, width))?;
        factor.p().inv_permute_rows(&mut coefficients);
        Some(coefficients)
    };
    let (first_qr, qr_first_target) = measure(|| solve(&y));
    let (second_qr, qr_reused_target) = measure(|| solve(&next_y));
    let maximum_coefficient_difference = first_qr.zip(second_qr).map(|(a, b)| {
        a.iter()
            .zip(model.coefficients())
            .chain(b.iter().zip(second_model.coefficients()))
            .map(|(qr, svd)| (qr - svd).abs())
            .fold(0., f64::max)
    });
    if singular {
        assert!(maximum_coefficient_difference.is_none());
        assert!(report.solve.rank < width);
    } else if maximum_coefficient_difference.is_none_or(|error| error > SOLUTION_TOLERANCE) {
        return Err(Error::Numerical(
            "QR profiling design disagrees with weighted ridge".into(),
        ));
    }

    let input: Vec<_> = x.row(0).iter().copied().collect();
    let (predicted, predictions) = measure(|| -> Result<()> {
        for _ in 0..PREDICTIONS {
            black_box(model.predict(black_box(&input))?);
        }
        Ok(())
    });
    predicted?;
    Ok(CaseReport {
        rows,
        base_width,
        gates,
        design,
        svd_prepare,
        svd_first_target,
        svd_reused_target,
        svd_rank: report.solve.rank,
        augmented_matrix,
        qr_factor,
        qr_first_target,
        qr_reused_target,
        maximum_coefficient_difference,
        prediction_count: PREDICTIONS,
        predictions,
    })
}

fn main() -> Result<()> {
    let mut evidence = Vec::new();
    for (rows, width, gates, penalty, singular) in [
        (256, 12, 1, 0.005, false),
        (1024, 32, 1, 0.005, false),
        (1024, 16, 4, 0.005, false),
        (2048, 32, 4, 0.005, false),
        (256, 12, 1, 0., true),
    ] {
        eprintln!("profiling {rows} weighted rows, {width} basis columns and {gates} gates");
        evidence.push(profile(rows, width, gates, penalty, singular)?);
    }
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    Ok(())
}
