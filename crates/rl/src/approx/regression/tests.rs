use super::*;

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn solves_a_nonsymmetric_system() {
    let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 3.0, 0.0, 4.0]);
    let rhs = DVector::from_column_slice(&[-1.0, -4.0]);
    let (solution, diagnostics) = solve_system(matrix, rhs, 1e-12).unwrap();
    close(solution[0], 1.0, 1e-12);
    close(solution[1], -1.0, 1e-12);
    assert_eq!(diagnostics.rank, 2);
    close(
        diagnostics.condition_number.unwrap(),
        (29.0 + 585.0_f64.sqrt()) / 16.0,
        1e-12,
    );
    close(diagnostics.relative_residual, 0.0, 1e-12);
}

#[test]
fn solver_cutoff_is_relative_to_matrix_scale() {
    for scale in [1e-120, 1e120] {
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![scale, 1e-8 * scale]));
        let rhs = DVector::from_vec(vec![2.0 * scale, 3e-8 * scale]);
        let (solution, diagnostics) = solve_system(matrix, rhs, 1e-6).unwrap();
        close(solution[0], 2.0, 1e-12);
        assert_eq!(solution[1], 0.0);
        assert_eq!(diagnostics.rank, 1);
        assert_eq!(diagnostics.condition_number, None);
        close(diagnostics.singular_values[0] / scale, 1.0, 1e-12);
        close(diagnostics.singular_values[1] / scale, 1e-8, 1e-12);
        close(diagnostics.relative_residual, 1.5e-8, 1e-12);
    }
}

#[test]
fn zero_system_returns_a_zero_minimum_norm_solution() {
    let (solution, diagnostics) =
        solve_system(DMatrix::zeros(2, 2), DVector::repeat(2, 1.0), 1e-12).unwrap();
    assert_eq!(solution.as_slice(), &[0.0, 0.0]);
    assert_eq!(diagnostics.rank, 0);
    assert_eq!(diagnostics.condition_number, None);
    close(diagnostics.relative_residual, 1.0, 1e-12);
}

#[test]
fn rejects_an_unrepresentable_singular_value() {
    let matrix = DMatrix::from_column_slice(2, 1, &[f64::MAX, f64::MAX]);
    let rhs = DVector::from_vec(vec![0.0, 0.0]);
    assert!(matches!(
        RankAwareSvd::new(matrix, 1e-12).and_then(|factor| factor.solve(&rhs)),
        Err(Error::Numerical(_))
    ));
}
