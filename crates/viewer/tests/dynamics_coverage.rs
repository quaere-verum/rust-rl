#[path = "support/dynamics.rs"]
mod support;
use rl::diagnostics::schema::*;
use rl_viewer::dynamics::{
    Axes, Grid, Limits,
    history::{CheckpointOccupancy, DomainSpec, FixedGrid, HistoryLimits, coverage},
    occupancy, prepare_episode,
};

#[test]
fn manual_domains_validate_finite_ordered_spans_and_preserve_exact_identity() {
    for bounds in [
        [[0., 0.], [0., 1.]],
        [[2., 1.], [0., 1.]],
        [[f64::NAN, 1.], [0., 1.]],
        [[0., f64::INFINITY], [0., 1.]],
        [[-f64::MAX, f64::MAX], [0., 1.]],
    ] {
        assert!(DomainSpec::manual(bounds).is_err());
    }
    let manual = DomainSpec::manual([[-2., 3.], [-4., 5.]]).unwrap();
    assert_eq!(manual.manual_bounds(), Some([[-2., 3.], [-4., 5.]]));
    assert_eq!(manual, DomainSpec::manual([[-2., 3.], [-4., 5.]]).unwrap());
    assert_ne!(DomainSpec::automatic(0), DomainSpec::automatic(1));
    assert_ne!(manual, DomainSpec::automatic(0));
}

#[test]
fn descriptor_bounds_are_used_and_missing_bounds_freeze_to_the_reference_population() {
    let (_dir, mut manifest, chunk) = support::saved(support::bounded(1));
    manifest
        .environment_descriptor
        .as_mut()
        .unwrap()
        .observations[0]
        .range = Some([-5., 5.]);
    manifest
        .environment_descriptor
        .as_mut()
        .unwrap()
        .observations[1]
        .range = None;
    let episode = prepare_episode(&chunk, &manifest, &Limits::default()).unwrap();
    let axes = Axes { x: 0, y: 1 };
    let fixed = FixedGrid::from_reference(&[episode], axes, [2, 2], &Limits::default()).unwrap();
    assert_eq!(fixed.grid.bounds(), [[-5., 5.], [0., 2.]]);
    assert_eq!(fixed.empirical, [false, true]);
    let mut later = chunk.clone();
    later.context.identity.learner_update_revision = 99;
    later.arrays.insert(
        "obs".into(),
        Array::f64(vec![5, 3], vec![100.; 15]).unwrap(),
    );
    let later = prepare_episode(&later, &manifest, &Limits::default()).unwrap();
    let histogram = occupancy(&[later], axes, &fixed.grid).unwrap();
    assert_eq!(histogram.excluded, 4);
    assert_eq!(fixed.grid.bounds(), [[-5., 5.], [0., 2.]]);
}

#[test]
fn empirical_bounds_include_the_terminal_endpoint_and_pad_constant_axes() {
    let (_dir, mut manifest, mut chunk) = support::saved(support::bounded(1));
    manifest.environment_descriptor = None;
    chunk.arrays.insert(
        "obs".into(),
        Array::f64(
            vec![5, 3],
            vec![1., 0., 0., 1., 1., 0., 1., 2., 0., 1., 3., 0., 1., 9., 0.],
        )
        .unwrap(),
    );
    let episode = prepare_episode(&chunk, &manifest, &Limits::default()).unwrap();
    let fixed = FixedGrid::from_reference(
        std::slice::from_ref(&episode),
        Axes { x: 0, y: 1 },
        [4, 4],
        &Limits::default(),
    )
    .unwrap();
    assert_eq!(fixed.grid.bounds(), [[0.5, 1.5], [0., 9.]]);
    assert_eq!(fixed.empirical, [true, true]);
    assert_eq!(
        occupancy(&[episode], Axes { x: 0, y: 1 }, &fixed.grid)
            .unwrap()
            .included,
        4
    );
}

fn histogram(revision: u64, counts: Vec<u64>, grid: &Grid) -> CheckpointOccupancy {
    let included = counts.iter().sum::<u64>();
    CheckpointOccupancy {
        episodes: 1,
        revision,
        axes: Axes { x: 0, y: 1 },
        grid: grid.clone(),
        occupancy: rl_viewer::dynamics::Occupancy {
            mass: counts
                .iter()
                .map(|&n| {
                    if included == 0 {
                        0.
                    } else {
                        n as f64 / included as f64
                    }
                })
                .collect(),
            counts,
            included,
            excluded: 0,
        },
    }
}

#[test]
fn cumulative_coverage_is_a_cell_union_on_one_fixed_grid() {
    let grid = Grid::new([[0., 1.], [0., 1.]], [2, 2], &Limits::default()).unwrap();
    let checkpoints = vec![
        histogram(2, vec![4, 0, 0, 0], &grid),
        histogram(8, vec![2, 3, 0, 0], &grid),
        histogram(20, vec![0, 0, 7, 0], &grid),
    ];
    let history = coverage(
        &grid,
        Axes { x: 0, y: 1 },
        &checkpoints,
        &HistoryLimits::default(),
    )
    .unwrap();
    assert_eq!(
        history
            .points
            .iter()
            .map(|p| (p.revision, p.occupied, p.new_cells, p.cumulative))
            .collect::<Vec<_>>(),
        [(2, 1, 1, 1), (8, 2, 1, 2), (20, 1, 1, 3)]
    );
    assert_eq!(history.first_seen, [Some(2), Some(8), Some(20), None]);
    assert_eq!(history.display_grid_cells, 4);
    assert!(history.retained_bytes <= HistoryLimits::default().max_bytes);
}

#[test]
fn coverage_rejects_different_grids_axes_unsorted_revisions_and_excessive_preparation() {
    let grid = Grid::new([[0., 1.], [0., 1.]], [2, 2], &Limits::default()).unwrap();
    let mut checkpoints = vec![
        histogram(2, vec![1, 0, 0, 0], &grid),
        histogram(8, vec![1, 0, 1, 0], &grid),
    ];
    let limited = HistoryLimits {
        max_checkpoints: 1,
        ..Default::default()
    };
    assert!(coverage(&grid, Axes { x: 0, y: 1 }, &checkpoints, &limited).is_err());
    assert!(
        coverage(
            &grid,
            Axes { x: 0, y: 1 },
            &checkpoints,
            &HistoryLimits {
                max_bytes: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    checkpoints[1].revision = 1;
    assert!(
        coverage(
            &grid,
            Axes { x: 0, y: 1 },
            &checkpoints,
            &HistoryLimits::default()
        )
        .is_err()
    );
    checkpoints[1].revision = 8;
    checkpoints[1].axes = Axes { x: 1, y: 0 };
    assert!(
        coverage(
            &grid,
            Axes { x: 0, y: 1 },
            &checkpoints,
            &HistoryLimits::default()
        )
        .is_err()
    );
    checkpoints[1].axes = Axes { x: 0, y: 1 };
    checkpoints[1].grid = Grid::new([[0., 2.], [0., 1.]], [2, 2], &Limits::default()).unwrap();
    assert!(
        coverage(
            &grid,
            Axes { x: 0, y: 1 },
            &checkpoints,
            &HistoryLimits::default()
        )
        .is_err()
    );
}

#[test]
fn missing_history_has_no_invented_curve_and_reference_without_data_is_unavailable() {
    let grid = Grid::new([[0., 1.], [0., 1.]], [2, 2], &Limits::default()).unwrap();
    let history = coverage(&grid, Axes { x: 0, y: 1 }, &[], &HistoryLimits::default()).unwrap();
    assert!(history.points.is_empty());
    assert_eq!(history.first_seen, [None; 4]);
    assert!(
        FixedGrid::from_reference(&[], Axes { x: 0, y: 1 }, [64, 64], &Limits::default()).is_err()
    );
}
#[test]
fn sparse_coverage_preserves_unequal_evidence_and_step_geometry() {
    use rl_viewer::dynamics::history::CoverageMode;
    let grid = Grid::new([[0., 1.], [0., 1.]], [2, 2], &Limits::default()).unwrap();
    let mut checkpoints = vec![
        histogram(2, vec![2, 0, 0, 0], &grid),
        histogram(11, vec![1, 3, 0, 0], &grid),
        histogram(97, vec![0, 0, 1, 0], &grid),
    ];
    checkpoints[1].episodes = 2;
    checkpoints[1].occupancy.excluded = 3;
    let coverage = coverage(
        &grid,
        Axes { x: 0, y: 1 },
        &checkpoints,
        &HistoryLimits::default(),
    )
    .unwrap();
    assert_eq!(
        coverage
            .points
            .iter()
            .map(|p| (p.episodes, p.decisions, p.included, p.excluded))
            .collect::<Vec<_>>(),
        [(1, 2, 2, 0), (2, 7, 4, 3), (1, 1, 1, 0)]
    );
    assert_eq!(
        CoverageMode::Cumulative.edges(&coverage.points),
        vec![
            [[2., 1.], [11., 1.]],
            [[11., 1.], [11., 2.]],
            [[11., 2.], [97., 2.]],
            [[97., 2.], [97., 3.]]
        ]
    );
    for mode in [CoverageMode::Current, CoverageMode::New] {
        let edges = mode.edges(&coverage.points);
        assert!(edges.iter().all(|[a, b]| a[0] == b[0] && a[1] == 0.));
        assert_eq!(
            edges.iter().map(|[_, b]| b[0]).collect::<Vec<_>>(),
            [2., 11., 97.]
        );
    }
    assert_eq!(CoverageMode::Current.value(&coverage.points[1]), 2);
    assert_eq!(CoverageMode::New.value(&coverage.points[1]), 1);
}
