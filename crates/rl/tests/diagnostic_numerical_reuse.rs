#![cfg(feature = "diagnostics")]
use rl::diagnostics::{policy::GaussianMarginal, schema::*, value::*};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    collections::BTreeMap,
};

struct Counting;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}
fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}
// SAFETY: Delegate every pointer/layout unchanged; the thread-local counter
// observes calls without allocating or accessing the allocated memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_allocation();
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_allocation();
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, size) };
        if !ptr.is_null() {
            record_allocation();
        }
        ptr
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
fn allocations<T>(f: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let result = f();
    let count = ALLOCATIONS.with(|count| count.replace(None).unwrap());
    (result, count)
}
fn quantile_chunk() -> Chunk {
    Chunk {
        context: RecordContext::new("reuse", 0),
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Quantiles {
                values: "z".into(),
                locations: "p".into(),
                masses: "m".into(),
            },
            action_condition: ActionCondition::Fixed,
            actions: None,
            continuation_policy: None,
            continuation_stochastic: None,
        }),
        arrays: BTreeMap::from([
            (
                "z".into(),
                Array::f64(vec![1, 4], vec![3., -0., 2., -2.]).unwrap(),
            ),
            (
                "p".into(),
                Array::f64(vec![4], vec![0.1, 0.3, 0.6, 0.9]).unwrap(),
            ),
            (
                "m".into(),
                Array::f64(vec![4], vec![0.1, 0.2, 0.3, 0.4000004]).unwrap(),
            ),
        ]),
    }
}
#[test]
fn bounded_quadrature_uses_no_heap_scratch() {
    // The wide Gaussian admits all nine transition points, exercising the
    // maximum 34-point stack scratch in the selected-statistics pass.
    for (mean, sigma) in [(0.7, 1_f64), (2., 100.)] {
        let g = GaussianMarginal {
            mean,
            log_std: sigma.ln(),
            bounds: Some([-1., 1.]),
        };
        let (mean, count) = allocations(|| g.expected_action().unwrap());
        assert!(mean.is_finite());
        assert_eq!(
            count, 0,
            "bounded expectation must not allocate breakpoint scratch"
        );
        let (_, count) = allocations(|| g.action_statistics().unwrap());
        assert_eq!(
            count, 0,
            "selected statistics must not allocate breakpoint scratch"
        );
    }
}
#[test]
fn quantile_extraction_allocates_only_returned_rows() {
    let chunk = quantile_chunk();
    let (prediction, count) =
        allocations(|| ValuePrediction::from_critic(&chunk, 0, None).unwrap());
    assert!(matches!(prediction, ValuePrediction::Quantiles { .. }));
    assert_eq!(
        count, 3,
        "extraction must not clone/sort a discarded forecast"
    );
}
fn assert_wide_quantile_extraction_allocations(width: usize) {
    for use_f32 in [false, true] {
        let mut chunk = quantile_chunk();
        let mut raw: Vec<_> = (0..2 * width).map(|i| i as f64 * 0.1 - 5.).collect();
        raw[width + 1] = -0.;
        let (values, expected_raw) = if use_f32 {
            let values: Vec<_> = raw.iter().map(|&v| v as f32).collect();
            let expected: Vec<_> = values[width..].iter().map(|&v| f64::from(v)).collect();
            (Array::f32(vec![2, width], values).unwrap(), expected)
        } else {
            let expected = raw[width..].to_vec();
            (Array::f64(vec![2, width], raw).unwrap(), expected)
        };
        let locations: Vec<_> = (0..width)
            .map(|i| (i as f64 + 0.5) / width as f64)
            .collect();
        let masses = vec![1. / width as f64; width];
        chunk.arrays.insert("z".into(), values);
        chunk.arrays.insert(
            "p".into(),
            Array::f64(vec![width], locations.clone()).unwrap(),
        );
        chunk
            .arrays
            .insert("m".into(), Array::f64(vec![width], masses.clone()).unwrap());
        let (prediction, count) =
            allocations(|| ValuePrediction::from_critic(&chunk, 1, None).unwrap());
        let ValuePrediction::Quantiles {
            raw,
            locations: actual_locations,
            masses: actual_masses,
        } = prediction
        else {
            panic!("quantile fixture")
        };
        for (actual, expected) in [
            (&raw, &expected_raw),
            (&actual_locations, &locations),
            (&actual_masses, &masses),
        ] {
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(
            count, 3,
            "q{width}, f32={use_f32}: extraction must allocate only three returned buffers"
        );
    }
}
#[test]
fn q32_extraction_allocates_only_returned_rows() {
    assert_wide_quantile_extraction_allocations(32);
}
#[test]
fn q256_extraction_allocates_only_returned_rows() {
    assert_wide_quantile_extraction_allocations(256);
}
#[test]
fn row_conversion_errors_precede_later_grid_and_legality_checks() {
    for index in [0, 3] {
        let mut values = vec![1.; 4];
        values[index] = f64::NAN;
        let mut chunk = quantile_chunk();
        chunk.arrays.remove("p");
        chunk
            .arrays
            .insert("z".into(), Array::f64(vec![1, 4], values.clone()).unwrap());
        assert_eq!(
            ValuePrediction::from_critic(&chunk, 0, None)
                .unwrap_err()
                .to_string(),
            "invalid configuration: nonfinite/overflowed value prediction"
        );

        let mut chunk = quantile_chunk();
        chunk.arrays.remove("m");
        chunk
            .arrays
            .insert("p".into(), Array::f64(vec![4], values.clone()).unwrap());
        assert_eq!(
            ValuePrediction::from_critic(&chunk, 0, None)
                .unwrap_err()
                .to_string(),
            "invalid configuration: nonfinite/overflowed value prediction"
        );

        chunk.record = Record::Critic(CriticRecord {
            representation: CriticRepresentation::Discrete { values: "z".into() },
            action_condition: ActionCondition::AllDiscreteActions,
            actions: None,
            continuation_policy: None,
            continuation_stochastic: None,
        });
        chunk
            .arrays
            .insert("z".into(), Array::f64(vec![1, 4], values).unwrap());
        assert_eq!(
            ValuePrediction::from_critic(&chunk, 0, None)
                .unwrap_err()
                .to_string(),
            "invalid configuration: nonfinite/overflowed value prediction"
        );
    }
    let mut chunk = quantile_chunk();
    chunk.arrays.remove("m");
    chunk
        .arrays
        .insert("p".into(), Array::f64(vec![2, 2], vec![0.1; 4]).unwrap());
    assert_eq!(
        ValuePrediction::from_critic(&chunk, 0, None)
            .unwrap_err()
            .to_string(),
        "invalid configuration: invalid critic probability grid"
    );
}
#[test]
fn crossed_nonuniform_heads_preserve_raw_and_ordered_results() {
    let prediction = ValuePrediction::from_critic(&quantile_chunk(), 0, None).unwrap();
    let ValuePrediction::Quantiles { raw, masses, .. } = &prediction else {
        panic!("quantile fixture")
    };
    assert_eq!(
        raw.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [
            3_f64.to_bits(),
            (-0_f64).to_bits(),
            2_f64.to_bits(),
            (-2_f64).to_bits()
        ]
    );
    assert_eq!(masses[3].to_bits(), 0.4000004_f64.to_bits());
    let ordered = prediction.ordered_quantiles().unwrap();
    assert_eq!(ordered.source_crossing_count(), 2);
    assert_eq!(ordered.values(), [-2., -0., 2., 3.]);
    assert_eq!(ordered.values()[1].to_bits(), (-0_f64).to_bits());
    // Captured from the original normalization/ordering before borrowed validation.
    assert_eq!(ordered.mean().to_bits(), 0x3ff9999a2fec7dd8);
    assert_eq!(
        prediction
            .scalar(&ValueOutput::ExpectedReturn, None)
            .unwrap()
            .to_bits(),
        0x3ff9999a2fec7dd8,
        "nonuniform masses must follow ordered probability bins, not raw heads"
    );
    assert_eq!(
        ordered.lower_cvar(0.5).unwrap().to_bits(),
        0x3fd9999c48cb63dd
    );
    assert_eq!(
        prediction
            .scalar(&ValueOutput::Cvar { probability: 0.5 }, None)
            .unwrap()
            .to_bits(),
        0x3fd9999c48cb63dd
    );
    assert_eq!(
        ordered
            .masses()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [
            0x3fb99998edcd2709,
            0x3fc99998edcd2709,
            0x3fd33332b259dd46,
            0x3fd9999a9b4c4572
        ]
    );
    assert_eq!(
        prediction
            .scalar(&ValueOutput::Quantile { probability: 0.5 }, None)
            .unwrap(),
        2.
    );
}
#[test]
fn quantile_extraction_preserves_validation_precedence() {
    let mut chunk = quantile_chunk();
    chunk.arrays.insert(
        "p".into(),
        Array::f64(vec![4], vec![0.3, 0.1, 0.6, 0.9]).unwrap(),
    );
    chunk
        .arrays
        .insert("m".into(), Array::f64(vec![4], vec![0.1; 4]).unwrap());
    assert_eq!(
        ValuePrediction::from_critic(&chunk, 0, None)
            .unwrap_err()
            .to_string(),
        "invalid configuration: invalid quantile values, locations or masses"
    );
    chunk.arrays.insert(
        "p".into(),
        Array::f64(vec![4], vec![0.1, 0.3, 0.6, 0.9]).unwrap(),
    );
    assert_eq!(
        ValuePrediction::from_critic(&chunk, 0, None)
            .unwrap_err()
            .to_string(),
        "invalid configuration: quantile probability masses must sum to one"
    );
}
