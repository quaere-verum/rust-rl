//! Allocation regression for complete numerical forwards; no timing threshold.
#[path = "../../../benches/support/allocation.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{diagnostics::network_geometry::NumericalMlp, nn::network::Mlp};

fn complete_forward_owns_each_row_once_for_a_multisample_workload() {
    let net = NumericalMlp::from_model(
        &Mlp::new(3, &[24, 6], 2, &mut ChaCha8Rng::seed_from_u64(42)).unwrap(),
    )
    .unwrap();
    let inputs: Vec<_> = (0..256).map(|i| [i as f64 / 255., -0.28, 0.47]).collect();
    let before = allocation::start();
    for input in &inputs {
        std::hint::black_box(net.forward_layers(input).unwrap());
    }
    let measured = allocation::finish(before);
    println!(
        "allocating forward x256: calls={} requested={} live={} peak={}",
        measured.calls, measured.requested, measured.live, measured.peak
    );
    assert!(
        measured.calls <= 256 * (net.layer_count() as u64 + 2),
        "complete forwards must allocate each activation row once, got {}",
        measured.calls
    );
    let mut scratch = Vec::new();
    net.forward_layers_into(&inputs[0], &mut scratch).unwrap();
    let before = allocation::start();
    for input in &inputs {
        net.forward_layers_into(input, &mut scratch).unwrap();
        std::hint::black_box(&scratch);
    }
    let reused = allocation::finish(before);
    println!(
        "reused forward x256: calls={} requested={} live={} peak={}",
        reused.calls, reused.requested, reused.live, reused.peak
    );
    assert_eq!(
        reused.calls, 0,
        "complete rows should reuse their allocations"
    );
}

fn complete_neighborhood_does_not_allocate_scratch_per_sample() {
    use rl_viewer::network::{Preparation, Request};
    let net = NumericalMlp::from_model(
        &Mlp::new(2, &[2], 2, &mut ChaCha8Rng::seed_from_u64(42)).unwrap(),
    )
    .unwrap();
    let request = Request {
        network: "allocation".into(),
        anchor: vec![0.13, -0.28],
        endpoint: vec![-0.4, 0.2],
        probes: vec![vec![-1., 0.], vec![1., 1.]],
        axes: vec![0, 1],
        axis_scales: vec![2., 2.],
        ranges: [[-1., 1.], [-1., 1.]],
        direction_angle: 0.4,
        t: 1.,
        radius: 0.75,
        action_mask: None,
        layer: 1,
        grid: [2, 2],
        previous_basis: vec![],
    };
    // Zero budget forces a genuine complete neighborhood, without cache hits.
    let mut preparation = Preparation::with_cache_budget(net, 0);
    let before = allocation::start();
    let frame = preparation.prepare(&request).unwrap();
    let measured = allocation::finish(before);
    assert_eq!(frame.neighborhood.positions.len(), 256);
    assert_eq!(frame.neighborhood.errors.len(), 3);
    println!(
        "complete Identity neighborhood preparation: calls={} requested={} live={} peak={}",
        measured.calls, measured.requested, measured.live, measured.peak
    );
    assert!(
        measured.calls < 2000,
        "request-local finite scratch must avoid thousands of per-sample allocations, got {}",
        measured.calls
    );
}

fn network_metric_allocation_evidence_after_scratch() {
    use rl_viewer::network::{Preparation, Request};
    let net = NumericalMlp::from_model(
        &Mlp::new(2, &[24], 2, &mut ChaCha8Rng::seed_from_u64(42)).unwrap(),
    )
    .unwrap();
    let mut counts = Vec::new();
    for population in [2, 3] {
        // The same model/widths use Identity with two reference probes and a
        // LowRank hidden metric with three. Warm only the reference/PCA bases;
        // changing radius forces all 256 finite samples to run again.
        let mut request = Request {
            network: "metric-allocation".into(),
            anchor: vec![0.13, -0.28],
            endpoint: vec![-0.4, 0.2],
            probes: vec![vec![-1., 0.], vec![1., 1.], vec![0., -1.]],
            axes: vec![0, 1],
            axis_scales: vec![2., 2.],
            ranges: [[-1., 1.], [-1., 1.]],
            direction_angle: 0.4,
            t: 1.,
            radius: 0.5,
            action_mask: None,
            layer: 1,
            grid: [2, 2],
            previous_basis: vec![],
        };
        request.probes.truncate(population);
        let mut preparation = Preparation::new(net.clone());
        let warm = preparation.prepare(&request).unwrap();
        request.previous_basis = warm.pca_bases.clone();
        request.radius = 0.75;
        let before = allocation::start();
        let frame = preparation.prepare(&request).unwrap();
        let measured = allocation::finish(before);
        assert_eq!(frame.neighborhood.positions.len(), 256);
        assert!(
            frame
                .neighborhood
                .errors
                .iter()
                .flatten()
                .all(Option::is_some)
        );
        println!(
            "recomputed neighborhood population={population}: calls={} requested={} live={} peak={}",
            measured.calls, measured.requested, measured.live, measured.peak
        );
        counts.push(measured.calls);
    }
    println!(
        "remaining LowRank extra allocation calls={}",
        counts[1].saturating_sub(counts[0])
    );
    assert!(counts[0] < 1000);
    assert!(
        counts[1] < 1000,
        "LowRank norms must reuse coordinate buffers for all 256 cells"
    );
}

fn wide_low_rank_preparations_release_request_storage() {
    use rl_viewer::network::{Preparation, Request};
    let net = NumericalMlp::from_model(
        &Mlp::new(2, &[4096], 2, &mut ChaCha8Rng::seed_from_u64(42)).unwrap(),
    )
    .unwrap();
    let request = Request {
        network: "wide-metric-allocation".into(),
        anchor: vec![0.13, -0.28],
        endpoint: vec![-0.4, 0.2],
        probes: vec![vec![-1., 0.], vec![1., 1.], vec![0., -1.]],
        axes: vec![0, 1],
        axis_scales: vec![2., 2.],
        ranges: [[-1., 1.], [-1., 1.]],
        direction_angle: 0.4,
        t: 1.,
        radius: 0.75,
        action_mask: None,
        layer: 1,
        grid: [2, 2],
        previous_basis: vec![],
    };
    let mut preparation = Preparation::with_cache_budget(net, 0);
    let retained = preparation.retained_bytes();
    let before = allocation::start();
    let frame = preparation.prepare(&request).unwrap();
    let measured = allocation::finish(before);
    assert!(
        frame
            .neighborhood
            .errors
            .iter()
            .flatten()
            .all(Option::is_some)
    );
    assert_eq!(preparation.retained_bytes(), retained);
    drop(frame);
    let released = allocation::finish(before);
    assert_eq!(
        released.live, 0,
        "uncached preparation must release all request storage"
    );
    println!(
        "wide4096 uncached complete preparation: calls={} requested={} live={} peak={} after_drop={}",
        measured.calls, measured.requested, measured.live, measured.peak, released.live
    );
    assert!(
        measured.calls < 1000,
        "wide LowRank norms must reuse coordinate buffers"
    );
}

// One harness test keeps unrelated libtest worker activity out of global
// allocator intervals. Each synchronous helper retains its full workload.
#[test]
fn network_scratch_allocation_scenarios() {
    complete_forward_owns_each_row_once_for_a_multisample_workload();
    complete_neighborhood_does_not_allocate_scratch_per_sample();
    network_metric_allocation_evidence_after_scratch();
    wide_low_rank_preparations_release_request_storage();
}
