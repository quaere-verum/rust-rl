use super::{cache::*, view::PolicyPane};
use crate::data::RunIndex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        models::SnapshotBuilder,
        policy_cache::{CacheLimits, PolicyCache},
        policy_precompute::{PrecomputeOptions, precompute},
        policy_slice::{PolicySliceSpec, SliceConfig},
        schema::{RecordContext, RunManifest},
        storage::{ReadLimits, Recorder, StorageConfig},
        visualization::EnvironmentSemantics,
    },
    nn::policy::{Deterministic, Policy},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

fn fixture(stochastic: bool, space: ActionSpace) -> (tempfile::TempDir, RunIndex, PolicyCache) {
    fixture_revisions(stochastic, space, &[3, 7, 11])
}
fn fixture_revisions(
    stochastic: bool,
    space: ActionSpace,
    revisions: &[u64],
) -> (tempfile::TempDir, RunIndex, PolicyCache) {
    fixture_resolution(stochastic, space, revisions, [8, 8])
}
fn fixture_resolution(
    stochastic: bool,
    space: ActionSpace,
    revisions: &[u64],
    resolution: [usize; 2],
) -> (tempfile::TempDir, RunIndex, PolicyCache) {
    let dir = tempfile::tempdir().unwrap();
    let manifest = RunManifest::new("prepared", "test", "ordinary", 2, space.clone(), 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for &revision in revisions {
        let mut builder = SnapshotBuilder::new(
            "saved",
            RecordContext::new("prepared", revision),
            2,
            space.clone(),
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(revision);
        if stochastic {
            builder
                .policy(
                    "actor",
                    &Policy::new(2, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
        } else {
            builder
                .deterministic(
                    "actor",
                    &Deterministic::new(2, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
        }
        writer.append(&builder.finish().unwrap()).unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut config = SliceConfig::new(
        EnvironmentSemantics::new(run.recording().manifest()),
        &Observation::new(vec![0., 0.]),
        &[],
    )
    .unwrap();
    config.resolution = resolution;
    let spec = PolicySliceSpec::from_config("online", "actor", &config);
    let cache = PolicyCache::open_or_create(
        dir.path().join("cache"),
        run.recording(),
        CacheLimits::default(),
    )
    .unwrap();
    assert_eq!(
        precompute(
            run.recording(),
            &cache,
            &[spec],
            &PrecomputeOptions::default(),
            |_| {}
        )
        .unwrap()
        .computed,
        revisions.len()
    );
    // The provider must function with no possible checkpoint restoration path.
    for source in run.recording().entries() {
        std::fs::remove_file(run.root().join(&source.file)).unwrap();
    }
    (dir, run, cache)
}
fn settle(provider: &mut PreparedProvider, catalog: bool) {
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        provider.poll();
        if if catalog {
            !provider.catalog_pending()
        } else {
            !provider.pending()
        } {
            break;
        }
        assert!(
            Instant::now() < until,
            "provider timed out: {:?}",
            provider.error()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(provider.error().is_none(), "{:?}", provider.error());
}

#[test]
fn cached_frames_use_the_same_renderer_and_exact_numerics_without_weights() {
    for (stochastic, space) in [
        (false, ActionSpace::Unbounded { dimensions: 2 }),
        (true, ActionSpace::Discrete { count: 3 }),
        (
            true,
            ActionSpace::Box {
                low: vec![-3., -1.],
                high: vec![7., 2.],
            },
        ),
    ] {
        let (_dir, run, cache) = fixture(stochastic, space);
        let mut provider = PreparedProvider::new(
            run.clone(),
            cache.root().to_owned(),
            PreparedLimits::default(),
            || {},
        )
        .unwrap();
        settle(&mut provider, true);
        let entries = provider.catalog().unwrap().entries.clone();
        assert_eq!(entries.len(), 3);
        let current = entries[1].clone();
        let reference = entries[0].clone();
        provider
            .request(&current.key, 1, 5, Some(&reference.key), None, &[])
            .unwrap();
        settle(&mut provider, false);
        let frame = provider.frame().unwrap();
        assert!(frame.grid.is_cached());
        let numerical = cache.load(&current).unwrap();
        let reference_data = cache.load(&reference).unwrap();
        for row in 0..64 {
            assert_eq!(frame.grid.values[row], numerical.value(row, 1).unwrap());
            assert_eq!(
                frame.grid.reference_at(row).unwrap(),
                reference_data.value(row, 1).unwrap()
            );
        }
        assert_eq!(
            frame.selected.as_ref().unwrap().input,
            numerical.input(5).unwrap()
        );
        assert_eq!(
            frame.selected.as_ref().unwrap().distribution,
            numerical.distribution(5, 1).unwrap()
        );
        assert_eq!(frame.grid.current.key.sequence, current.source.sequence);
        assert_eq!(frame.grid.cache_entry().unwrap().key, current.key);
        assert_eq!(
            frame.grid.reference_cache_entry().unwrap().key,
            reference.key
        );
        if matches!(numerical.action_space(), ActionSpace::Discrete { .. }) {
            let (batch, row) = numerical.checked_row(5).unwrap();
            let (other, other_row) = reference_data.checked_row(5).unwrap();
            assert_eq!(
                frame.selected.as_ref().unwrap().chosen_action,
                Some(batch.actions().as_u64().unwrap()[row] as usize)
            );
            assert_eq!(
                frame.selected.as_ref().unwrap().reference_chosen_action,
                Some(other.actions().as_u64().unwrap()[other_row] as usize)
            );
        }
        let ctx = eframe::egui::Context::default();
        let mut pane = PolicyPane::new();
        pane.config = Some((*frame.grid.config).clone());
        pane.ready = true;
        let paint = |pane: &mut PolicyPane| {
            let mut output = ctx.run_ui(Default::default(), |ui| {
                pane.show(
                    ui,
                    EnvironmentSemantics::new(run.recording().manifest()),
                    Some(frame),
                );
            });
            output.textures_delta.clear();
        };
        paint(&mut pane);
        let uploads = pane.texture_uploads;
        for _ in 0..10 {
            paint(&mut pane);
        }
        assert_eq!(pane.texture_uploads, uploads);
        let before = provider.metrics();
        provider
            .request(&current.key, 1, 5, Some(&reference.key), None, &[])
            .unwrap();
        provider.poll();
        assert_eq!(provider.metrics().preparations, before.preparations);
        assert_eq!(provider.metrics().loads, before.loads);
    }
}

#[test]
fn backward_navigation_prefetch_and_point_edits_reuse_owned_surfaces() {
    let (_dir, run, cache) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let mut provider = PreparedProvider::new(
        run,
        cache.root().to_owned(),
        PreparedLimits::default(),
        || {},
    )
    .unwrap();
    settle(&mut provider, true);
    let entries = provider.catalog().unwrap().entries.clone();
    provider
        .request(
            &entries[0].key,
            0,
            0,
            None,
            None,
            &[entries[1].key.clone(), entries[2].key.clone()],
        )
        .unwrap();
    settle(&mut provider, false);
    let until = Instant::now() + Duration::from_secs(10);
    while provider.metrics().loads < 3 {
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    let first = provider.frame().unwrap().grid.clone();
    provider
        .request(&entries[0].key, 0, 17, None, None, &[])
        .unwrap();
    settle(&mut provider, false);
    assert!(Arc::ptr_eq(&first, &provider.frame().unwrap().grid));
    for entry in [&entries[1], &entries[2], &entries[0]] {
        provider
            .request(&entry.key, 1, 17, None, None, &[])
            .unwrap();
        settle(&mut provider, false);
        assert_eq!(
            provider.frame().unwrap().grid.current.key.sequence,
            entry.source.sequence
        );
    }
    assert_eq!(provider.metrics().loads, 3);
    provider.retire();
    let until = Instant::now() + Duration::from_secs(10);
    drop(first);
    while !provider.retired() {
        assert!(Instant::now() < until);
        provider.poll();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(provider.metrics().retained_bytes, 0);
}

#[test]
fn scrubbing_missing_and_corrupt_entries_never_publish_under_another_identity() {
    let (_dir, run, cache) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let mut provider = PreparedProvider::new(
        run,
        cache.root().to_owned(),
        PreparedLimits::default(),
        || {},
    )
    .unwrap();
    settle(&mut provider, true);
    let entries = provider.catalog().unwrap().entries.clone();
    for i in 0..50 {
        provider
            .request(&entries[i % 3].key, i % 2, i % 64, None, None, &[])
            .unwrap();
    }
    settle(&mut provider, false);
    assert_eq!(
        provider.frame().unwrap().grid.current.key.sequence,
        entries[49 % 3].source.sequence
    );
    assert_eq!(
        provider
            .frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .cached_row,
        Some(49)
    );
    let old = provider.frame().unwrap().grid.current.clone();
    assert!(provider.request("missing", 0, 0, None, None, &[]).is_err());
    assert!(!provider.ready());
    assert_eq!(provider.frame().unwrap().grid.current, old);
    assert!(provider.metrics().retained_bytes <= PreparedLimits::default().max_bytes);
}

#[test]
fn verified_catalog_exceeds_the_historical_256_limit_and_excludes_corruption() {
    let revisions: Vec<_> = (0..260).collect();
    let (_dir, run, cache) =
        fixture_revisions(false, ActionSpace::Unbounded { dimensions: 2 }, &revisions);
    let entries = cache.catalog().unwrap().entries;
    let payload = cache
        .root()
        .join("frames")
        .join(&entries[1].key)
        .join("payload.bin");
    let mut corrupted = std::fs::read(&payload).unwrap();
    corrupted[10] ^= 1;
    std::fs::write(payload, corrupted).unwrap();
    let mut provider = PreparedProvider::new(
        run,
        cache.root().to_owned(),
        PreparedLimits::default(),
        || {},
    )
    .unwrap();
    settle(&mut provider, true);
    assert_eq!(provider.catalog().unwrap().entries.len(), 259);
    assert_eq!(provider.catalog().unwrap().issues.len(), 1);
    assert_eq!(
        provider.metrics().loads,
        0,
        "catalog integrity streams bytes without materializing numerical frames"
    );
    let last = provider.catalog().unwrap().entries.last().unwrap().clone();
    provider.request(&last.key, 0, 0, None, None, &[]).unwrap();
    settle(&mut provider, false);
    assert_eq!(provider.frame().unwrap().grid.current.revision, 259);
}

#[test]
fn presentation_leases_track_external_owners_and_low_budgets_fail_explicitly() {
    let (_dir, run, cache) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let mut provider = PreparedProvider::new(
        run.clone(),
        cache.root().to_owned(),
        PreparedLimits::default(),
        || {},
    )
    .unwrap();
    settle(&mut provider, true);
    let key = provider.catalog().unwrap().entries[0].key.clone();
    provider.request(&key, 0, 0, None, None, &[]).unwrap();
    settle(&mut provider, false);
    let held = provider.frame().unwrap().clone();
    provider.retire();
    let until = Instant::now() + Duration::from_secs(10);
    while !provider.retired() {
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(provider.metrics().retained_bytes > 0);
    drop(held);
    assert_eq!(provider.metrics().retained_bytes, 0);
    let mut limited = PreparedProvider::new(
        run,
        cache.root().to_owned(),
        PreparedLimits {
            max_bytes: 16 * 1024,
            max_frames: 1,
            ..Default::default()
        },
        || {},
    )
    .unwrap();
    settle(&mut limited, true);
    limited.request(&key, 0, 0, None, None, &[]).unwrap();
    while limited.pending() {
        assert!(Instant::now() < until);
        limited.poll();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(limited.error().is_some());
    assert!(!limited.ready());
    assert!(limited.metrics().retained_bytes <= 16 * 1024);
}

#[test]
fn non_power_of_two_grid_capacities_remain_fully_charged() {
    let (_dir, run, cache) = fixture_resolution(
        false,
        ActionSpace::Unbounded { dimensions: 2 },
        &[3, 7],
        [129, 129],
    );
    let mut provider = PreparedProvider::new(
        run,
        cache.root().to_owned(),
        PreparedLimits::default(),
        || {},
    )
    .unwrap();
    settle(&mut provider, true);
    let entries = provider.catalog().unwrap().entries.clone();
    provider
        .request(&entries[1].key, 0, 0, Some(&entries[0].key), None, &[])
        .unwrap();
    settle(&mut provider, false);
    let raw_bytes = cache.load(&entries[0]).unwrap().owned_bytes()
        + cache.load(&entries[1]).unwrap().owned_bytes();
    let projection_bytes = provider.frame().unwrap().grid.projection_bytes();
    assert!(
        provider.metrics().retained_bytes >= raw_bytes + projection_bytes,
        "all retained projection capacities must be charged"
    );
}

#[test]
fn point_updates_evict_unused_prefetch_before_reporting_memory_exhaustion() {
    let (_dir, run, cache) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let mut probe = PreparedProvider::new(
        run.clone(),
        cache.root().to_owned(),
        PreparedLimits::default(),
        || {},
    )
    .unwrap();
    settle(&mut probe, true);
    let entries = probe.catalog().unwrap().entries.clone();
    probe
        .request(&entries[0].key, 0, 0, None, None, &[])
        .unwrap();
    settle(&mut probe, false);
    let base = probe.metrics().retained_bytes;
    let raw = cache.load(&entries[0]).unwrap().owned_bytes();
    probe.retire();
    let until = Instant::now() + Duration::from_secs(20);
    while !probe.retired() {
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    let budget = 2 * base - raw;
    let mut provider = PreparedProvider::new(
        run,
        cache.root().to_owned(),
        PreparedLimits {
            max_bytes: budget,
            ..Default::default()
        },
        || {},
    )
    .unwrap();
    settle(&mut provider, true);
    provider
        .request(
            &entries[0].key,
            0,
            0,
            None,
            None,
            &[entries[1].key.clone(), entries[2].key.clone()],
        )
        .unwrap();
    settle(&mut provider, false);
    while provider.metrics().loads < 3 {
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    provider
        .request(&entries[0].key, 0, 1, None, None, &[])
        .unwrap();
    settle(&mut provider, false);
    assert!(provider.ready());
    assert_eq!(
        provider
            .frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .cached_row,
        Some(1)
    );
    assert!(provider.metrics().retained_bytes <= budget);
}

#[test]
fn plot_clicks_select_cached_rows_instead_of_arbitrary_coordinates() {
    use eframe::egui::{self, Event, PointerButton};
    let (_dir, run, cache) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let mut provider = PreparedProvider::new(
        run.clone(),
        cache.root().to_owned(),
        PreparedLimits::default(),
        || {},
    )
    .unwrap();
    settle(&mut provider, true);
    let key = provider.catalog().unwrap().entries[0].key.clone();
    provider.request(&key, 0, 0, None, None, &[]).unwrap();
    settle(&mut provider, false);
    let frame = provider.frame().unwrap();
    let ctx = egui::Context::default();
    let mut pane = PolicyPane::new();
    pane.config = Some((*frame.grid.config).clone());
    pane.ready = true;
    let paint = |pane: &mut PolicyPane, events| {
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1100., 700.),
                )),
                events,
                ..Default::default()
            },
            |ui| {
                pane.show(
                    ui,
                    EnvironmentSemantics::new(run.recording().manifest()),
                    Some(frame),
                );
            },
        );
        out.textures_delta.clear();
    };
    paint(&mut pane, vec![]);
    let position = pane.main_rect.unwrap().center() + egui::vec2(17.3, 11.7);
    paint(
        &mut pane,
        vec![
            Event::PointerMoved(position),
            Event::PointerButton {
                pos: position,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: Default::default(),
            },
        ],
    );
    paint(
        &mut pane,
        vec![Event::PointerButton {
            pos: position,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        }],
    );
    let row = pane
        .cached_row
        .expect("cached click must select an evaluated row");
    assert!(row < frame.grid.values.len());
    assert!(
        pane.pin.is_none(),
        "cached navigation cannot create arbitrary synthetic pins"
    );
    assert!(pane.dirty);
    assert_eq!(
        frame.selected.as_ref().unwrap().cached_row,
        Some(0),
        "previous complete point remains coherent until publication"
    );
}
