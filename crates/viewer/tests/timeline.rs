use rl_viewer::timeline::{Direction, RevisionRange, Timeline};
use std::time::Duration;

fn range(min: u64, max: u64) -> RevisionRange {
    RevisionRange::new(min, max).unwrap()
}

fn timeline() -> Timeline {
    let mut timeline = Timeline::new(Some(range(1, 61)));
    timeline.set_snapshot_positions(&[41, 21, 1, 21, 61]);
    timeline
}

#[test]
fn seeking_uses_the_nearest_real_position_with_lower_tie_breaking() {
    let mut timeline = timeline();
    assert_eq!(timeline.bounds(), Some(range(1, 61)));
    timeline.set_revision(37);
    assert_eq!(timeline.revision(), 41);
    timeline.set_revision(51);
    assert_eq!(timeline.revision(), 41);
    timeline.set_revision(0);
    assert_eq!(timeline.revision(), 1);
    timeline.set_revision(u64::MAX);
    assert_eq!(timeline.revision(), 61);
    assert!(RevisionRange::new(4, 3).is_none());
}

#[test]
fn elapsed_fractions_accumulate_and_pause_freezes_training_time() {
    let mut timeline = timeline();
    assert!(timeline.set_speed(2.0));
    timeline.play();
    assert!(!timeline.advance_ready(Duration::from_millis(200), true));
    assert_eq!(timeline.revision(), 1);
    assert!(timeline.advance_ready(Duration::from_millis(300), true));
    assert_eq!(timeline.revision(), 21);
    timeline.pause();
    assert!(!timeline.advance_ready(Duration::from_secs(20), true));
    timeline.play();
    assert!(timeline.advance_ready(Duration::from_millis(500), true));
    assert_eq!(timeline.revision(), 41);
}

#[test]
fn positions_step_in_both_directions_without_duplicates() {
    let mut timeline = timeline();
    timeline.set_revision(21);
    assert!(timeline.step(Direction::Forward));
    assert_eq!(timeline.revision(), 41);
    assert!(timeline.step(Direction::Backward));
    assert_eq!(timeline.revision(), 21);
    assert!(timeline.step(Direction::Backward));
    assert_eq!(timeline.revision(), 1);
    assert!(!timeline.step(Direction::Backward));
    assert!(!timeline.is_playing());
}

#[test]
fn speed_counts_prepared_positions_without_catching_up_over_numeric_gaps() {
    let mut timeline = timeline();
    timeline.set_speed(2.0);
    timeline.play();
    assert!(timeline.advance_ready(Duration::from_secs(1), true));
    assert_eq!(timeline.revision(), 21);
    assert!(timeline.advance_ready(Duration::from_millis(500), true));
    assert_eq!(timeline.revision(), 41);
    assert!(timeline.advance_ready(Duration::MAX, true));
    assert_eq!(timeline.revision(), 61);
    assert!(!timeline.is_playing());
}

#[test]
fn stepping_filters_positions_to_inclusive_run_bounds() {
    let mut timeline = Timeline::new(Some(range(20, 60)));
    timeline.set_snapshot_positions(&[1, 21, 41, 61]);
    timeline.set_revision(41);
    assert!(timeline.step(Direction::Backward));
    assert_eq!(timeline.revision(), 21);
    assert!(!timeline.step(Direction::Backward));
    assert_eq!(timeline.revision(), 21);
    assert!(timeline.step(Direction::Forward));
    assert_eq!(timeline.revision(), 41);
    assert!(!timeline.step(Direction::Forward));
    assert_eq!(timeline.revision(), 41);
}

#[test]
fn loop_wraps_the_real_positions_for_playback_and_manual_steps() {
    let mut timeline = timeline();
    timeline.set_looping(true);
    assert!(timeline.looping());
    timeline.set_revision(61);
    timeline.play();
    assert!(timeline.advance_ready(Duration::from_secs(6), true));
    assert_eq!(timeline.revision(), 1);
    assert!(timeline.is_playing());
    assert!(timeline.step(Direction::Backward));
    assert_eq!(timeline.revision(), 61);
    assert!(!timeline.is_playing());
    assert!(timeline.step(Direction::Forward));
    assert_eq!(timeline.revision(), 1);
}

#[test]
fn empty_and_single_position_playback_stays_bounded_and_paused() {
    let mut timeline = Timeline::default();
    timeline.play();
    assert!(!timeline.is_playing());
    assert!(!timeline.advance_ready(Duration::MAX, true));
    assert!(!timeline.step(Direction::Forward));
    timeline.set_bounds(Some(range(1, 100)));
    timeline.set_snapshot_positions(&[]);
    timeline.play();
    assert!(!timeline.is_playing());
    timeline.set_snapshot_positions(&[7]);
    assert_eq!(timeline.revision(), 7);
    timeline.play();
    assert!(!timeline.is_playing());
    assert!(!timeline.step(Direction::Backward));
    assert!(!timeline.step(Direction::Forward));
    timeline.set_looping(true);
    timeline.play();
    assert!(!timeline.is_playing());
}

#[test]
fn live_growth_preserves_selection_and_extends_the_position_domain() {
    let mut timeline = timeline();
    timeline.set_revision(41);
    timeline.set_bounds(Some(range(1, 81)));
    timeline.set_snapshot_positions(&[1, 21, 41, 61, 81]);
    assert_eq!(timeline.revision(), 41);
    assert_eq!(timeline.bounds(), Some(range(1, 81)));
    timeline.set_revision(61);
    timeline.play();
    assert!(timeline.advance_ready(Duration::from_secs(1), true));
    assert_eq!(timeline.revision(), 81);
}

#[test]
fn live_bounds_refresh_does_not_discard_fractional_playback_time() {
    let mut timeline = timeline();
    timeline.set_speed(2.0);
    timeline.play();
    for (maximum, expected) in [61, 61, 62, 63, 64, 64]
        .into_iter()
        .zip([1, 1, 21, 21, 21, 41])
    {
        timeline.set_bounds(Some(range(1, maximum)));
        timeline.advance_ready(Duration::from_millis(200), true);
        assert_eq!(timeline.revision(), expected);
    }
}

#[test]
fn unchanged_selection_and_position_refresh_preserve_fractional_playback_time() {
    let mut timeline = timeline();
    timeline.play();
    for _ in 0..4 {
        timeline.set_revision(timeline.revision());
        timeline.set_snapshot_positions(&[1, 21, 41, 61]);
        timeline.advance_ready(Duration::from_millis(250), true);
    }
    assert_eq!(timeline.revision(), 21);
}

#[test]
fn removing_run_bounds_stops_playback() {
    let mut timeline = timeline();
    timeline.play();
    assert!(timeline.is_playing());
    timeline.set_bounds(None);
    assert!(!timeline.is_playing());
    assert_eq!(timeline.bounds(), None);
    assert_eq!(timeline.revision(), 0);
}

#[test]
fn playing_again_after_nonlooping_endpoint_restarts_at_first_position() {
    let mut timeline = timeline();
    timeline.set_revision(61);
    timeline.play();
    assert_eq!(timeline.revision(), 1);
    assert!(timeline.is_playing());
    timeline.advance_ready(Duration::from_secs(1), true);
    assert_eq!(timeline.revision(), 21);
}

#[test]
fn invalid_speeds_do_not_poison_transport() {
    let mut timeline = timeline();
    for speed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(!timeline.set_speed(speed));
    }
    assert_eq!(timeline.speed(), 1.0);
    timeline.play();
    timeline.advance_ready(Duration::from_secs(1), true);
    assert_eq!(timeline.revision(), 21);
}

#[test]
fn seeking_and_manual_steps_discard_old_fractional_playback_time() {
    let mut timeline = timeline();
    timeline.play();
    timeline.advance_ready(Duration::from_millis(900), true);
    timeline.set_revision(21);
    timeline.advance_ready(Duration::from_millis(200), true);
    assert_eq!(timeline.revision(), 21);
    timeline.step(Direction::Forward);
    assert_eq!(timeline.revision(), 41);
    assert!(!timeline.is_playing());
    timeline.play();
    timeline.advance_ready(Duration::from_millis(900), true);
    assert_eq!(timeline.revision(), 41);
}

#[test]
fn huge_numeric_gaps_and_u64_endpoints_do_not_allocate_or_overflow() {
    let mut timeline = Timeline::new(Some(range(0, u64::MAX)));
    timeline.set_snapshot_positions(&[0, u64::MAX]);
    timeline.play();
    timeline.advance_ready(Duration::from_secs(1), true);
    assert_eq!(timeline.revision(), u64::MAX);
    assert!(!timeline.is_playing());
    timeline.set_looping(true);
    timeline.play();
    timeline.advance_ready(Duration::from_secs(1), true);
    assert_eq!(timeline.revision(), 0);
    timeline.advance_ready(Duration::MAX, true);
    assert_eq!(timeline.revision(), u64::MAX);
}
