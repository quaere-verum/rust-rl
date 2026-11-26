use rl_viewer::timeline::{Direction, RevisionRange, Timeline};
use std::time::Duration;

#[test]
fn snapshot_positions_wait_without_catchup_and_preserve_manual_navigation() {
    let mut t = Timeline::new(RevisionRange::new(0, 300));
    t.set_snapshot_positions(&[0, 120, 240]);
    t.play();
    assert!(!t.advance_ready(Duration::from_secs(100), false));
    assert_eq!(t.revision(), 0);
    assert!(!t.advance_ready(Duration::from_millis(100), true));
    assert!(t.advance_ready(Duration::from_secs(100), true));
    assert_eq!(t.revision(), 120, "one prepared position, no catch-up");
    t.step(Direction::Backward);
    assert_eq!(t.revision(), 0);
    t.set_looping(true);
    t.step(Direction::Backward);
    assert_eq!(t.revision(), 240);
    t.set_revision(121);
    assert_eq!(t.revision(), 120, "scrubbing cannot invent a position");
}
