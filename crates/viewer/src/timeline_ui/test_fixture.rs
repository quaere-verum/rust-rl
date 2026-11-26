// Snapshot transport setup for unit tests. Production interaction goes through the UI.
use super::*;
impl TrainingTransport {
    /// Start over the same eligible positions and playback clock as show_snapshots.
    pub(crate) fn start_snapshot_playback(
        &mut self,
        run: &RunIndex,
        selection: &mut Selection,
        positions: &[u64],
        speed: f64,
    ) -> Result<()> {
        if !speed.is_finite() || speed <= 0. || speed > 100_000. {
            return Err(rl::Error::Config("invalid playback speed".into()));
        }
        self.sync(run, selection);
        self.timeline.set_snapshot_positions(positions);
        self.timeline.set_speed(speed);
        self.timeline.set_looping(false);
        self.timeline.play();
        self.last_tick = Some(Instant::now());
        selection.select_revision(run, self.timeline.revision())
    }
}
