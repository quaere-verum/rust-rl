//! Playback over validated panel positions, independent of rendering.
//!
//! Positions can come from model snapshots or recorded measurements. Numeric
//! revision gaps never cause intermediate data allocation.

use std::time::Duration;

/// Inclusive learner-revision interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevisionRange {
    min: u64,
    max: u64,
}

impl RevisionRange {
    pub fn new(min: u64, max: u64) -> Option<Self> {
        (min <= max).then_some(Self { min, max })
    }

    fn clamp(self, revision: u64) -> u64 {
        revision.clamp(self.min, self.max)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Backward,
    Forward,
}

/// CPU-only playback state. Advance only after the requested frame is ready.
#[derive(Debug, Clone)]
pub struct Timeline {
    bounds: Option<RevisionRange>,
    revision: u64,
    playing: bool,
    speed: f64,
    looping: bool,
    fractional_steps: f64,
    snapshots: Option<Vec<u64>>,
    prefetch_direction: Direction,
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Timeline {
    pub fn new(bounds: Option<RevisionRange>) -> Self {
        Self {
            bounds,
            revision: bounds.map_or(0, |range| range.min),
            playing: false,
            speed: 1.0,
            looping: false,
            fractional_steps: 0.0,
            snapshots: None,
            prefetch_direction: Direction::Forward,
        }
    }

    pub fn bounds(&self) -> Option<RevisionRange> {
        self.bounds
    }

    /// Refresh live run bounds without snapping to a checkpoint or following its tail.
    pub fn set_bounds(&mut self, bounds: Option<RevisionRange>) {
        self.bounds = bounds;
        let revision = bounds.map_or(0, |range| range.clamp(self.revision));
        if revision != self.revision {
            self.revision = revision;
            self.fractional_steps = 0.0;
        }
        if bounds.is_none() {
            self.pause();
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Seek to the nearest validated position after constraining to run bounds.
    pub fn set_revision(&mut self, revision: u64) {
        let revision = self.bounds.map_or(0, |range| range.clamp(revision));
        let revision = self
            .snapshots
            .as_ref()
            .and_then(|positions| positions.iter().min_by_key(|&&k| k.abs_diff(revision)))
            .copied()
            .unwrap_or(revision);
        if self.revision != revision {
            self.prefetch_direction = if revision < self.revision {
                Direction::Backward
            } else {
                Direction::Forward
            };
            self.revision = revision;
            self.fractional_steps = 0.0;
        }
    }

    /// Validated panel positions; unchecked metadata never enters this list.
    pub fn set_snapshot_positions(&mut self, positions: &[u64]) {
        if self.snapshots.as_deref() == Some(positions) {
            self.set_revision(self.revision);
            return;
        }
        self.snapshots = Some(positions.to_vec());
        self.snapshots.as_mut().unwrap().sort_unstable();
        self.snapshots.as_mut().unwrap().dedup();
        self.set_revision(self.revision);
    }
    pub(crate) fn clear_snapshot_positions(&mut self) -> bool {
        self.snapshots.take().is_some()
    }
    /// Keep an explicitly selected but now unavailable checkpoint visible until
    /// the user navigates. Manual steps still use only eligible positions.
    pub(crate) fn hold_unavailable_snapshot(&mut self, revision: u64) {
        self.revision = self.bounds.map_or(0, |range| range.clamp(revision));
        self.pause();
    }
    /// A slow frame discards elapsed time. At most one new frame is requested.
    pub fn advance_ready(&mut self, elapsed: Duration, ready: bool) -> bool {
        if !ready {
            self.fractional_steps = 0.;
            return false;
        }
        if !self.playing {
            return false;
        }
        self.fractional_steps += elapsed.as_secs_f64() * self.speed;
        if self.fractional_steps < 1. {
            return false;
        }
        self.fractional_steps = 0.;
        self.move_step(Direction::Forward)
    }

    /// Same clock/positions for prepared frames. Carry sub-frame time across
    /// ready frames and bounded loading delays, requesting at most one frame
    /// per paint. A long stall cannot accumulate an unbounded catch-up burst.
    pub fn advance_cached_ready(&mut self, elapsed: Duration, ready: bool) -> bool {
        if !self.playing {
            return false;
        }
        self.fractional_steps += elapsed.as_secs_f64() * self.speed;
        if !ready {
            self.fractional_steps = self.fractional_steps.min(1.);
            return false;
        }
        if self.fractional_steps < 1. {
            return false;
        }
        self.fractional_steps = (self.fractional_steps - 1.).min(1.);
        self.move_step(Direction::Forward)
    }

    /// Validated positions per second.
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// Reject nonpositive or nonfinite speeds, retaining the previous setting.
    pub fn set_speed(&mut self, speed: f64) -> bool {
        if !speed.is_finite() || speed <= 0.0 {
            return false;
        }
        self.speed = speed;
        true
    }

    pub fn looping(&self) -> bool {
        self.looping
    }

    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Resume within the run bounds; replay from the first position if
    /// already at the nonlooping endpoint. A singleton at its endpoint stays paused.
    pub fn play(&mut self) {
        self.prefetch_direction = Direction::Forward;
        let Some(range) = self.bounds else {
            self.pause();
            return;
        };
        self.revision = range.clamp(self.revision);
        let checkpoints = self.active_checkpoints(range);
        let (Some(&first), Some(&last)) = (checkpoints.first(), checkpoints.last()) else {
            self.pause();
            return;
        };
        if !self.looping && self.revision >= last && first < last {
            self.revision = first;
        }
        self.playing = first < last || self.revision < last;
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.fractional_steps = 0.0;
    }

    /// Move one position and pause. Looping applies to manual steps as well.
    /// Return whether requested time changed.
    pub fn step(&mut self, direction: Direction) -> bool {
        self.pause();
        self.prefetch_direction = direction;
        self.move_step(direction)
    }

    fn active_checkpoints(&self, range: RevisionRange) -> &[u64] {
        let checkpoints = self.snapshots.as_deref().unwrap_or_default();
        let start = checkpoints.partition_point(|&revision| revision < range.min);
        let end = checkpoints.partition_point(|&revision| revision <= range.max);
        &checkpoints[start..end]
    }

    /// Predict the next snapshot in the most recent navigation direction.
    /// Playback is always forward; this does not advance or mutate the clock.
    pub(crate) fn next_snapshot(&self) -> Option<u64> {
        self.snapshots.as_ref()?;
        let range = self.bounds?;
        let positions = self.active_checkpoints(range);
        let current = range.clamp(self.revision);
        let direction = if self.playing {
            Direction::Forward
        } else {
            self.prefetch_direction
        };
        let next = match direction {
            Direction::Forward => {
                let index = positions.partition_point(|&revision| revision <= current);
                positions
                    .get(index)
                    .copied()
                    .or_else(|| self.looping.then(|| positions.first().copied()).flatten())
            }
            Direction::Backward => {
                let index = positions.partition_point(|&revision| revision < current);
                index
                    .checked_sub(1)
                    .and_then(|i| positions.get(i).copied())
                    .or_else(|| self.looping.then(|| positions.last().copied()).flatten())
            }
        };
        next.filter(|&next| next != self.revision)
    }

    fn move_step(&mut self, direction: Direction) -> bool {
        let Some(range) = self.bounds else {
            self.pause();
            return false;
        };
        let previous = self.revision;
        let current = range.clamp(previous);
        let checkpoints = self.active_checkpoints(range);
        if checkpoints.is_empty() {
            self.pause();
            return false;
        }
        let insertion = match direction {
            Direction::Forward => checkpoints.partition_point(|&revision| revision <= current),
            Direction::Backward => checkpoints.partition_point(|&revision| revision < current),
        };
        let target = if self.looping {
            match direction {
                Direction::Forward => insertion % checkpoints.len(),
                Direction::Backward => (insertion + checkpoints.len() - 1) % checkpoints.len(),
            }
        } else {
            match direction {
                Direction::Forward => insertion.min(checkpoints.len() - 1),
                Direction::Backward => insertion.saturating_sub(1),
            }
        };
        // At an edge, retain an unavailable requested position rather than
        // stepping in the opposite direction to reach the nearest position.
        let next = checkpoints[target];
        if !self.looping
            && ((direction == Direction::Forward && next < current)
                || (direction == Direction::Backward && next > current))
        {
            self.pause();
            return false;
        }
        let endpoint = target == checkpoints.len() - 1;
        self.revision = next;
        if endpoint && !self.looping {
            self.pause();
        }
        self.revision != previous
    }
}

#[cfg(test)]
mod cached_clock_tests {
    use super::*;
    #[test]
    fn prefetch_follows_backward_navigation_then_forward_playback() {
        let mut timeline = Timeline::new(RevisionRange::new(0, 300));
        timeline.set_snapshot_positions(&[0, 100, 200, 300]);
        timeline.set_revision(300);
        timeline.set_revision(200);
        assert_eq!(timeline.next_snapshot(), Some(100));
        assert_eq!(timeline.revision(), 200);
        assert!(timeline.step(Direction::Backward));
        assert_eq!(timeline.next_snapshot(), Some(0));
        timeline.play();
        assert_eq!(timeline.next_snapshot(), Some(200));
        timeline.set_revision(0);
        assert!(timeline.is_playing());
        assert_eq!(timeline.next_snapshot(), Some(100));
    }

    #[test]
    fn backward_prefetch_respects_boundaries_and_looping() {
        let mut timeline = Timeline::new(RevisionRange::new(100, 300));
        timeline.set_snapshot_positions(&[0, 100, 200, 300, 400]);
        timeline.set_revision(200);
        assert!(timeline.step(Direction::Backward));
        assert_eq!(timeline.next_snapshot(), None);
        timeline.set_looping(true);
        assert_eq!(timeline.next_snapshot(), Some(300));
    }

    #[test]
    fn ready_cached_frames_keep_the_requested_rate_across_one_paint_loading_delays() {
        let mut timeline = Timeline::new(RevisionRange::new(0, 100));
        timeline.set_snapshot_positions(&(0..=100).collect::<Vec<_>>());
        timeline.set_speed(10.);
        timeline.play();
        let mut ready = true;
        let mut advanced = 0;
        for _ in 0..50 {
            let changed = timeline.advance_cached_ready(Duration::from_millis(20), ready);
            if changed {
                advanced += 1;
            }
            ready = !changed;
        }
        assert_eq!(advanced, 10);
        assert_eq!(timeline.revision(), 10);
    }
}
