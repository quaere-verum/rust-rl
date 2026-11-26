//! Checked recorded trajectories and empirical projected decision-state occupancy.
//!
//! Preparation and aggregation belong on a bounded worker. Cursor access borrows
//! exact decoded rows and never scans or decodes arrays. No samples are discarded:
//! an episode that cannot fit the byte budget is rejected in its entirety.
pub mod history;
mod prepared;
pub mod source;
pub mod work;
pub use prepared::{DecisionAction, Episode, Step, prepare_episode};

use rl::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axes {
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionAxis {
    /// Categorical IDs, not a continuous projection coordinate.
    Discrete,
    Continuous(usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionValue {
    Discrete(u64),
    Continuous(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeStatus {
    Terminated,
    /// Reached the declared evaluation objective without environment termination.
    FiniteHorizon,
    Truncated,
    /// A recorded prefix with no episode boundary flag.
    Prefix,
}

#[derive(Clone, Debug)]
pub struct Coordinate {
    pub label: String,
    pub unit: Option<String>,
    pub range: Option<[f64; 2]>,
    /// Undescribed floating observation coordinates are usable numerically;
    /// explicitly categorical coordinates are never eligible for a projection.
    pub numeric: bool,
}

#[derive(Clone, Debug)]
pub struct Limits {
    /// Conservative retained ownership charge: exact decoded trajectory arrays
    /// + context/record metadata + coordinate/collection overhead.
    ///
    /// Input chunk/loading ownership is a separate loader budget.
    pub max_episode_bytes: usize,
    /// Counts and probabilities each occupy eight bytes per cell.
    pub max_grid_cells: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_episode_bytes: 32 * 1024 * 1024,
            max_grid_cells: 256 * 256,
        }
    }
}

pub(super) fn checked(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Shape(message.into()))
    }
}

/// A fixed partition, independent of episode values or action selection.
/// Cells are row-major `y * width + x`, lower edges inclusive. The last cell
/// includes the upper domain edge. Outliers are excluded, never clamped.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    bounds: [[f64; 2]; 2],
    cells: [usize; 2],
    len: usize,
}
impl Grid {
    pub fn new(bounds: [[f64; 2]; 2], cells: [usize; 2], limits: &Limits) -> Result<Self> {
        checked(
            bounds.iter().all(|&[lo, hi]| {
                lo.is_finite() && hi.is_finite() && lo < hi && (hi - lo).is_finite()
            }),
            "invalid dynamics grid bounds",
        )?;
        let len = cells[0]
            .checked_mul(cells[1])
            .ok_or_else(|| Error::Shape("dynamics grid size overflow".into()))?;
        checked(
            cells.iter().all(|&n| n > 0) && len <= limits.max_grid_cells,
            "dynamics grid exceeds cell budget or is empty",
        )?;
        checked(len.checked_mul(16).is_some(), "dynamics grid byte overflow")?;
        Ok(Self { bounds, cells, len })
    }
    pub fn bounds(&self) -> [[f64; 2]; 2] {
        self.bounds
    }
    pub fn cells(&self) -> [usize; 2] {
        self.cells
    }
    pub fn cell_count(&self) -> usize {
        self.len
    }
    pub fn bin(&self, point: [f64; 2]) -> Option<usize> {
        let mut index = [0; 2];
        for axis in 0..2 {
            let value = point[axis];
            let [lo, hi] = self.bounds[axis];
            if !value.is_finite() || value < lo || value > hi {
                return None;
            }
            index[axis] = (((value - lo) / (hi - lo) * self.cells[axis] as f64).floor() as usize)
                .min(self.cells[axis] - 1);
        }
        Some(index[1] * self.cells[0] + index[0])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Occupancy {
    pub counts: Vec<u64>,
    /// Probability mass N_b / sum(N_c), never density per unit area.
    pub mass: Vec<f64>,
    pub included: u64,
    pub excluded: u64,
}
impl Occupancy {
    /// Empty cells have no log value and must receive an explicit empty color.
    pub fn log_mass(&self, cell: usize) -> Option<f64> {
        self.mass
            .get(cell)
            .copied()
            .filter(|&x| x > 0.)
            .map(f64::ln)
    }
}

/// Aggregate exactly s_0,...,s_(T-1); final endpoint s_T is not a visit.
/// This primitive does not pool protocols implicitly: callers must select a
/// compatible evaluation population before passing episodes here.
pub fn occupancy(episodes: &[Episode], axes: Axes, grid: &Grid) -> Result<Occupancy> {
    let mut counts = vec![0_u64; grid.len];
    let mut included = 0_u64;
    let mut excluded = 0_u64;
    for episode in episodes {
        episode.validate_axes(axes)?;
        for t in 0..episode.transitions() {
            if let Some(bin) = grid.bin(episode.project(t, axes)?) {
                counts[bin] = counts[bin]
                    .checked_add(1)
                    .ok_or_else(|| Error::Shape("dynamics occupancy count overflow".into()))?;
                included = included
                    .checked_add(1)
                    .ok_or_else(|| Error::Shape("dynamics occupancy count overflow".into()))?;
            } else {
                excluded = excluded
                    .checked_add(1)
                    .ok_or_else(|| Error::Shape("dynamics occupancy count overflow".into()))?;
            }
        }
    }
    let mass = counts
        .iter()
        .map(|&n| {
            if included == 0 {
                0.
            } else {
                n as f64 / included as f64
            }
        })
        .collect();
    Ok(Occupancy {
        counts,
        mass,
        included,
        excluded,
    })
}
