//! Fixed plotting domains and cumulative observed evaluation-cell coverage.
use super::{ActionAxis, ActionValue, Axes, Episode, Grid, Limits, Occupancy, checked};
use rl::{Error, Result};

/// Exact requested domain identity. Recompute increments the automatic epoch;
/// validated manual coordinates retain their bits rather than a float hash.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DomainSpec {
    epoch: u64,
    manual: Option<[[u64; 2]; 2]>,
}
impl DomainSpec {
    pub fn automatic(epoch: u64) -> Self {
        Self {
            epoch,
            manual: None,
        }
    }
    pub fn manual(bounds: [[f64; 2]; 2]) -> Result<Self> {
        checked(
            bounds.iter().all(|&[lo, hi]| {
                lo.is_finite() && hi.is_finite() && lo < hi && (hi - lo).is_finite()
            }),
            "manual Dynamics bounds require finite increasing spans",
        )?;
        Ok(Self {
            epoch: 0,
            manual: Some(bounds.map(|range| range.map(f64::to_bits))),
        })
    }
    pub fn manual_bounds(self) -> Option<[[f64; 2]; 2]> {
        self.manual
            .map(|bounds| bounds.map(|range| range.map(f64::from_bits)))
    }
}

/// Streaming exact extrema. Storage depends only on coordinate/action width,
/// never the number of decisions or episodes in the frozen population.
#[derive(Debug)]
pub(super) struct ReferenceExtrema {
    pub state: Vec<Option<[f64; 2]>>,
    pub declared: Vec<Option<[f64; 2]>>,
    pub actions: Vec<[f64; 2]>,
    pub episodes: usize,
    pub decisions: usize,
    pub retained_bytes: usize,
}
impl ReferenceExtrema {
    pub fn new(first: &Episode, max_bytes: usize) -> Result<Self> {
        let action_count = first
            .action_axes()
            .iter()
            .filter(|a| matches!(a, ActionAxis::Continuous(_)))
            .count();
        let retained_bytes = first
            .coordinates()
            .len()
            .checked_mul(2 * std::mem::size_of::<Option<[f64; 2]>>())
            .and_then(|n| action_count.checked_mul(16).and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or_else(|| Error::Shape("reference extrema byte overflow".into()))?;
        checked(
            retained_bytes <= max_bytes,
            "Dynamics reference extrema exceed ownership budget",
        )?;
        Ok(Self {
            state: first
                .coordinates()
                .iter()
                .map(|c| c.numeric.then_some([f64::INFINITY, f64::NEG_INFINITY]))
                .collect(),
            declared: first.coordinates().iter().map(|c| c.range).collect(),
            actions: vec![[f64::INFINITY, f64::NEG_INFINITY]; action_count],
            episodes: 0,
            decisions: 0,
            retained_bytes,
        })
    }
    pub fn add(&mut self, episode: &Episode, check: impl Fn() -> Result<()>) -> Result<()> {
        checked(
            episode.coordinates().len() == self.state.len()
                && episode
                    .coordinates()
                    .iter()
                    .zip(&self.declared)
                    .all(|(c, range)| c.range == *range)
                && episode
                    .coordinates()
                    .iter()
                    .zip(&self.state)
                    .all(|(c, range)| c.numeric == range.is_some())
                && episode
                    .action_axes()
                    .iter()
                    .filter(|a| matches!(a, ActionAxis::Continuous(_)))
                    .count()
                    == self.actions.len(),
            "reference trajectory coordinate/action domains differ",
        )?;
        for t in 0..=episode.transitions() {
            if t % 1024 == 0 {
                check()?;
            }
            for (value, range) in episode.step(t)?.observation.iter().zip(&mut self.state) {
                if let Some(range) = range {
                    range[0] = range[0].min(*value);
                    range[1] = range[1].max(*value);
                }
            }
            if t < episode.transitions() {
                for (component, range) in self.actions.iter_mut().enumerate() {
                    if let Some(ActionValue::Continuous(value)) =
                        episode.action_value(t, ActionAxis::Continuous(component))?
                    {
                        range[0] = range[0].min(value);
                        range[1] = range[1].max(value);
                    }
                }
            }
        }
        self.episodes += 1;
        self.decisions = self
            .decisions
            .checked_add(episode.transitions())
            .ok_or_else(|| Error::Shape("reference decision count overflow".into()))?;
        Ok(())
    }
    pub fn grid(
        &self,
        axes: Axes,
        cells: [usize; 2],
        domain: DomainSpec,
        limits: &Limits,
    ) -> Result<FixedGrid> {
        let mut bounds = [[0.; 2]; 2];
        let mut empirical = [false; 2];
        for (axis, coordinate) in [axes.x, axes.y].into_iter().enumerate() {
            let observed = self
                .state
                .get(coordinate)
                .copied()
                .flatten()
                .ok_or_else(|| Error::Shape("reference requires numerical coordinates".into()))?;
            bounds[axis] = self.declared[coordinate].unwrap_or(observed);
            empirical[axis] = self.declared[coordinate].is_none();
            if empirical[axis] && bounds[axis][0] == bounds[axis][1] {
                let pad = (bounds[axis][0].abs() * 0.05).max(0.5);
                bounds[axis] = [bounds[axis][0] - pad, bounds[axis][1] + pad];
            }
        }
        if let Some(manual) = domain.manual_bounds() {
            bounds = manual;
            empirical = [false; 2];
        }
        Ok(FixedGrid {
            grid: Grid::new(bounds, cells, limits)?,
            empirical,
        })
    }
}

#[derive(Clone, Debug)]
pub struct FixedGrid {
    pub grid: Grid,
    /// True when this axis uses an identified empirical reference population,
    /// not a declared coordinate domain. Callers retain its exact source keys.
    pub empirical: [bool; 2],
}
impl FixedGrid {
    /// Establish bounds once, outside painting. The final observation is useful
    /// for plotting endpoints even though it is not an occupancy visit.
    pub fn from_reference(
        episodes: &[Episode],
        axes: Axes,
        cells: [usize; 2],
        limits: &Limits,
    ) -> Result<Self> {
        let first = episodes
            .first()
            .ok_or_else(|| Error::State("no reference trajectory for plotting bounds".into()))?;
        first.validate_axes(axes)?;
        let coordinates = [axes.x, axes.y];
        let mut bounds = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
        let mut empirical = [true; 2];
        for (axis, &coordinate) in coordinates.iter().enumerate() {
            if let Some(range) = first.coordinates()[coordinate].range {
                bounds[axis] = range;
                empirical[axis] = false;
            }
        }
        for episode in episodes {
            episode.validate_axes(axes)?;
            for (axis, &coordinate) in coordinates.iter().enumerate() {
                checked(
                    episode.coordinates()[coordinate].range
                        == first.coordinates()[coordinate].range,
                    "reference trajectory coordinate domains differ",
                )?;
                if empirical[axis] {
                    for t in 0..=episode.transitions() {
                        let value = episode.step(t)?.observation[coordinate];
                        bounds[axis][0] = bounds[axis][0].min(value);
                        bounds[axis][1] = bounds[axis][1].max(value);
                    }
                }
            }
        }
        for axis in 0..2 {
            if empirical[axis] && bounds[axis][0] == bounds[axis][1] {
                let pad = (bounds[axis][0].abs() * 0.05).max(0.5);
                bounds[axis] = [bounds[axis][0] - pad, bounds[axis][1] + pad];
            }
        }
        Ok(Self {
            grid: Grid::new(bounds, cells, limits)?,
            empirical,
        })
    }
}

#[derive(Clone, Debug)]
pub struct CheckpointOccupancy {
    pub episodes: usize,
    pub revision: u64,
    pub axes: Axes,
    pub grid: Grid,
    pub occupancy: Occupancy,
}
#[derive(Clone, Copy, Debug)]
pub struct HistoryLimits {
    pub max_checkpoints: usize,
    /// Output retained ownership. Input checkpoint histograms are independently
    /// charged to the preparation/cache owner, never hidden in this limit.
    pub max_bytes: usize,
}
impl Default for HistoryLimits {
    fn default() -> Self {
        Self {
            max_checkpoints: 512,
            max_bytes: 8 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoveragePoint {
    pub revision: u64,
    pub occupied: usize,
    pub new_cells: usize,
    pub cumulative: usize,
    pub episodes: usize,
    pub decisions: u64,
    pub included: u64,
    pub excluded: u64,
}
/// Rendering choices share the same immutable evidence and never submit work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoverageMode {
    #[default]
    Cumulative,
    Current,
    New,
}
impl CoverageMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cumulative => "Cumulative",
            Self::Current => "Current",
            Self::New => "New",
        }
    }
    pub fn value(self, point: &CoveragePoint) -> usize {
        match self {
            Self::Cumulative => point.cumulative,
            Self::Current => point.occupied,
            Self::New => point.new_cells,
        }
    }
    /// Right-continuous cumulative steps; discrete modes have only vertical stems.
    pub fn edges(self, points: &[CoveragePoint]) -> Vec<[[f64; 2]; 2]> {
        if self == Self::Cumulative {
            points
                .windows(2)
                .flat_map(|p| {
                    let a = [p[0].revision as f64, p[0].cumulative as f64];
                    let b = [p[1].revision as f64, p[0].cumulative as f64];
                    let c = [p[1].revision as f64, p[1].cumulative as f64];
                    [[a, b], [b, c]]
                })
                .collect()
        } else {
            points
                .iter()
                .map(|p| {
                    [
                        [p.revision as f64, 0.],
                        [p.revision as f64, self.value(p) as f64],
                    ]
                })
                .collect()
        }
    }
}
#[derive(Clone, Debug)]
pub struct Coverage {
    pub points: Vec<CoveragePoint>,
    pub first_seen: Vec<Option<u64>>,
    /// A percentage using this denominator refers only to this display grid.
    pub display_grid_cells: usize,
    pub retained_bytes: usize,
}

/// C_k = |union_{j<=k} B_j| for a caller-selected compatible evaluation
/// population. Strictly ordered actual evaluation revisions avoid manufacturing
/// metric/snapshot history; episodes at one revision must be aggregated first.
/// An empty population has no plotted points, not a fabricated zero curve.
pub fn coverage(
    grid: &Grid,
    axes: Axes,
    checkpoints: &[CheckpointOccupancy],
    limits: &HistoryLimits,
) -> Result<Coverage> {
    checked(
        checkpoints.len() <= limits.max_checkpoints,
        "evaluation coverage exceeds checkpoint budget",
    )?;
    let retained_bytes = grid
        .cell_count()
        .checked_mul(std::mem::size_of::<Option<u64>>())
        .and_then(|n| {
            checkpoints
                .len()
                .checked_mul(std::mem::size_of::<CoveragePoint>())
                .and_then(|m| n.checked_add(m))
        })
        .and_then(|n| n.checked_add(std::mem::size_of::<Coverage>()))
        .ok_or_else(|| Error::Shape("evaluation coverage byte overflow".into()))?;
    checked(
        retained_bytes <= limits.max_bytes,
        "evaluation coverage exceeds byte budget",
    )?;
    let mut first_seen = vec![None; grid.cell_count()];
    let mut points = Vec::with_capacity(checkpoints.len());
    let mut previous = None;
    let mut cumulative = 0;
    for checkpoint in checkpoints {
        checked(
            checkpoint.grid == *grid && checkpoint.axes == axes,
            "evaluation coverage requires one fixed coordinate grid",
        )?;
        checked(
            checkpoint.occupancy.counts.len() == grid.cell_count(),
            "evaluation coverage bin shape mismatch",
        )?;
        checked(
            previous.is_none_or(|k| k < checkpoint.revision),
            "evaluation revisions must increase strictly",
        )?;
        let mut occupied = 0;
        let mut new_cells = 0;
        for (&visits, first) in checkpoint.occupancy.counts.iter().zip(&mut first_seen) {
            if visits > 0 {
                occupied += 1;
                if first.is_none() {
                    *first = Some(checkpoint.revision);
                    new_cells += 1;
                }
            }
        }
        cumulative += new_cells;
        points.push(CoveragePoint {
            revision: checkpoint.revision,
            occupied,
            new_cells,
            cumulative,
            episodes: checkpoint.episodes,
            decisions: checkpoint
                .occupancy
                .included
                .checked_add(checkpoint.occupancy.excluded)
                .ok_or_else(|| Error::Shape("coverage decision overflow".into()))?,
            included: checkpoint.occupancy.included,
            excluded: checkpoint.occupancy.excluded,
        });
        previous = Some(checkpoint.revision);
    }
    Ok(Coverage {
        points,
        first_seen,
        display_grid_cells: grid.cell_count(),
        retained_bytes,
    })
}
