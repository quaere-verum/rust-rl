//! Selected-state numerical evidence prepared away from the paint loop.
use egui_plot::PlotPoint;
use rl::{
    Error, Result,
    diagnostics::{
        evaluation::{
            EvaluationHorizon, TrajectoryProgress, evaluation_horizon, trajectory_progress,
            trajectory_scenario_seed,
        },
        schema::{Array, Chunk, DType, Record},
    },
};

const MAX_ROW_COMPONENTS: usize = 64;
const MAX_REWARD_POINTS: usize = 512;
const MAX_OBSERVATION_TRACES: usize = 16;
const MAX_ACTION_TRACES: usize = 4;
const MAX_FRAME_TRACES: usize = 16;
const MAX_OBSERVATION_POINTS: usize = 256;

/// Retain each interval's extrema and endpoints in a bounded public trace.
/// A component containing a non-finite value is omitted entirely.
pub(super) fn observation_traces(chunk: &Chunk) -> Result<Vec<Vec<PlotPoint>>> {
    let Record::Trajectory(record) = &chunk.record else {
        return Err(Error::State(
            "Environment playback requires a trajectory".into(),
        ));
    };
    let observations = chunk
        .arrays
        .get(&record.observations)
        .ok_or_else(|| Error::Shape("Missing episode observations".into()))?;
    let &[rows, width] = observations.shape() else {
        return Err(Error::Shape("Episode observations require a matrix".into()));
    };
    match observations.dtype() {
        DType::F32 => Ok(sample_observations(
            observations.as_f32()?,
            rows,
            width,
            MAX_OBSERVATION_TRACES,
        )),
        DType::F64 => Ok(sample_observations(
            observations.as_f64()?,
            rows,
            width,
            MAX_OBSERVATION_TRACES,
        )),
        _ => Err(Error::Shape(
            "Episode observations require floating-point values".into(),
        )),
    }
}

pub(super) fn action_traces(chunk: &Chunk) -> Result<Vec<Vec<PlotPoint>>> {
    let Record::Trajectory(record) = &chunk.record else {
        return Err(Error::State(
            "Environment playback requires a trajectory".into(),
        ));
    };
    let actions = chunk
        .arrays
        .get(&record.actions)
        .ok_or_else(|| Error::Shape("Missing episode actions".into()))?;
    let &[rows, width] = actions.shape() else {
        return Ok(Vec::new());
    };
    match actions.dtype() {
        DType::F32 => Ok(sample_observations(
            actions.as_f32()?,
            rows,
            width,
            MAX_ACTION_TRACES,
        )),
        DType::F64 => Ok(sample_observations(
            actions.as_f64()?,
            rows,
            width,
            MAX_ACTION_TRACES,
        )),
        _ => Ok(Vec::new()),
    }
}

pub(crate) struct FrameTrace {
    pub name: String,
    pub component: usize,
    pub points: Vec<PlotPoint>,
}

pub(super) fn frame_traces(chunk: &Chunk) -> Result<Vec<FrameTrace>> {
    let Record::Trajectory(record) = &chunk.record else {
        return Err(Error::State(
            "Environment playback requires a trajectory".into(),
        ));
    };
    let mut result = Vec::new();
    for (name, key) in &record.frames {
        if result.len() == MAX_FRAME_TRACES {
            break;
        }
        if name.len() > 128 {
            continue;
        }
        let Some(array) = chunk.arrays.get(key) else {
            continue;
        };
        let &[rows, width] = array.shape() else {
            continue;
        };
        let remaining = MAX_FRAME_TRACES - result.len();
        let traces = match array.dtype() {
            DType::F32 => sample_observations(array.as_f32()?, rows, width, remaining),
            DType::F64 => sample_observations(array.as_f64()?, rows, width, remaining),
            _ => continue,
        };
        result.extend(
            traces
                .into_iter()
                .enumerate()
                .filter(|(_, trace)| !trace.is_empty())
                .map(|(component, points)| FrameTrace {
                    name: name.clone(),
                    component,
                    points,
                }),
        );
    }
    Ok(result)
}

#[derive(Clone, Copy)]
pub(super) enum RecordedTimes<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl RecordedTimes<'_> {
    pub(super) fn len(self) -> usize {
        match self {
            Self::F32(times) => times.len(),
            Self::F64(times) => times.len(),
        }
    }
    pub(super) fn get(self, index: usize) -> Option<f64> {
        match self {
            Self::F32(times) => times.get(index).map(|value| f64::from(*value)),
            Self::F64(times) => times.get(index).copied(),
        }
    }
    pub(super) fn partition_point(self, time: f64) -> usize {
        match self {
            Self::F32(times) => times.partition_point(|value| f64::from(*value) < time),
            Self::F64(times) => times.partition_point(|value| *value < time),
        }
    }
    fn strictly_increasing(self) -> bool {
        match self {
            Self::F32(times) => {
                times.iter().all(|time| time.is_finite())
                    && times.windows(2).all(|pair| pair[0] < pair[1])
            }
            Self::F64(times) => {
                times.iter().all(|time| time.is_finite())
                    && times.windows(2).all(|pair| pair[0] < pair[1])
            }
        }
    }
}

pub(super) fn recorded_times(chunk: &Chunk) -> Option<RecordedTimes<'_>> {
    let Record::Trajectory(record) = &chunk.record else {
        return None;
    };
    let key = record.simulation_times.as_ref()?;
    let array = chunk.arrays.get(key)?;
    let times = match array.dtype() {
        DType::F32 => RecordedTimes::F32(array.as_f32().ok()?),
        DType::F64 => RecordedTimes::F64(array.as_f64().ok()?),
        _ => return None,
    };
    let observations = chunk.arrays.get(&record.observations)?;
    let rows = *observations.shape().first()?;
    (times.len() == rows).then_some(times)
}

pub(super) fn simulation_times(chunk: &Chunk) -> Option<RecordedTimes<'_>> {
    let times = recorded_times(chunk)?;
    times.strictly_increasing().then_some(times)
}

pub(super) fn convert_to_time(points: &mut [PlotPoint], times: RecordedTimes<'_>) {
    for point in points {
        if let Some(time) = times.get(point.x as usize) {
            point.x = time;
        }
    }
}

fn sample_observations<T: Copy + Into<f64>>(
    values: &[T],
    rows: usize,
    width: usize,
    max_components: usize,
) -> Vec<Vec<PlotPoint>> {
    (0..width.min(max_components))
        .map(|component| {
            let value = |row: usize| -> f64 { values[row * width + component].into() };
            let mut points = Vec::with_capacity(rows.min(MAX_OBSERVATION_POINTS));
            if rows == 0 {
                return points;
            }
            if (0..rows).any(|row| !value(row).is_finite()) {
                return points;
            }
            let point = |row: usize| PlotPoint::new(row as f64, value(row));
            if rows <= MAX_OBSERVATION_POINTS {
                points.extend((0..rows).map(point));
                return points;
            }
            points.push(point(0));
            let bucket_width = (rows - 2).div_ceil((MAX_OBSERVATION_POINTS - 2) / 2);
            for start in (1..rows - 1).step_by(bucket_width) {
                let end = start.saturating_add(bucket_width).min(rows - 1);
                let (mut low, mut high) = (start, start);
                for row in start + 1..end {
                    if value(row) < value(low) {
                        low = row;
                    }
                    if value(row) > value(high) {
                        high = row;
                    }
                }
                points.push(point(low.min(high)));
                if low != high {
                    points.push(point(low.max(high)));
                }
            }
            points.push(point(rows - 1));
            points
        })
        .collect()
}

#[cfg(test)]
#[test]
fn observation_telemetry_is_bounded_and_keeps_episode_endpoints() {
    let rows = 10_000;
    let mut values = Vec::with_capacity(rows * 2);
    for row in 0..rows {
        values.extend([row as f64, -(row as f64)]);
    }
    let traces = sample_observations(&values, rows, 2, MAX_OBSERVATION_TRACES);
    assert_eq!(traces.len(), 2);
    for (component, trace) in traces.iter().enumerate() {
        assert!(trace.len() <= MAX_OBSERVATION_POINTS);
        assert_eq!(trace.first().unwrap().x, 0.);
        assert_eq!(trace.last().unwrap().x, (rows - 1) as f64);
        assert_eq!(
            trace.last().unwrap().y,
            if component == 0 { 9999. } else { -9999. }
        );
        assert!(trace.windows(2).all(|points| points[0].x < points[1].x));
    }
    values[0] = f64::NAN;
    let traces = sample_observations(&values, rows, 2, MAX_OBSERVATION_TRACES);
    assert!(traces[0].is_empty());
    assert!(!traces[1].is_empty());
}

#[cfg(test)]
#[test]
fn observation_telemetry_preserves_short_lived_extremes() {
    let mut values = vec![0.0; 10_000];
    values[1_234] = 9.0;
    values[5_678] = -7.0;
    let trace = &sample_observations(&values, values.len(), 1, MAX_OBSERVATION_TRACES)[0];
    assert!(trace.len() <= MAX_OBSERVATION_POINTS);
    assert!(trace.iter().any(|point| point.x == 1_234. && point.y == 9.));
    assert!(
        trace
            .iter()
            .any(|point| point.x == 5_678. && point.y == -7.)
    );
    assert!(trace.windows(2).all(|points| points[0].x < points[1].x));
}

/// Retain both extremes of each interval so an isolated reward spike remains visible.
pub(super) fn episode_rewards(chunk: &Chunk) -> Result<Vec<PlotPoint>> {
    let Record::Trajectory(record) = &chunk.record else {
        return Err(Error::State(
            "Environment playback requires a trajectory".into(),
        ));
    };
    let rewards = chunk
        .arrays
        .get(&record.rewards)
        .ok_or_else(|| Error::Shape("Missing episode rewards".into()))?;
    match rewards.dtype() {
        DType::F32 => Ok(reward_points(rewards.as_f32()?)),
        DType::F64 => Ok(reward_points(rewards.as_f64()?)),
        _ => Err(Error::Shape(
            "Episode rewards require floating-point values".into(),
        )),
    }
}

fn reward_points<T: Copy + Into<f64>>(rewards: &[T]) -> Vec<PlotPoint> {
    let point = |step: usize| PlotPoint::new(step as f64, rewards[step].into());
    if rewards.len() <= MAX_REWARD_POINTS {
        return (0..rewards.len()).map(point).collect();
    }
    let bucket_width = rewards.len().div_ceil((MAX_REWARD_POINTS - 2) / 2);
    let mut points = Vec::with_capacity(MAX_REWARD_POINTS);
    points.push(point(0));
    for (bucket, values) in rewards.chunks(bucket_width).enumerate() {
        let first = [0, 0];
        let [low, high] =
            values
                .iter()
                .enumerate()
                .skip(1)
                .fold(first, |[low, high], (i, &value)| {
                    [
                        if value.into() < values[low].into() {
                            i
                        } else {
                            low
                        },
                        if value.into() > values[high].into() {
                            i
                        } else {
                            high
                        },
                    ]
                });
        let start = bucket * bucket_width;
        let earlier = start + low.min(high);
        if earlier > 0 {
            points.push(point(earlier));
        }
        if low != high {
            points.push(point(start + low.max(high)));
        }
    }
    let last = point(rewards.len() - 1);
    if points.last().is_none_or(|point| point.x < last.x) {
        points.push(last);
    }
    points
}

#[cfg(test)]
#[test]
fn bounded_reward_points_keep_spikes_and_episode_endpoints() {
    let mut values = vec![0.; 10_000];
    values[391] = 74.;
    values[392] = -81.;
    let points = reward_points(&values);
    assert!(points.len() <= MAX_REWARD_POINTS);
    assert_eq!(points[0], PlotPoint::new(0., 0.));
    assert_eq!(points.last(), Some(&PlotPoint::new(9_999., 0.)));
    assert!(points.contains(&PlotPoint::new(391., 74.)));
    assert!(points.contains(&PlotPoint::new(392., -81.)));
    assert!(points.windows(2).all(|rows| rows[0].x < rows[1].x));
}

pub struct ObservedRow {
    pub values: Vec<f64>,
    pub dimensions: usize,
}

impl ObservedRow {
    fn new(array: &Array, index: usize) -> Result<Self> {
        let &[rows, dimensions] = array.shape() else {
            return Err(Error::Shape("episode rows require a numeric matrix".into()));
        };
        if index >= rows {
            return Err(Error::Shape(
                "episode row is outside the recorded matrix".into(),
            ));
        }
        let start = index * dimensions;
        let end = start + dimensions.min(MAX_ROW_COMPONENTS);
        let values = match array.dtype() {
            DType::F32 => array.as_f32()?[start..end]
                .iter()
                .copied()
                .map(f64::from)
                .collect(),
            DType::F64 => array.as_f64()?[start..end].to_vec(),
            _ => {
                return Err(Error::Shape(
                    "episode rows require floating-point values".into(),
                ));
            }
        };
        Ok(Self { values, dimensions })
    }
}

pub enum ExecutedAction {
    Discrete(u64),
    Continuous(ObservedRow),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeEnd {
    Terminated,
    Truncated,
    ObjectiveHorizon,
    Incomplete,
}

pub struct PlaybackState {
    pub observations: ObservedRow,
    pub executed_action: Option<ExecutedAction>,
    pub reward: Option<f64>,
    pub progress: TrajectoryProgress,
    pub end: Option<EpisodeEnd>,
    pub scenario_seed: Option<u64>,
    pub horizon: EvaluationHorizon,
}

impl PlaybackState {
    pub(super) fn new(chunk: &Chunk, step: usize) -> Result<Self> {
        let Record::Trajectory(record) = &chunk.record else {
            return Err(Error::State(
                "Environment playback requires a trajectory".into(),
            ));
        };
        let array = |key: &str| {
            chunk
                .arrays
                .get(key)
                .ok_or_else(|| Error::Shape(format!("missing episode array {key}")))
        };
        let rewards = array(&record.rewards)?;
        let transitions = rewards.len();
        let progress = trajectory_progress(chunk, step)?;
        let observations = ObservedRow::new(array(&record.observations)?, step)?;
        let horizon = evaluation_horizon(chunk)?;
        let scenario_seed = trajectory_scenario_seed(chunk)?;
        let (executed_action, reward, end) = if step < transitions {
            let actions = array(&record.actions)?;
            let action = match actions.dtype() {
                DType::U64 => ExecutedAction::Discrete(
                    *actions
                        .as_u64()?
                        .get(step)
                        .ok_or_else(|| Error::Shape("missing episode action".into()))?,
                ),
                _ => ExecutedAction::Continuous(ObservedRow::new(actions, step)?),
            };
            let reward = match rewards.dtype() {
                DType::F32 => f64::from(rewards.as_f32()?[step]),
                DType::F64 => rewards.as_f64()?[step],
                _ => {
                    return Err(Error::Shape(
                        "episode rewards require floating-point values".into(),
                    ));
                }
            };
            (Some(action), Some(reward), None)
        } else {
            let terminated = array(&record.terminated)?.as_u8()?.last() == Some(&1);
            let truncated = array(&record.truncated)?.as_u8()?.last() == Some(&1);
            let end = if terminated {
                EpisodeEnd::Terminated
            } else if truncated {
                EpisodeEnd::Truncated
            } else if horizon == EvaluationHorizon::FiniteDecisions(transitions) {
                EpisodeEnd::ObjectiveHorizon
            } else {
                EpisodeEnd::Incomplete
            };
            (None, None, Some(end))
        };
        Ok(Self {
            observations,
            executed_action,
            reward,
            progress,
            end,
            scenario_seed,
            horizon,
        })
    }
}
