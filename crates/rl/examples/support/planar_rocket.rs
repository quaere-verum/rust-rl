//! Presentation-sized port of the historical fixed-mass planar landing simulator.
use super::{Run, off_policy_config};
use burn::module::Param;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rl::{
    diagnostics::{
        evaluation::{EnvironmentFrame, FrameSource},
        models::{ModelBundle, ModelLimits},
        probes::ProbeSet,
        reference::{
            BenchmarkActor, ConfigValue, ReferenceContext, ReferenceHorizon, ReferenceInformation,
            ReferenceKind, ReferenceLifetime, ReferenceObjective, ReferenceQuality, ReferenceSpec,
            ReferenceTarget,
        },
        render::*,
        schema::{Chunk, MetricRecord, ProbePopulation, Record, RecordKind, RunManifest},
        storage::{AppendOutcome, ReadLimits, Recorder, Recording, StorageConfig},
        value::ValueConventions,
        visualization::*,
    },
    prelude::*,
};
use std::{collections::BTreeMap, f64::consts::FRAC_PI_2};

const HORIZON: usize = 240;
const ROCKET_GAMMA: f32 = 0.995;
const TIME_COST_RATE: f64 = 12.;
const DT: f64 = 0.05;
const RADIUS: f64 = 0.25;
const X_LIMIT: f64 = 8.;
const Y_LIMIT: f64 = 20.;
const PAD_RADIUS: f64 = 0.75;
const GRAVITY: f64 = 9.81;
const MAIN_ACCELERATION: f64 = 20.;
const ANGULAR_ACCELERATION: f64 = 2.;
const ANGULAR_DRAG: f64 = 0.4;
const FUEL_CAPACITY: f64 = 5.;
const SCALES: [f64; 7] = [X_LIMIT, Y_LIMIT, 5., 5., FRAC_PI_2, 3., FUEL_CAPACITY];
const HISTORY_METADATA: &str = "evaluation.rocket_history";
const HISTORY_METRICS: &[&str] = &[
    "evaluation.episodes",
    "evaluation.landing_success_rate",
    "evaluation.flight_duration_seconds",
    "evaluation.terminal_abs_x",
    "evaluation.terminal_abs_vx",
    "evaluation.terminal_abs_vy",
    "evaluation.success_fuel_used",
    "evaluation.success_duration_seconds",
    "evaluation.success_abs_x",
    "evaluation.success_abs_vx",
    "evaluation.success_abs_vy",
];

#[derive(Clone, Copy, Default)]
struct State {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    theta: f64,
    omega: f64,
    fuel: f64,
}

impl State {
    fn observation(self, decisions: usize) -> Observation {
        let values = [
            self.x, self.y, self.vx, self.vy, self.theta, self.omega, self.fuel,
        ];
        let mut values: Vec<_> = values
            .into_iter()
            .zip(SCALES)
            .map(|(v, scale)| (v / scale) as f32)
            .collect();
        values.push((HORIZON - decisions) as f32 / HORIZON as f32);
        Observation::new(values)
    }

    fn potential(self) -> f64 {
        -5. * self.x.abs()
            - 2. * (self.y - RADIUS).max(0.)
            - 2. * self.vx.abs()
            - 2. * self.vy.abs()
            - 10. * self.theta.abs()
            - self.omega.abs()
    }

    fn safe_landing(self) -> bool {
        self.x.abs() <= PAD_RADIUS
            && self.vx.abs() <= 0.75
            && self.vy.abs() <= 1.
            && self.theta.abs() <= 0.15
            && self.omega.abs() <= 0.5
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Flying,
    Landed,
    GroundCrash,
    Escape,
    AttitudeFailure,
    Horizon,
}

impl Event {
    fn label(self) -> &'static str {
        match self {
            Self::Flying => "Flying",
            Self::Landed => "Landed",
            Self::GroundCrash => "Ground crash",
            Self::Escape => "Out of bounds",
            Self::AttitudeFailure => "Attitude limit",
            Self::Horizon => "Time limit",
        }
    }

    fn reward(self) -> f64 {
        match self {
            Self::Flying => 0.,
            Self::Landed => 100.,
            Self::Horizon => -50.,
            _ => -100.,
        }
    }
}

fn shaping_reward(previous: f64, next: f64, terminal: bool) -> f64 {
    let next = if terminal { 0. } else { next };
    f64::from(ROCKET_GAMMA) * next - previous
}

#[derive(Clone, Copy)]
enum RocketObjective {
    Historical,
    Guarded,
}
impl RocketObjective {
    fn from_flag(guarded: bool) -> Self {
        if guarded {
            Self::Guarded
        } else {
            Self::Historical
        }
    }
    fn discount(self) -> f32 {
        match self {
            Self::Historical => 0.99,
            Self::Guarded => ROCKET_GAMMA,
        }
    }
    fn id(self) -> &'static str {
        match self {
            Self::Historical => "planar_rocket",
            Self::Guarded => "planar_rocket_guarded_v1",
        }
    }
    fn time_cost(self) -> f64 {
        match self {
            Self::Historical => 0.3,
            Self::Guarded => TIME_COST_RATE,
        }
    }
    fn shaping(self, previous: f64, next: f64, terminal: bool) -> f64 {
        match self {
            Self::Historical => next - previous,
            Self::Guarded => shaping_reward(previous, next, terminal),
        }
    }
}

#[derive(Clone)]
struct PlanarRocket {
    objective: RocketObjective,
    state: State,
    event: Event,
    space: ActionSpace,
    rng: ChaCha8Rng,
    decisions: usize,
    time: f64,
    throttle: f64,
}

impl PlanarRocket {
    fn new(seed: u64) -> Self {
        Self::with_objective(seed, RocketObjective::Historical)
    }

    fn with_objective(seed: u64, objective: RocketObjective) -> Self {
        Self {
            objective,
            state: State::default(),
            event: Event::Horizon,
            space: ActionSpace::Box {
                low: vec![0., -1.],
                high: vec![1., 1.],
            },
            rng: ChaCha8Rng::seed_from_u64(seed),
            decisions: 0,
            time: 0.,
            throttle: 0.,
        }
    }

    /// Freeze acceleration twice per decision, splitting again at fuel exhaustion.
    fn advance(&mut self, throttle: f64, attitude: f64) -> f64 {
        let previous = self.state.potential();
        let noise = [0.05, 0.05, 0.01].map(|std| self.rng.sample::<f64, _>(StandardNormal) * std);
        let mut duration = 0.;
        let mut effort = 0.;
        let mut throttle_time = 0.;
        for _ in 0..2 {
            let mut remaining = DT * 0.5;
            while remaining > 0. && self.event == Event::Flying {
                let state = self.state;
                let (u, c) = if state.fuel > 0. {
                    (throttle, attitude)
                } else {
                    (0., 0.)
                };
                let burn = u + 0.1 * c.abs();
                let exhaustion = if burn > 0. {
                    state.fuel / burn
                } else {
                    f64::INFINITY
                };
                let segment = remaining.min(exhaustion);
                let position = [state.x, state.y, state.theta];
                let velocity = [state.vx, state.vy, state.omega];
                let acceleration = [
                    MAIN_ACCELERATION * u * state.theta.sin() + noise[0],
                    MAIN_ACCELERATION * u * state.theta.cos() - GRAVITY + noise[1],
                    ANGULAR_ACCELERATION * c - ANGULAR_DRAG * state.omega + noise[2],
                ];
                // Ground wins exact ties, followed by translation and attitude limits.
                let mut hit = None;
                let mut elapsed = segment;
                for (axis, limit, event) in [
                    (1, RADIUS, Event::GroundCrash),
                    (0, -X_LIMIT, Event::Escape),
                    (0, X_LIMIT, Event::Escape),
                    (1, Y_LIMIT, Event::Escape),
                    (2, -FRAC_PI_2, Event::AttitudeFailure),
                    (2, FRAC_PI_2, Event::AttitudeFailure),
                ] {
                    if let Some(t) = first_root(
                        position[axis] - limit,
                        velocity[axis],
                        acceleration[axis],
                        elapsed,
                    ) && (hit.is_none() || t < elapsed)
                    {
                        elapsed = t;
                        hit = Some((axis, limit, event));
                    }
                }
                let mut next = std::array::from_fn(|i| {
                    (0.5 * acceleration[i] * elapsed)
                        .mul_add(elapsed, velocity[i].mul_add(elapsed, position[i]))
                });
                if let Some((axis, limit, _)) = hit {
                    next[axis] = limit;
                }
                [self.state.x, self.state.y, self.state.theta] = next;
                [self.state.vx, self.state.vy, self.state.omega] =
                    std::array::from_fn(|i| acceleration[i].mul_add(elapsed, velocity[i]));
                self.state.fuel = if elapsed == exhaustion {
                    0.
                } else {
                    (state.fuel - burn * elapsed).max(0.)
                };
                duration += elapsed;
                throttle_time += elapsed * u;
                effort += elapsed * (0.2 * u * u + 0.05 * c * c);
                remaining -= elapsed;
                if let Some((_, _, event)) = hit {
                    self.event = if event == Event::GroundCrash && self.state.safe_landing() {
                        Event::Landed
                    } else {
                        event
                    };
                }
            }
        }
        self.time += duration;
        self.decisions += 1;
        self.throttle = if duration > 0. {
            throttle_time / duration
        } else {
            0.
        };
        if self.event == Event::Flying && self.decisions == HORIZON {
            self.event = Event::Horizon;
        }
        self.objective.shaping(
            previous,
            self.state.potential(),
            self.event != Event::Flying,
        ) - self.objective.time_cost() * duration
            - effort
            + self.event.reward()
    }
}

/// First crossing for a constant-acceleration segment, including tangent contact.
fn first_root(c: f64, v: f64, acceleration: f64, end: f64) -> Option<f64> {
    let a = 0.5 * acceleration;
    if a == 0. {
        let t = -c / v;
        return (0. ..=end).contains(&t).then_some(t);
    }
    let vv = v * v;
    let ac = 4. * a * c;
    let discriminant = (vv - ac) + (v.mul_add(v, -vv) - (4. * a).mul_add(c, -ac));
    if discriminant < 0. {
        return None;
    }
    let numerator = -0.5 * (v + discriminant.sqrt().copysign(v));
    let roots = if numerator == 0. {
        [-v / (2. * a), f64::INFINITY]
    } else {
        [numerator / a, c / numerator]
    };
    roots
        .into_iter()
        .filter(|t| (0. ..=end).contains(t))
        .min_by(f64::total_cmp)
}

impl Env for PlanarRocket {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(8)
    }

    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.state = State {
            x: self.rng.random_range(-1. ..1.),
            y: self.rng.random_range(6. ..8.),
            vx: self.rng.random_range(-0.3..0.3),
            vy: self.rng.random_range(-1.5..-0.5),
            theta: self.rng.random_range(-0.08..0.08),
            omega: self.rng.random_range(-0.05..0.05),
            fuel: FUEL_CAPACITY,
        };
        self.decisions = 0;
        self.time = 0.;
        self.throttle = 0.;
        self.event = Event::Flying;
        Ok(EpisodeStart::fresh(self.state.observation(self.decisions)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.event != Event::Flying {
            return Err(Error::State(
                "reset required after planar terminal event".into(),
            ));
        }
        self.space.check(action, None)?;
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        let reward = self.advance(f64::from(action[0]), f64::from(action[1]));
        Ok(Step {
            observation: self.state.observation(self.decisions),
            reward: (reward as f32).into(),
            terminated: self.event != Event::Flying,
            truncated: false,
        })
    }
}

impl FrameSource for PlanarRocket {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        let state = self.state;
        Ok(EnvironmentFrame {
            environment: self.objective.id().into(),
            simulation_time: Some(self.time),
            channels: BTreeMap::from([
                ("horizontal_position".into(), vec![state.x]),
                ("altitude".into(), vec![state.y]),
                ("horizontal_velocity".into(), vec![state.vx]),
                ("vertical_velocity".into(), vec![state.vy]),
                ("attitude".into(), vec![state.theta]),
                ("angular_velocity".into(), vec![state.omega]),
                ("fuel".into(), vec![state.fuel]),
                ("applied_throttle".into(), vec![self.throttle]),
            ]),
        })
    }
}

/// Evaluation wrapper accumulates a bounded flight path independently of training.
struct RenderedRocket {
    inner: PlanarRocket,
    path: Vec<RenderPoint>,
}

impl RenderedRocket {
    fn new(seed: u64, objective: RocketObjective) -> Self {
        Self {
            inner: PlanarRocket::with_objective(seed, objective),
            path: Vec::with_capacity(HORIZON + 1),
        }
    }

    fn remember(&mut self) {
        self.path.push([self.inner.state.x, self.inner.state.y]);
    }
}

impl Env for RenderedRocket {
    fn observation_space(&self) -> ObservationSpace {
        self.inner.observation_space()
    }

    fn action_space(&self) -> &ActionSpace {
        self.inner.action_space()
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        let start = self.inner.reset(seed)?;
        self.path.clear();
        self.remember();
        Ok(start)
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        let step = self.inner.step(action)?;
        self.remember();
        Ok(step)
    }
}

impl FrameSource for RenderedRocket {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        self.inner.diagnostic_frame()
    }
}

impl RenderSource for RenderedRocket {
    fn render_scene(&self) -> Result<RenderScene> {
        let state = self.inner.state;
        let center = [state.x, state.y];
        let (sin, cos) = state.theta.sin_cos();
        // The silhouette is enlarged for readability; the outlined circle remains the physical collision radius.
        let body = |x: f64, y: f64| {
            [
                state.x + 4. * (x * cos + y * sin),
                state.y + 4. * (-x * sin + y * cos),
            ]
        };
        let stroke = |color, width| RenderStroke { color, width };
        let green = [91, 226, 165, 255];
        let blue = [96, 207, 246, 255];
        let white = [229, 238, 247, 255];
        let mut items = Vec::new();
        let mut draw = |layer, primitive| items.push(RenderItem { layer, primitive });
        draw(
            0,
            RenderPrimitive::Rectangle {
                min: [-X_LIMIT, -0.7],
                max: [X_LIMIT, 0.],
                fill: Some([42, 59, 76, 255]),
                stroke: None,
            },
        );
        draw(
            1,
            RenderPrimitive::Line {
                from: [-PAD_RADIUS, 0.],
                to: [PAD_RADIUS, 0.],
                stroke: stroke(green, 4.),
            },
        );
        let count = self.path.len().min(48);
        if count > 1 {
            draw(
                2,
                RenderPrimitive::Polyline {
                    points: (0..count)
                        .map(|i| self.path[i * (self.path.len() - 1) / (count - 1)])
                        .collect(),
                    stroke: stroke([99, 174, 214, 180], 1.5),
                },
            );
        }
        draw(
            6,
            RenderPrimitive::Circle {
                center,
                radius: RADIUS,
                fill: None,
                stroke: Some(stroke([219, 232, 242, 190], 1.)),
            },
        );
        let vehicle = match self.inner.event {
            Event::Flying => blue,
            Event::Landed => green,
            _ => [244, 102, 90, 255],
        };
        draw(
            4,
            RenderPrimitive::Polygon {
                points: vec![
                    body(-0.12, -0.18),
                    body(0.12, -0.18),
                    body(0.12, 0.15),
                    body(-0.12, 0.15),
                ],
                fill: Some(vehicle),
                stroke: Some(stroke(white, 1.)),
            },
        );
        draw(
            4,
            RenderPrimitive::Polygon {
                points: vec![body(-0.12, 0.15), body(0., 0.29), body(0.12, 0.15)],
                fill: Some(white),
                stroke: Some(stroke(white, 1.)),
            },
        );
        for side in [-1., 1.] {
            draw(
                3,
                RenderPrimitive::Polygon {
                    points: vec![
                        body(side * 0.11, -0.02),
                        body(side * 0.22, -0.20),
                        body(side * 0.11, -0.16),
                    ],
                    fill: Some([104, 139, 167, 255]),
                    stroke: Some(stroke(white, 1.)),
                },
            );
            draw(
                4,
                RenderPrimitive::Line {
                    from: body(side * 0.10, -0.14),
                    to: body(side * 0.23, -0.24),
                    stroke: stroke(white, 1.),
                },
            );
        }
        draw(
            5,
            RenderPrimitive::Circle {
                center: body(0., 0.07),
                radius: 0.055,
                fill: Some([21, 55, 76, 255]),
                stroke: Some(stroke(white, 1.)),
            },
        );
        let throttle = self.inner.throttle;
        if self.inner.event == Event::Flying && throttle > 0. {
            draw(
                3,
                RenderPrimitive::Polygon {
                    points: vec![
                        body(-0.075, -0.18),
                        body(0., -0.2 - 0.34 * throttle),
                        body(0.075, -0.18),
                    ],
                    fill: Some([255, 175, 76, 220]),
                    stroke: None,
                },
            );
        }
        if state.vx != 0. || state.vy != 0. {
            draw(
                3,
                RenderPrimitive::Line {
                    from: center,
                    to: [state.x + 0.25 * state.vx, state.y + 0.25 * state.vy],
                    stroke: stroke(green, 1.5),
                },
            );
        }
        draw(
            5,
            RenderPrimitive::Text {
                position: [0., Y_LIMIT - 0.5],
                text: format!(
                    "{}  •  {:.2} s  •  {:.2} kg fuel",
                    self.inner.event.label(),
                    self.inner.time,
                    state.fuel
                ),
                color: white,
                size: 13.,
            },
        );
        draw(
            5,
            RenderPrimitive::Text {
                position: [0., Y_LIMIT - 0.95],
                text: "Vehicle ×4 · circle: physical radius".into(),
                color: [159, 180, 199, 255],
                size: 11.,
            },
        );
        Ok(RenderScene {
            version: RENDER_VERSION,
            bounds: RenderBounds {
                min: [-X_LIMIT, -0.7],
                max: [X_LIMIT, Y_LIMIT],
            },
            background: [14, 23, 38, 255],
            items,
        })
    }
}

struct LandingController(ActionSpace);

impl Actor for LandingController {
    fn action_space(&self) -> &ActionSpace {
        &self.0
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.clear();
        for observation in observations.iter() {
            observation.validate(8, &self.0, true)?;
            let [x, y, vx, vy, theta, omega, fuel] =
                std::array::from_fn(|i| f64::from(observation.dense_values()[i]) * SCALES[i]);
            if fuel <= 0. {
                actions.push(Action::Continuous(vec![0., 0.]));
                continue;
            }
            let ax = (-0.8 * x - 1.5 * vx).clamp(-3., 3.);
            let target_vy = -(0.3 + 0.55 * (y - RADIUS).max(0.)).min(2.);
            let vertical = (GRAVITY + 3. * (target_vy - vy)).max(0.);
            let target_theta = ax.atan2(vertical).clamp(-0.3, 0.3);
            let throttle = (ax.hypot(vertical) / MAIN_ACCELERATION).clamp(0., 1.) as f32;
            let torque = ((6. * (target_theta - theta) - 3. * omega) / ANGULAR_ACCELERATION)
                .clamp(-1., 1.) as f32;
            actions.push(Action::Continuous(vec![throttle, torque]));
        }
        Ok(())
    }
}

#[derive(Debug, Default, PartialEq)]
struct BenchmarkSummary {
    // Landed, ground crash, escape, attitude failure, horizon.
    outcomes: [usize; 5],
    return_total: f64,
    return_squared_total: f64,
    duration_total: f64,
    terminal_abs_x_total: f64,
    terminal_abs_vx_total: f64,
    terminal_abs_vy_total: f64,
    terminal_fuel_total: f64,
    success_fuel_total: f64,
    success_duration_total: f64,
    success_abs_x_total: f64,
    success_abs_vx_total: f64,
    success_abs_vy_total: f64,
}

fn historical_metrics(summary: &BenchmarkSummary) -> Result<MetricRecord> {
    let count: usize = summary.outcomes.iter().sum();
    if count == 0 {
        return Err(Error::Config(
            "Historical evaluation has no episodes".into(),
        ));
    }
    let n = count as f64;
    let successes = summary.outcomes[0] as f64;
    let mut values = BTreeMap::from([
        ("evaluation.episodes".into(), n),
        ("evaluation.landing_success_rate".into(), successes / n),
        (
            "evaluation.flight_duration_seconds".into(),
            summary.duration_total / n,
        ),
        (
            "evaluation.terminal_abs_x".into(),
            summary.terminal_abs_x_total / n,
        ),
        (
            "evaluation.terminal_abs_vx".into(),
            summary.terminal_abs_vx_total / n,
        ),
        (
            "evaluation.terminal_abs_vy".into(),
            summary.terminal_abs_vy_total / n,
        ),
    ]);
    let mut unavailable = BTreeMap::new();
    for (name, total) in [
        ("evaluation.success_fuel_used", summary.success_fuel_total),
        (
            "evaluation.success_duration_seconds",
            summary.success_duration_total,
        ),
        ("evaluation.success_abs_x", summary.success_abs_x_total),
        ("evaluation.success_abs_vx", summary.success_abs_vx_total),
        ("evaluation.success_abs_vy", summary.success_abs_vy_total),
    ] {
        if successes == 0. {
            unavailable.insert(
                name.into(),
                "No successful landing in this evaluation".into(),
            );
        } else {
            values.insert(name.into(), total / successes);
        }
    }
    Ok(MetricRecord {
        values,
        unavailable,
    })
}

#[cfg(test)]
fn evaluate_benchmark(seeds: &[u64]) -> Result<BenchmarkSummary> {
    let actor = LandingController(PlanarRocket::new(0).action_space().clone());
    evaluate_actor(&actor, seeds, RocketObjective::Historical)
}

fn evaluate_actor(
    actor: &dyn Actor,
    seeds: &[u64],
    objective: RocketObjective,
) -> Result<BenchmarkSummary> {
    if seeds.is_empty() {
        return Err(Error::Config(
            "rocket evaluation requires at least one scenario".into(),
        ));
    }
    let mut summary = BenchmarkSummary::default();
    for (index, &seed) in seeds.iter().enumerate() {
        let mut env = PlanarRocket::with_objective(seed, objective);
        let mut observation = env.reset(Some(seed))?.observation;
        let mut actor_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut actions = Vec::new();
        let mut discounted_return = 0.;
        let mut discount = 1.;
        while env.event == Event::Flying {
            actor.act_into(
                ObservationBatch::all(std::slice::from_ref(&observation)),
                ActMode::Deterministic,
                &mut actor_rng,
                &mut actions,
            )?;
            let step = env.step(&actions[0])?;
            discounted_return += discount * f64::from(step.reward.total());
            discount *= f64::from(objective.discount());
            observation = step.observation;
        }
        let outcome = match env.event {
            Event::Landed => 0,
            Event::GroundCrash => 1,
            Event::Escape => 2,
            Event::AttitudeFailure => 3,
            Event::Horizon => 4,
            Event::Flying => unreachable!(),
        };
        summary.outcomes[outcome] += 1;
        summary.return_total += discounted_return;
        summary.return_squared_total += discounted_return * discounted_return;
        summary.duration_total += env.time;
        summary.terminal_abs_x_total += env.state.x.abs();
        summary.terminal_abs_vx_total += env.state.vx.abs();
        summary.terminal_abs_vy_total += env.state.vy.abs();
        summary.terminal_fuel_total += env.state.fuel;
        if env.event == Event::Landed {
            summary.success_fuel_total += FUEL_CAPACITY - env.state.fuel;
            summary.success_duration_total += env.time;
            summary.success_abs_x_total += env.state.x.abs();
            summary.success_abs_vx_total += env.state.vx.abs();
            summary.success_abs_vy_total += env.state.vy.abs();
        }
        if seeds.len() >= 50 && ((index + 1) % 50 == 0 || index + 1 == seeds.len()) {
            eprintln!(
                "rocket evaluation: {}/{} episodes complete",
                index + 1,
                seeds.len()
            );
        }
    }
    Ok(summary)
}

pub(super) fn benchmark(run: &Run) -> Result<()> {
    let seeds: Vec<_> = (0..run.steps)
        .map(|offset| run.seed.wrapping_add(offset))
        .collect();
    let actor = LandingController(PlanarRocket::new(0).action_space().clone());
    let result = evaluate_actor(
        &actor,
        &seeds,
        RocketObjective::from_flag(run.rocket_guarded),
    )?;
    print_evaluation("landing-feedback", &seeds, &result)
}

pub(super) fn evaluate_learner(actor: &dyn Actor, seeds: &[u64], guarded: bool) -> Result<()> {
    let result = evaluate_actor(actor, seeds, RocketObjective::from_flag(guarded))?;
    print_evaluation("learner", seeds, &result)
}

fn validate_history_manifest(manifest: &RunManifest, objective: RocketObjective) -> Result<()> {
    let space = PlanarRocket::new(0).action_space().clone();
    let expected = context(space.clone(), objective);
    if manifest.environment != objective.id()
        || manifest.observation_dim != 8
        || manifest.action_space != space
        || manifest.discount != f64::from(objective.discount())
        || rl::diagnostics::reference::manifest_context(manifest)? != Some(expected)
    {
        return Err(Error::Config(
            "source rocket simulator context is incompatible".into(),
        ));
    }
    Ok(())
}

fn validate_history_destination(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> Result<()> {
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    if parent.canonicalize()?.starts_with(source.canonicalize()?) {
        return Err(Error::Config(
            "history output must be outside the source recording".into(),
        ));
    }
    Ok(())
}

/// Copy the complete immutable recording, then add independent aggregate
/// evaluation metrics for retained historical policies. Original record
/// sequences are kept intact so existing cross-record identities stay valid.
pub(super) fn historical_evaluation(run: &Run, source: Option<&std::path::Path>) -> Result<()> {
    if run.evaluation_episodes == 0 {
        return Err(Error::Config(
            "rocket-history requires --evaluation-episodes".into(),
        ));
    }
    if run.reference
        || run.live
        || run.render
        || run.replay_population
        || run.wide_lqg
        || run.profile != "standard"
        || run.training_preset != "baseline"
    {
        return Err(Error::Config(
            "unsupported option for rocket-history".into(),
        ));
    }
    let source = Recording::open(
        source.ok_or_else(|| Error::Config("rocket-history requires --source".into()))?,
        ReadLimits::default(),
    )?;
    validate_history_destination(source.root(), &run.path)?;
    let objective = RocketObjective::from_flag(run.rocket_guarded);
    if !source.complete() || !source.issues().is_empty() {
        return Err(Error::Config("source recording is incomplete".into()));
    }
    validate_history_manifest(source.manifest(), objective)?;
    if source.manifest().metadata.contains_key(HISTORY_METADATA) {
        return Err(Error::Config(
            "source already contains rocket-history evaluations".into(),
        ));
    }
    if source
        .entries()
        .iter()
        .enumerate()
        .any(|(i, entry)| entry.sequence != i as u64 || !source.is_retained(i))
    {
        return Err(Error::Config(
            "source has missing or retired records".into(),
        ));
    }
    let mut checked_metrics = 0;
    for (i, entry) in source.entries().iter().enumerate() {
        if entry.kind != RecordKind::Metrics || entry.model_role != "online" {
            continue;
        }
        let chunk = source.load(i)?;
        let Record::Metrics(metrics) = &chunk.record else {
            unreachable!()
        };
        if HISTORY_METRICS.iter().any(|name| {
            metrics.values.contains_key(*name) || metrics.unavailable.contains_key(*name)
        }) {
            return Err(Error::Config(
                "source already contains a rocket-history metric name".into(),
            ));
        }
        checked_metrics += 1;
        if checked_metrics % 200 == 0 {
            eprintln!("rocket history: checked {checked_metrics} existing metric records");
        }
    }
    let snapshots: Vec<_> = source
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            entry.kind == RecordKind::ModelSnapshot && entry.model_role == "online"
        })
        .map(|(i, entry)| (i, entry.revision))
        .collect();
    if snapshots.is_empty() {
        return Err(Error::Config(
            "source contains no retained online snapshots".into(),
        ));
    }
    let first = snapshots.len().saturating_sub(run.steps as usize);
    let snapshots = &snapshots[first..];
    let last_seed = run
        .evaluation_seed
        .checked_add((run.evaluation_episodes - 1) as u64)
        .ok_or_else(|| Error::Config("evaluation seed range overflows".into()))?;
    let seeds: Vec<_> = (run.evaluation_seed..=last_seed).collect();
    let mut manifest = source.manifest().clone();
    manifest.metadata.insert(HISTORY_METADATA.into(), format!(
        "Independent deterministic policy evaluations; episodes={}; seeds={}..={}; snapshots={}/{}; source_run={}",
        seeds.len(), run.evaluation_seed, last_seed, snapshots.len(), first + snapshots.len(), manifest.run_id
    ));
    let mut writer = Recorder::create(&run.path, manifest, StorageConfig::default())?;
    for (i, entry) in source.entries().iter().enumerate() {
        let chunk = source.load(i)?;
        match writer.append(&chunk)? {
            AppendOutcome::Committed { sequence } | AppendOutcome::Buffered { sequence }
                if sequence == entry.sequence => {}
            _ => {
                return Err(Error::State(
                    "recording copy did not preserve sequence".into(),
                ));
            }
        }
        if (i + 1) % 200 == 0 || i + 1 == source.entries().len() {
            eprintln!(
                "rocket history: copied {}/{} source records",
                i + 1,
                source.entries().len()
            );
        }
    }
    let started = std::time::Instant::now();
    for (position, &(index, revision)) in snapshots.iter().enumerate() {
        let snapshot = source.load(index)?;
        let bundle = ModelBundle::load(&snapshot, ModelLimits::default())?;
        let actor = bundle.evaluation_actor("policy")?;
        eprintln!(
            "rocket history: evaluating checkpoint {}/{} (k={revision})",
            position + 1,
            snapshots.len()
        );
        let summary = evaluate_actor(actor.as_ref(), &seeds, objective)?;
        let metrics = Chunk {
            context: snapshot.context.clone(),
            record: Record::Metrics(historical_metrics(&summary)?),
            arrays: BTreeMap::new(),
        };
        if !matches!(
            writer.append(&metrics)?,
            AppendOutcome::Committed { .. } | AppendOutcome::Buffered { .. }
        ) {
            return Err(Error::State(
                "historical evaluation metric was not recorded".into(),
            ));
        }
        eprintln!(
            "rocket history: k={revision}, landed={}/{}, elapsed={:.1}s",
            summary.outcomes[0],
            seeds.len(),
            started.elapsed().as_secs_f64()
        );
    }
    writer.finish()?;
    Ok(())
}

pub(super) fn historical_report(run: &Run) -> Result<()> {
    let recording = Recording::open(&run.path, ReadLimits::default())?;
    if !recording.complete()
        || !recording.issues().is_empty()
        || !recording.manifest().metadata.contains_key(HISTORY_METADATA)
    {
        return Err(Error::Config(
            "recording has no complete rocket-history evaluation".into(),
        ));
    }
    let fields = [
        ("episodes", "evaluation.episodes"),
        ("landing_success_rate", "evaluation.landing_success_rate"),
        (
            "flight_duration_seconds",
            "evaluation.flight_duration_seconds",
        ),
        ("terminal_abs_x_m", "evaluation.terminal_abs_x"),
        ("terminal_abs_vx_m_per_s", "evaluation.terminal_abs_vx"),
        ("terminal_abs_vy_m_per_s", "evaluation.terminal_abs_vy"),
        ("success_fuel_used_kg", "evaluation.success_fuel_used"),
        (
            "success_duration_seconds",
            "evaluation.success_duration_seconds",
        ),
        ("success_abs_x_m", "evaluation.success_abs_x"),
        ("success_abs_vx_m_per_s", "evaluation.success_abs_vx"),
        ("success_abs_vy_m_per_s", "evaluation.success_abs_vy"),
    ];
    println!(
        "revision,{}",
        fields
            .iter()
            .map(|(label, _)| *label)
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut found = 0;
    for (i, entry) in recording.entries().iter().enumerate() {
        if entry.kind != RecordKind::Metrics || entry.model_role != "online" {
            continue;
        }
        let chunk = recording.load(i)?;
        let Record::Metrics(metrics) = &chunk.record else {
            unreachable!()
        };
        if !metrics
            .values
            .contains_key("evaluation.landing_success_rate")
        {
            continue;
        }
        let values: Vec<_> = fields
            .iter()
            .map(|(_, name)| {
                metrics
                    .values
                    .get(*name)
                    .map_or_else(String::new, |value| format!("{value:.9}"))
            })
            .collect();
        println!("{},{}", entry.revision, values.join(","));
        found += 1;
    }
    if found == 0 {
        return Err(Error::Config(
            "recording has no rocket-history metric rows".into(),
        ));
    }
    Ok(())
}

fn print_evaluation(label: &str, seeds: &[u64], result: &BenchmarkSummary) -> Result<()> {
    let n = seeds.len() as f64;
    let successes = result.outcomes[0] as f64;
    let mean = result.return_total / n;
    let variance = (result.return_squared_total / n - mean * mean).max(0.);
    let conditional = |sum: f64| {
        if successes > 0. {
            format!("{:.6}", sum / successes)
        } else {
            "unavailable".into()
        }
    };
    println!(
        "benchmark,episodes,landed,ground_crash,escape,attitude,horizon,mean_discounted_return,return_std,mean_duration_s,terminal_mean_abs_x,terminal_mean_abs_vx,terminal_mean_abs_vy,terminal_mean_fuel,success_mean_fuel_used,success_mean_duration_s,success_mean_abs_x,success_mean_abs_vx,success_mean_abs_vy"
    );
    println!(
        "{label},{},{},{},{},{},{},{mean:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}",
        seeds.len(),
        result.outcomes[0],
        result.outcomes[1],
        result.outcomes[2],
        result.outcomes[3],
        result.outcomes[4],
        variance.sqrt(),
        result.duration_total / n,
        result.terminal_abs_x_total / n,
        result.terminal_abs_vx_total / n,
        result.terminal_abs_vy_total / n,
        result.terminal_fuel_total / n,
        conditional(result.success_fuel_total),
        conditional(result.success_duration_total),
        conditional(result.success_abs_x_total),
        conditional(result.success_abs_vx_total),
        conditional(result.success_abs_vy_total)
    );
    Ok(())
}

fn context(action_space: ActionSpace, objective: RocketObjective) -> ReferenceContext {
    ReferenceContext {
        environment: objective.id().into(),
        observation_dim: 8,
        typed_observation_space: None,
        action_space,
        conventions: ValueConventions {
            dynamics: "Fixed nominal mass, circular envelope, and two constant-acceleration substeps with independent decision noise".into(),
            observation: "Scaled x, height, horizontal and vertical velocity, clockwise attitude and rate, fuel, and remaining decision fraction".into(),
            action: "Normalized main throttle and signed attitude command".into(),
            reward_units: "reward units".into(),
            reward_convention: match objective { RocketObjective::Historical => "Plain potential difference minus elapsed-time and powered effort charges, plus terminal reward; not discount-invariant shaping", RocketObjective::Guarded => "Discount-consistent potential shaping with zero terminal potential, minus elapsed-time and powered effort charges, plus terminal reward" }.into(),
            discount: f64::from(objective.discount()),
            horizon: format!("Up to {HORIZON} decisions"),
            terminal_treatment: "Landing, crash, boundary crossing or finite-horizon expiry ends the episode".into(),
            action_constraints: "Throttle in [0, 1] and attitude command in [-1, 1]. Both require fuel to produce acceleration.".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: HORIZON as u32 },
        environment_config: BTreeMap::from([
            ("model".into(), ConfigValue::Text(objective.id().into())),
            ("horizon".into(), ConfigValue::Integer(HORIZON as i64)),
            ("decision_seconds".into(), ConfigValue::Real(DT)),
            ("substeps".into(), ConfigValue::Integer(2)),
            ("fuel_capacity".into(), ConfigValue::Real(FUEL_CAPACITY)),
            ("time_cost_per_second".into(), ConfigValue::Real(objective.time_cost())),
        ]),
        markov_observation: false,
        decision_time_coordinate: Some(7),
    }
}

fn descriptor(objective: RocketObjective) -> EnvironmentDescriptor {
    let mut observations: Vec<_> = [
        ("x", "Horizontal position", "m"),
        ("y", "Height", "m"),
        ("vx", "Horizontal velocity", "m/s"),
        ("vy", "Vertical velocity", "m/s"),
        ("theta", "Clockwise attitude from upright", "rad"),
        ("omega", "Attitude rate", "rad/s"),
        ("fuel", "Fuel remaining", "kg"),
    ]
    .into_iter()
    .zip(SCALES)
    .map(|((id, label, unit), scale)| ObservationField {
        id: id.into(),
        label: label.into(),
        unit: Some("scaled".into()),
        description: Some(format!("Physical value divided by {scale:.4} {unit}")),
        kind: ObservationType::Numerical,
        range: None,
        coordinate: true,
    })
    .collect();
    observations.push(ObservationField {
        id: "remaining_fraction".into(),
        label: "Remaining decision fraction".into(),
        unit: None,
        description: None,
        kind: ObservationType::Numerical,
        range: Some([0., 1.]),
        coordinate: true,
    });
    EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations,
        actions: ActionDescriptor::Continuous {
            components: vec![
                ActionComponent { label: "Main throttle".into(), unit: None, meaning: Some("Main thrust along the body axis".into()) },
                ActionComponent { label: "Attitude command".into(), unit: None, meaning: Some("Signed command to an ideal torque couple".into()) },
            ],
            preferred_component: Some(0),
        },
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel { label: "Learner revision".into(), unit: None },
            decision_step: QuantityLabel { label: "Decision".into(), unit: None },
            simulation: Some(QuantityLabel { label: "Flight time".into(), unit: Some("s".into()) }),
        },
        reward: RewardDescriptor {
            label: "Landing and flight reward".into(), unit: Some("reward units".into()),
            description: Some(match objective { RocketObjective::Historical => "Plain potential difference, elapsed-time and powered-effort costs, and terminal outcome reward.", RocketObjective::Guarded => "Discount-consistent potential shaping with zero terminal potential, elapsed-time and powered-effort costs, and terminal outcome reward." }.into()),
            components: Vec::new(),
        },
        frames: [
            ("horizontal_position", "Horizontal position", "m"),
            ("altitude", "Altitude", "m"),
            ("horizontal_velocity", "Horizontal velocity", "m/s"),
            ("vertical_velocity", "Vertical velocity", "m/s"),
            ("attitude", "Attitude", "rad"),
            ("angular_velocity", "Angular velocity", "rad/s"),
            ("fuel", "Fuel remaining", "kg"),
            ("applied_throttle", "Mean applied throttle over prior decision", "fraction"),
        ].into_iter().map(|(name, label, unit)| FrameChannel {
            name: name.into(), label: label.into(), width: 1, unit: Some(unit.into()),
            visibility: FrameVisibility::DiagnosticOnly, representation: Some(FrameRepresentation::Scalar),
        }).collect(),
        preferences: VisualizationPreferences {
            coordinates: vec!["x".into(), "y".into()],
            trajectories: ["altitude", "vertical_velocity", "attitude", "fuel", "applied_throttle"].into_iter().map(|channel| FrameReference { channel: channel.into(), index: 0 }).collect(),
            ..Default::default()
        },
    }
}

fn historical_ppo_policy(space: ActionSpace, rng: &mut ChaCha8Rng) -> Result<Policy> {
    let mut policy = Policy::new(8, space, &[64, 64], rng)?;
    let last = policy
        .network
        .layers
        .last_mut()
        .expect("policy output layer");
    last.weight = Param::from_tensor((last.weight.val() * 0.01).detach());
    policy.log_std = Param::from_tensor(rl::nn::vector(&[0.35f32.ln(); 2]));
    Ok(policy)
}

pub(super) fn run(run: &Run) -> Result<()> {
    let objective = RocketObjective::from_flag(run.rocket_guarded);
    let environment = PlanarRocket::with_objective(run.seed, objective);
    let space = environment.action_space().clone();
    let paired = context(space.clone(), objective);
    let mut config = run.config(HORIZON)?;
    config.evaluation.mode = ActMode::Deterministic;
    if run.profile == "detailed" && run.algorithm == "planar-ppo" {
        let each_rollout = Some(rl::diagnostics::schedule::Interval { first: 1, every: 1 });
        config.schedule.probes = each_rollout;
        config.schedule.representations = each_rollout;
        config.schedule.evaluation = each_rollout;
        config.schedule.snapshots = each_rollout;
        config.snapshot_retention = 64;
    }
    let mut fixed_states: Vec<_> = [-2., -1., 0., 1., 2.]
        .into_iter()
        .flat_map(|x| {
            [1., 3., 5., 7.].into_iter().map(move |y| {
                State {
                    x,
                    y,
                    vy: -1.,
                    fuel: FUEL_CAPACITY,
                    ..State::default()
                }
                .observation(0)
            })
        })
        .collect();
    if run.profile == "detailed" {
        for decision in [0, 120] {
            for x in [-2., 0., 2.] {
                for y in [1., 4., 7.] {
                    for vy in [-2., -0.5] {
                        for theta in [-0.15, 0., 0.15] {
                            fixed_states.push(
                                State {
                                    x,
                                    y,
                                    vy,
                                    theta,
                                    fuel: if decision == 0 { FUEL_CAPACITY } else { 2.5 },
                                    ..State::default()
                                }
                                .observation(decision),
                            );
                        }
                    }
                }
            }
        }
    }
    let probes = ProbeSet::new(
        "position-height-slice",
        ProbePopulation::FixedReference,
        "Fixed physical candidates spanning position, altitude, descent, attitude, fuel and remaining time; not a visitation distribution",
        fixed_states,
        None,
        space.clone(),
    )?;
    let mut builder = run
        .builder(&environment, config)
        .discount(f64::from(objective.discount()))
        .environment_descriptor(descriptor(objective))
        .reference_context(paired.clone())
        .metadata(
            "evaluation.scenario_protocol",
            "Independent seeded position, velocity and attitude draws and Gaussian disturbances",
        )
        .probe_set(probes);
    builder = if run.render {
        builder.evaluation_factory_with_frames_and_render(
            move |seed| Ok(RenderedRocket::new(seed, objective)),
            RenderCaptureConfig::default(),
        )
    } else {
        builder.evaluation_factory_with_frames(move |seed| {
            Ok(PlanarRocket::with_objective(seed, objective))
        })
    };
    if run.reference {
        builder = builder.reference(BenchmarkActor::new(
            ReferenceSpec {
                id: "landing-feedback".into(), context: paired,
                quality: ReferenceQuality {
                    kind: ReferenceKind::Heuristic,
                    method: "Height-dependent descent target with horizontal and attitude feedback".into(),
                    assumptions: Vec::new(),
                },
                target: ReferenceTarget::BenchmarkPolicy, lifetime: ReferenceLifetime::Fixed,
                information: ReferenceInformation::LearnerObservation, uncertainty: None,
            }, LandingController(space.clone()),
        )?.with_evaluation_mode(ActMode::Deterministic)).reference_trajectories(1);
    }
    if run.algorithm == "planar-ppo" {
        let mut rng = ChaCha8Rng::seed_from_u64(run.seed ^ 0x494e_4954);
        let policy = historical_ppo_policy(space.clone(), &mut rng)?;
        let models = ActorCritic {
            policy,
            value: Value::new(8, &[64, 64], &mut rng)?,
        };
        let learner = Ppo::new(
            models,
            PpoConfig {
                minibatch_size: 256,
                ..Default::default()
            },
            run.seed ^ 0x004c_4541_524e,
        )?;
        let mut trainer = OnPolicyTrainer::new(
            OnPolicyCollector::new(
                SyncVectorEnv::new(vec![environment; 16])?,
                run.seed ^ 0x434f_4c4c,
            )?,
            OnPolicyConfig {
                rollout_steps: HORIZON,
                gamma: objective.discount(),
                gae_lambda: Some(0.95),
                normalize_advantages: true,
                episode_guard: None,
            },
        )?;
        let budget = Budget::vector_steps(run.steps);
        let mut next_report = 5.;
        let mut progress = |p: &rl::train::trainer::Progress| {
            if p.elapsed_seconds >= next_report {
                next_report = p.elapsed_seconds + 5.;
                let eta = p.elapsed_seconds
                    * run.steps.saturating_sub(p.counters.vector_steps) as f64
                    / p.counters.vector_steps.max(1) as f64;
                eprintln!(
                    "rocket PPO: {}/{} vector steps, {} transitions, {} updates, {:.1}s elapsed, {:.1}s estimated remaining; collection {:.1}s, learning {:.1}s",
                    p.counters.vector_steps,
                    run.steps,
                    p.counters.env_steps,
                    p.counters.updates,
                    p.elapsed_seconds,
                    eta,
                    p.collection_seconds,
                    p.learning_seconds
                );
            }
        };
        return run.train(learner, builder, |learner, diagnostics| match diagnostics {
            Some(session) => trainer.train_observed(learner, budget, Some(&mut progress), session),
            None => trainer.train(learner, budget, Some(&mut progress)),
        });
    }
    let mut rng = ChaCha8Rng::seed_from_u64(run.stream_seed(1));
    let width = if run.training_preset == "balanced" {
        32
    } else {
        16
    };
    let policy = Deterministic::new(8, space, &[width, width], &mut rng)?;
    let mut critic = || QValue::new(8, 2, 1, &[width, width], &mut rng);
    let training = if run.training_preset == "balanced" {
        OffPolicyConfig {
            replay_capacity: 20_000,
            batch_size: 32,
            warmup_transitions: 512,
            gamma: objective.discount(),
            updates_per_transition: 0.5,
            exploration: rl::train::schedule::Schedule::Constant(0.2),
            ..Default::default()
        }
    } else {
        OffPolicyConfig {
            gamma: objective.discount(),
            ..off_policy_config()
        }
    };
    run.train_off_policy(
        Td3::new(
            policy,
            critic()?,
            critic()?,
            Default::default(),
            run.stream_seed(2),
        )?,
        builder,
        environment,
        training,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_evaluation_metrics_use_all_episodes_and_explicit_success_denominators() {
        let summary = BenchmarkSummary {
            outcomes: [2, 2, 0, 0, 0],
            duration_total: 10.,
            success_fuel_total: 4.,
            success_duration_total: 6.,
            ..Default::default()
        };
        let metrics = historical_metrics(&summary).unwrap();
        assert_eq!(metrics.values["evaluation.landing_success_rate"], 0.5);
        assert_eq!(metrics.values["evaluation.flight_duration_seconds"], 2.5);
        assert_eq!(metrics.values["evaluation.success_fuel_used"], 2.);
        assert_eq!(metrics.values["evaluation.success_duration_seconds"], 3.);
        let failed = historical_metrics(&BenchmarkSummary {
            outcomes: [0, 1, 0, 0, 0],
            ..Default::default()
        })
        .unwrap();
        assert!(!failed.values.contains_key("evaluation.success_fuel_used"));
        assert!(
            failed
                .unavailable
                .contains_key("evaluation.success_fuel_used")
        );
        assert!(historical_metrics(&BenchmarkSummary::default()).is_err());
    }

    #[test]
    fn benchmark_evaluation_is_seeded_and_accounts_for_every_terminal_outcome() -> Result<()> {
        let seeds: Vec<_> = (50_000..50_008).collect();
        let first = evaluate_benchmark(&seeds)?;
        let second = evaluate_benchmark(&seeds)?;
        assert_eq!(first, second);
        assert_eq!(first.outcomes.iter().sum::<usize>(), seeds.len());
        assert!(first.duration_total > 0.);
        assert!(first.return_total.is_finite());
        assert!(first.success_fuel_total >= 0.);
        Ok(())
    }

    #[test]
    fn independent_actor_evaluation_matches_the_benchmark_controller() -> Result<()> {
        let seeds = [200_010, 200_011, 200_012];
        let space = PlanarRocket::new(0).action_space().clone();
        assert_eq!(
            evaluate_actor(
                &LandingController(space),
                &seeds,
                RocketObjective::Historical
            )?,
            evaluate_benchmark(&seeds)?
        );
        Ok(())
    }

    #[test]
    fn discounted_shaping_telescopes_even_when_the_episode_terminates() {
        let first = shaping_reward(-100., -90., false);
        let terminal = shaping_reward(-90., -80., true);
        assert!((first + f64::from(ROCKET_GAMMA) * terminal - 100.).abs() < 1e-10);
        assert!(f64::from(ROCKET_GAMMA).powi(200) > 0.3);
    }

    #[test]
    fn delaying_an_inevitable_identical_crash_does_not_improve_base_return() {
        let gamma = f64::from(ROCKET_GAMMA);
        let failure_return = |steps: usize| {
            -(0..steps)
                .map(|step| gamma.powi(step as i32) * TIME_COST_RATE * DT)
                .sum::<f64>()
                - 100. * gamma.powi((steps - 1) as i32)
        };
        assert!(failure_return(20) > failure_return(200));
    }

    #[test]
    fn feedback_controller_lands_nominal_validation_scenario_at_safe_speed() -> Result<()> {
        let result = evaluate_benchmark(&[200_000])?;
        assert_eq!(result.outcomes, [1, 0, 0, 0, 0]);
        assert!(result.success_abs_vy_total <= 1.);
        Ok(())
    }

    #[test]
    fn rocket_frame_reports_physical_state_and_actual_elapsed_time() -> Result<()> {
        use rl::diagnostics::evaluation::FrameSource;
        let mut rocket = RenderedRocket::new(19, RocketObjective::Historical);
        rocket.reset(Some(19))?;
        rocket.step(&Action::Continuous(vec![0.6, 0.]))?;
        let frame = rocket.diagnostic_frame()?;
        assert_eq!(frame.simulation_time, Some(rocket.inner.time));
        assert_eq!(frame.channels["altitude"], vec![rocket.inner.state.y]);
        assert_eq!(
            frame.channels["vertical_velocity"],
            vec![rocket.inner.state.vy]
        );
        assert_eq!(frame.channels["fuel"], vec![rocket.inner.state.fuel]);
        assert!((frame.channels["applied_throttle"][0] - 0.6).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn historical_rocket_reward_remains_the_original_potential_difference() -> Result<()> {
        let mut environment = PlanarRocket::new(7);
        environment.reset(Some(7))?;
        let before = environment.state.potential();
        let step = environment.step(&Action::Continuous(vec![0., 0.]))?;
        let expected = environment.state.potential() - before - 0.3 * environment.time
            + environment.event.reward();
        assert!((f64::from(step.reward.total()) - expected).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn historical_ppo_protocol_initializes_near_hover_with_bounded_controls() -> Result<()> {
        let environment = PlanarRocket::new(7);
        let mut rng = ChaCha8Rng::seed_from_u64(7 ^ 0x494e_4954);
        let policy = historical_ppo_policy(environment.action_space().clone(), &mut rng)?;
        let actor = rl::nn::policy::PolicyActor::new(policy)?;
        let observations = [Observation::new(vec![0.; 8])];
        let mut actions = Vec::new();
        actor.act_into(
            ObservationBatch::all(&observations),
            ActMode::Deterministic,
            &mut ChaCha8Rng::seed_from_u64(1),
            &mut actions,
        )?;
        let Action::Continuous(values) = &actions[0] else {
            panic!("expected continuous controls")
        };
        assert!((values[0] - 0.5).abs() < 0.05);
        assert!(values[1].abs() < 0.05);
        Ok(())
    }

    #[test]
    fn rocket_scene_shows_distinct_vehicle_parts_and_only_applied_thrust() -> Result<()> {
        let mut rocket = RenderedRocket::new(7, RocketObjective::Historical);
        rocket.reset(Some(7))?;
        rocket.inner.throttle = 0.;
        let idle = rocket.render_scene()?;
        assert!(
            idle.items.iter().any(|item| matches!(
                &item.primitive,
                RenderPrimitive::Circle { center, radius, .. }
                    if *center == [rocket.inner.state.x, rocket.inner.state.y] && *radius == RADIUS
            )),
            "the physical collision circle must remain at the actual state"
        );
        assert!(idle.items.iter().any(|item| matches!(
            &item.primitive,
            RenderPrimitive::Text { text, .. } if text.contains("Vehicle ×4") && text.contains("physical radius")
        )), "the display enlargement must be disclosed in the scene");
        let polygons = idle
            .items
            .iter()
            .filter(|item| matches!(item.primitive, RenderPrimitive::Polygon { .. }))
            .count();
        assert!(polygons >= 3, "the vehicle needs a hull, nose, and fins");
        rocket.inner.throttle = 0.5;
        let firing = rocket.render_scene()?;
        assert!(
            firing.items.len() > idle.items.len(),
            "applied thrust must add a visible plume"
        );
        rocket.inner.event = Event::Landed;
        assert_eq!(
            rocket.render_scene()?.items.len(),
            idle.items.len(),
            "landed vehicles have no plume"
        );
        Ok(())
    }

    #[test]
    fn historical_ppo_training_records_physical_frames_and_rendered_scenes() -> Result<()> {
        use rl::diagnostics::{
            schema::RecordKind,
            storage::{ReadLimits, Recording},
        };
        let directory = tempfile::tempdir()?;
        let run = Run {
            algorithm: "planar-ppo".into(),
            path: directory.path().join("rocket"),
            profile: "standard".into(),
            reference: false,
            live: false,
            wide_lqg: false,
            render: true,
            replay_population: false,
            seed: 7,
            steps: 16,
            evaluation_episodes: 0,
            evaluation_seed: 200_000,
            training_preset: "baseline".into(),
            rocket_guarded: false,
            rich_conditional: false,
        };
        super::run(&run)?;
        let recording = Recording::open(&run.path, ReadLimits::default())?;
        assert_eq!(recording.manifest().environment, "planar_rocket");
        assert_eq!(recording.manifest().discount, f64::from(0.99f32));
        assert!(
            validate_history_manifest(recording.manifest(), RocketObjective::Historical).is_ok()
        );
        let mut wrong_manifest = recording.manifest().clone();
        let mut wrong_context =
            rl::diagnostics::reference::manifest_context(&wrong_manifest)?.unwrap();
        wrong_context
            .environment_config
            .insert("decision_seconds".into(), ConfigValue::Real(0.1));
        wrong_manifest
            .metadata
            .remove(rl::diagnostics::reference::REFERENCE_CONTEXT_METADATA_KEY);
        rl::diagnostics::reference::attach_manifest_context(&mut wrong_manifest, &wrong_context)?;
        assert!(validate_history_manifest(&wrong_manifest, RocketObjective::Historical).is_err());
        assert!(
            recording
                .entries()
                .iter()
                .any(|entry| entry.kind == RecordKind::ModelSnapshot && entry.revision > 0)
        );
        let index = recording
            .entries()
            .iter()
            .position(|entry| entry.kind == RecordKind::Trajectory && entry.model_role == "online")
            .unwrap();
        let trajectory = recording.load(index)?;
        assert!(trajectory.arrays.contains_key("frames/altitude"));
        assert!(trajectory.arrays.contains_key("simulation_times"));
        assert!(
            matches!(&trajectory.record, rl::diagnostics::schema::Record::Trajectory(record) if record.render.is_some())
        );
        let source_path = run.path.clone();
        assert!(validate_history_destination(&source_path, &source_path.join("nested")).is_err());
        assert!(
            validate_history_destination(&source_path, &directory.path().join("sibling")).is_ok()
        );
        let history = Run {
            path: directory.path().join("rocket-history"),
            render: false,
            steps: 2,
            evaluation_episodes: 4,
            evaluation_seed: 900_000,
            ..run
        };
        historical_evaluation(&history, Some(&source_path))?;
        let augmented = Recording::open(&history.path, ReadLimits::default())?;
        assert!(augmented.complete());
        let selected = recording
            .entries()
            .iter()
            .filter(|entry| entry.kind == RecordKind::ModelSnapshot && entry.model_role == "online")
            .count()
            .min(2);
        assert_eq!(
            augmented.entries().len(),
            recording.entries().len() + selected
        );
        for (before, after) in recording.entries().iter().zip(augmented.entries()) {
            assert_eq!(
                (before.sequence, before.kind, before.revision),
                (after.sequence, after.kind, after.revision)
            );
            assert_eq!(before.model_role, after.model_role);
            assert_eq!(before.checksum, after.checksum);
        }
        for i in recording.entries().len()..augmented.entries().len() {
            let chunk = augmented.load(i)?;
            let Record::Metrics(metrics) = chunk.record else {
                panic!("expected evaluation metrics")
            };
            assert_eq!(metrics.values["evaluation.episodes"], 4.);
            assert!((0. ..=1.).contains(&metrics.values["evaluation.landing_success_rate"]));
        }
        historical_report(&history)?;
        let derived_path = history.path.clone();
        let duplicate = Run {
            path: directory.path().join("duplicate-history"),
            ..history
        };
        assert!(historical_evaluation(&duplicate, Some(&derived_path)).is_err());
        assert!(!duplicate.path.exists());
        let contaminated_path = directory.path().join("contaminated");
        let mut contaminated = Recorder::create(
            &contaminated_path,
            recording.manifest().clone(),
            StorageConfig::default(),
        )?;
        for i in 0..recording.entries().len() {
            contaminated.append(&recording.load(i)?)?;
        }
        contaminated.append(&Chunk {
            context: recording.load(0)?.context,
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::from([("evaluation.landing_success_rate".into(), 0.5)]),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        })?;
        contaminated.finish()?;
        let collision = Run {
            path: directory.path().join("collision-history"),
            ..duplicate
        };
        assert!(historical_evaluation(&collision, Some(&contaminated_path)).is_err());
        assert!(!collision.path.exists());
        Ok(())
    }

    #[test]
    fn free_fall_resolves_first_ground_contact_and_ends_without_truncating() -> Result<()> {
        let mut environment = PlanarRocket::new(1);
        environment.reset(Some(1))?;
        environment.state = State {
            y: RADIUS + 0.005,
            vy: -0.25,
            fuel: 0.,
            ..State::default()
        };
        let action = Action::Continuous(vec![0., 0.]);
        let step = environment.step(&action)?;
        assert!(step.terminated);
        assert!(!step.truncated);
        assert_eq!(environment.event, Event::Landed);
        assert_eq!(environment.state.y, RADIUS);
        assert!((0.01..0.02).contains(&environment.time));
        assert!(f64::from(step.reward.total()) > 99.);
        assert!(environment.step(&action).is_err());
        Ok(())
    }

    #[test]
    fn two_actuators_control_distinct_motion_and_observation_components() -> Result<()> {
        let mut environment = PlanarRocket::new(7);
        environment.reset(Some(7))?;
        let mut throttle = environment.clone();
        let mut torque = environment.clone();
        let neutral = environment.step(&Action::Continuous(vec![0., 0.]))?;
        let thrust = throttle.step(&Action::Continuous(vec![1., 0.]))?;
        let attitude = torque.step(&Action::Continuous(vec![0., 1.]))?;
        assert!(!neutral.ended() && !thrust.ended() && !attitude.ended());
        let neutral = neutral.observation.dense_values();
        let thrust = thrust.observation.dense_values();
        let attitude = attitude.observation.dense_values();
        assert!(thrust[3] > neutral[3]);
        assert_eq!(attitude[3], neutral[3]);
        assert!(attitude[5] > neutral[5]);
        assert_eq!(thrust[5], neutral[5]);
        assert!(thrust[6] < attitude[6] && attitude[6] < neutral[6]);
        Ok(())
    }

    #[test]
    fn fuel_exhaustion_splits_a_substep_and_stops_thrust() -> Result<()> {
        let mut coasting = PlanarRocket::new(7);
        coasting.reset(Some(7))?;
        coasting.state = State {
            y: 5.,
            ..State::default()
        };
        let mut powered = coasting.clone();
        powered.state.fuel = 0.01;
        coasting.step(&Action::Continuous(vec![0., 0.]))?;
        powered.step(&Action::Continuous(vec![1., 0.]))?;
        assert_eq!(powered.state.fuel, 0.);
        // The upright craft gets 20 m/s² of additional acceleration for 0.01 s.
        assert!((powered.state.vy - coasting.state.vy - 0.2).abs() < 1e-12);
        assert!((powered.time - 0.05).abs() < 1e-12);
        assert!((powered.throttle - 0.2).abs() < 1e-12);
        Ok(())
    }
}
