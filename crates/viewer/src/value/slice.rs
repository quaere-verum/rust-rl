use super::ensure;
use crate::policy::slice::{GridEligibility, SliceConfig, SliceMode};
use rl::{
    Result,
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::visualization::EnvironmentSemantics,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueCoordinate {
    Observation(usize),
    Action(usize),
    DiscreteAction,
}
#[derive(Debug, Clone, PartialEq)]
pub enum ActionConditioning {
    Policy,
    Fixed(Action),
}
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSlice {
    pub state: SliceConfig,
    pub mode: SliceMode,
    pub x: ValueCoordinate,
    pub y: Option<ValueCoordinate>,
    pub resolution: [usize; 2],
    pub action_ranges: Vec<[f32; 2]>,
    pub conditioning: ActionConditioning,
    space: ActionSpace,
}
impl ValueSlice {
    /// Replace only unconfigured fallback ranges with the extent of genuine
    /// recorded inputs (at most 8192 rows / one million scalars). Once
    /// established, ranges stay fixed across checkpoints.
    pub(crate) fn adopt_recorded_ranges(
        &mut self,
        observations: impl IntoIterator<Item = Result<Observation>>,
    ) -> Result<bool> {
        use crate::policy::slice::RangeSource;
        if !self
            .state
            .range_sources
            .contains(&RangeSource::ModelInputDefault)
        {
            return Ok(false);
        }
        let mut bounds = vec![[f32::INFINITY, f32::NEG_INFINITY]; self.state.fixed.len()];
        for (row, observation) in observations.into_iter().enumerate() {
            if row >= 8192 || (row + 1).saturating_mul(bounds.len()) > 1024 * 1024 {
                break;
            }
            let observation = observation?;
            ensure(
                observation.dense_values().len() == bounds.len(),
                "recorded range width differs",
            )?;
            for (range, &value) in bounds.iter_mut().zip(observation.dense_values()) {
                ensure(value.is_finite(), "non-finite recorded range")?;
                range[0] = range[0].min(value);
                range[1] = range[1].max(value);
            }
        }
        let mut changed = false;
        for (i, range) in bounds.into_iter().enumerate() {
            if self.state.range_sources[i] == RangeSource::ModelInputDefault && range[0] < range[1]
            {
                self.state.ranges[i] = range;
                self.state.range_sources[i] = RangeSource::ReferencePopulation;
                changed = true;
            }
        }
        Ok(changed)
    }
    pub fn new(
        environment: EnvironmentSemantics<'_>,
        anchor: &Observation,
        population: &[Observation],
    ) -> Result<Self> {
        anchor.validate(
            environment.observation_dim(),
            environment.action_space(),
            false,
        )?;
        let state = SliceConfig::new(environment, anchor, population)?;
        let action_ranges = match environment.action_space() {
            ActionSpace::Box { low, high } => low.iter().zip(high).map(|(&l, &h)| [l, h]).collect(),
            ActionSpace::Unbounded { dimensions } => vec![[-1., 1.]; *dimensions],
            ActionSpace::Discrete { .. } => vec![],
            ActionSpace::Conditional(_) => {
                return Err(rl::Error::Config(
                    "value slices require a numeric action space".into(),
                ));
            }
        };
        Ok(Self {
            mode: state.mode,
            x: ValueCoordinate::Observation(state.x),
            y: state.y.map(ValueCoordinate::Observation),
            resolution: state.resolution,
            state,
            action_ranges,
            conditioning: ActionConditioning::Policy,
            space: environment.action_space().clone(),
        })
    }
    pub fn state_only(&self) -> bool {
        self.mode == SliceMode::SamplesOnly
            || (matches!(self.x, ValueCoordinate::Observation(_))
                && self
                    .y
                    .is_none_or(|a| matches!(a, ValueCoordinate::Observation(_))))
    }
    pub(crate) fn has_discrete_axis(&self) -> bool {
        self.x == ValueCoordinate::DiscreteAction || self.y == Some(ValueCoordinate::DiscreteAction)
    }

    pub(crate) fn default_first_action(&self) -> Option<Action> {
        match &self.space {
            ActionSpace::Discrete { count } => (0..*count)
                .find(|&i| self.state.anchor_mask().is_none_or(|mask| mask[i]))
                .map(Action::Discrete),
            ActionSpace::Box { low, high } => Some(Action::Continuous(
                low.iter()
                    .zip(high)
                    .map(|(&l, &h)| 0f32.clamp(l, h))
                    .collect(),
            )),
            ActionSpace::Unbounded { dimensions } => {
                Some(Action::Continuous(vec![0.; *dimensions]))
            }
            ActionSpace::Conditional(_) => None,
        }
    }

    pub fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    pub fn validate(&self, environment: EnvironmentSemantics<'_>) -> Result<()> {
        self.state.validate(environment)?;
        ensure(
            &self.space == environment.action_space(),
            "Value action encoding differs",
        )?;
        self.structural()
    }
    fn structural(&self) -> Result<()> {
        if self.mode != SliceMode::SamplesOnly {
            ensure(
                self.y != Some(self.x)
                    && (self.mode == SliceMode::OneDimensional && self.y.is_none()
                        || self.mode == SliceMode::TwoDimensional && self.y.is_some()),
                "invalid Value axes",
            )?;
            for axis in std::iter::once(self.x).chain(self.y) {
                ensure(
                    match axis {
                        ValueCoordinate::Observation(i) => self.state.numeric(i),
                        ValueCoordinate::Action(i) => {
                            !matches!(self.space, ActionSpace::Discrete { .. })
                                && i < self.space.dimensions()
                        }
                        ValueCoordinate::DiscreteAction => {
                            matches!(self.space, ActionSpace::Discrete { .. })
                        }
                    },
                    "invalid Value coordinate",
                )?;
            }
        }
        let width = if matches!(self.space, ActionSpace::Discrete { .. }) {
            0
        } else {
            self.space.dimensions()
        };
        ensure(
            self.action_ranges.len() == width,
            "invalid Value action ranges",
        )?;
        for (i, &[l, h]) in self.action_ranges.iter().enumerate() {
            ensure(
                l.is_finite() && h.is_finite() && l < h,
                "invalid Value action range",
            )?;
            if let ActionSpace::Box { low, high } = &self.space {
                ensure(
                    l >= low[i] && h <= high[i],
                    "Value action range violates bounds",
                )?;
            }
        }
        if let ActionConditioning::Fixed(a) = &self.conditioning {
            self.space.check(a, self.state.anchor_mask())?;
        }
        self.anchor()
            .validate(self.state.fixed.len(), &self.space, false)?;
        self.len()?;
        Ok(())
    }
    pub fn axis_len(&self, axis: ValueCoordinate, index: usize) -> Result<usize> {
        ensure(
            self.mode != SliceMode::SamplesOnly,
            "sample population has no physical axes",
        )?;
        if axis == ValueCoordinate::DiscreteAction {
            let ActionSpace::Discrete { count } = self.space else {
                return Err(rl::Error::Config(
                    "discrete axis requires discrete actions".into(),
                ));
            };
            let n = self
                .state
                .anchor_mask()
                .map_or(count, |m| m.iter().filter(|&&b| b).count());
            ensure(
                n > 0 && n <= 4096,
                "no legal categories or too many actions",
            )?;
            Ok(n)
        } else {
            let n = self.resolution[index];
            let max = if self.mode == SliceMode::OneDimensional {
                4096
            } else {
                256
            };
            ensure((2..=max).contains(&n), "Value resolution exceeds limits")?;
            Ok(n)
        }
    }
    pub fn len(&self) -> Result<usize> {
        if self.mode == SliceMode::SamplesOnly {
            return Ok(0);
        }
        let nx = self.axis_len(self.x, 0)?;
        let ny = self
            .y
            .map(|a| self.axis_len(a, 1))
            .transpose()?
            .unwrap_or(1);
        let n = nx
            .checked_mul(ny)
            .ok_or_else(|| rl::Error::Shape("Value grid overflow".into()))?;
        ensure(n <= 65536, "Value grid exceeds cell admission")?;
        Ok(n)
    }
    pub fn is_empty(&self) -> bool {
        self.mode == SliceMode::SamplesOnly
    }
    pub fn anchor(&self) -> Observation {
        Observation::dense(
            self.state.fixed.clone(),
            self.state.anchor_mask().map(<[bool]>::to_vec),
        )
    }
    pub fn coordinate_at(&self, axis: ValueCoordinate, index: usize, n: usize) -> Result<f32> {
        ensure(
            self.mode != SliceMode::SamplesOnly,
            "sample population has no physical coordinates",
        )?;
        if axis == ValueCoordinate::DiscreteAction {
            let ActionSpace::Discrete { count } = self.space else {
                return Err(rl::Error::Config("not discrete".into()));
            };
            return (0..count)
                .filter(|&i| self.state.anchor_mask().is_none_or(|m| m[i]))
                .nth(index)
                .map(|a| a as f32)
                .ok_or_else(|| rl::Error::Shape("category out of bounds".into()));
        }
        let [l, h] = match axis {
            ValueCoordinate::Observation(i) => *self
                .state
                .ranges
                .get(i)
                .ok_or_else(|| rl::Error::Shape("state axis".into()))?,
            ValueCoordinate::Action(i) => *self
                .action_ranges
                .get(i)
                .ok_or_else(|| rl::Error::Shape("action axis".into()))?,
            _ => unreachable!(),
        };
        Ok((f64::from(l) + (f64::from(h) - f64::from(l)) * index as f64 / (n - 1) as f64) as f32)
    }
    pub fn position(&self, index: usize) -> Result<[f32; 2]> {
        ensure(index < self.len()?, "Value row outside grid")?;
        let nx = self.axis_len(self.x, 0)?;
        Ok([
            self.coordinate_at(self.x, index % nx, nx)?,
            self.y
                .map(|a| self.coordinate_at(a, index / nx, self.axis_len(a, 1)?))
                .transpose()?
                .unwrap_or(0.),
        ])
    }
    pub fn model_eligible(&self) -> bool {
        self.mode != SliceMode::SamplesOnly
            && self.state.grid_eligibility() != GridEligibility::HistoryBound
            && (self.state.grid_eligibility() == GridEligibility::ContinuousFeedforward
                || !std::iter::once(self.x)
                    .chain(self.y)
                    .any(|a| matches!(a, ValueCoordinate::Observation(_))))
    }
    pub fn observation_at(&self, index: usize) -> Result<Observation> {
        self.structural()?;
        ensure(
            self.state.grid_eligibility() != GridEligibility::HistoryBound,
            "history adapter unavailable",
        )?;
        let state_axes = std::iter::once(self.x)
            .chain(self.y)
            .any(|a| matches!(a, ValueCoordinate::Observation(_)));
        ensure(
            !state_axes || self.state.grid_eligibility() == GridEligibility::ContinuousFeedforward,
            "synthetic state generation unavailable for masked or discrete inputs",
        )?;
        let pos = self.position(index)?;
        // Reuse Policy reconstruction for observation-only grids, including arbitrary axes.
        if let ValueCoordinate::Observation(x) = self.x
            && self
                .y
                .is_none_or(|a| matches!(a, ValueCoordinate::Observation(_)))
        {
            let mut state = self.state.clone();
            state.x = x;
            state.y = self.y.map(|a| match a {
                ValueCoordinate::Observation(i) => i,
                _ => unreachable!(),
            });
            state.mode = self.mode;
            state.resolution = self.resolution;
            return state.observation_at(index);
        }
        let mut o = self.anchor();
        for (axis, v) in std::iter::once(self.x).chain(self.y).zip(pos) {
            if let ValueCoordinate::Observation(i) = axis {
                o.as_dense_mut().expect("dense diagnostic input").values[i] = v;
            }
        }
        Ok(o)
    }
    pub fn action_at(
        &self,
        index: usize,
        policy_action: Option<&Action>,
    ) -> Result<Option<Action>> {
        self.structural()?;
        let pos = self.position(index)?;
        if self.has_discrete_axis() {
            return Ok(Some(Action::Discrete(
                if self.x == ValueCoordinate::DiscreteAction {
                    pos[0]
                } else {
                    pos[1]
                } as usize,
            )));
        }
        let base = match &self.conditioning {
            ActionConditioning::Policy => policy_action,
            ActionConditioning::Fixed(a) => Some(a),
        };
        let Some(base) = base else {
            ensure(
                self.state_only(),
                "action coordinates require a complete action",
            )?;
            return Ok(None);
        };
        self.space.check(base, self.state.anchor_mask())?;
        let mut a = base.clone();
        for (axis, v) in std::iter::once(self.x).chain(self.y).zip(pos) {
            if let ValueCoordinate::Action(i) = axis {
                let Action::Continuous(values) = &mut a else {
                    unreachable!()
                };
                values[i] = v;
            }
        }
        self.space.check(&a, self.state.anchor_mask())?;
        Ok(Some(a))
    }
    pub fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.state.owned_bytes()
            + self.action_ranges.capacity() * 8
            + self.space.dimensions() * 8
            + match &self.conditioning {
                ActionConditioning::Fixed(Action::Continuous(a)) => a.capacity() * 4,
                _ => 0,
            }
    }
}

#[cfg(test)]
mod color_domain_tests {
    use super::*;
    use crate::policy::slice::RangeSource;
    use rl::diagnostics::schema::RunManifest;

    #[test]
    fn recorded_ranges_replace_defaults_once_and_preserve_manual_limits() {
        let manifest = RunManifest::new(
            "range",
            "test",
            "test",
            3,
            ActionSpace::Unbounded { dimensions: 1 },
            1.,
        )
        .unwrap();
        let mut slice = ValueSlice::new(
            EnvironmentSemantics::new(&manifest),
            &Observation::new(vec![1.; 3]),
            &[],
        )
        .unwrap();
        slice.state.range_sources[2] = RangeSource::Declared;
        slice.state.ranges[2] = [0., 1.];
        assert!(
            slice
                .adopt_recorded_ranges([
                    Ok(Observation::new(vec![-0.25, 1., 0.5])),
                    Ok(Observation::new(vec![1.25, 1., 1.])),
                ])
                .unwrap()
        );
        assert_eq!(slice.state.ranges[0], [-0.25, 1.25]);
        assert_eq!(slice.state.range_sources[1], RangeSource::ModelInputDefault);
        // A trajectory supplies the price variation missing from fixed probes.
        assert!(
            slice
                .adopt_recorded_ranges([
                    Ok(Observation::new(vec![-10., 0.9, 0.2])),
                    Ok(Observation::new(vec![10., 1.1, 0.8])),
                ])
                .unwrap()
        );
        assert_eq!(
            slice.state.ranges,
            vec![[-0.25, 1.25], [0.9, 1.1], [0., 1.]]
        );
        slice.state.ranges[1] = [0.8, 1.2];
        slice.state.range_sources[1] = RangeSource::UserConfigured;
        assert!(
            !slice
                .adopt_recorded_ranges([Ok(Observation::new(vec![100.; 3]))])
                .unwrap()
        );
        assert_eq!(slice.state.ranges[1], [0.8, 1.2]);
    }
}
