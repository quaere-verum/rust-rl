use super::{BranchingEnv, EnvironmentFrame, FrameSource};
use crate::{
    Result,
    core::{
        action::ActionSpace,
        ensure,
        env::{ClipAction, Env, Observation},
    },
};

impl<E: BranchingEnv> BranchingEnv for ClipAction<E> {
    fn branch(&self, future_seed: u64) -> Result<Self> {
        self.validate()?;
        let branch = Self::new(self.inner().branch(future_seed)?)?;
        ensure(
            branch.physical_action_space() == self.physical_action_space(),
            "branched physical action space changed",
        )?;
        Ok(branch)
    }
    fn branch_observation(&self) -> Result<Observation> {
        self.validate()?;
        self.inner().branch_observation()
    }
    fn branch_state(&self) -> Result<Vec<f64>> {
        self.validate()?;
        self.inner().branch_state()
    }
    fn branch_description(&self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "{}; recorded policy actions are raw samples clipped to physical bounds {:?}",
            self.inner().branch_description()?,
            self.physical_action_space()
        ))
    }
}

impl<E: Env + FrameSource> FrameSource for ClipAction<E> {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        self.validate()?;
        let mut frame = self.inner().diagnostic_frame()?;
        ensure(
            !frame.channels.contains_key("clip_action.low")
                && !frame.channels.contains_key("clip_action.high"),
            "inner frame uses reserved clip-action channel names",
        )?;
        let ActionSpace::Box { low, high } = self.physical_action_space() else {
            unreachable!()
        };
        frame.channels.insert(
            "clip_action.low".into(),
            low.iter().map(|&x| x as f64).collect(),
        );
        frame.channels.insert(
            "clip_action.high".into(),
            high.iter().map(|&x| x as f64).collect(),
        );
        Ok(frame)
    }
}
