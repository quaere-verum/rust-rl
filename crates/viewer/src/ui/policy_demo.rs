//! Policy view setup for headless tests.
use super::*;
use rl::diagnostics::{policy_slice::PolicySliceSpec, visualization::EnvironmentSemantics};

impl Laboratory {
    pub(super) fn configure_policy_demo_slice(&mut self, spec: &PolicySliceSpec) -> Result<()> {
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| rl::Error::State("no run".into()))?;
        let config = spec.to_config(EnvironmentSemantics::new(run.recording().manifest()))?;
        if self.policy_mode() != Some(PolicyMode::Interactive)
            || self.export_policy_slice()?.policy != spec.policy
            || self.selection.model_role != spec.model_role
        {
            return Err(rl::Error::Config(
                "demo slice requires the selected Interactive policy".into(),
            ));
        }
        let input = config.observation_at(0)?;
        self.policy_pane.pin = Some(crate::policy::prepared::PolicyPin::Synthetic(input.clone()));
        self.policy_pane.pin_input = Some(input);
        self.policy_pane.config = Some(config);
        self.policy_pane.resolution_explicit = [true; 2];
        self.policy_pane.dirty = true;
        self.policy_pane.ready = false;
        Ok(())
    }
    pub(super) fn configure_policy_demo_component(
        &mut self,
        action: usize,
        row: usize,
    ) -> Result<()> {
        let frame = self
            .policy_frame()
            .ok_or_else(|| rl::Error::State("demo frame unavailable".into()))?;
        let run = self.run.as_ref().unwrap();
        let count = match &run.recording().manifest().action_space {
            rl::core::action::ActionSpace::Discrete { count } => *count,
            space => space.dimensions(),
        };
        if !frame.grid.is_cached() || action >= count || row >= frame.grid.values.len() {
            return Err(rl::Error::Config(
                "demo component or cached row unavailable".into(),
            ));
        }
        self.policy_pane.config.as_mut().unwrap().action = action;
        // Mirror the normal action selector: a new component has a different
        // distribution domain, especially when Box bounds differ by component.
        self.policy_pane.domain = None;
        self.policy_pane.cached_row = Some(row);
        self.policy_pane.ready = false;
        Ok(())
    }
}
