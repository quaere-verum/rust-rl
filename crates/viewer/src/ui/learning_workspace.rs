//! Apply bounded initial defaults while preserving deliberate selections.
use super::*;

impl Laboratory {
    pub(super) fn prepare_workspace_defaults(&mut self, run: &RunIndex) -> Result<()> {
        let Some((pending, original, reference_serial)) = &mut self.workspace_defaults else {
            return Ok(());
        };
        // Any deliberate selection wins over the asynchronous initial default.
        if self.selection.input != *original
            || self.comparison.reference_interaction_serial() != *reference_serial
        {
            self.workspace_defaults = None;
            return Ok(());
        }
        pending.update_revision(run, &self.selection);
        if !pending.poll(run, &mut self.loader)? {
            return Ok(());
        }
        let (pending, _, _) = self.workspace_defaults.take().unwrap();
        if let Some(plan) = pending.finish() {
            self.selection.input = plan.input;
            self.selection.model_role = plan.role;
            self.comparison.use_local_defaults(
                run,
                &self.selection.model_role,
                &self.selection.input,
                plan.reference_revision,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl::{
        core::{action::ActionSpace, env::Observation},
        diagnostics::{probes::ProbeSet, storage::*},
    };

    fn app() -> (tempfile::TempDir, Laboratory) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("defaults");
        let space = ActionSpace::Discrete { count: 2 };
        let manifest =
            RunManifest::new("defaults", "defaults", "ordinary", 1, space.clone(), 0.9).unwrap();
        let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
        for revision in [1, 7] {
            let mut context = RecordContext::new("defaults", revision);
            context.probe_set = Some("fixed".into());
            let probes = ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "fixed inputs",
                vec![Observation::new(vec![0.])],
                None,
                space.clone(),
            )
            .unwrap()
            .record(context)
            .unwrap();
            writer.append(&probes).unwrap();
        }
        writer.finish().unwrap();
        let context = egui::Context::default();
        let app = Laboratory::from_context(&context, path.display().to_string()).unwrap();
        assert!(app.workspace_defaults.is_some());
        (directory, app)
    }

    #[test]
    fn deliberate_input_wins_over_pending_defaults() {
        let (_directory, mut app) = app();
        app.selection.input = SelectedInput::None;
        let run = app.run.take().unwrap();
        app.prepare_workspace_defaults(&run).unwrap();
        assert!(app.workspace_defaults.is_none());
        assert_eq!(app.selection.input, SelectedInput::None);
    }

    #[test]
    fn deliberate_reference_wins_even_if_requested_revision_is_unchanged() {
        let (_directory, mut app) = app();
        let revision = app.comparison.reference_revision();
        app.comparison.set_reference_revision(revision).unwrap();
        let run = app.run.take().unwrap();
        let selected = app.selection.clone();
        app.prepare_workspace_defaults(&run).unwrap();
        assert!(app.workspace_defaults.is_none());
        assert_eq!(app.selection.input, selected.input);
        assert_eq!(app.selection.model_role, selected.model_role);
        assert_eq!(app.selection.revision, selected.revision);
        assert_eq!(app.comparison.reference_revision(), revision);
    }
}
