//! Compact revision transport shared by the recorded workspaces.

use crate::{
    data::{RunIndex, Selection},
    timeline::{Direction, RevisionRange, Timeline},
};
use eframe::egui;
use rl::Result;
use std::{path::PathBuf, sync::Arc, time::Instant};

/// Immutable presentation state supplied by the currently active panel.
#[derive(Clone, Copy)]
pub struct SnapshotPresentation<'a> {
    pub positions: &'a [u64],
    pub published: Option<u64>,
    pub ready: bool,
    pub checking: bool,
    pub error: Option<&'a str>,
}

/// Rendering and clock adapter for the CPU-only training timeline.
#[derive(Default)]
pub struct TrainingTransport {
    timeline: Timeline,
    last_tick: Option<Instant>,
    published_snapshot: Option<(u64, usize, usize)>,
    unavailable_snapshot: Option<u64>,
    cached_root: Option<Arc<PathBuf>>,
    cached_role: String,
}

impl TrainingTransport {
    /// Predict the next step in the current navigation direction, without a clock.
    pub(crate) fn next_snapshot(&self) -> Option<u64> {
        self.timeline.next_snapshot()
    }
    #[cfg(feature = "screenshots")]
    pub(crate) fn capture_publication(&self) -> Option<(u64, usize, usize)> {
        self.published_snapshot
    }
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Retiring a workspace also retires its publication and position domain.
    /// Same-panel revalidation does not call this: its complete frame stays held.
    pub(crate) fn clear_snapshot_state(&mut self) {
        let had_positions = self.timeline.clear_snapshot_positions();
        self.published_snapshot = None;
        self.unavailable_snapshot = None;
        if had_positions {
            self.pause();
        }
    }
    pub(crate) fn hold_unavailable_snapshot(&mut self, revision: Option<u64>) {
        self.unavailable_snapshot = revision;
        if revision.is_some() {
            self.pause();
            self.published_snapshot = None;
        }
    }

    pub fn is_playing(&self) -> bool {
        self.timeline.is_playing()
    }

    pub fn pause(&mut self) {
        self.timeline.pause();
        self.last_tick = None;
    }
    fn sync(&mut self, run: &RunIndex, selection: &Selection) {
        let changed_run = self
            .cached_root
            .as_ref()
            .is_none_or(|root| !Arc::ptr_eq(root, run.root()));
        if changed_run {
            self.reset();
        }
        let changed_role = self.cached_role != selection.model_role;
        if changed_run || changed_role {
            self.cached_root = Some(run.root().clone());
            self.cached_role.clone_from(&selection.model_role);
        }
        if changed_role {
            self.timeline.pause();
        }
        let bounds = run
            .revisions()
            .first()
            .zip(run.revisions().last())
            .and_then(|(&min, &max)| RevisionRange::new(min, max));
        self.timeline.set_bounds(bounds);
        self.timeline.set_revision(selection.revision);
    }

    /// Uses the existing clock, with an independently published complete identity.
    pub fn show_snapshots(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
        selection: &mut Selection,
        presentation: SnapshotPresentation<'_>,
    ) -> Result<bool> {
        self.show_positions(ui, run, selection, presentation, false)
    }

    /// The same sparse-revision clock over compatible recorded evaluations.
    pub fn show_evaluations(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
        selection: &mut Selection,
        presentation: SnapshotPresentation<'_>,
    ) -> Result<bool> {
        self.show_positions(ui, run, selection, presentation, true)
    }

    fn show_positions(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
        selection: &mut Selection,
        presentation: SnapshotPresentation<'_>,
        evaluations: bool,
    ) -> Result<bool> {
        let positions = presentation.positions;
        let now = Instant::now();
        let elapsed = self.last_tick.map_or(std::time::Duration::ZERO, |last| {
            now.saturating_duration_since(last)
        });
        // A caller can request an exact unavailable checkpoint before this run's
        // first paint. Run synchronization resets old state, but not that request.
        let requested_hold = self
            .unavailable_snapshot
            .filter(|&revision| revision == selection.revision);
        self.sync(run, selection);
        self.unavailable_snapshot = requested_hold;
        self.last_tick = Some(now);
        self.timeline.set_snapshot_positions(positions);
        // A partial catalogue is not evidence that an explicitly requested
        // checkpoint is unavailable. Keep that request until validation catches
        // up, and resynchronize against the newly admitted positions afterwards.
        if presentation.checking && !positions.contains(&selection.revision) {
            self.timeline.hold_unavailable_snapshot(selection.revision);
        } else if positions.contains(&selection.revision) {
            self.timeline.set_revision(selection.revision);
        }
        // The selected artifact can be unavailable even when a different
        // artifact at the same revision has since entered the catalog.
        if self.unavailable_snapshot == Some(selection.revision) {
            self.timeline.hold_unavailable_snapshot(selection.revision);
        } else {
            self.unavailable_snapshot = None;
        }
        let retry = self.position_controls(ui, presentation, elapsed, evaluations, false)?;
        if self.timeline.revision() != selection.revision {
            selection.select_revision(run, self.timeline.revision())?;
        }
        Ok(retry)
    }

    /// Exact cache-source sequences use this transport's existing clock.
    pub fn show_prepared(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
        sequence: &mut u64,
        presentation: SnapshotPresentation<'_>,
    ) -> Result<bool> {
        let changed = self
            .cached_root
            .as_ref()
            .is_none_or(|root| !Arc::ptr_eq(root, run.root()))
            || self.cached_role != "prepared-policy";
        if changed {
            self.reset();
            self.cached_root = Some(run.root().clone());
            self.cached_role = "prepared-policy".into();
        }
        let now = Instant::now();
        let elapsed = self.last_tick.map_or(std::time::Duration::ZERO, |last| {
            now.saturating_duration_since(last)
        });
        self.last_tick = Some(now);
        let entries = run.recording().entries();
        self.timeline.set_bounds(
            entries
                .first()
                .zip(entries.last())
                .and_then(|(a, b)| RevisionRange::new(a.sequence, b.sequence)),
        );
        self.timeline.set_snapshot_positions(presentation.positions);
        self.timeline.set_revision(*sequence);
        if !presentation.positions.contains(sequence) {
            self.timeline.hold_unavailable_snapshot(*sequence);
        }
        let retry = self.position_controls(ui, presentation, elapsed, false, true)?;
        *sequence = self.timeline.revision();
        Ok(retry)
    }

    fn position_controls(
        &mut self,
        ui: &mut egui::Ui,
        presentation: SnapshotPresentation<'_>,
        elapsed: std::time::Duration,
        evaluations: bool,
        cached: bool,
    ) -> Result<bool> {
        let SnapshotPresentation {
            positions,
            published,
            ready,
            checking,
            error,
        } = presentation;
        let mut retry = false;
        if cached && !positions.is_empty() {
            self.timeline.advance_cached_ready(
                elapsed,
                ready && published == Some(self.timeline.revision()),
            );
        } else if !positions.is_empty() {
            self.timeline.advance_ready(
                elapsed,
                ready && published == Some(self.timeline.revision()),
            );
        }
        if !positions.is_empty() && !ui.ctx().egui_wants_keyboard_input() {
            if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space)) {
                self.toggle_play();
            }
            for (key, direction) in [
                (egui::Key::ArrowLeft, Direction::Backward),
                (egui::Key::ArrowRight, Direction::Forward),
            ] {
                if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, key)) {
                    self.timeline.step(direction);
                }
            }
        }
        let slider_width = (ui.available_width() - 360.).clamp(72., 580.);
        ui.horizontal(|ui| {
            ui.spacing_mut().slider_width = slider_width;
            ui.add_enabled_ui(!positions.is_empty(), |ui| {
                if ui.button("‹").on_hover_text("Previous checkpoint · Left").clicked() {
                    self.timeline.step(Direction::Backward);
                }
                if ui.button(if self.is_playing() { "Pause" } else { "Play" })
                    .on_hover_text("Play / pause · Space").clicked() {
                    self.toggle_play();
                }
                if ui.button("›").on_hover_text("Next checkpoint · Right").clicked() {
                    self.timeline.step(Direction::Forward);
                }
            });
            if let Some((k, position)) =
                published.and_then(|k| positions.iter().position(|&p| p == k).map(|p| (k, p)))
            {
                self.published_snapshot = Some((k, position + 1, positions.len()));
            }
            let visible = self.published_snapshot;
            ui.label(egui::RichText::new(visible.map_or_else(
                || "—".into(),
                |(_, position, count)| format!("{position} / {count}"),
            )).small().color(crate::theme::MUTED))
            .on_hover_text(published.map_or_else(
                || "Waiting for a complete frame".into(),
                |k| {
                    if cached {
                        format!("source = {k}")
                    } else {
                        format!("k = {k}")
                    }
                },
            ));
            let mut index = positions
                .iter()
                .position(|&k| k == self.timeline.revision())
                .unwrap_or(0);
            if ui
                .add_enabled(!positions.is_empty(),egui::Slider::new(&mut index, 0..=positions.len().saturating_sub(1)).show_value(false))
                .on_hover_text("Recorded checkpoints only · drag to scrub")
                .changed()
                && !positions.is_empty()
            {
                self.timeline.pause();
                self.timeline.set_revision(positions[index]);
            }
            crate::landscape::popover(ui, "Rate", |ui| {
                ui.label("Checkpoints per second");
                let mut speed = self.timeline.speed();
                if ui.add(egui::DragValue::new(&mut speed).range(0.1..=100.)).changed() {
                    self.timeline.set_speed(speed);
                }
                let mut looping = self.timeline.looping();
                if ui.checkbox(&mut looping,"Loop").changed() { self.timeline.set_looping(looping); }
            });
            let unavailable = error.is_some() || positions.is_empty() && !checking;
            ui.allocate_ui_with_layout(egui::vec2(146.,26.),egui::Layout::left_to_right(egui::Align::Center),|ui| {
                if unavailable {
                    crate::landscape::popover(ui,"Unavailable",|ui| {
                        ui.set_max_width(360.);
                        ui.label(error.unwrap_or(if cached {"No cached checkpoints for this slice"}
                            else if evaluations {"No compatible evaluation trajectories"} else {"No usable model snapshots"}));
                        retry = ui.button("Retry").clicked();
                        if !positions.is_empty() && ui.button("Skip").clicked() { self.timeline.step(Direction::Forward); }
                    });
                } else {
                    if let Some(actual) = published.or(visible.map(|v|v.0))
                        && actual != self.timeline.revision() {
                        ui.label(egui::RichText::new(if cached {format!("source #{actual}")}
                            else {format!("shown k{actual}")}).small().color(crate::theme::EMPIRICAL))
                            .on_hover_text("The previous complete publication remains visible while the requested checkpoint is prepared.");
                    }
                    if checking {
                        ui.weak("Checking").on_hover_text(if cached {"Checking prepared data…"}
                            else if evaluations {"Checking evaluations…"} else {"Checking model snapshots…"});
                    } else if !ready {
                        ui.weak("Loading").on_hover_text("Preparing a complete frame at the requested checkpoint");
                    }
                }
            });
        });
        if self.is_playing() || checking || !ready && !positions.is_empty() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(16));
        }
        Ok(retry)
    }
    fn toggle_play(&mut self) {
        if self.is_playing() {
            self.timeline.pause();
        } else {
            self.timeline.play();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl::{
        core::action::ActionSpace,
        diagnostics::{schema::*, storage::*},
    };
    use std::collections::BTreeMap;

    fn append(writer: &mut Recorder, revision: u64, role: &str, activation: bool) {
        let mut context = RecordContext::new("transport", revision);
        context.identity.model_role = role.into();
        let (record, arrays) = if activation {
            (
                Record::Activations(ActivationRecord {
                    network: "policy".into(),
                    layers: BTreeMap::from([("hidden".into(), "a".into())]),
                    ..Default::default()
                }),
                BTreeMap::from([("a".into(), Array::f32(vec![1, 1], vec![1.0]).unwrap())]),
            )
        } else {
            (
                Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("loss".into(), 1.0)]),
                    unavailable: BTreeMap::new(),
                }),
                BTreeMap::new(),
            )
        };
        writer
            .append(&Chunk {
                context,
                record,
                arrays,
            })
            .unwrap();
    }

    fn fixture() -> (tempfile::TempDir, Recorder, RunIndex) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run");
        let mut writer = Recorder::create(
            &path,
            RunManifest::new(
                "transport",
                "test",
                "test",
                1,
                ActionSpace::Discrete { count: 2 },
                0.9,
            )
            .unwrap(),
            StorageConfig::default(),
        )
        .unwrap();
        append(&mut writer, 1, "online", false);
        append(&mut writer, 21, "online", true);
        append(&mut writer, 61, "online", false);
        append(&mut writer, 41, "target", true);
        let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
        (dir, writer, run)
    }

    #[test]
    fn explicit_checkpoint_waits_for_partial_catalog_validation() {
        let (_dir, _writer, run) = fixture();
        let mut transport = TrainingTransport::default();
        let mut selection = Selection {
            revision: 1,
            ..Default::default()
        };
        let context = egui::Context::default();
        let mut paint = |selection: &mut Selection, positions: &[u64], checking| {
            let mut output = context.run_ui(egui::RawInput::default(), |ui| {
                transport
                    .show_snapshots(
                        ui,
                        &run,
                        selection,
                        SnapshotPresentation {
                            positions,
                            published: Some(1),
                            ready: true,
                            checking,
                            error: None,
                        },
                    )
                    .unwrap();
            });
            output.textures_delta.clear();
        };
        paint(&mut selection, &[1], true);
        selection.select_revision(&run, 21).unwrap();
        paint(&mut selection, &[1], true);
        assert_eq!(
            selection.revision, 21,
            "pending validation must not snap a deliberate selection back to the old frame"
        );
        paint(&mut selection, &[1, 21], false);
        assert_eq!(selection.revision, 21);
    }

    #[test]
    fn empty_revalidation_keeps_the_held_complete_snapshot_counter() {
        let (_dir, _writer, run) = fixture();
        let mut transport = TrainingTransport::default();
        let mut selection = Selection {
            revision: 21,
            ..Default::default()
        };
        let context = egui::Context::default();
        let mut first = context.run_ui(egui::RawInput::default(), |ui| {
            transport
                .show_snapshots(
                    ui,
                    &run,
                    &mut selection,
                    SnapshotPresentation {
                        positions: &[1, 21, 61],
                        published: Some(21),
                        ready: true,
                        checking: false,
                        error: None,
                    },
                )
                .unwrap();
        });
        first.textures_delta.clear();
        let mut output = context.run_ui(egui::RawInput::default(), |ui| {
            transport
                .show_snapshots(
                    ui,
                    &run,
                    &mut selection,
                    SnapshotPresentation {
                        positions: &[],
                        published: None,
                        ready: false,
                        checking: true,
                        error: None,
                    },
                )
                .unwrap();
        });
        output.textures_delta.clear();
        fn contains(shape: &egui::Shape, text: &str) -> bool {
            match shape {
                egui::Shape::Text(t) => t.galley.job.text == text,
                egui::Shape::Vec(shapes) => shapes.iter().any(|s| contains(s, text)),
                _ => false,
            }
        }
        assert!(output.shapes.iter().any(|s| contains(&s.shape, "2 / 3")));
        assert!(output.shapes.iter().any(|s| contains(&s.shape, "Checking")));
        assert_eq!(selection.revision, 21);
    }

    #[test]
    fn unavailable_checkpoint_stays_held_until_real_navigation() {
        let (_dir, _writer, run) = fixture();
        let mut transport = TrainingTransport::default();
        let mut selection = Selection {
            revision: 21,
            ..Default::default()
        };
        transport
            .start_snapshot_playback(&run, &mut selection, &[1, 21, 61], 100.)
            .unwrap();
        transport.hold_unavailable_snapshot(Some(21));
        let context = egui::Context::default();
        for (key, expected) in [(None, 21), (Some(egui::Key::ArrowRight), 61)] {
            let mut output = context.run_ui(
                egui::RawInput {
                    events: key
                        .into_iter()
                        .map(|key| egui::Event::Key {
                            key,
                            physical_key: None,
                            pressed: true,
                            repeat: false,
                            modifiers: egui::Modifiers::NONE,
                        })
                        .collect(),
                    ..Default::default()
                },
                |ui| {
                    transport
                        .show_snapshots(
                            ui,
                            &run,
                            &mut selection,
                            SnapshotPresentation {
                                positions: &[1, 61],
                                published: None,
                                ready: false,
                                checking: false,
                                error: Some("Unavailable"),
                            },
                        )
                        .unwrap();
                },
            );
            output.textures_delta.clear();
            assert_eq!(selection.revision, expected);
            assert!(!transport.is_playing());
        }
    }

    fn paint_unavailable_transport(
        transport: &mut TrainingTransport,
        run: &RunIndex,
        selection: &mut Selection,
    ) {
        let context = egui::Context::default();
        let mut output = context.run_ui(egui::RawInput::default(), |ui| {
            transport
                .show_snapshots(
                    ui,
                    run,
                    selection,
                    SnapshotPresentation {
                        positions: &[1, 21, 61],
                        published: None,
                        ready: false,
                        checking: false,
                        error: Some("Unavailable"),
                    },
                )
                .unwrap();
        });
        output.textures_delta.clear();
    }

    #[test]
    fn first_paint_preserves_current_request_hold_across_run_initialization() {
        let (_dir, _writer, run) = fixture();
        let mut transport = TrainingTransport::default();
        let mut selection = Selection {
            revision: 11,
            ..Default::default()
        };
        transport.hold_unavailable_snapshot(Some(11));
        paint_unavailable_transport(&mut transport, &run, &mut selection);
        assert_eq!(selection.revision, 11);
        assert!(!transport.is_playing());
    }

    #[test]
    fn mismatched_or_reset_holds_do_not_survive_run_initialization() {
        for (requested, reset, expected) in [(61, false, 61), (11, true, 1)] {
            let (_dir, _writer, run) = fixture();
            let mut transport = TrainingTransport::default();
            transport.hold_unavailable_snapshot(Some(11));
            if reset {
                transport.reset();
            }
            let mut selection = Selection {
                revision: requested,
                ..Default::default()
            };
            paint_unavailable_transport(&mut transport, &run, &mut selection);
            assert_eq!(selection.revision, expected);
            assert_eq!(transport.unavailable_snapshot, None);
            assert!(!transport.is_playing());
        }
    }

    #[test]
    fn unavailable_edge_never_steps_in_the_opposite_direction() {
        let mut timeline = Timeline::new(RevisionRange::new(1, 100));
        timeline.set_snapshot_positions(&[21, 61]);
        timeline.hold_unavailable_snapshot(81);
        assert!(!timeline.step(Direction::Forward));
        assert_eq!(timeline.revision(), 81);
        assert!(timeline.step(Direction::Backward));
        assert_eq!(timeline.revision(), 61);
        timeline.hold_unavailable_snapshot(1);
        assert!(!timeline.step(Direction::Backward));
        assert_eq!(timeline.revision(), 1);
        assert!(timeline.step(Direction::Forward));
        assert_eq!(timeline.revision(), 21);
    }

    #[test]
    fn role_and_live_refresh_use_only_the_supplied_panel_positions() {
        let (_dir, mut writer, mut run) = fixture();
        let mut transport = TrainingTransport::default();
        let mut selection = Selection {
            revision: 21,
            ..Default::default()
        };
        transport
            .start_snapshot_playback(&run, &mut selection, &[1, 21, 61], 1.)
            .unwrap();
        assert!(transport.is_playing());
        selection.model_role = "target".into();
        transport.sync(&run, &selection);
        transport.timeline.set_snapshot_positions(&[41]);
        assert!(!transport.is_playing());
        assert_eq!(transport.timeline.revision(), 41);
        assert_eq!(transport.next_snapshot(), None);
        selection.select_revision(&run, 41).unwrap();
        append(&mut writer, 81, "target", true);
        run.refresh().unwrap();
        transport.sync(&run, &selection);
        transport.timeline.set_snapshot_positions(&[41, 81]);
        assert_eq!(transport.timeline.revision(), 41);
        assert_eq!(transport.next_snapshot(), Some(81));
        transport.timeline.play();
        transport.last_tick = Some(Instant::now());
        transport.clear_snapshot_state();
        assert!(!transport.is_playing());
        assert!(transport.last_tick.is_none());
        assert!(transport.published_snapshot.is_none());
        assert_eq!(transport.next_snapshot(), None);
    }

    #[test]
    fn reset_clears_playback_and_run_specific_state() {
        let (_dir, _writer, run) = fixture();
        let mut transport = TrainingTransport::default();
        transport.sync(
            &run,
            &Selection {
                revision: 1,
                ..Default::default()
            },
        );
        transport.timeline.set_snapshot_positions(&[1, 21, 61]);
        transport.timeline.play();
        assert!(transport.is_playing());
        transport.reset();
        assert!(!transport.is_playing());
        assert_eq!(transport.next_snapshot(), None);
        assert_eq!(transport.timeline.bounds(), None);
    }

    #[test]
    fn warmup_preview_obeys_exact_snapshot_bounds_loop_and_endpoint() {
        let mut t = TrainingTransport::default();
        t.timeline.set_bounds(RevisionRange::new(0, 360));
        t.timeline.set_snapshot_positions(&[0, 120, 240, 360]);
        assert_eq!(t.next_snapshot(), Some(120));
        assert_eq!(t.timeline.revision(), 0);
        t.timeline.set_bounds(RevisionRange::new(120, 240));
        t.timeline.set_revision(120);
        assert_eq!(t.next_snapshot(), Some(240));
        t.timeline.set_revision(240);
        assert_eq!(t.next_snapshot(), None);
        t.timeline.set_looping(true);
        assert_eq!(t.next_snapshot(), Some(120));
        assert_eq!(t.timeline.revision(), 240);
    }
}

#[cfg(test)]
mod test_fixture;
