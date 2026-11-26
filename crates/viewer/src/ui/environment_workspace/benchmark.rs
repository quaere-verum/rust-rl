//! Playback for a separately executed episode sharing an initial scenario.
use super::*;
use crate::dynamics::source::TrajectorySource;
use rl::diagnostics::reference::ReferenceLifetime;

pub(super) struct BenchmarkPlayback {
    catalog: TrajectoryCatalog,
    worker: SceneWorker,
    source: Option<CacheKey>,
    step: usize,
    playing: bool,
    last_tick: Instant,
}

impl BenchmarkPlayback {
    pub(super) fn new(
        run: &RunIndex,
        role: &str,
        gate: crate::dynamics::work::DecodeGate,
        worker: &SceneWorker,
        repaint: egui::Context,
    ) -> Result<Self> {
        Ok(Self {
            catalog: TrajectoryCatalog::with_decode_gate(
                run,
                role,
                CatalogLimits::default(),
                gate,
            )?,
            worker: worker.peer(move || repaint.request_repaint())?,
            source: None,
            step: 0,
            playing: false,
            last_tick: Instant::now(),
        })
    }

    pub(super) fn matches(&self, run: &RunIndex, role: &str) -> bool {
        self.catalog.matches(run, role)
    }

    pub(super) fn pending(&self) -> bool {
        self.catalog.pending()
            || self.worker.pending()
            || self.worker.error().is_none()
                && self.source.as_ref().is_some_and(|source| {
                    self.worker.frame().is_none_or(|frame| {
                        &frame.identity.key != source || frame.identity.step != self.step
                    })
                })
    }

    pub(super) fn recheck_source(&mut self, run: &RunIndex) {
        if let Some(frame) = self.worker.frame()
            && let Some(index) = run.by_sequence(frame.identity.key.sequence)
        {
            let path = run.root().join(&run.recording().entries()[index].file);
            let _ = self.worker.check_path(path);
        }
    }

    #[cfg(feature = "screenshots")]
    pub(super) fn evidence(&self) -> Option<(u64, usize, Option<u64>)> {
        self.worker.frame().map(|frame| {
            (
                frame.identity.revision,
                frame.identity.step,
                frame.playback.scenario_seed,
            )
        })
    }

    pub(super) fn show(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
        loader: &mut Loader,
        learner: &TrajectorySource,
        session: u64,
    ) -> Result<Option<Arc<PreparedScene>>> {
        self.catalog.poll(run, loader)?;
        self.worker.poll();
        let Some(scenario) = &learner.scenario else {
            ui.weak("The learner episode has no recorded initial scenario");
            return Ok(None);
        };
        if self.catalog.pending() {
            ui.weak("Checking benchmark episodes…");
            return Ok(None);
        }
        let source = self
            .catalog
            .entries()
            .iter()
            .filter(|source| {
                source.environment == learner.environment
                    && source.scenario.as_ref() == Some(scenario)
                    && source.benchmark.as_ref().is_some_and(|benchmark| {
                        source.revision == learner.revision
                            || benchmark.lifetime == ReferenceLifetime::Fixed
                                && source.revision <= learner.revision
                    })
            })
            .max_by_key(|source| (source.revision, source.key.sequence));
        let Some(source) = source else {
            self.worker.retire();
            self.source = None;
            self.playing = false;
            ui.weak("No benchmark episode matches the recorded scenario and return clock");
            return Ok(None);
        };
        let benchmark = source
            .benchmark
            .as_ref()
            .expect("matched benchmark declaration");
        ui.horizontal_wrapped(|ui| {
            ui.strong("Benchmark episode");
            ui.label(format!(
                "{} · scenario {}",
                crate::reference::quality_label(Some(benchmark.quality)),
                scenario.seed
            ))
            .on_hover_text(
                "Matched initial scenario. Learner and benchmark use independent playback clocks.",
            );
        });
        if self.source.as_ref() != Some(&source.key) {
            self.worker.retire();
            self.source = Some(source.key.clone());
            self.step = 0;
            self.playing = false;
        }

        let end = source.transitions;
        ui.horizontal_wrapped(|ui| {
            if ui
                .small_button("Previous")
                .on_hover_text("Previous benchmark state")
                .clicked()
            {
                self.step = self.step.saturating_sub(1);
                self.playing = false;
            }
            if ui
                .small_button(if self.playing {
                    "Pause benchmark"
                } else {
                    "Play benchmark"
                })
                .clicked()
            {
                if self.step == end {
                    self.step = 0;
                }
                self.playing = !self.playing;
                self.last_tick = Instant::now();
            }
            if ui
                .small_button("Next")
                .on_hover_text("Next benchmark state")
                .clicked()
            {
                self.step = self.step.saturating_add(1).min(end);
                self.playing = false;
            }
            if ui.add(egui::Slider::new(&mut self.step, 0..=end)).changed() {
                self.playing = false;
            }
        });
        if self.playing
            && self.last_tick.elapsed() >= Duration::from_millis(100)
            && self.worker.frame().is_some_and(|frame| {
                frame.identity.key == source.key && frame.identity.step == self.step
            })
        {
            self.step = self.step.saturating_add(1).min(end);
            self.last_tick = Instant::now();
            self.playing = self.step < end;
        }
        if self.playing {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }
        if let Some(error) = self.worker.error() {
            self.playing = false;
            ui.weak(error);
            if ui.button("Retry benchmark episode").clicked() {
                self.worker.retire();
                loader.invalidate();
            }
            return Ok(None);
        }
        let index = run
            .by_sequence(source.key.sequence)
            .ok_or_else(|| rl::Error::State("Recorded benchmark trajectory was removed".into()))?;
        let Some(chunk) = loader.request_admitted(run, index, crate::environment::MAX_RAW_BYTES)?
        else {
            ui.weak("Loading benchmark episode…");
            return Ok(None);
        };
        let identity = SceneIdentity {
            session,
            key: source.key.clone(),
            revision: source.revision,
            role: source.role.clone(),
            step: self.step,
        };
        self.worker.submit(identity, &chunk, Vec::new())?;
        if let Some(frame) = self.worker.frame().filter(|frame| {
            frame.identity.session == session
                && frame.identity.key == source.key
                && frame.identity.revision == source.revision
                && frame.identity.role == source.role
        }) {
            ui.label(format!("Benchmark state {} of {end}", frame.identity.step));
            paint_comparison_readouts(ui, frame, run.recording().manifest());
            Ok(Some(Arc::clone(frame)))
        } else {
            ui.weak("Preparing benchmark state…");
            Ok(None)
        }
    }
}

pub(super) fn reference_role<'a>(run: &'a RunIndex, selected: &'a str) -> Option<&'a str> {
    let mut roles = run
        .record_roles(RecordKind::Trajectory)
        .filter(|role| crate::reference::provider_role(role));
    if crate::reference::provider_role(selected) {
        return roles.find(|role| *role == selected);
    }
    if selected != "online" {
        return None;
    }
    // The ordinary unselected reference defaults to online; only a unique
    // recorded benchmark is eligible for the local convenience action.
    let candidate = roles.next()?;
    roles.next().is_none().then_some(candidate)
}
