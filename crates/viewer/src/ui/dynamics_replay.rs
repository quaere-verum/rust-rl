//! Bounded factual replay coverage over immutable observation and action rows.
use super::*;
use crate::{landscape, pane::SelectionIntent, timeline_ui::SnapshotPresentation};
use egui_plot::{MarkerShape, Plot, Points};
use rl::core::action::Action;

const MAX_CHUNK_BYTES: usize = 512 * 1024;

pub(super) struct ReplayView {
    pub(super) enabled: bool,
    pub(super) actual: Option<u64>,
    axes: [usize; 2],
    plot: Option<ReplayPlot>,
}

struct ReplayPlot {
    key: CacheKey,
    axes: [usize; 2],
    points: Vec<[f64; 2]>,
}

impl ReplayView {
    pub(super) fn new() -> Self {
        Self {
            enabled: cfg!(feature = "screenshots")
                && std::env::var("RL_VIEWER_DYNAMICS_REPLAY").as_deref() == Ok("true"),
            actual: None,
            axes: [0, 1],
            plot: None,
        }
    }
}

pub(super) fn probe_id(run: &RunIndex) -> Option<&str> {
    run.recording()
        .manifest()
        .metadata
        .get("coverage.replay_probe")
        .map(String::as_str)
}

impl Laboratory {
    pub(super) fn replay_transport(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let mut positions = run
            .candidates(
                RecordKind::ProbeSet,
                &self.selection.model_role,
                probe_id(run),
                u64::MAX,
            )
            .take(128)
            .map(|i| run.recording().entries()[i].revision)
            .collect::<Vec<_>>();
        positions.sort_unstable();
        positions.dedup();
        self.transport.show_snapshots(
            ui,
            run,
            &mut self.selection,
            SnapshotPresentation {
                positions: &positions,
                published: self.dynamics_pane.replay.actual,
                ready: self.dynamics_pane.replay.actual.is_some(),
                checking: self.loader.pending() > 0,
                error: None,
            },
        )?;
        Ok(())
    }

    pub(super) fn show_replay(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        self.dynamics_pane.replay.actual = None;
        let Some(index) = run
            .candidates(
                RecordKind::ProbeSet,
                &self.selection.model_role,
                probe_id(run),
                self.selection.revision,
            )
            .next()
        else {
            ui.weak(format!(
                "No replay sample at checkpoint k {}.",
                self.selection.revision
            ));
            return Ok(());
        };
        let entry = &run.recording().entries()[index];
        let Some(chunk) = self.loader.request_admitted(run, index, MAX_CHUNK_BYTES)? else {
            ui.weak("Loading the recorded replay sample…");
            return Ok(());
        };
        let Record::ProbeSet(probes) = &chunk.record else {
            return Err(rl::Error::State(
                "Replay source has no recorded probe population".into(),
            ));
        };
        let Some(action_key) = &probes.fixed_actions else {
            return Err(rl::Error::State(
                "Replay probe has no recorded executed actions".into(),
            ));
        };
        if probes.population != ProbePopulation::ReplayRetained {
            return Err(rl::Error::Config(
                "Selected evidence does not declare retained factual replay".into(),
            ));
        }
        let observations = crate::plot_data::array(&chunk, &probes.observations)?;
        let [rows, width] = observations.shape() else {
            return Err(rl::Error::Shape("Invalid replay observations".into()));
        };
        let manifest = run.recording().manifest();
        if *rows > 512 || *width == 0 || *width != manifest.observation_dim {
            return Err(rl::Error::Shape(
                "Replay observations differ from the admitted run encoding".into(),
            ));
        }
        let semantics = EnvironmentSemantics::new(run.recording().manifest());
        let action_width = semantics.action_space().dimensions();
        if width.saturating_add(action_width) > 128 {
            ui.weak(
                "Replay coverage supports up to 128 recorded observation and action coordinates.",
            );
            return Ok(());
        }
        let labels = (0..*width)
            .map(|i| landscape::observation_axis_label(semantics, i))
            .chain((0..action_width).map(|i| {
                semantics.action(i).map_or_else(
                    || format!("Executed action {i}"),
                    |d| {
                        format!(
                            "Executed {}",
                            landscape::with_unit(d.label.into_owned(), d.unit)
                        )
                    },
                )
            }))
            .collect::<Vec<_>>();
        if labels.len() < 2 {
            ui.weak("Replay coverage needs two recorded observation or action coordinates.");
            return Ok(());
        }
        let axes = &mut self.dynamics_pane.replay.axes;
        for axis in &mut *axes {
            *axis = (*axis).min(labels.len() - 1);
        }
        if axes[0] == axes[1] {
            axes[1] = (axes[0] + 1) % labels.len();
        }
        ui.horizontal_wrapped(|ui| {
            for i in 0..2 {
                let other = axes[1 - i];
                egui::ComboBox::from_id_salt(("replay-coverage-axis", i))
                    .selected_text(&labels[axes[i]])
                    .show_ui(ui, |ui| {
                        for (j, label) in labels.iter().enumerate() {
                            ui.add_enabled_ui(j != other, |ui| {
                                ui.selectable_value(&mut axes[i], j, label);
                            });
                        }
                    });
            }
            ui.weak(format!(
                "{rows} sampled replay rows · data k {}",
                entry.revision
            ))
            .on_hover_text(&probes.methodology);
        });
        let axes = *axes;
        let key = run.key(index)?;
        let replay = &mut self.dynamics_pane.replay;
        if replay
            .plot
            .as_ref()
            .is_none_or(|p| p.key != key || p.axes != axes)
        {
            let mut points = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let observation = crate::policy::prepared::exact_observation(&chunk, row)?;
                let action = crate::value::visitation::action_row(
                    &chunk,
                    action_key,
                    row,
                    semantics.action_space(),
                )?;
                observation.check_action(semantics.action_space(), &action)?;
                let coordinate = |axis: usize| {
                    if axis < *width {
                        f64::from(observation.dense_values()[axis])
                    } else {
                        match &action {
                            Action::Discrete(a) => *a as f64,
                            Action::Continuous(a) => f64::from(a[axis - width]),
                            Action::Conditional(_) => unreachable!("portable replay action"),
                        }
                    }
                };
                points.push([coordinate(axes[0]), coordinate(axes[1])]);
            }
            replay.plot = Some(ReplayPlot { key, axes, points });
        }
        let points = &replay.plot.as_ref().unwrap().points;
        let response = Plot::new(("factual-replay-coverage", axes))
            .height((ui.available_height() - 24.).max(220.))
            .x_axis_label(&labels[axes[0]])
            .y_axis_label(&labels[axes[1]])
            .show(ui, |plot| {
                plot.points(
                    Points::new("Factual replay sample", points.clone())
                        .color(GOLD)
                        .radius(4.),
                );
                if let SelectedInput::ProbeRow { sequence, row, .. } = self.selection.input
                    && sequence == entry.sequence
                    && let Some(point) = points.get(row)
                {
                    plot.points(
                        Points::new("Selected", vec![*point])
                            .shape(MarkerShape::Diamond)
                            .color(crate::theme::SELECTED)
                            .radius(6.),
                    );
                }
                plot.pointer_coordinate()
            });
        if response.response.clicked()
            && let Some(cursor) = response.inner
            && let Some((row, _)) = points.iter().enumerate().min_by(|(_, a), (_, b)| {
                let distance = |p: &[f64; 2]| {
                    let screen = response
                        .transform
                        .position_from_point(&egui_plot::PlotPoint::new(p[0], p[1]));
                    screen.distance(response.transform.position_from_point(&cursor))
                };
                distance(a).total_cmp(&distance(b))
            })
        {
            SelectionIntent::Input(SelectedInput::ProbeRow {
                set: probes.id.clone(),
                sequence: entry.sequence,
                row,
            })
            .apply(run, &mut self.selection)?;
            self.workspace_defaults = None;
        }
        self.dynamics_pane.plot_bounds.clear();
        self.dynamics_pane
            .plot_bounds
            .push(*response.transform.frame());
        self.dynamics_pane.replay.actual = Some(entry.revision);
        Ok(())
    }
}
