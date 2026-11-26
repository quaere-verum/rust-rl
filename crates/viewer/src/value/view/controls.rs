use super::*;
pub fn available_outputs(kind: ValueCriticKind, twin: bool, utility: bool) -> Vec<ValueOutput> {
    if twin {
        return vec![
            ValueOutput::Q1,
            ValueOutput::Q2,
            ValueOutput::Minimum,
            ValueOutput::Disagreement,
        ];
    }
    let mut v = vec![ValueOutput::ExpectedReturn];
    match kind {
        ValueCriticKind::QuantileQ | ValueCriticKind::CategoricalReturnQ => {
            v.push(ValueOutput::Quantile { probability: 0.5 });
            v.push(ValueOutput::Cvar { probability: 0.1 });
            if kind == ValueCriticKind::CategoricalReturnQ {
                v.extend([ValueOutput::Variance, ValueOutput::StandardDeviation]);
            }
        }
        ValueCriticKind::MomentQ => {
            v.extend([
                ValueOutput::RawSecondMoment,
                ValueOutput::Variance,
                ValueOutput::StandardDeviation,
            ]);
            if utility {
                v.push(ValueOutput::Utility);
            }
        }
        _ => {}
    }
    v
}
impl ValueView {
    pub(crate) fn decision_slice(c: &mut ValueSlice) {
        if !c.state_only() {
            c.x = ValueCoordinate::Observation(c.state.x);
            c.y = (c.mode == SliceMode::TwoDimensional)
                .then_some(c.state.y)
                .flatten()
                .map(ValueCoordinate::Observation);
        }
        if c.mode == SliceMode::TwoDimensional && c.y.is_none() {
            c.mode = SliceMode::OneDimensional;
            c.resolution[1] = 1;
        }
        c.conditioning = ActionConditioning::Policy;
    }
    pub(crate) fn sync_decision_controls(&mut self, c: &mut ValueSlice, output: &ValueOutput) {
        if *output == ValueOutput::DecisionLoss {
            if self.decision_controls.is_none() {
                self.decision_controls = Some(DecisionControls {
                    x: c.x,
                    y: c.y,
                    conditioning: c.conditioning.clone(),
                    display: self.display,
                    comparison: self.comparison(),
                    colors: self.colors.mode,
                    mode: c.mode,
                    resolution: c.resolution,
                });
            }
            Self::decision_slice(c);
            self.display = Display::Current;
            self.condition = Some(ComparisonCondition::SameInput);
            self.colors.mode = colors::ColorMode::Adaptive;
        } else if let Some(saved) = self.decision_controls.take() {
            c.x = saved.x;
            c.y = saved.y;
            c.mode = saved.mode;
            c.resolution = saved.resolution;
            c.conditioning = saved.conditioning;
            if let ActionConditioning::Fixed(action) = &c.conditioning
                && c.action_space()
                    .check(action, c.state.anchor_mask())
                    .is_err()
            {
                c.conditioning = ActionConditioning::Policy;
            }
            self.display = saved.display;
            self.condition = Some(saved.comparison);
            self.colors.mode = saved.colors;
        }
    }

    pub(crate) fn controls(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        extra: impl FnOnce(&mut egui::Ui),
    ) {
        if self.colors.provisional(self.display == Display::Difference) {
            ui.weak("Provisional colour scale").on_hover_text(self.colors.message.as_deref().unwrap_or("Calibration is pending or unavailable; using the first available slice provisionally."));
        }
        let Some(mut c) = self.config.clone() else {
            extra(ui);
            return;
        };
        self.sync_decision_controls(&mut c, &self.quantity());
        let before = c.clone();
        let output_before = self.quantity();
        let condition_before = self.comparison();
        let choice_before = self.choice.clone();
        let kind = self
            .request
            .as_ref()
            .map(|r| r.current.kind())
            .or_else(|| self.choice.as_ref().map(|c| c.kind))
            .unwrap_or(ValueCriticKind::StateValue);
        let twin = self.choice.as_ref().is_some_and(|c| c.twin.is_some());
        let utility = self
            .request
            .as_ref()
            .and_then(|r| r.current.declaration())
            .and_then(|d| {
                d.critics
                    .get(self.request.as_ref()?.current.critic_names().first()?)
            })
            .and_then(|c| c.utility.as_ref())
            .is_some_and(|u| !matches!(u, MomentUtility::Unsupported { .. }));
        let mut axes: Vec<_> = (0..c.state.fixed.len())
            .filter(|&i| c.state.numeric(i))
            .map(ValueCoordinate::Observation)
            .collect();
        if kind != ValueCriticKind::StateValue && self.quantity() != ValueOutput::DecisionLoss {
            match e.action_space() {
                ActionSpace::Discrete { .. } => axes.push(ValueCoordinate::DiscreteAction),
                _ => axes.extend((0..e.action_space().dimensions()).map(ValueCoordinate::Action)),
            }
        }
        if kind == ValueCriticKind::StateValue
            && (self.choice.is_some() || self.request.is_some())
            && !c.state_only()
        {
            c.x = ValueCoordinate::Observation(c.state.x);
            c.y = if c.mode == SliceMode::TwoDimensional {
                c.state.y.map(ValueCoordinate::Observation)
            } else {
                None
            };
            c.mode = if c.y.is_some() {
                SliceMode::TwoDimensional
            } else {
                SliceMode::OneDimensional
            };
            c.resolution = if c.y.is_some() { [64, 64] } else { [256, 1] };
        }
        let soft = self
            .request
            .as_ref()
            .is_some_and(|r| r.current.return_target.is_some_and(ReturnTarget::is_soft));
        let label =
            |output: &ValueOutput| -> &'static str { output_label_for_target(soft, output) };
        let mut output = self.quantity();
        let mut outputs = available_outputs(kind, twin, utility);
        if self
            .request
            .as_ref()
            .is_some_and(ValueRequest::decision_loss_available)
        {
            outputs.push(ValueOutput::DecisionLoss);
        }
        if output != ValueOutput::DecisionLoss
            && !outputs
                .iter()
                .any(|o| std::mem::discriminant(o) == std::mem::discriminant(&output))
        {
            output = outputs[0].clone();
        }
        if c.mode == SliceMode::SamplesOnly {
            ui.horizontal(|ui| {
                ui.weak("Recorded samples · no numerical observation axes");
                egui::ComboBox::from_id_salt("value-output")
                    .selected_text(label(&output))
                    .show_ui(ui, |ui| {
                        for o in &outputs {
                            ui.selectable_value(&mut output, o.clone(), label(o));
                        }
                    });
                if let ValueOutput::Quantile { probability } | ValueOutput::Cvar { probability } =
                    &mut output
                {
                    ui.add(
                        egui::DragValue::new(probability)
                            .speed(0.01)
                            .range(0.0001..=1.)
                            .prefix("p = "),
                    );
                }
                if self.choices.len() > 1 {
                    egui::ComboBox::from_id_salt("value-critic")
                        .selected_text(
                            self.choice
                                .as_ref()
                                .map_or("Choose critic", |c| c.twin.as_deref().unwrap_or(&c.name)),
                        )
                        .show_ui(ui, |ui| {
                            for choice in &self.choices {
                                ui.selectable_value(
                                    &mut self.choice,
                                    Some(choice.clone()),
                                    choice.twin.as_deref().unwrap_or(&choice.name),
                                );
                            }
                        });
                }
                ui.checkbox(&mut self.reference_enabled, "Reference");
            });
            if output != self.quantity() {
                self.output = Some(output);
                self.dirty = true;
                self.ready = false;
            }
            if choice_before != self.choice {
                self.current_search = None;
                self.request = None;
                self.output = Some(if self.choice.as_ref().is_some_and(|c| c.twin.is_some()) {
                    ValueOutput::Q1
                } else {
                    ValueOutput::ExpectedReturn
                });
                self.dirty = true;
                self.ready = false;
            }
            return;
        }
        if output != ValueOutput::DecisionLoss
            && !outputs
                .iter()
                .any(|o| std::mem::discriminant(o) == std::mem::discriminant(&output))
        {
            output = outputs[0].clone();
        }
        let plot_settings = |ui: &mut egui::Ui,
                             this: &mut Self,
                             c: &mut ValueSlice,
                             output: &ValueOutput| {
            landscape::popover(ui, "Slice settings", |ui| {
                let height = (ui.ctx().content_rect().height() - 120.).max(180.);
                egui::ScrollArea::vertical().max_height(height).show(ui, |ui| {
                ui.set_max_width(380.);
                ui.strong("Axes and conditioning");
                for (label, axis) in [("X axis", Some(&mut c.x)), ("Y axis", c.y.as_mut())] {
                    if let Some(axis) = axis {
                        ui.label(label);
                        egui::ComboBox::from_id_salt(("value-axis", label))
                            .selected_text(coordinate_label(e, *axis))
                            .show_ui(ui, |ui| {
                                for a in &axes {
                                    ui.selectable_value(axis, *a, coordinate_label(e, *a));
                                }
                            });
                    }
                }

                ui.small(format!(
                    "Evaluated grid: {} x {} · {}",
                    c.resolution[0],
                    c.resolution[1],
                    if this.resolution_explicit[usize::from(c.mode == SliceMode::TwoDimensional)] {
                        "explicit"
                    } else {
                        "automatic admission"
                    }
                ));
                if ui.button("Automatic resolution").clicked() {
                    this.resolution_explicit[usize::from(c.mode == SliceMode::TwoDimensional)] =
                        false;
                    c.resolution = if c.mode == SliceMode::TwoDimensional {
                        [128, 128]
                    } else {
                        [256, 1]
                    };
                    this.dirty = true;
                    this.ready = false;
                }
                ui.horizontal_wrapped(|ui| {
                    for (n, a) in std::iter::once(c.x).chain(c.y).enumerate() {
                        if a == ValueCoordinate::DiscreteAction {
                            continue;
                        }
                        ui.label(format!("{} limits", coordinate_label(e, a)));
                        let range = match a {
                            ValueCoordinate::Observation(i) => &mut c.state.ranges[i],
                            ValueCoordinate::Action(i) => &mut c.action_ranges[i],
                            _ => unreachable!(),
                        };
                        ui.add(egui::DragValue::new(&mut range[0]).speed(0.05));
                        ui.add(egui::DragValue::new(&mut range[1]).speed(0.05));
                        ui.label("Samples");
                        let max = if c.mode == SliceMode::OneDimensional {
                            4096
                        } else {
                            256
                        };
                        if ui
                            .add(egui::DragValue::new(&mut c.resolution[n]).range(2..=max))
                            .changed()
                        {
                            this.resolution_explicit
                                [usize::from(c.mode == SliceMode::TwoDimensional)] = true;
                        }
                    }
                    ui.label("Fixed-input tolerance");
                    ui.add(
                        egui::DragValue::new(&mut c.state.tolerance)
                            .range(0.0..=100.)
                            .speed(0.001),
                    );
                });
                if kind != ValueCriticKind::StateValue && *output != ValueOutput::DecisionLoss {
                    let mut fixed = matches!(c.conditioning, ActionConditioning::Fixed(_));
                    let requires_fixed = this.request.as_ref().is_some_and(|r| {
                        r.current.return_target.is_some_and(ReturnTarget::is_soft)
                            && !r.current.identity().recorded && !c.has_discrete_axis()
                    });
                    ui.add_enabled(!requires_fixed, egui::Checkbox::new(&mut fixed, "Fixed first action"))
                        .on_disabled_hover_text("This critic requires an explicit action for value prediction.");
                    if fixed != matches!(c.conditioning, ActionConditioning::Fixed(_)) {
                        c.conditioning = if fixed {
                            let Some(action) = c.default_first_action() else { return; };
                            ActionConditioning::Fixed(action)
                        } else {
                            ActionConditioning::Policy
                        };
                    }
                    if let ActionConditioning::Fixed(Action::Discrete(action)) = &mut c.conditioning
                        && let ActionSpace::Discrete { count } = e.action_space()
                    {
                        egui::ComboBox::from_id_salt("value-fixed-action")
                            .selected_text(format!("Action {action}"))
                            .show_ui(ui, |ui| {
                                for i in 0..*count {
                                    if c.state.anchor_mask().is_none_or(|m| m[i]) {
                                        ui.selectable_value(action, i, format!("Action {i}"));
                                    }
                                }
                            });
                    }
                }
                let mut condition = this.comparison();
                let controller_admitted = c.state_only()
                    && matches!(c.conditioning, ActionConditioning::Policy)
                    && matches!(
                        *output,
                        ValueOutput::ExpectedReturn | ValueOutput::Q1 | ValueOutput::Q2
                    );
                if !controller_admitted {
                    condition = ComparisonCondition::SameInput;
                }
                ui.add_enabled_ui(controller_admitted, |ui| {
                    ui.selectable_value(
                        &mut condition,
                        ComparisonCondition::SameInput,
                        "Same complete input",
                    );
                    ui.selectable_value(
                        &mut condition,
                        ComparisonCondition::ControllerValues,
                        "Controller values",
                    );
                });
                this.condition = Some(condition);
                if condition == ComparisonCondition::ControllerValues {
                    ui.small("Each critic uses its own controller's first action and continuation; conditioning differs.");
                }
                use super::colors::ColorMode;
                ui.horizontal(|ui| {
                    ui.label("Color scale");
                    for (mode, label) in [
                        (ColorMode::FinalSnapshot, "Final snapshot"),
                        (ColorMode::Adaptive, "Adaptive"),
                        (ColorMode::Manual, "Manual"),
                    ] {
                        if ui
                            .add_enabled_ui(
                                *output != ValueOutput::DecisionLoss || mode == ColorMode::Adaptive,
                                |ui| ui.selectable_value(&mut this.colors.mode, mode, label),
                            )
                            .inner
                            .changed()
                        {
                            if mode == ColorMode::Manual {
                                let _ = this.set_color_limits(this.range.unwrap_or([-1., 1.]));
                            }
                            this.dirty = true;
                        }
                    }
                });
                if this.colors.mode == ColorMode::FinalSnapshot {
                    ui.horizontal_wrapped(|ui| {
                        if let Some(source) = &this.colors.source {
                            ui.label(format!("Fixed at k={}", source.identity().revision));
                        }
                        if ui.button("Refresh latest").clicked() {
                            this.colors.refresh(None);
                            this.dirty = true;
                        }
                        if ui.button("Use selected checkpoint").clicked()
                            && let Some(request) = &this.request
                        {
                            this.colors
                                .refresh(Some(request.current.identity().revision));
                            this.dirty = true;
                        }
                    });
                }
                ui.small(if this.display == Display::Difference {
                    "98th percentile of absolute differences + 10%; centered at zero."
                } else {
                    "2nd–98th percentiles + 10% padding; fixed inputs stay fixed."
                });
                let mut range = this.range.unwrap_or([-1., 1.]);
                ui.horizontal(|ui| {
                    ui.label("Color limits");
                    ui.add_enabled_ui(this.colors.mode == ColorMode::Manual, |ui| {
                        let changed = ui
                            .add(egui::DragValue::new(&mut range[0]).speed(0.01))
                            .changed()
                            | ui.add(egui::DragValue::new(&mut range[1]).speed(0.01))
                                .changed();
                        if changed {
                            let _ = this.set_color_limits(range);
                        }
                    });
                    if ui.button("Fit axes").clicked() {
                        this.initialized = false;
                    }
                });
                });
            });
        };
        let fixed_editor = |ui: &mut egui::Ui, c: &mut ValueSlice, search: &str| {
            for i in 0..c.state.fixed.len() {
                if std::iter::once(c.x)
                    .chain(c.y)
                    .any(|a| a == ValueCoordinate::Observation(i))
                {
                    continue;
                }
                if !landscape::observation_label(e, i)
                    .to_lowercase()
                    .contains(&search.to_lowercase())
                {
                    continue;
                }
                landscape::fixed_input(ui, e, i, &mut c.state.fixed[i], "value-fixed");
            }
            if let ActionConditioning::Fixed(Action::Continuous(a)) = &mut c.conditioning {
                for (i, v) in a.iter_mut().enumerate() {
                    if std::iter::once(c.x)
                        .chain(c.y)
                        .any(|a| a == ValueCoordinate::Action(i))
                    {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        ui.label(coordinate_label(e, ValueCoordinate::Action(i)));
                        ui.add(egui::DragValue::new(v).speed(0.01));
                    });
                }
            }
        };
        ui.horizontal_wrapped(|ui| {
            let mut two = c.mode == SliceMode::TwoDimensional;
            ui.add_enabled_ui(axes.len() > 1, |ui| {
                landscape::segment(ui, &mut two, true, "Map");
            });
            landscape::segment(ui, &mut two, false, "Slice");
            if two != (c.mode == SliceMode::TwoDimensional) {
                self.resolutions[usize::from(c.mode == SliceMode::TwoDimensional)] = c.resolution;
                c.mode = if two {
                    SliceMode::TwoDimensional
                } else {
                    SliceMode::OneDimensional
                };
                c.y = if two {
                    axes.iter().copied().find(|a| *a != c.x)
                } else {
                    None
                };
                c.resolution = self.resolutions[usize::from(two)];
                self.initialized = false;
            }
            ui.label("Statistic");
            egui::ComboBox::from_id_salt("value-output")
                .selected_text(label(&output))
                .show_ui(ui, |ui| {
                    for o in outputs.clone() {
                        let selected =
                            std::mem::discriminant(&output) == std::mem::discriminant(&o);
                        if ui.selectable_label(selected, label(&o)).clicked() {
                            output = o;
                        }
                    }
                });
            if let ValueOutput::Quantile { probability } | ValueOutput::Cvar { probability } =
                &mut output
            {
                ui.add(
                    egui::DragValue::new(probability)
                        .speed(0.01)
                        .range(0.0001..=1.)
                        .prefix("p = "),
                );
            }
            if ui.ctx().content_rect().width() < 1100. {
                let label = match self.display {
                    Display::Current => "Current",
                    Display::Reference => "Reference",
                    Display::Difference => "Difference",
                };
                egui::ComboBox::from_id_salt("value-display-mode")
                    .width(85.)
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.display, Display::Current, "Current");
                        ui.add_enabled_ui(
                            output != ValueOutput::DecisionLoss && self.comparison_available,
                            |ui| {
                                ui.selectable_value(
                                    &mut self.display,
                                    Display::Reference,
                                    "Reference",
                                );
                                ui.add_enabled_ui(self.difference_available, |ui| {
                                    ui.selectable_value(
                                        &mut self.display,
                                        Display::Difference,
                                        "Difference",
                                    );
                                });
                            },
                        );
                    });
            } else {
                landscape::segment(ui, &mut self.display, Display::Current, "Current");
                ui.add_enabled_ui(
                    output != ValueOutput::DecisionLoss && self.comparison_available,
                    |ui| {
                        landscape::segment(ui, &mut self.display, Display::Reference, "Reference");
                        ui.add_enabled_ui(self.difference_available, |ui| {
                            landscape::segment(
                                ui,
                                &mut self.display,
                                Display::Difference,
                                "Difference",
                            );
                        });
                    },
                );
            }
            ui.toggle_value(&mut self.visits_enabled, "Visited states");
            ui.toggle_value(&mut self.reference_enabled, "Reference");
            if self.choices.len() > 1 {
                ui.label("Critic");
                egui::ComboBox::from_id_salt("value-critic")
                    .selected_text(
                        self.choice
                            .as_ref()
                            .map_or("Choose critic", |c| c.twin.as_deref().unwrap_or(&c.name)),
                    )
                    .show_ui(ui, |ui| {
                        for choice in &self.choices {
                            ui.selectable_value(
                                &mut self.choice,
                                Some(choice.clone()),
                                choice.twin.as_deref().unwrap_or(&choice.name),
                            );
                        }
                    });
            }
            plot_settings(ui, self, &mut c, &output);
            landscape::popover(ui, "Fixed inputs", |ui| {
                ui.set_width(300.);
                if c.state.fixed.len() + e.action_space().dimensions() > 8 {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.search)
                            .char_limit(128)
                            .hint_text("Search inputs"),
                    );
                }
                egui::ScrollArea::vertical()
                    .id_salt("value-fixed-scroll")
                    .max_height(280.)
                    .show(ui, |ui| fixed_editor(ui, &mut c, &self.search));
            });
            extra(ui);
        });
        if c.y == Some(c.x) {
            c.y = axes.iter().copied().find(|a| *a != c.x);
        }
        if let Err(err) = c.validate(e) {
            self.status = Some(err.to_string());
            return;
        }
        for i in 0..c.state.ranges.len() {
            if c.state.ranges[i] != before.state.ranges[i] {
                c.state.range_sources[i] = crate::policy::slice::RangeSource::UserConfigured;
            }
        }
        if c.x != before.x
            || c.y != before.y
            || c.state.ranges != before.state.ranges
            || c.action_ranges != before.action_ranges
        {
            self.initialized = false;
        }
        let fixed_changed =
            c.state.fixed != before.state.fixed || c.conditioning != before.conditioning;
        self.output = Some(output.clone());
        self.sync_decision_controls(&mut c, &output);
        self.config = Some(c.clone());
        if before != c
            || output_before != output
            || condition_before != self.comparison()
            || choice_before != self.choice
        {
            self.dirty = true;
            self.ready = false;
        }
        if fixed_changed {
            let mut input = match &self.pin {
                Some(ValuePin::State(o)) | Some(ValuePin::Input { observation: o, .. }) => {
                    o.clone()
                }
                _ => c.anchor(),
            };
            for i in 0..input.dense_values().len() {
                if !std::iter::once(c.x)
                    .chain(c.y)
                    .any(|a| a == ValueCoordinate::Observation(i))
                {
                    input.as_dense_mut().expect("dense diagnostic input").values[i] =
                        c.state.fixed[i];
                }
            }
            let action = match &c.conditioning {
                ActionConditioning::Fixed(a) => Some(a.clone()),
                _ => None,
            };
            let _ = self.pin_input(input, action);
        }
        if choice_before != self.choice {
            self.current_search = None;
            self.request = None;
            self.output = Some(if self.choice.as_ref().is_some_and(|c| c.twin.is_some()) {
                ValueOutput::Q1
            } else {
                ValueOutput::ExpectedReturn
            });
        }
    }
}
