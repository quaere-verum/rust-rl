//! Small presentation helpers shared by Policy and Value; no inference or family semantics.
use eframe::egui::{self, Color32};
use rl::diagnostics::visualization::{EnvironmentSemantics, ObservationType};

/// Overlay status for a plot that already has a complete frame.
pub(crate) fn complete_frame_status(status: Option<&str>, ready: bool) -> Option<String> {
    let status = status.filter(|status| *status != "Loading…")?;
    Some(if ready {
        status.to_owned()
    } else {
        format!("Showing previous complete frame · {status}")
    })
}

/// Compact precision for statistical summaries; do not print a positive spread as zero.
pub fn statistic_number(value: f64) -> String {
    if value != 0. && (value.abs() < 1e-3 || value.abs() >= 1e5) {
        format!("{value:.2e}")
    } else {
        format!("{value:.3}")
    }
}
/// Original source row identity, never a physical plot coordinate. Constant UI work.
pub fn sample_navigator(ui: &mut egui::Ui, count: usize, selected: usize) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let mut next = selected.min(count - 1);
    ui.horizontal(|ui| {
        if ui
            .add_enabled(next > 0, egui::Button::new("Previous sample"))
            .clicked()
        {
            next -= 1;
        }
        ui.label(format!("Sample {} of {count}", selected + 1));
        if ui
            .add_enabled(next + 1 < count, egui::Button::new("Next sample"))
            .clicked()
        {
            next += 1;
        }
        let mut row = next + 1;
        if ui
            .add(
                egui::DragValue::new(&mut row)
                    .range(1..=count)
                    .prefix("Go to "),
            )
            .changed()
        {
            next = row - 1;
        }
    });
    (next != selected).then_some(next)
}
#[derive(Debug, Clone, Copy)]
pub struct LandscapeLayout {
    pub stacked: bool,
    pub main_width: f32,
    pub selected_width: f32,
    pub plot_height: f32,
}
pub fn layout(width: f32, height: f32, inspector_open: bool) -> LandscapeLayout {
    let stacked = width < 850.;
    let usable = (width - 16.).max(0.);
    LandscapeLayout {
        stacked,
        main_width: if stacked || !inspector_open {
            width
        } else {
            usable * 0.73
        },
        selected_width: if !inspector_open {
            0.
        } else if stacked {
            width
        } else {
            usable * 0.27
        },
        plot_height: if stacked {
            if inspector_open {
                height * 0.5
            } else {
                height - 36.
            }
        } else {
            height
        },
    }
}
pub fn axis_tick(mark: egui_plot::GridMark, _: &std::ops::RangeInclusive<f64>) -> String {
    let value = mark.value;
    if value != 0. && !(1e-4..1e6).contains(&value.abs()) {
        format!("{value:e}")
    } else {
        let decimals = ((-mark.step_size.abs().log10().floor()).max(0.) + 1.).min(10.) as usize;
        egui::emath::format_with_decimals_in_range(value, 0..=decimals)
    }
}

#[cfg(test)]
mod axis_label_tests {
    use super::*;

    #[test]
    fn compact_statistics_preserve_small_nonzero_spreads() {
        assert_eq!(statistic_number(0.145581), "0.146");
        assert_eq!(statistic_number(0.00001), "1.00e-5");
        assert_eq!(statistic_number(0.), "0.000");
    }

    #[test]
    fn binary_grid_labels_are_compact_and_adjacent_ticks_remain_distinct() {
        let label = |value| {
            axis_tick(
                egui_plot::GridMark {
                    value,
                    step_size: 0.0625,
                },
                &(0.75..=1.25),
            )
        };
        let a = label(1.0625);
        let b = label(1.125);
        assert!(a.len() <= 5, "tick label is unnecessarily precise: {a}");
        assert_ne!(a, b);
    }
}
pub fn numeric_bounds(values: impl Iterator<Item = f64>, symmetric: bool) -> [f64; 2] {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values.filter(|v| v.is_finite()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() {
        return [0., 1.];
    }
    if symmetric {
        let m = lo.abs().max(hi.abs()).max(1e-6);
        [-m, m]
    } else {
        [lo, hi.max(lo + 1e-6)]
    }
}
pub fn cell_index(point: [f64; 2], ranges: [[f32; 2]; 2], size: [usize; 2]) -> Option<usize> {
    let mut ix = [0; 2];
    for a in 0..2 {
        if size[a] == 0 {
            return None;
        }
        if size[a] == 1 {
            continue;
        }
        let [lo, hi] = ranges[a].map(f64::from);
        if !point[a].is_finite() || point[a] < lo || point[a] > hi || hi <= lo {
            return None;
        }
        ix[a] = (((point[a] - lo) / (hi - lo) * (size[a] - 1) as f64).round() as usize)
            .min(size[a] - 1);
    }
    Some(ix[1] * size[0] + ix[0])
}
pub fn nearest_point(
    points: impl Iterator<Item = (usize, [f64; 2])>,
    cursor: egui_plot::PlotPoint,
    transform: &egui_plot::PlotTransform,
    radius: f32,
) -> Option<usize> {
    let p = transform.position_from_point(&cursor);
    points
        .map(|(i, q)| (i, transform.position_from_point(&q.into()).distance_sq(p)))
        .filter(|(_, d)| *d <= radius * radius)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}
/// Meaning is explicit: the sign of a value alone never chooses its palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Palette {
    Sequential,
    Value,
    Probability,
    SignedAction,
    Difference,
    Categorical,
}

#[cfg(test)]
mod value_palette_tests {
    use super::*;
    #[test]
    fn negative_value_range_uses_the_full_sequential_scale() {
        let scale = LandscapeScale::new([-2., -0.01], Palette::Value);
        assert_eq!(scale.range, [-2., -0.01]);
        assert_eq!(scale.color(-2.), sequential_color(0.));
        assert_eq!(scale.color(-0.01), sequential_color(1.));
        assert_eq!(scale.color(f64::NAN), Color32::TRANSPARENT);
    }

    #[test]
    fn sequential_scales_have_increasing_luminance() {
        for palette in [Palette::Sequential, Palette::Value, Palette::Probability] {
            let scale = LandscapeScale::new([0., 1.], palette);
            let luma = |value| {
                let c = scale.color(value);
                0.2126 * f64::from(c.r()) + 0.7152 * f64::from(c.g()) + 0.0722 * f64::from(c.b())
            };
            let values: Vec<_> = (0..=32).map(|i| luma(f64::from(i) / 32.)).collect();
            assert!(
                values.windows(2).all(|pair| pair[0] < pair[1]),
                "{palette:?}"
            );
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandscapeScale {
    pub range: [f64; 2],
    pub palette: Palette,
}
impl LandscapeScale {
    pub fn new(range: [f64; 2], palette: Palette) -> Self {
        let range = match palette {
            Palette::Probability => [0., 1.],
            Palette::SignedAction | Palette::Difference => {
                let m = range[0].abs().max(range[1].abs()).max(1e-6);
                [-m, m]
            }
            _ => range,
        };
        Self { range, palette }
    }
    pub fn color(self, v: f64) -> Color32 {
        palette_color(v, self)
    }
}
pub fn color(v: f64, range: [f64; 2]) -> Color32 {
    LandscapeScale::new(range, Palette::Sequential).color(v)
}
fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> Color32 {
    let blend = |i: usize| f32::from(a[i]) + (f32::from(b[i]) - f32::from(a[i])) * t;
    Color32::from_rgb(
        blend(0).round() as u8,
        blend(1).round() as u8,
        blend(2).round() as u8,
    )
}

/// Navy-to-yellow scalar ramp with increasing luminance throughout.
pub(crate) fn sequential_color(fraction: f32) -> Color32 {
    const STOPS: [[u8; 3]; 5] = [
        [12, 34, 72],
        [66, 78, 109],
        [124, 123, 120],
        [187, 174, 106],
        [253, 231, 90],
    ];
    let position = fraction.clamp(0., 1.) * (STOPS.len() - 1) as f32;
    let index = (position as usize).min(STOPS.len() - 2);
    mix(STOPS[index], STOPS[index + 1], position - index as f32)
}

fn palette_color(v: f64, scale: LandscapeScale) -> Color32 {
    let range = scale.range;
    if !v.is_finite() {
        return Color32::TRANSPARENT;
    }
    let t = ((v - range[0]) / (range[1] - range[0])).clamp(0., 1.) as f32;
    if scale.palette == Palette::Categorical {
        const COLORS: [[u8; 3]; 8] = [
            [65, 145, 235],
            [237, 178, 76],
            [91, 200, 173],
            [192, 133, 224],
            [234, 123, 128],
            [94, 194, 218],
            [192, 202, 104],
            [151, 170, 201],
        ];
        let c = COLORS[v.round().max(0.) as usize % COLORS.len()];
        return Color32::from_rgb(c[0], c[1], c[2]);
    }
    if matches!(scale.palette, Palette::SignedAction | Palette::Difference) {
        let zero = crate::theme::SIGNED_ZERO;
        let neutral = [zero.r(), zero.g(), zero.b()];
        if t < 0.5 {
            mix([103, 163, 229], neutral, t * 2.)
        } else {
            mix(neutral, [231, 146, 153], (t - 0.5) * 2.)
        }
    } else if scale.palette == Palette::Value {
        sequential_color(t)
    } else {
        mix([13, 32, 66], [66, 159, 255], t)
    }
}
pub fn color_bar(
    ui: &egui::Ui,
    frame: egui::Rect,
    scale: LandscapeScale,
    label: &str,
) -> egui::Rect {
    let rect = egui::Rect::from_min_size(
        egui::pos2(frame.right() + 12., frame.top()),
        egui::vec2(11., frame.height()),
    );
    for i in 0..128 {
        let y = rect.bottom() - (i + 1) as f32 * rect.height() / 128.;
        ui.painter().rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(rect.left(), y),
                egui::vec2(rect.width(), rect.height() / 128. + 1.),
            ),
            0.,
            scale.color(scale.range[0] + (scale.range[1] - scale.range[0]) * i as f64 / 127.),
        );
    }
    for i in 0..=4 {
        let t = i as f64 / 4.;
        let y = rect.bottom() - t as f32 * rect.height();
        let value = scale.range[0] + t * (scale.range[1] - scale.range[0]);
        let step = (scale.range[1] - scale.range[0]) / 4.;
        let text = if value != 0. && !(1e-3..1e5).contains(&value.abs()) {
            let precision = (value.abs() / step.abs()).log10().ceil().clamp(2., 6.) as usize;
            format!("{value:.precision$e}")
        } else {
            let precision = (1. - step.abs().log10().floor()).clamp(0., 8.) as usize;
            egui::emath::format_with_decimals_in_range(value, 0..=precision)
        };
        ui.painter().text(
            egui::pos2(rect.right() + 5., y),
            egui::Align2::LEFT_CENTER,
            text,
            egui::FontId::proportional(11.),
            crate::theme::MUTED,
        );
    }
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        egui::FontId::proportional(11.),
        crate::theme::MUTED,
    );
    let mut text = egui::epaint::TextShape::new(
        egui::pos2(rect.right() + 47., rect.center().y + galley.size().x * 0.5),
        galley,
        crate::theme::MUTED,
    );
    text.angle = -std::f32::consts::FRAC_PI_2;
    ui.painter().add(text);
    ui.interact(
        rect.expand(5.),
        ui.scope_id().with("landscape-scale"),
        egui::Sense::hover(),
    )
    .on_hover_text(label);
    rect
}
pub fn segment<T: PartialEq>(ui: &mut egui::Ui, value: &mut T, selected: T, label: &str) -> bool {
    ui.selectable_value(value, selected, label).changed()
}
/// A bool-owned popover can contain ComboBoxes: egui's single memory-owned
/// popup slot remains available for their menus instead of replacing the parent.
pub fn popover<R>(
    ui: &mut egui::Ui,
    label: &str,
    body: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    let button = ui.button(label);
    // Conditional status labels can change automatic widget IDs. Keep both the
    // popup and its open state attached to this named control across reflow.
    let id = ui.make_persistent_id(("context-popover", label));
    let mut open = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(id))
        .unwrap_or_else(|| {
            (label == "Details" || label.ends_with("details"))
                && crate::theme::inspection_default(ui)
        });
    if button.clicked() {
        open = !open;
    }
    let result = egui::Popup::from_response(&button)
        .id(id.with("popup"))
        .kind(egui::PopupKind::Menu)
        .open_bool(&mut open)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(body)
        .map(|r| r.inner);
    ui.ctx().data_mut(|d| d.insert_temp(id, open));
    result
}
pub fn observation_label(e: EnvironmentSemantics<'_>, i: usize) -> String {
    e.observation(i)
        .map_or_else(|| format!("obs[{i}]"), |o| o.label.into_owned())
}
pub fn fixed_input(
    ui: &mut egui::Ui,
    e: EnvironmentSemantics<'_>,
    i: usize,
    value: &mut f32,
    id: &str,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(observation_label(e, i));
        if let Some(ObservationType::Categorical { categories }) =
            e.observation(i).and_then(|o| o.kind)
        {
            egui::ComboBox::from_id_salt((id, i))
                .selected_text(
                    categories
                        .iter()
                        .find(|x| x.code == *value)
                        .map_or("?", |x| x.label.as_str()),
                )
                .show_ui(ui, |ui| {
                    for c in categories {
                        changed |= ui.selectable_value(value, c.code, &c.label).changed();
                    }
                });
        } else {
            changed |= ui.add(egui::DragValue::new(value).speed(0.01)).changed();
        }
    });
    changed
}

/// Axis metadata includes declared units; controls keep their short field labels.
pub fn observation_axis_label(e: EnvironmentSemantics<'_>, i: usize) -> String {
    with_unit(
        observation_label(e, i),
        e.observation(i).and_then(|o| o.unit),
    )
}
pub fn with_unit(label: String, unit: Option<&str>) -> String {
    unit.filter(|u| !u.is_empty())
        .map_or_else(|| label.clone(), |u| format!("{label} ({u})"))
}
