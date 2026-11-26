//! Shared scientific presentation tokens. No data or model semantics live here.
use eframe::egui::{self, Color32, FontFamily, FontId, Stroke, TextStyle};

pub const BG: Color32 = Color32::from_rgb(11, 17, 26);
pub const SURFACE: Color32 = Color32::from_rgb(17, 26, 39);
pub const INSET: Color32 = Color32::from_rgb(11, 19, 32);
pub const TEXT: Color32 = Color32::from_rgb(228, 235, 244);
pub const MUTED: Color32 = Color32::from_rgb(152, 169, 190);
pub const CURRENT: Color32 = Color32::from_rgb(55, 224, 208);
pub const REFERENCE: Color32 = Color32::from_rgb(255, 193, 55);
pub const EMPIRICAL: Color32 = Color32::from_rgb(185, 151, 238);
pub const SELECTED: Color32 = Color32::from_rgb(236, 242, 250);
pub const WARNING: Color32 = Color32::from_rgb(240, 180, 90);
pub const GRID: Color32 = Color32::from_rgb(34, 46, 63);
/// Opaque neutral for a measured signed zero; missing values remain transparent.
pub const SIGNED_ZERO: Color32 = Color32::from_rgb(52, 64, 83);

/// Apply once per context; an explicit preference also prevents an OS theme reset.
pub fn install(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();
    for (kind, size, family) in [
        (TextStyle::Body, 15., FontFamily::Proportional),
        (TextStyle::Button, 14., FontFamily::Proportional),
        (TextStyle::Small, 12.5, FontFamily::Proportional),
        (TextStyle::Heading, 18., FontFamily::Proportional),
        (TextStyle::Monospace, 13., FontFamily::Monospace),
    ] {
        style.text_styles.insert(kind, FontId::new(size, family));
    }
    style.spacing.item_spacing = egui::vec2(8., 5.);
    style.spacing.button_padding = egui::vec2(8., 4.);
    style.spacing.interact_size.y = 26.;
    style.animation_time = 0.12;
    let visuals = &mut style.visuals;
    visuals.override_text_color = Some(TEXT);
    visuals.weak_text_color = Some(MUTED);
    visuals.panel_fill = BG;
    visuals.window_fill = SURFACE;
    visuals.extreme_bg_color = INSET;
    visuals.faint_bg_color = SURFACE;
    visuals.hyperlink_color = CURRENT;
    visuals.selection.bg_fill = Color32::from_rgb(31, 79, 91);
    visuals.selection.stroke = Stroke::new(1., CURRENT);
    visuals.window_stroke = Stroke::new(1., GRID);
    visuals.widgets.noninteractive.bg_fill = SURFACE;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(0.6, GRID);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1., MUTED);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(28, 42, 59);
    visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(28, 42, 59);
    visuals.widgets.inactive.bg_stroke = Stroke::NONE;
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(38, 61, 78);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(38, 61, 78);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1., CURRENT);
    visuals.widgets.active.bg_fill = Color32::from_rgb(31, 79, 91);
    visuals.widgets.active.weak_bg_fill = Color32::from_rgb(31, 79, 91);
    visuals.widgets.active.bg_stroke = Stroke::new(1., CURRENT);
    ctx.set_style_of(egui::Theme::Dark, style);
}

pub fn panel_frame() -> egui::Frame {
    egui::Frame::new()
        .inner_margin(12)
        .fill(SURFACE)
        .stroke(Stroke::new(1., Color32::from_rgb(55, 75, 98)))
        .corner_radius(6)
}

/// Native acceptance can start an existing inspector expanded without simulating
/// a hover. Ordinary application defaults remain collapsed.
pub(crate) fn inspection_default(ui: &egui::Ui) -> bool {
    ui.ctx()
        .data(|data| data.get_temp::<bool>(egui::Id::unique("native-inspection-default")))
        .unwrap_or(false)
}
