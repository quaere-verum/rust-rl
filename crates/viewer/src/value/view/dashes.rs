//! Value-only dashed strokes: clip and admit with the final plot transform.
use eframe::egui::{self, Color32, Shape};
use egui_plot::{Line, PlotBounds, PlotGeometry, PlotItem, PlotItemBase, PlotPoint, PlotTransform};
use std::{cell::Cell, ops::RangeInclusive, rc::Rc};
const DASH: f32 = 5.;
const GAP: f32 = DASH * (std::f32::consts::GOLDEN_RATIO - 1.);
const SHAPE_BYTES: usize =
    2 * std::mem::size_of::<Shape>() + 2 * std::mem::size_of::<egui::epaint::ClippedShape>();
#[derive(Default)]
pub(super) struct DashAdmission {
    remaining: Cell<usize>,
    pub rejected: Cell<bool>,
}
impl DashAdmission {
    pub fn reset(&self, remaining: usize) {
        self.remaining.set(remaining);
        self.rejected.set(false);
    }
    fn admit(&self, bytes: usize) -> bool {
        if self.rejected.get() || bytes > self.remaining.get() {
            self.rejected.set(true);
            false
        } else {
            self.remaining.set(self.remaining.get() - bytes);
            true
        }
    }
}
pub(super) struct DashedLine {
    line: Line<'static>,
    vertical: Option<f64>,
    width: f32,
    admission: Rc<DashAdmission>,
}
impl DashedLine {
    pub fn curve(line: Line<'static>, width: f32, admission: Rc<DashAdmission>) -> Self {
        Self {
            line,
            vertical: None,
            width,
            admission,
        }
    }
    pub fn vertical(
        name: &str,
        x: f64,
        color: Color32,
        width: f32,
        admission: Rc<DashAdmission>,
    ) -> Self {
        Self {
            line: Line::new(name, Vec::<[f64; 2]>::new()).color(color),
            vertical: Some(x),
            width,
            admission,
        }
    }
    /// No allocations: both passes use exactly the final transform supplied by PlotItem::shapes.
    fn segments(
        &self,
        ui: &egui::Ui,
        transform: &PlotTransform,
        mut visit: impl FnMut([egui::Pos2; 2]),
    ) {
        let clip = ui.clip_rect().intersect(*transform.frame());
        if !clip.is_positive() {
            return;
        }
        if let Some(x) = self.vertical {
            let x = transform
                .position_from_point(&PlotPoint::new(x, transform.bounds().min()[1]))
                .x;
            if x.is_finite() && clip.x_range().contains(x) {
                visit([egui::pos2(x, clip.top()), egui::pos2(x, clip.bottom())]);
            }
        } else if let PlotGeometry::Points(points) = self.line.geometry() {
            for pair in points.array_windows::<2>() {
                if let Some(segment) = clipped(*pair, clip, transform) {
                    visit(segment);
                }
            }
        }
    }
}
// Clip in f64 plot coordinates before conversion to f32 screen positions: enormous
// off-range values must not create enormous dash vectors or overflow screen math.
fn clipped(
    points: [PlotPoint; 2],
    clip: egui::Rect,
    transform: &PlotTransform,
) -> Option<[egui::Pos2; 2]> {
    let a = transform.value_from_position(clip.left_bottom());
    let b = transform.value_from_position(clip.right_top());
    let low = [a.x.min(b.x), a.y.min(b.y)];
    let high = [a.x.max(b.x), a.y.max(b.y)];
    let start = [points[0].x, points[0].y];
    let end = [points[1].x, points[1].y];
    let mut first = 0f64;
    let mut last = 1f64;
    let mut first_plane = None;
    let mut last_plane = None;
    for axis in 0..2 {
        let delta = end[axis] - start[axis];
        if delta == 0. {
            if start[axis] < low[axis] || start[axis] > high[axis] {
                return None;
            }
        } else {
            let l = (low[axis] - start[axis]) / delta;
            let h = (high[axis] - start[axis]) / delta;
            if l.min(h) > first {
                first = l.min(h);
                first_plane = Some((axis, if delta > 0. { low[axis] } else { high[axis] }));
            }
            if l.max(h) < last {
                last = l.max(h);
                last_plane = Some((axis, if delta > 0. { high[axis] } else { low[axis] }));
            }
        }
    }
    if first > last {
        return None;
    }
    let at = |t: f64, plane: Option<(usize, f64)>| {
        let mut coordinates = [
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        ];
        // Preserve the exact intersected face even if enormous opposite endpoints
        // make both intersection parameters round to the same f64 value.
        if let Some((axis, value)) = plane {
            coordinates[axis] = value;
        }
        let point = PlotPoint::new(
            coordinates[0].clamp(low[0], high[0]),
            coordinates[1].clamp(low[1], high[1]),
        );
        transform
            .position_from_point(&point)
            .clamp(clip.min, clip.max)
    };
    let result = [at(first, first_plane), at(last, last_plane)];
    result.iter().all(|p| p.is_finite()).then_some(result)
}
impl PlotItem for DashedLine {
    fn shapes(&self, ui: &egui::Ui, transform: &PlotTransform, shapes: &mut Vec<Shape>) {
        let mut count = 0usize;
        self.segments(ui, transform, |[a, b]| {
            // Each source segment can split a dash at both ends. Capacity growth
            // in the plot and clipped output vectors is included per shape.
            count = count
                .saturating_add((a.distance(b) / DASH).ceil() as usize)
                .saturating_add(2);
        });
        if !self.admission.admit(count.saturating_mul(SHAPE_BYTES)) {
            return;
        }
        let mut previous = None;
        let mut drawing = true;
        let mut remaining = DASH;
        let stroke = egui::Stroke::new(
            self.width * if self.highlighted() { 2. } else { 1. },
            self.color(),
        );
        self.segments(ui, transform, |[a, b]| {
            if previous != Some(a) {
                drawing = true;
                remaining = DASH;
            }
            let length = a.distance(b);
            let mut distance = 0.;
            while distance < length {
                let step = remaining.min(length - distance);
                if drawing && step > 0. {
                    shapes.push(Shape::line_segment(
                        [
                            a + (b - a) * (distance / length),
                            a + (b - a) * ((distance + step) / length),
                        ],
                        stroke,
                    ));
                }
                distance += step;
                remaining -= step;
                if remaining <= f32::EPSILON {
                    drawing = !drawing;
                    remaining = if drawing { DASH } else { GAP };
                }
            }
            previous = Some(b);
        });
    }
    fn initialize(&mut self, x_range: RangeInclusive<f64>) {
        self.line.initialize(x_range);
    }
    fn color(&self) -> Color32 {
        PlotItem::color(&self.line)
    }
    fn geometry(&self) -> PlotGeometry<'_> {
        if self.vertical.is_some() {
            PlotGeometry::None
        } else {
            self.line.geometry()
        }
    }
    fn bounds(&self) -> PlotBounds {
        if let Some(x) = self.vertical {
            PlotBounds::from_min_max([x, f64::INFINITY], [x, f64::NEG_INFINITY])
        } else {
            self.line.bounds()
        }
    }
    fn base(&self) -> &PlotItemBase {
        self.line.base()
    }
    fn base_mut(&mut self) -> &mut PlotItemBase {
        self.line.base_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn paint(points: Vec<[f64; 2]>, bounds: PlotBounds, budget: usize) -> (bool, usize) {
        let ctx = egui::Context::default();
        let gate = Rc::new(DashAdmission::default());
        gate.reset(budget);
        let input_len = points.len();
        let line = DashedLine::curve(
            Line::new("reference", points.clone()).color(Color32::YELLOW),
            1.,
            gate.clone(),
        );
        assert!(matches!(line.geometry(),PlotGeometry::Points(p) if p.len()==input_len));
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(600., 400.),
                )),
                ..Default::default()
            },
            |ui| {
                egui_plot::Plot::new("bounded-dash-test")
                    .height(300.)
                    .show(ui, |p| {
                        p.set_plot_bounds(bounds);
                        p.add(DashedLine::curve(
                            Line::new("reference", points.clone()).color(Color32::YELLOW),
                            1.,
                            gate.clone(),
                        ));
                    });
            },
        );
        output.textures_delta.clear();
        fn strokes(shapes: impl Iterator<Item = Shape>) -> usize {
            shapes
                .map(|s| match s {
                    Shape::LineSegment { stroke, .. } if stroke.color == Color32::YELLOW => 1,
                    Shape::Vec(v) => strokes(v.into_iter()),
                    _ => 0,
                })
                .sum()
        }
        (
            gate.rejected.get(),
            strokes(output.shapes.into_iter().map(|s| s.shape)),
        )
    }
    #[test]
    fn off_range_curve_and_zoomed_cdf_clip_before_dash_expansion() {
        for (points, bounds) in [
            (
                vec![[-1., -1e20], [1., 1e20]],
                PlotBounds::from_min_max([-1., -1.], [1., 1.]),
            ),
            (
                vec![[0., 0.], [0., 0.5], [1., 0.5], [1., 1.]],
                PlotBounds::from_min_max([-1e-7, 0.25], [1e-7, 0.25000001]),
            ),
        ] {
            let (rejected, strokes) = paint(points, bounds, 64 * 1024);
            assert!(!rejected);
            assert!(
                (1..100).contains(&strokes),
                "visible dashed strokes={strokes}"
            );
        }
    }
    #[test]
    fn excessive_visible_zigzags_are_rejected_before_any_dash_is_created() {
        let points = (0..4096)
            .map(|i| [i as f64 / 4095., if i % 2 == 0 { -1. } else { 1. }])
            .collect();
        let (rejected, strokes) = paint(
            points,
            PlotBounds::from_min_max([0., -1.], [1., 1.]),
            64 * 1024,
        );
        assert!(rejected);
        assert_eq!(strokes, 0);
    }
}
