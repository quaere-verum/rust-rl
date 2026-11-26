//! Frozen November 4 plot-input checks and independent full-render equivalence.
use super::*;
use egui_plot::{PlotGeometry, PlotItem};

mod original {
    use super::*;
    pub(super) fn ellipse_shape(
        gram: [[f64; 2]; 2],
        position: [f64; 2],
        scales: [f64; 2],
        multiplier: f64,
    ) -> Vec<[f64; 2]> {
        let [a, b] = gram;
        let angle = (2. * a[1]).atan2(a[0] - b[1]) / 2.;
        let trace = a[0] + b[1];
        let discriminant = ((a[0] - b[1]).powi(2) + 4. * a[1] * a[1]).sqrt();
        let eig1 = (0.5 * (trace + discriminant)).max(1e-12);
        let eig2 = (0.5 * (trace - discriminant)).max(1e-12);
        let long = (multiplier / eig2.sqrt()).min(0.4);
        let short = (multiplier / eig1.sqrt()).min(0.4);
        (0..=32)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 32.;
                let u = short * t.cos();
                let v = long * t.sin();
                [
                    position[0] + scales[0] * (angle.cos() * u - angle.sin() * v),
                    position[1] + scales[1] * (angle.sin() * u + angle.cos() * v),
                ]
            })
            .collect()
    }
    pub(super) fn show_ellipse_field(
        ui: &mut egui::Ui,
        frame: &network::Frame,
        ranges: &[[f64; 2]; 2],
        axes: [String; 2],
    ) {
        let (dx, dy) = (
            (ranges[0][1] - ranges[0][0]) / 11.,
            (ranges[1][1] - ranges[1][0]) / 7.,
        );
        let log_strengths = frame
            .grid
            .iter()
            .map(|g| (g.strength + 1e-12).log10())
            .collect::<Vec<_>>();
        let lo = log_strengths.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = log_strengths
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_sensitivity = frame
            .grid
            .iter()
            .map(|g| g.strength)
            .fold(0., f64::max)
            .max(1e-12);
        let height = network_plot_height(ui, 188.);
        ui.horizontal_top(|ui| {
            EguiPlot::new("network-local-geometry")
                .width((ui.available_width() - 70.).max(120.))
                .height(height)
                .show_grid(false)
                .x_axis_label(&axes[0])
                .y_axis_label(&axes[1])
                .show(ui, |plot| {
                    plot.set_plot_bounds(PlotBounds::from_min_max(
                        [ranges[0][0] - dx, ranges[1][0] - dy],
                        [ranges[0][1] + dx, ranges[1][1] + dy],
                    ));
                    for (index, g) in frame.grid.iter().enumerate() {
                        let strength = if hi > lo {
                            (log_strengths[index] - lo) / (hi - lo)
                        } else {
                            0.5
                        };
                        let color = ellipse_color(strength);
                        let shape =
                            ellipse_shape(g.gram, g.position, [dx, dy], 0.15 * max_sensitivity);
                        plot.line(
                            PlotLine::new(format!("ellipse-{index}"), shape)
                                .color(color)
                                .width(1.5),
                        );
                    }
                    plot.points(
                        PlotPoints::new("A", vec![frame.slice_anchor])
                            .radius(4.5)
                            .filled(false)
                            .color(crate::theme::SELECTED),
                    );
                    plot.points(
                        PlotPoints::new("B", vec![frame.slice_endpoint])
                            .radius(5.)
                            .shape(egui_plot::MarkerShape::Diamond)
                            .filled(false)
                            .color(crate::theme::REFERENCE),
                    );
                    plot.points(
                        PlotPoints::new(
                            if frame.position_on_slice {
                                "xₜ"
                            } else {
                                "xₜ projected"
                            },
                            vec![frame.slice_position],
                        )
                        .radius(4.)
                        .shape(egui_plot::MarkerShape::Cross)
                        .color(crate::theme::CURRENT),
                    );
                    for (name, position, label, color) in [
                        (
                            "geometry-a-label",
                            frame.slice_anchor,
                            "A",
                            crate::theme::SELECTED,
                        ),
                        (
                            "geometry-b-label",
                            frame.slice_endpoint,
                            "B",
                            crate::theme::REFERENCE,
                        ),
                        (
                            "geometry-x-label",
                            frame.slice_position,
                            if frame.position_on_slice {
                                "x_t"
                            } else {
                                "x_t*"
                            },
                            crate::theme::CURRENT,
                        ),
                    ] {
                        plot.text(
                            egui_plot::Text::new(name, position.into(), label)
                                .anchor(egui::Align2::LEFT_BOTTOM)
                                .color(color),
                        );
                    }
                });
            ui.vertical(|ui| {
                ui.set_min_width(58.);
                ui.small("log s")
                    .on_hover_text("Log sensitivity, scaled within the displayed field.");
                let (bar, _) = ui.allocate_exact_size(
                    egui::vec2(12., (height - 34.).max(60.)),
                    egui::Sense::hover(),
                );
                for i in 0..100 {
                    let y = bar.bottom() - (i + 1) as f32 * bar.height() / 100.;
                    let strength = if hi > lo { i as f64 / 99. } else { 0.5 };
                    let color = ellipse_color(strength);
                    ui.painter().rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(bar.left(), y),
                            egui::vec2(bar.width(), bar.height() / 100. + 1.),
                        ),
                        0.,
                        color,
                    );
                }
                for (value, y) in [(hi, bar.top()), (lo, bar.bottom())] {
                    ui.painter().text(
                        egui::pos2(bar.right() + 4., y),
                        egui::Align2::LEFT_CENTER,
                        format!("{value:.2}"),
                        egui::FontId::proportional(11.),
                        crate::theme::MUTED,
                    );
                }
            });
        });
    }
    pub(super) fn show_paths(ui: &mut egui::Ui, frame: &network::Frame) {
        if frame.paths.is_empty() {
            ui.weak("PCA unavailable: the fixed probe population has no usable variance.");
            return;
        }
        let min_width = if ui.ctx().content_rect().height() < 720. {
            105.
        } else {
            125.
        };
        let count = (((ui.available_width() + ui.spacing().item_spacing.x)
            / (min_width + ui.spacing().item_spacing.x))
            .floor()
            .max(1.) as usize)
            .min(frame.paths.len());
        for (row_index, paths) in frame.paths.chunks(count).enumerate() {
            ui.columns(count, |columns| {
            for (column, path) in paths.iter().enumerate() {
                let index = row_index * count + column;
                let ui = &mut columns[column];
                ui.strong(&path.layer).on_hover_text(format!(
                    "Independent fixed-reference PCA plane. Reference variance: {:.2}%. Projected path-step energy fidelity: {:.2}%. Exact values and definitions are also in Details.",
                    100. * path.variance,
                    100. * path.projected_energy
                ));
                EguiPlot::new(("network-path", index))
                    .height(network_plot_height(ui, 140.))
                    .show_grid(false)
                    .grid_spacing(30.0..=60.0)
                    .y_grid_spacer(egui_plot::log_grid_spacer(2))
                    .x_axis_label("PC 1")
                    .y_axis_label("PC 2")
                    .x_axis_formatter(crate::landscape::axis_tick)
                    .y_axis_formatter(crate::landscape::axis_tick)
                    .show(ui, |plot| {
                        for (i, pair) in path.points.windows(2).enumerate() {
                            let t = i as f64 / (path.points.len() - 1) as f64;
                            let color =
                                crate::theme::CURRENT.gamma_multiply((0.5 + 0.5 * t) as f32);
                            plot.line(
                                PlotLine::new(format!("segment-{index}-{i}"), pair.to_vec())
                                    .color(color)
                                    .width(2.),
                            );
                        }
                        plot.points(
                            PlotPoints::new("A", vec![path.points[0]])
                                .radius(4.)
                                .filled(false)
                                .color(crate::theme::SELECTED),
                        );
                        plot.points(
                            PlotPoints::new("B", vec![*path.points.last().unwrap()])
                                .radius(4.)
                                .shape(egui_plot::MarkerShape::Diamond)
                                .filled(false)
                                .color(crate::theme::SELECTED),
                        );
                        plot.text(egui_plot::Text::new("endpoint-a-label", path.points[0].into(), "A")
                            .anchor(egui::Align2::LEFT_BOTTOM).color(crate::theme::SELECTED));
                        plot.text(egui_plot::Text::new("endpoint-b-label", (*path.points.last().unwrap()).into(), "B")
                            .anchor(egui::Align2::RIGHT_TOP).color(crate::theme::SELECTED));
                    });
                ui.small(format!("variance {:.3}", path.variance));
            }
        });
            ui.add_space(6.);
        }
    }
}

fn glyphs() -> Vec<network::Glyph> {
    [
        ([[4., 1.25], [1.25, 2.]], [-0.3, 0.7], 2.75),
        ([[1e8, -7.], [-7., 1e-6]], [12., -3.], 1e4),
        ([[0., -0.], [-0., 0.]], [-0., 0.], 0.),
        ([[2., 2.], [2., 2.]], [1e-5, -1e-8], 0.5),
        ([[1., -0.], [-0., 1.]], [0., -0.], 1.),
    ]
    .into_iter()
    .map(|(gram, position, strength)| network::Glyph {
        gram,
        position,
        strength,
    })
    .collect()
}

fn geometry_bits(item: &dyn PlotItem) -> Vec<[u64; 2]> {
    let PlotGeometry::Points(points) = item.geometry() else {
        panic!("expected point geometry")
    };
    points
        .iter()
        .map(|p| [p.x.to_bits(), p.y.to_bits()])
        .collect()
}

// A changed endpoint, closed-circle normalization, clamp, scale or rotated axis
// must fail even when the resulting pixels round to the same screen position.
#[test]
fn every_ellipse_coordinate_bit_matches_the_frozen_original() {
    let grid = glyphs();
    for scales in [[1., 1.], [0.13, 9.7], [-3., 0.], [1e-9, 1e5]] {
        for multiplier in [0., 0.037, 12.] {
            let actual = ellipse_coordinates(&grid, scales, multiplier);
            assert_eq!(actual.len(), grid.len() * 33);
            for (glyph, points) in grid.iter().zip(actual.chunks_exact(33)) {
                let expected =
                    original::ellipse_shape(glyph.gram, glyph.position, scales, multiplier);
                for (index, (actual, expected)) in points.iter().zip(expected).enumerate() {
                    assert_eq!(
                        [actual.x.to_bits(), actual.y.to_bits()],
                        expected.map(f64::to_bits),
                        "point {index}, scales {scales:?}, multiplier {multiplier}"
                    );
                }
            }
        }
    }
    // Analytical circles retain their axes and the original nonzero last sine.
    let grid = [network::Glyph {
        gram: [[1., 0.], [0., 1.]],
        position: [0., 0.],
        strength: 1.,
    }];
    let points = ellipse_coordinates(&grid, [1., 1.], 0.2);
    assert_eq!([points[0].x, points[0].y], [0.2, 0.]);
    assert_eq!(points[8].y, 0.2);
    assert_eq!(points[16].x, -0.2);
    assert_ne!(points[32].y.to_bits(), points[0].y.to_bits());
}

fn same_line(actual: &PlotLine<'_>, expected: &PlotLine<'_>) {
    assert_eq!(actual.base(), expected.base());
    assert_eq!(actual.name(), expected.name());
    assert_eq!(actual.id(), expected.id());
    assert_eq!(actual.color(), expected.color());
    assert_eq!(actual.allow_hover(), expected.allow_hover());
    assert_eq!(geometry_bits(actual), geometry_bits(expected));
    // Width and other private Line style fields are observable in these shapes.
    let transform = egui_plot::PlotTransform::new(
        egui::Rect::from_min_size(egui::pos2(0., 0.), egui::vec2(600., 400.)),
        PlotBounds::from_min_max([-20., -20.], [20., 20.]),
        [false, false],
    );
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(Default::default(), |ui| {
        let mut a = Vec::new();
        let mut b = Vec::new();
        actual.shapes(ui, &transform, &mut a);
        expected.shapes(ui, &transform, &mut b);
        assert_eq!(a, b);
        for p in [
            egui::pos2(0., 0.),
            egui::pos2(280., 210.),
            egui::pos2(600., 400.),
        ] {
            let a = actual
                .find_closest(p, &transform)
                .map(|v| (v.index, v.dist_sq.to_bits()));
            let b = expected
                .find_closest(p, &transform)
                .map(|v| (v.index, v.dist_sq.to_bits()));
            assert_eq!(a, b);
        }
    });
    output.textures_delta.clear();
}

#[test]
fn ellipse_items_keep_complete_identity_style_and_order() {
    let grid = glyphs();
    let logs: Vec<_> = grid.iter().map(|g| (g.strength + 1e-12).log10()).collect();
    for (lo, hi) in [(-12., 4.), (1., 1.)] {
        let coordinates = ellipse_coordinates(&grid, [0.3, 1.7], 0.09);
        let actual: Vec<_> = ellipse_lines(&coordinates, &logs, lo, hi).collect();
        assert_eq!(actual.len(), grid.len());
        for (index, (actual, g)) in actual.iter().zip(&grid).enumerate() {
            let strength = if hi > lo {
                (logs[index] - lo) / (hi - lo)
            } else {
                0.5
            };
            let color = ellipse_color(strength);
            let expected = PlotLine::new(
                format!("ellipse-{index}"),
                original::ellipse_shape(g.gram, g.position, [0.3, 1.7], 0.09),
            )
            .color(color)
            .width(1.5);
            same_line(actual, &expected);
        }
    }
}

#[test]
fn geometry_sensitivity_palette_remains_visible_and_ordered() {
    let luminance = |color: egui::Color32| {
        0.2126 * f64::from(color.r())
            + 0.7152 * f64::from(color.g())
            + 0.0722 * f64::from(color.b())
    };
    let low = luminance(ellipse_color(0.));
    let middle = luminance(ellipse_color(0.5));
    let high = luminance(ellipse_color(1.));
    assert!(low >= 70., "low sensitivity must remain visible: {low}");
    assert!(
        low < middle && middle < high && high - low >= 80.,
        "sensitivity needs an ordered scale: {low}, {middle}, {high}"
    );
}

fn paths() -> Vec<network::Path> {
    [
        vec![[-0., 0.], [0.25, -0.5], [0.25, -0.5], [1., 2.]],
        vec![[12.3, -4.5], [-0., -0.]],
        (0..33)
            .map(|i| [i as f64 / 32., (i as f64 * 0.13).sin()])
            .collect(),
        vec![[0.125, -0.]],
        vec![[9., -3.], [-2., 1.], [0., -0.]],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, points)| network::Path {
        layer: format!("layer-{i}"),
        points,
        variance: 0.75,
        projected_energy: 0.8,
    })
    .collect()
}

#[test]
fn path_items_keep_each_segment_endpoint_identity_color_and_row_order() {
    let paths = paths();
    for columns in [1, 2, 3] {
        for (row_index, row) in paths.chunks(columns).enumerate() {
            for (column, path) in row.iter().enumerate() {
                let index = row_index * columns + column;
                let coordinates = path_coordinates(&path.points);
                let actual: Vec<_> = path_segments(index, &coordinates).collect();
                assert_eq!(actual.len(), path.points.len() - 1);
                if path.points.len() == 1 {
                    assert_eq!(coordinates.capacity(), 0);
                }
                for (i, (actual, pair)) in actual.iter().zip(path.points.windows(2)).enumerate() {
                    let t = i as f64 / (path.points.len() - 1) as f64;
                    let expected = PlotLine::new(format!("segment-{index}-{i}"), pair.to_vec())
                        .color(crate::theme::CURRENT.gamma_multiply((0.5 + 0.5 * t) as f32))
                        .width(2.);
                    same_line(actual, &expected);
                }
            }
        }
    }
}

fn frame() -> network::Frame {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{diagnostics::network_geometry::NumericalMlp, nn::network::Mlp};
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = NumericalMlp::from_model(&Mlp::new(2, &[4], 1, &mut rng).unwrap()).unwrap();
    let mut frame = network::prepare(
        &net,
        &network::Request {
            network: "plot".into(),
            anchor: vec![0., 0.],
            endpoint: vec![1., 1.],
            probes: vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
            axes: vec![0, 1],
            axis_scales: vec![1., 1.],
            ranges: [[-1., 1.], [-1., 1.]],
            direction_angle: 0.4,
            t: 1.,
            radius: 0.2,
            action_mask: None,
            layer: 1,
            grid: [2, 2],
            previous_basis: vec![],
        },
    )
    .unwrap();
    frame.grid = glyphs();
    frame.paths = paths();
    frame.slice_anchor = [-0.3, 0.7];
    frame
}

// Includes all markers, headings, plot IDs, plot layout and line widths. The
// independently allocated legacy path renderer uses the same presentation
// settings, including responsive plot height, so this still tests exact
// optimized-versus-original geometry.
// Separate contexts replay pointers so tooltip identities are covered.
#[test]
fn complete_ellipse_and_path_paints_match_the_frozen_original() {
    let frame = frame();
    for width in [580., 900., 1500.] {
        let old = egui::Context::default();
        let new = egui::Context::default();
        for pointer in [
            None,
            Some(egui::pos2(170., 90.)),
            Some(egui::pos2(280., 310.)),
            Some(egui::pos2(100., 510.)),
        ] {
            let paint = |ctx: &egui::Context, frozen| {
                let mut output = None;
                for _ in 0..3 {
                    let mut painted = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(width, 1200.),
                            )),
                            time: Some(1.),
                            events: pointer.map(egui::Event::PointerMoved).into_iter().collect(),
                            ..Default::default()
                        },
                        |ui| {
                            if frozen {
                                original::show_ellipse_field(
                                    ui,
                                    &frame,
                                    &[[-2., 14.], [-5., 4.]],
                                    ["x".into(), "y".into()],
                                );
                                original::show_paths(ui, &frame);
                            } else {
                                show_ellipse_field(
                                    ui,
                                    &frame,
                                    &[[-2., 14.], [-5., 4.]],
                                    ["x".into(), "y".into()],
                                );
                                show_paths(ui, &frame);
                            }
                        },
                    );
                    painted.textures_delta.clear();
                    output = Some(painted);
                }
                output.unwrap().shapes
            };
            assert_eq!(
                paint(&new, false),
                paint(&old, true),
                "width {width}, pointer {pointer:?}"
            );
        }
    }
}
