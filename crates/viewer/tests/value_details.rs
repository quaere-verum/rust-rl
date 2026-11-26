#[path = "support/value_ui.rs"]
#[allow(dead_code)]
mod family;

use eframe::egui;
use rl::core::{action::Action, env::Observation};
use rl_viewer::ui::{Laboratory, View};
use std::time::{Duration, Instant};

struct PaintedText {
    text: String,
    rect: egui::Rect,
    clip: egui::Rect,
}
struct Harness {
    ctx: egui::Context,
    app: Laboratory,
    size: egui::Vec2,
    time: f64,
}
impl Harness {
    fn paint(&mut self, events: Vec<egui::Event>) -> Vec<PaintedText> {
        self.time += 0.1;
        let mut output = self.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, self.size)),
                time: Some(self.time),
                events,
                ..Default::default()
            },
            |ui| self.app.show(ui),
        );
        output.textures_delta.clear();
        assert!(self.app.notice().is_none(), "{:?}", self.app.notice());
        let mut text = Vec::new();
        for shape in &output.shapes {
            collect(&shape.shape, shape.clip_rect, &mut text);
        }
        text
    }
    fn settle(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            self.paint(Vec::new());
            if self.app.value_view().ready()
                && !self.app.value_pending()
                && self.app.snapshot_preparation_idle()
            {
                self.paint(Vec::new());
                return;
            }
            assert!(
                Instant::now() < deadline,
                "{:?}",
                self.app.value_view().status()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    fn click_last(&mut self, label: &str) {
        let text = self.paint(Vec::new());
        let position = text
            .iter()
            .rev()
            .find(|item| item.text == label && item.clip.contains(item.rect.center()))
            .unwrap_or_else(|| panic!("Missing visible control {label}"))
            .rect
            .center();
        for pressed in [true, false] {
            self.paint(vec![
                egui::Event::PointerMoved(position),
                egui::Event::PointerButton {
                    pos: position,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        self.paint(Vec::new());
    }
}
fn collect(shape: &egui::Shape, clip: egui::Rect, text: &mut Vec<PaintedText>) {
    match shape {
        egui::Shape::Text(shape) => text.push(PaintedText {
            text: shape.galley.text().into(),
            rect: shape.galley.rect.translate(shape.pos.to_vec2()),
            clip,
        }),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect(shape, clip, text);
            }
        }
        _ => {}
    }
}

#[test]
fn expanded_value_details_wrap_complete_inputs_predictions_and_sources_inside_the_inspector() {
    let directory = family::family("quantile", true);
    let ctx = egui::Context::default();
    let app =
        Laboratory::from_context(&ctx, directory.path().join("run").display().to_string()).unwrap();
    let mut h = Harness {
        ctx,
        app,
        size: egui::vec2(1440., 900.),
        time: 0.,
    };
    h.app.select_view(View::Value);
    h.settle();
    let observation = Observation::new(vec![0.125, -0.375, 0.625]);
    let action = Action::Continuous(vec![0.125, -0.75]);
    h.app
        .value_view_mut()
        .pin_input(observation.clone(), Some(action.clone()))
        .unwrap();
    h.settle();
    h.click_last("Reference");
    h.settle();
    assert!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_prediction
            .is_some()
    );
    h.click_last("Details");
    let expected_observation = format!(
        "Complete observation {:?}\nMask {:?}",
        observation.dense_values(),
        observation.dense_mask()
    );
    for size in [egui::vec2(1440., 900.), egui::vec2(1280., 720.)] {
        h.size = size;
        let mut seen = [false; 4];
        for _ in 0..14 {
            let text = h.paint(Vec::new());
            for item in &text {
                let field = if item.text.starts_with("Complete observation") {
                    Some(0)
                } else if item.text.starts_with("Current:") {
                    Some(1)
                } else if item.text.starts_with("Complete current prediction") {
                    Some(2)
                } else if item.text.starts_with("Complete reference prediction") {
                    Some(3)
                } else {
                    None
                };
                if let Some(field) = field {
                    assert!(
                        item.rect.left() >= item.clip.left() - 1.
                            && item.rect.right() <= item.clip.right() + 1.,
                        "expanded Details text is horizontally clipped at {size:?}: {:?} outside {:?}: {}",
                        item.rect,
                        item.clip,
                        item.text
                    );
                    seen[field] = true;
                    if field == 0 {
                        assert_eq!(item.text, expected_observation);
                    }
                    if field == 1 {
                        assert!(item.text.contains("actual k=7 / requested k=7"));
                        assert!(
                            item.text.contains("action[0] = 0.125")
                                && item.text.contains("action[1] = -0.75")
                        );
                    }
                }
            }
            let inspector = h.app.value_view().inspector_rect.unwrap();
            assert!(
                (400.0..=530.0).contains(&inspector.width()),
                "{inspector:?}"
            );
            let position = inspector.left_top() + egui::vec2(6., 32.);
            h.paint(vec![
                egui::Event::PointerMoved(position),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0., -200.),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Unknown,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        assert!(
            seen.into_iter().all(|field| field),
            "all complete Details fields remain reachable at {size:?}: {seen:?}"
        );
        let inspector = h.app.value_view().inspector_rect.unwrap();
        let position = inspector.left_top() + egui::vec2(6., 32.);
        h.paint(vec![
            egui::Event::PointerMoved(position),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0., 10000.),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Unknown,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
    }
    let point = h.app.value_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.input, observation);
    assert_eq!(point.action, Some(action));
}
