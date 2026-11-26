//! Actual serialized policies and egui texture uploads, without selected-only semantics.
use super::*;
use crate::{data::RunIndex, interactive::Interactive, landscape::Palette};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{models::SnapshotBuilder, schema::*, storage::*},
    nn::{critics::scalar::DiscreteQ, matrix, network::Mlp, policy::Policy, vector},
};
use std::time::{Duration, Instant};

fn fixture(current_indicator: bool, reference_indicator: bool) -> (tempfile::TempDir, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 2 };
    let manifest = RunManifest::new(
        "texture-semantics",
        "generic",
        "fixture",
        2,
        space.clone(),
        0.9,
    )
    .unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for (revision, indicator) in [(3, reference_indicator), (7, current_indicator)] {
        let mut builder = SnapshotBuilder::new(
            "actual",
            RecordContext::new("texture-semantics", revision),
            2,
            space.clone(),
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(revision);
        if indicator {
            let mut q = DiscreteQ::new(2, 2, &[], &mut rng).unwrap();
            q.network.layers[0].weight = q.network.layers[0]
                .weight
                .clone()
                .map(|_| matrix(vec![-1., 1., 0., 0.], 2, 2));
            q.network.layers[0].bias = q.network.layers[0]
                .bias
                .take()
                .map(|p| p.map(|_| vector(&[0., 0.])));
            builder.discrete("actor", &q).unwrap();
        } else {
            let mut network = Mlp::new(2, &[], 2, &mut rng).unwrap();
            network.layers[0].weight = network.layers[0]
                .weight
                .clone()
                .map(|_| matrix(vec![0.; 4], 2, 2));
            network.layers[0].bias = network.layers[0]
                .bias
                .take()
                .map(|p| p.map(|_| vector(&[0., 3f32.ln()])));
            builder
                .policy(
                    "actor",
                    &Policy::from_network(network, space.clone()).unwrap(),
                )
                .unwrap();
        }
        writer.append(&builder.finish().unwrap()).unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    assert_eq!(
        run.recording().manifest().action_space,
        ActionSpace::Discrete { count: 2 }
    );
    (dir, run)
}
fn prepare(run: &RunIndex) -> (Interactive, PreparedPolicyFrame) {
    let current = PolicySource::model(run, 1, Some("actor")).unwrap();
    let reference = PolicySource::model(run, 0, Some("actor")).unwrap();
    let input = Observation::new(vec![0.12345, -0.23456]);
    let mut slice = SliceConfig::new(current.semantics(), &input, &[]).unwrap();
    slice.resolution = [8, 8];
    let request = PolicyRequest::new(
        current,
        Some(reference),
        7,
        slice,
        PolicyPin::Synthetic(input.clone()),
        None,
    )
    .unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.submit_policy(request).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        interactive.policy_error().is_none(),
        "{:?}",
        interactive.policy_error()
    );
    let frame = interactive.policy_frame().unwrap().clone();
    assert!(frame.grid.regular());
    assert_eq!(frame.selected.as_ref().unwrap().input, input);
    (interactive, frame)
}
fn paint(
    ctx: &egui::Context,
    pane: &mut PolicyPane,
    run: &RunIndex,
    frame: &PreparedPolicyFrame,
) -> Option<(Vec<Color32>, TextureOptions)> {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440., 900.),
            )),
            ..Default::default()
        },
        |ui| {
            pane.show(
                ui,
                EnvironmentSemantics::new(run.recording().manifest()),
                Some(frame),
            );
        },
    );
    let texture = pane.texture.as_ref().unwrap().id();
    let pixels = output
        .textures_delta
        .set
        .iter()
        .find(|(id, _)| **id == texture)
        .map(|(_, deltas)| {
            let delta = deltas.last().unwrap();
            let egui::ImageData::Color(image) = &delta.image;
            (image.pixels.clone(), delta.options)
        });
    output.textures_delta.clear();
    pixels
}
#[test]
fn deterministic_indicator_current_reference_and_difference_upload_nearest() {
    let (_dir, run) = fixture(true, true);
    let (interactive, frame) = prepare(&run);
    assert!(frame.grid.values.contains(&0.) && frame.grid.values.contains(&1.));
    let ctx = egui::Context::default();
    let mut pane = PolicyPane::new();
    for display in [Display::Current, Display::Reference, Display::Difference] {
        pane.set_display(display);
        let (_, options) = paint(&ctx, &mut pane, &run, &frame).unwrap();
        assert_eq!(options, TextureOptions::NEAREST, "{display:?}");
        let uploads = pane.texture_uploads;
        assert!(paint(&ctx, &mut pane, &run, &frame).is_none());
        assert_eq!(pane.texture_uploads, uploads);
    }
    assert_eq!(
        interactive
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values(),
        vec![0.12345, -0.23456]
    );
}

#[test]
fn deterministic_discrete_inspector_reports_action_agreement_without_probability() {
    let (_dir, run) = fixture(true, true);
    let (_interactive, frame) = prepare(&run);
    let selected = frame.selected.as_ref().unwrap();
    assert!(selected.chosen_action.is_some());
    assert_eq!(selected.chosen_action, selected.reference_chosen_action);
    assert!(
        selected.kl.is_none() && selected.plot.is_none() && selected.action_statistics.is_none()
    );
    let mut pane = PolicyPane::new();
    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        pane.selected_contents(
            ui,
            EnvironmentSemantics::new(run.recording().manifest()),
            Some(&frame),
        )
    });
    fn text(shape: &egui::Shape, out: &mut String) {
        match shape {
            egui::Shape::Text(t) => {
                out.push_str(&t.galley.job.text);
                out.push('\n');
            }
            egui::Shape::Vec(v) => {
                for s in v {
                    text(s, out)
                }
            }
            _ => {}
        }
    }
    let mut labels = String::new();
    for s in &output.shapes {
        text(&s.shape, &mut labels);
    }
    output.textures_delta.clear();
    assert!(
        labels.contains("Chosen action") && labels.contains("Actions agree"),
        "{labels}"
    );
    assert!(
        !labels.contains("Output")
            && !labels.contains("probability")
            && !labels.contains("Action distribution"),
        "{labels}"
    );
}
#[test]
fn displayed_mixed_family_selects_its_own_palette_and_filter_with_or_without_selected() {
    for (current_indicator, reference_indicator) in [(true, false), (false, true), (false, false)] {
        let (_dir, run) = fixture(current_indicator, reference_indicator);
        let (_interactive, frame) = prepare(&run);
        let point = frame.selected.as_ref().unwrap();
        assert_eq!(
            matches!(point.distribution, SelectedDistribution::Deterministic),
            current_indicator
        );
        assert_eq!(
            matches!(
                point.reference_distribution,
                Some(SelectedDistribution::Deterministic)
            ),
            reference_indicator
        );
        for selected in [true, false] {
            let mut shown = frame.clone();
            if !selected {
                shown.selected = None;
            }
            let ctx = egui::Context::default();
            let mut pane = PolicyPane::new();
            for display in [Display::Reference, Display::Current, Display::Difference] {
                pane.set_display(display);
                let indicator = match display {
                    Display::Current => current_indicator,
                    Display::Reference => reference_indicator,
                    Display::Difference => current_indicator || reference_indicator,
                };
                let expected_palette = if display == Display::Difference {
                    Palette::Difference
                } else if indicator {
                    Palette::Categorical
                } else {
                    Palette::Probability
                };
                let (pixels, options) = paint(&ctx, &mut pane, &run, &shown).unwrap();
                assert_eq!(
                    pane.texture_key.unwrap().3,
                    expected_palette,
                    "current_indicator={current_indicator} reference_indicator={reference_indicator} selected={selected} display={display:?}"
                );
                assert_eq!(
                    options,
                    if indicator {
                        TextureOptions::NEAREST
                    } else {
                        TextureOptions::LINEAR
                    },
                    "{display:?}"
                );
                if display != Display::Difference {
                    assert_eq!(pane.color_limits(), Some([0., 1.]));
                    if !indicator {
                        let expected =
                            crate::landscape::LandscapeScale::new([0., 1.], Palette::Probability)
                                .color(0.25);
                        assert!(
                            pixels.iter().all(|c| *c == expected),
                            "genuine quarter-probability must keep its sequential shade"
                        );
                    }
                }
                let uploads = pane.texture_uploads;
                let mut without_selected = shown.clone();
                without_selected.selected = None;
                assert!(
                    paint(&ctx, &mut pane, &run, &without_selected).is_none(),
                    "Selected presence cannot change the field meaning"
                );
                assert_eq!(pane.texture_uploads, uploads);
            }
        }
    }
}
