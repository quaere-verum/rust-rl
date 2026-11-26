#[path = "support/network.rs"]
mod fixtures;
#[path = "support/policy_ui.rs"]
mod support;

use rl_viewer::ui::View;
use std::time::{Duration, Instant};

// This catches a regression back to derivative magnitudes without a finite
// displacement, and ensures Output can be inspected through the shared layer.
#[test]
fn network_exposes_finite_comparison_and_output_layer() {
    let dir = fixtures::fixture();
    let mut h = support::Harness::new(&dir, eframe::egui::vec2(1920., 1080.));
    h.app.select_view(View::Network);
    let end = Instant::now() + Duration::from_secs(20);
    while !h.app.network_ready() {
        h.frame(vec![]);
        assert!(Instant::now() < end, "{:?}", h.text);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(h.has("Propagation by layer"), "{:?}", h.text);
    assert!(h.has("Local nonlinearity"));
    let snapshot = h.app.network_frame().unwrap().snapshot.key.clone();
    h.click("Hidden 2");
    h.click_last("Output");
    let end = Instant::now() + Duration::from_secs(20);
    while !h.app.network_ready() {
        h.frame(vec![]);
        assert!(Instant::now() < end, "{:?}", h.text);
        std::thread::sleep(Duration::from_millis(2));
    }
    // The selected layer now lives in the compact toolbar. Check its actual
    // control value, independently of the Output card shown for every layer.
    let layer = h.text.iter().position(|(text, _)| text == "Layer").unwrap();
    assert_eq!(h.text[layer + 1].0, "Output");
    assert_eq!(h.app.network_frame().unwrap().snapshot.key, snapshot);
    assert!(h.has("Local geometry") && h.has("Representation paths"));
}
