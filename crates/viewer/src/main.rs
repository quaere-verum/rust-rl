fn main() -> eframe::Result {
    if std::env::args().nth(1).as_deref() == Some("--check") {
        let path = std::env::args().nth(2).unwrap_or_else(|| "runs".into());
        match rl_viewer::data::validate_directory(std::path::Path::new(&path)) {
            Ok(report) => {
                println!("{report}");
                return Ok(());
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }
    let path = std::env::args().nth(1).unwrap_or_default();
    let requested_wgpu = match std::env::var("RL_VIEWER_RENDERER").as_deref() {
        Ok("wgpu") => true,
        Ok("glow") => false,
        Err(_) => cfg!(feature = "wgpu"),
        Ok(value) => {
            eprintln!("Unknown RL_VIEWER_RENDERER={value}; use wgpu or glow");
            std::process::exit(2);
        }
    };
    #[cfg(not(feature = "wgpu"))]
    if requested_wgpu {
        eprintln!("WGPU rendering requires a viewer build with --features wgpu");
        std::process::exit(2);
    }
    let renderer = eframe::Renderer::Glow;
    #[cfg(feature = "wgpu")]
    let renderer = if requested_wgpu {
        eframe::Renderer::Wgpu
    } else {
        renderer
    };
    let size = [1440., 960.];
    #[cfg(feature = "screenshots")]
    let size = std::env::var("RL_VIEWER_SIZE")
        .ok()
        .and_then(|value| {
            let (width, height) = value.split_once('x')?;
            let width = width.parse::<f32>().ok()?;
            let height = height.parse::<f32>().ok()?;
            (width.is_finite() && height.is_finite() && width >= 640. && height >= 480.)
                .then_some([width.min(3840.), height.min(2160.)])
        })
        .unwrap_or(size);
    let viewport = eframe::egui::ViewportBuilder::default().with_inner_size(size);
    // Native visual acceptance can use the full physical display at1080p.
    #[cfg(feature = "screenshots")]
    let viewport =
        viewport.with_decorations(std::env::var("RL_VIEWER_DECORATIONS").as_deref() != Ok("false"));
    let options = eframe::NativeOptions {
        viewport,
        renderer,
        depth_buffer: 0,
        stencil_buffer: 0,
        multisampling: 0,
        ..Default::default()
    };
    eframe::run_native(
        "RL Visualization Laboratory",
        options,
        Box::new(move |cc| {
            #[cfg(feature = "screenshots")]
            if let Some(zoom) = std::env::var("RL_VIEWER_UI_ZOOM")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| (0.75..=2.).contains(value))
            {
                cc.egui_ctx.set_zoom_factor(zoom);
            }
            Ok(Box::new(rl_viewer::ui::Laboratory::new(cc, path)?))
        }),
    )
}
