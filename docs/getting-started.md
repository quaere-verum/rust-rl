# Install rust-rl

## Windows viewer

Download `rust-rl-v0.1.0-windows-x86_64.zip` and extract the entire folder. Launch `rl-viewer.exe`, select **Open demonstration**, and use **Run** to switch between the bundled recordings. Keep `demonstrations` beside the executable when you move it. A directory name may contain spaces. The viewer can inspect these recordings offline, using the files in the archive. [GitHub Releases](https://github.com/quaere-verum/rust-rl/releases)

Use **Open recording** to browse to a different recording directory or a parent folder containing several runs. You can also validate the bundled recordings from PowerShell.

```powershell
.\rl-viewer.exe --check ".\demonstrations"
```

A completed bundle reports `complete=true` for each of its three recordings. [Viewer manual](viewer.md)

## Linux viewer

Download `rust-rl-v0.1.0-linux-x86_64.tar.gz` from the same release.

```sh
tar -xzf rust-rl-v0.1.0-linux-x86_64.tar.gz
cd rust-rl-v0.1.0-linux-x86_64
./rl-viewer
```

The binary requires an x86-64 system with glibc 2.43 or newer and `libgcc_s.so.1`. Use a Wayland or X11 desktop with libxkbcommon and graphics drivers that provide a compatible Vulkan or OpenGL/EGL adapter. X11 also requires `libxkbcommon-x11.so.0`. The archive was verified on Ubuntu 26.04 LTS under WSL2 with WSLg.

Select **Open demonstration** and switch recordings from **Run**. For headless validation, run `./rl-viewer --check ./demonstrations`.

## Build from source

Use Rust 1.95 or newer. The repository's toolchain file pins 1.95.0 for workspace verification. On Windows, install the MSVC C++ build tools and a Windows SDK for source builds. From the repository root, build the default library member and the separate viewer package.

```sh
cargo build --locked --release
cargo build --locked --release -p rl-viewer
cargo run --locked --release -p rl-viewer
cargo run --locked --release -p rl-viewer -- path/to/recording
cargo run --locked --release -p rl-viewer -- --check path/to/recording
```

Replace `path/to/recording` with an actual recording directory. Quote the path if it contains spaces. Platform-specific package availability and verification are listed in the version's release notes.

Add `--no-default-features` to the viewer build or run command to use the Glow backend shipped in the Windows archive. For a run command, put it before the `--` that separates application arguments.

## Use the library from your project

Keep a source checkout next to your project and point Cargo at `crates/rl`.

```toml
[dependencies]
rl = { package = "rust-rl", path = "../rust-rl/crates/rl" }
```

The release tag can also be used as a Git dependency with the same package and import names.

```toml
[dependencies]
rl = { package = "rust-rl", git = "https://github.com/quaere-verum/rust-rl.git", tag = "v0.1.0" }
```

| Feature | What it enables |
| --- | --- |
| `neural` (default) | Burn Flex models and neural learners. |
| `default-features = false` | Tabular methods, statistical approximation, replay, and planning without Burn. |
| `diagnostics` | Portable recording, independent evaluation and analysis. It works without `neural`. |
| `diagnostics` plus `neural` | Built-in neural diagnostic adapters and supported historical model inference. The viewer uses this combination. |
| `experimental-acer` | Experimental ACER. |
| `probes` | Optional neural timing instrumentation. |

Run the small [Q-learning example](../crates/rl/examples/tabular_q_learning.rs) or build a [custom environment](environment-api.md). The [independent consumer example](../examples/external_consumer/README.md) demonstrates using the library from a separate Cargo project.

```sh
cargo run --locked -p rust-rl --no-default-features --example tabular_q_learning
```

## Record and inspect a run

From the repository root, choose a fresh destination and run the recording example. The source example constructs the environment and learner, trains, and publishes metrics, model records and evaluation data through the public APIs. [Diagnostic recording API](diagnostics-viewer.md)

```sh
cargo run --locked -p rust-rl --features diagnostics --example diagnostic_generic -- runs/my-experiment
cargo run --locked --release -p rl-viewer -- --check runs/my-experiment
cargo run --locked --release -p rl-viewer -- runs/my-experiment
```
