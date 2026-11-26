# Package the Windows viewer

Build on Windows x86-64 with Rust 1.95.0, the MSVC C++ build tools and a Windows SDK. Use PowerShell 7 (`pwsh`) from the repository root. These flags remove developer paths and symbols and link the C runtime into the executable.

```powershell
$source = (Get-Location).Path
$env:CARGO_TARGET_DIR = Join-Path $source 'target'
$target = $env:CARGO_TARGET_DIR
$env:CARGO_ENCODED_RUSTFLAGS = (@(
    "--remap-path-prefix=$env:USERPROFILE=/build-home"
    "--remap-path-prefix=$source=/rust-rl"
    "--remap-path-prefix=$($source.Replace('\','/'))=/rust-rl"
    "--remap-path-prefix=$target=/rust-build"
    "--remap-path-prefix=$($target.Replace('\','/'))=/rust-build"
    '-Ctarget-feature=+crt-static'
) -join [char]31)
$env:CARGO_PROFILE_RELEASE_DEBUG = '0'
$env:CARGO_PROFILE_RELEASE_STRIP = 'symbols'
cargo build --locked --release -p rl-viewer --no-default-features
```

The portable Windows package uses the viewer's Glow backend. The regular Cargo build supports WGPU through the viewer's default feature.

Prepare a directory with complete recordings named `rocket-landing`, `alpha-execution` and `alpha-execution-cvar`. Place a matching `<name>.sha256` inventory beside each directory. The release download contains the inventories for its bundled data. To package independently reproduced data, use the demonstration guide and write fresh inventories. [Reproduction commands](demonstrations.md)

Place the reviewed binary distribution notices in a separate directory containing `THIRD-PARTY-NOTICES` and its accompanying licence and source directories. Preserve the original licence files and include the corresponding source for MPL components.

```powershell
./scripts/package-windows.ps1 -Viewer target/release/rl-viewer.exe -Demonstrations path/to/demonstrations -Notices path/to/notices -OutputDirectory artifacts
```

The script checks the per-recording inventories and runs headless validation before creating the ZIP and its SHA-256 file. Extract the ZIP into a fresh path outside the source tree and launch the included executable. Check that **Open demonstration** can find the three recordings before distributing the archive. [Installation guide](getting-started.md)
