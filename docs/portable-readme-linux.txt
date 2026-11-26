rust-rl v0.1.0
Linux x86-64 portable viewer

Extract the whole archive and run ./rl-viewer. Select Open demonstration,
then use Run to choose Rocket Landing · PPO, Alpha Execution · Expected
Return, or Alpha Execution · CVaR. Keep the demonstrations folder beside
the executable. The bundled recordings are available offline.

Use an x86-64 system with glibc 2.43 or newer, libgcc_s.so.1, a Wayland or
X11 desktop, libxkbcommon, and graphics drivers providing a compatible
Vulkan or OpenGL/EGL adapter. Verified on Ubuntu 26.04 LTS under WSL2
with WSLg. X11 also requires libxkbcommon-x11.so.0.

Validate the recordings from a terminal with this command.

    ./rl-viewer --check ./demonstrations

The .sha256 files contain the SHA-256 of every bundled recording file.
See LICENSE for the project's terms and THIRD-PARTY-NOTICES.txt for
dependency notices.

Source, user guide and release notes are available at
https://github.com/quaere-verum/rust-rl
