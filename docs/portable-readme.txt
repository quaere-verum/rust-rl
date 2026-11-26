rust-rl v0.1.0
Windows x86-64 portable viewer

Extract the whole archive and launch rl-viewer.exe. Select Open demonstration,
then use Run to choose Rocket Landing · PPO, Alpha Execution · Expected Return, or
Alpha Execution · CVaR. Keep the demonstrations folder beside the executable
when moving it. The bundled recordings are available offline.

Open recording browses to another recording directory or a parent folder.
Ctrl+1 through Ctrl+6 select Training, Policy, Value, Network, Dynamics and
Environment in that order. The top timeline advances the learner checkpoint.
The Environment workspace has separate controls for episode playback.

From PowerShell in this directory, validate the recordings with

    .\rl-viewer.exe --check ".\demonstrations"

The .sha256 files contain the SHA-256 of every bundled recording file.
See LICENSE for the project's terms and THIRD-PARTY-NOTICES for dependencies.

Source, user guide and release notes are available from the release repository
at https://github.com/quaere-verum/rust-rl.
