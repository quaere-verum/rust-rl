# Checkpoints and reproducibility

Choose the checkpoint that matches what needs to resume.

| Interface | Stored state |
| --- | --- |
| `train::checkpoint::{save_learner, load_learner}` | Algorithm state returned by `Checkpoint`, including parameters, optimizer state and learner-owned random streams where supported. |
| `train::checkpoint::{save_run, load_run}` | Learner state plus the supplied `Snapshot`. The snapshot owner specifies its environment, collector, replay, random streams and schedule. |
| `train::checkpoint::load_run_portable` | An explicit import path for compatible snapshots that implement bounded portable restoration of runtime memory. |
| `train::checkpoint::{export_policy, load_policy}` | Neural inference parameters loaded into a compatible model template. Requires `neural`. |

Save to a fresh path. The checkpoint helpers use a verified envelope and completion marker, and check the record kind before restoration. A learner checks compatibility and validates its replacement before installing it. Model architectures, action and observation schemas, checkpoint versions, and custom feature identities must match the consumer's requirements. `load_run_portable` reports the audits its snapshot performs. Use the ordinary native path for exact continuation on the same compatible runtime.

## Tabular execution

`QTable` implements `Checkpoint` for serializable keys. A learner record stores ordered state-action rows, visit counts, and the initial value. Reconstructing a `TabularActor` from a standalone table requires the same public key mapping and environment schemas. The Q-table is independent of the exploration setting and random number generator.

If the key builder and keys support Serde, `TabularActor` also supports run snapshots. The actor saves the builder, its bound schemas, immutable table, revision, and exploration setting. Include the collector and any factual store in the run snapshot to resume actor memory and lane RNG streams mid-episode. A custom loop owns any additional sampler state it needs to save.

## Deterministic execution

Seed the environment and each independent collection lane. Preserve the owned learner and replay-sampling streams when resuming. Use a distinct planner stream so a larger hypothetical search budget cannot draw from factual environment randomness. Keep physical reward traces and boundary flags with any recorded experience used for later targets.

Recurrent inference uses a causal prefix or a validated representation anchor. A fresh reset starts a new history. A caller-owned environment can return an `EpisodeStart` carrying a `ResumeContext` with a complete public prefix and its physical clock. Native run snapshots can keep open episodes and partial rollouts, allowing supported trainers to resume both collection and optimizer steps. Shape agreement alone does not establish compatibility between two observation layouts, recurrent memories, or policy checkpoints. See [environment schemas](environment-api.md) and the [run restoration tests](../crates/rl/tests/checkpoints.rs).
