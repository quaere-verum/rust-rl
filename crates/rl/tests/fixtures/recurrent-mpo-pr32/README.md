# Earlier recurrent MPO learner records

Runtime: Windows x86_64 MSVC, Rust 1.95.0, Burn 0.22 Flex f32. `SHA256SUMS`
identifies the immutable data. Both backups have a first terminal
update, a native v1 checkpoint, and a second truncated update with a target sync.
The original loader reproduced uninterrupted continuation before extraction.

| Backup | First target mean | Second target mean | Projection scale / trials |
|---|---:|---:|---|
| Policy evaluation | 1.08576238155365 | 1.4237780570983887 | 0.125 / 4 |
| Double-Q optimality | 1.08576238155365 | 1.6818002462387085 | 0.125 / 4 |

Every step accepts a fractional Adam displacement, improves fixed-posterior
cross entropy, and satisfies the forward KL budget. Each owner has 14 trained
Adam groups; the unused categorical `log_std` tensor has no optimizer group.
The terminal actor row is zero, singleton masks have exact support, and the
truncated endpoint bootstraps from two possible actions.

`recurrent_mpo_reference.rs` compares normalized model/Adam payloads after
explicit v1 import on every host, rejects ordinary implicit import, and checks
native resumed continuation. On Windows x86_64 MSVC it also checks both fresh
updates, E-step reports, projection trials, targets and all tensor/Adam bits
against the original arithmetic. It does not assert cross-runtime bit identity.

The v2 learner imports this envelope through `import_pr32_record` or
`import_pr32_checkpoint`. Numerical comparisons resolve Adam groups through
their owning model's stable parameter paths.
