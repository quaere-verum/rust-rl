# Earlier recurrent DDQN run records

Runtime: Windows x86_64 MSVC, Rust 1.95.0, Burn 0.22 Flex f32. `SHA256SUMS`
identifies the immutable data. Both original run restores were verified before
capturing expectations.

| Case | Learner metadata | Trainer snapshot | Coverage |
|---|---:|---:|---|
| exact-v1 | 1 | 2 | Exact episodes, scalar inference, reference update |
| anchored-v2 | 2 | 3 | Segmented replay, independent Q/target anchors, batched inference, optimized exact update |

Both runs use a width-two GRU, two-step returns, two lanes, target interval two,
nondefault GRU clipping, and an episode-pinned older actor. A lane is immediately
before a stochastic two-action decision. `expected.json` records the checkpoint
and the next three steps: reports, sampled windows, Q values, parameter bits,
Adam ownership/steps, environment and sampler RNGs, observations and memories.

`legacy_recurrent_ddqn.rs` checks byte markers and schema identities on every
host. Arithmetic against the captured golden is enabled on Windows x86_64 MSVC;
it is not a claim of cross-runtime floating-point identity. Other regression
tests check exact uninterrupted/resumed continuation on their native runtime.

Keep the fixtures immutable when comparing a new build. Packed-file ordering
and process-local parameter IDs can change raw hashes. The tests compare stable
model ownership, numerical values and temporal identities.
