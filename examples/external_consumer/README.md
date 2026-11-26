# External public API consumer

The door hall publishes a graph and a changing menu of entity-keyed actions. The entrance reveals a cue that disappears at the fork. The tabular policy remembers the cue through the public history and chooses the corresponding door even though both fork observations are identical.

The fixture's bounded TD learner clips each one-step Q update. It implements the public learner and behavior publication traits, samples factual experience from the ordinary episode store, and publishes a standard tabular actor. An independent predictive model uses the reconstructed public state for exact finite planning and POMCP. The two-tick passage pays a discounted immediate reward of 3 and reaches the fork with probability 0.75, giving the optimal entrance action a value of 6.375.

The optional neural path updates feedforward Candidate DDQN on the same factual graph observations and conditional actions. Its graph network sees each current frame, so the neural check exercises the shared interfaces and leaves cue-dependent control to the history-aware tabular policy.

Run from this directory. On a fresh checkout, `cargo fetch` generates the standalone lockfile and fetches dependencies. Use `cargo generate-lockfile --offline` instead if the dependencies are already cached.

```sh
cargo run --offline --no-default-features
cargo test --offline --no-default-features
cargo tree --offline --no-default-features -e normal
cargo run --offline --features neural
cargo test --offline --features neural
cargo clippy --offline --all-targets --no-default-features -- -D warnings
cargo clippy --offline --all-targets --features neural -- -D warnings
cargo fmt --check
```
