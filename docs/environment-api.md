# Environment API

Implement [`Env`](../crates/rl/src/core/env.rs) for a simulator instance. Declare a stable `ObservationSpace` and `ActionSpace`. `reset` starts a new episode and publishes the first observation. Given `Some(seed)`, reseed the environment's random stream. Given `None`, continue its owned stream. `step` checks the action against the current public observation and returns the actual successor before any reset.

## Boundaries and rewards

Set `terminated` when the modeled episode ends. Set `truncated` for an external limit and publish the real final observation, including legal actions for a continuation bootstrap. A collector can stop recording a batch while both flags are false. A single decision can consume several physical ticks. `Reward::from_primitives` records the chronological rewards, and `Reward::discounted_f64` returns both the accumulated reward and the discount carried to the successor.

Collectors validate schemas, track clocks and record only real environment transitions. `SyncVectorEnv` runs multiple independent instances. `StatefulCollector` accepts a `StatefulActor`, scalar or compatible batched inference, and a `TemporalSink` such as `EpisodeStore`.

## Public observations

| Observation | Use |
| --- | --- |
| `Observation::Discrete` | A finite `u64` symbol and optional action mask. Use `DiscreteObservationSpace` to declare the exclusive symbol bound. |
| `Observation::Dense` | A fixed-width `f32` vector with an optional finite action mask and optional versioned public payload. |
| `Observation::Structured` | Schema-bound spatial, scalar, entity and event fields, including semantic candidate actions. |
| `Observation::Graph` | Structured fields plus sparse typed nodes, directed edges, and a versioned graph schema. Emit both directed arcs when representing an undirected edge. |

Entity and region IDs describe public objects. Keep IDs stable when row order changes. Use schema namespaces and versions to distinguish different meanings or units. Publish only acquired information. An observation may omit latent simulator state and may alias several possible states.

## Actions and information states

`ActionSpace` supports discrete choices, bounded or unbounded continuous vectors, and conditional semantic actions. In the conditional grammar, each operation declares its argument kind. Enabled candidates supply complete actions, including entity or location arguments. A location or entity reference identifies a public object rather than a packed array position.

Use `Observation::finite_actions` for an ordered legal menu and `Observation::finite_action_key` to form stable keys for a table. The menu retains the action spelling supplied by the environment. A primary-map coordinate has the same table identity as its explicitly named primary-region coordinate. Pass the original legal action to `step`.

Implement `StateKeyBuilder` for policies that use a tabular information state. The builder starts memory at reset, assimilates `HistoryInput` containing the current observation and previous factual input, and produces a key from its resulting memory. Declare whether the mapping is exact, discretized, or aggregated. An exact key is Markov only if the represented information is sufficient. `DiscreteSymbolKey` uses the latest symbol directly. A partially observed environment may require a public statistic, event history, or recursive belief update.

The [tabular example](../crates/rl/examples/tabular_q_learning.rs) trains from real environment steps. The [cue-memory test](../crates/rl/tests/tabular_execution.rs) uses the same symbol for two different decisions and distinguishes them using the preceding public observation. It saves and reloads the values and evaluates the reconstructed actor through the ordinary collector.
