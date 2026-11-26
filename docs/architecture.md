# Architecture

The library has a lightweight interaction and planning foundation, with optional neural and diagnostic features. The modules describe responsibilities rather than selecting an execution framework for the caller.

| Module | Owns |
| --- | --- |
| `core` | Environment and actor contracts, public observations and actions, physical rewards, factual transitions, history, and checkpoint interfaces. |
| `collect` and `data` | Environment lanes and factual execution, then reusable dense replay, chronological episode storage, and validated batches. |
| `algorithms` and `approx` | Update rules, trainable state, feature encodings, and regression. |
| `models` and `planning` | Public predictive laws and belief updates, search budgets, exact and sampled solvers, guidance, and search targets. |
| `train` | Optional runtime composition, schedules, behavior publication, evaluation, and checkpoint I/O. |
| `nn` | Burn-specific tensors, models, packing, and optimization. Requires `neural`. |
| `diagnostics` and `rl-viewer` | Optional recording, analysis, and playback. |

An environment publishes observations and legal actions. An actor uses public input and its own information state to select an action. Collectors run that action and store the resulting factual transition. A learner can update directly or consume storage through its batch contract. Dense replay and chronological replay have different storage and reconstruction requirements and both use the same interaction semantics.

Factual storage and collectors share recording contracts in `core::recording`. Temporal actor identity is a core capability, and the temporal learner contract uses the generic batch directly. The `probes` feature supplies timing instrumentation without making data or collection depend on neural actor types. Planning implementations own their algorithms and use the public requests and budgets in `planning`. Optional diagnostic adapters implement their capabilities alongside the recorder.

A world model takes a public context and supplies hypothetical outcomes. A planner can use policy and value guidance to produce a decision and learning targets. Physical execution remains with the environment. Supervision can combine search-derived policy targets and factual return targets while retaining their separate provenance.

The default build includes neural algorithms. Disable default features for tabular, statistical, and model-based workflows without Burn. Portable recording can run with `diagnostics` alone. The viewer enables neural adapters for model-backed inspection. The optional recording observer reads completed trainer updates through an immutable boundary. DiagnosticSession combines that boundary with independently scheduled learner capabilities and the portable recorder. Its frozen evaluation actors and reference providers use diagnostic random streams. The viewer resolves recorded evidence and runs numerical preparation on bounded workers. Training uses the existing learning-history mailbox for immutable metric and evaluation summaries, with independent admission for peer runs. Policy evolution reuses the same fixed-population evidence and fits a single projection of checkpoint policy outputs. Network representation preparation compares complete input records and feature identities before reusing a PCA basis. Episode and learner transport use separate clocks. Render tracks and factual replay populations are optional recording capabilities.

See [environment contracts](environment-api.md), [adding a component](extending.md), and [planning](planning.md).
