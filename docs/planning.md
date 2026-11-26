# Planning with public models

The model describes hypothetical outcomes. The environment owns factual execution. Use public observations, histories, or belief summaries as predictive contexts. A planner can hold a simulated latent particle during a rollout, while policy and value guidance see the public context. Execute a selected action only after checking it against the real observation's current legal menu.

## Choosing a solver

| Need | Model and solver |
| --- | --- |
| Exact expectations on a finite state set | Implement `PlanningModel` and `EnumerableModel`. Prepare `planning::finite::FiniteModel` with the complete live state set and an outcome cap, then call `value_iteration` or `evaluate_policy`. |
| A bounded exact or interval-valued contingent decision | Use `planning::contingent` with an enumerable public model, return bounds, and a budget. Public outcome keys identify branches. |
| A sampled tree search | Implement `PlanningModel` and `PredictiveModel` and use `planning::pomcp`. Use guidance when policy priors or continuation values are available. A fitted predictor can implement the same sampling contract as an explicit model. |

The finite solver accepts discounted cycles and undiscounted acyclic live graphs. Each complete outcome law has positive probabilities summing to one and unique public outcome keys. Every live successor belongs to the supplied state set. The result includes the Bellman residual and whether the solve met its tolerance within the allowed sweeps. The policy evaluator uses a frozen `PolicyGuide` probability row for each state's legal menu.

Run `cargo run --locked -p rust-rl --no-default-features --example model_planning` for an explicit model and a model fitted from environment steps. The example compares their search decisions and verifies that search leaves the factual steps, physical clock, and environment RNG unchanged until execution.

## Outcomes and physical time

`PredictiveModel::simulate` updates a hypothetical particle and returns the resulting public context. `next: None` represents terminal model dynamics. A search cutoff leaves the context live and is accounted for by the planner. `PlanningModel::planning_returns` declares the reward objective and per-tick discount for either capability. If a prediction lasts several ticks, the model supplies the reward already discounted within the action. A backup discounts continuation by the declared discount raised to the duration.

A public observation key distinguishes outcomes that change the information required for future decisions. `EnumerableModel::enumerate` marginalizes latent branches that yield the same public outcome. Use `FiniteBelief` or a caller-provided sufficient information state when the real world is partially observed. The [finite planning tests](../crates/rl/tests/finite_planning.rs) include an exact belief example that takes opposite actions after opposite public readings.

## Guidance and learning

`PolicyGuide` supplies probabilities for a planner's ordered legal support. `ValueGuide` supplies a continuation value with the same declared return convention. POMCP supports both unguided UCT and prior-guided selection. The tree can emit a separate policy improvement target through `planning::targets`.

Search targets can supervise an ordinary policy or value learner without becoming factual replay. Produce a target with `PolicyTarget::from_visits`, verify its factual prefix with `SearchTarget::join_supervision`, and pass the joined row and factual sample to `SearchGuidedPolicyValue`. Publish the updated pair and execute it through an ordinary collector. The [direct search-guided test](../crates/rl/tests/search_guided_factual.rs) exercises planning failures, factual recording, optimizer restoration and execution of the updated policy. These components also support future belief-space search and distillation.

`search_with_options` configures public rollout policy, PUCT, and absolute value cutoffs. For a cutoff measured from the start of a rollout, use `pomcp::search_with_rollout` and read its rollout step index.
