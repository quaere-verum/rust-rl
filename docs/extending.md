# Extending rust-rl

## Extend diagnostic recording

Use `DiagnosticSession::builder(path).environment(&env)` with the environment's public spaces. A custom `DiagnosticLearner` supplies its immutable revision, description and an independent evaluation actor. `diagnostic_records` can publish recorded predictions without publishing reconstructable model weights. `diagnostic_metrics` adds learner-specific metrics to the existing scheduled channel. Publish completed non-neural updates or fits with `checkpoint` and send collection counters from the actual training stream.

For typed probes, build `ProbeSet::new_in` with the actual `ObservationSpace` and `ActionSpace`. Use `DiagnosticInput::Observation` for observation-sufficient models and `DiagnosticInput::History` for causal, reset-rooted public histories. A `DisplayProjection` adds viewer coordinates independently of either representation. Stable caller-owned row IDs let the viewer verify a fixed population across revisions. `FinitePredictions::record` requires one row per input and exactly the legal semantic keys in that row. Use `Observation::finite_action_key` to normalize regional location aliases. Preserve unavailable predictions as `None`.

An ordinary benchmark can use `BenchmarkActor`. A richer `ReferenceProvider` receives the frozen checkpoint and can implement `action_value_for_input` for typed histories. Set its authored objective, return conventions, observation schema and information requirements in `ReferenceSpec`. Assign verified optimal quality only to a provider with a verified optimum on its declared domain. The Gridworld example uses one exact finite model for the tabular and statistical runs.

Set or reconstruct recurrent memory from the **same factual public history** separately for each checkpoint. `PublicHistory::inputs` carries previous actions and their primitive rewards through the reset boundary. Existing stateful reconstruction APIs can replay this input into the current model. Publish the resulting decision law and predictions through the ordinary adapter, and the recorded representation with its input contract when available. The viewer consumes those results without reconstructing arbitrary recurrent state. `record_factual_episode` keeps typed training actions and rewards separate from independent evaluation. It requires a reset-rooted contiguous episode or prefix.

The schema uses the same recorder, retention policy and chunk index for both dense and typed data. Typed arrays have a separate byte ceiling in addition to the run and chunk budgets. `RenderSource` and `FrameSource` remain opt-in for the dense environment presentation. The [user manual](diagnostics-viewer.md) describes the resulting views and controls.

Add the smallest contract your component needs. An algorithm owns its math and trainable state. It can use a collected dataset, a live public transition, or a sampled sequence through its own update interface.

## Add a learning algorithm

1. Decide what the update consumes and what can bootstrap. Use the ordinary `Reward`, boundary flags, and factual transitions. Keep search targets in a separate input when learning from a planner.
2. Implement the update against that data. For direct use, an inherent method such as `QTable::q_learning` is sufficient. Implement `Learner` with your batch type to use a compatible fixed-width trainer, and add `OffPolicyLearner` or `OnPolicyLearner` for its collection requirements. Use `StatefulLearner` and the temporal capabilities for sequential batches.
3. Publish an executable actor. A learner can expose an immutable actor revision while updating a separate trainable model. Keep behavior likelihoods tied to the actual sampling call and declare any required action masks, replay returns, or history constraints.
4. Implement `Checkpoint` when the learner needs to persist state. Validate all replacement fields before installing them. Write an independent reference test for the smallest representative update, including termination, truncation, or multi-tick timing where they affect the target.

The [independent consumer](../examples/external_consumer/README.md) implements a bounded TD learner against public factual transitions and executes a policy with the same semantic keys. See [tabular updates](../crates/rl/src/algorithms/tabular.rs) for direct step-based learning and [FQI](../crates/rl/src/algorithms/fqi.rs) for batch updates with separately supplied features and regression.

## Add a policy or representation

Implement `Actor` for ordinary inference or `StatefulActor` for causal assimilation and action selection. Sampling from a policy, reporting likelihoods, and supplying planner priors are separate capabilities. Use `PolicyGuide` or `ValueGuide` when the planner needs a probability row or continuation estimates.

Use `core::recording::StatefulRecording` when a collector needs extra sampling metadata. The recorder keys metadata by episode and timestep, while the collector retains the factual clock and actor memory.

Use `StateKeyBuilder` to define a finite key from a public information state. Use `FeatureMap` and its versioned `FeatureSpec` to encode factual histories and complete actions for statistical regression. Tensor encoders and value models live behind `neural`. Preserve semantic entity and action IDs across packing, and test any claimed permutation invariance with reordered frames.

## Add a model or planner

Implement `PlanningModel` for the public contract. Add `PredictiveModel` for sampling or `EnumerableModel` for exact outcomes. A model can provide either capability independently. Name the public context, complete action and outcome key, and the predictive clock and reward conventions. A planner can use the request, result and work types in `rl::planning`. Keep simulated transitions and RNG streams separate from the factual collector. [The model planning example](../crates/rl/examples/model_planning.rs) uses the same search interface with explicit probabilities and probabilities fitted from factual steps.
