# Diagnostics and viewer

[Quick start](getting-started.md) · [Viewer manual](viewer.md).

Enable `diagnostics` for portable recording, evaluation and playback. `--no-default-features --features diagnostics` uses no Burn. Built-in neural learner adapters and historical model inference additionally require `neural`, which is enabled by default. The native viewer is a separate package and reads only the recording directory.

## Record and open an experiment

Train a bounded DDQN experiment on CartPole with the default Standard profile.

```sh
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- ddqn runs/cartpole
cargo run --locked --release -p rl-viewer -- --check runs/cartpole
cargo run --locked --release -p rl-viewer -- runs/cartpole
```

Use a fresh path. The viewer also accepts a parent directory containing recordings. Open an active recording with **Watch recording** enabled to keep indexing published data. The timeline keeps the requested checkpoint when new data arrives. Select the new checkpoint or play forward to inspect it. `--check` validates a directory without starting the viewer.

Launch the viewer without a path to choose **Open recording**. Browse to a recording or a directory containing several recordings and select **Open directory**. The application menu also offers this action while a run is open. Successfully opened directories appear under **Recent recordings** on the start screen. A directory that cannot be opened leaves the current run intact.

If a `demonstrations` directory is installed beside the viewer executable, the start screen offers **Open demonstration**. The directory may contain one recording or a collection of recordings.

For an ordinary dense environment, build a session from its public spaces, a frozen learner adapter, and optionally a factory for separate evaluation environments. The factory is invoked with an independent seed for each scenario. For example, the following program records a small CartPole run using the ordinary trainer and no reference.

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{diagnostics::session::DiagnosticSession, envs::cartpole::CartPole, prelude::*};

fn main() -> rl::Result<()> {
    let environment = CartPole::new(10);
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let mut learner = Ddqn::new(
        DiscreteQ::new(4, 2, &[16], &mut rng)?,
        DdqnConfig::default(),
    )?;
    let mut trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(vec![environment.clone()])?, 12)?,
        OffPolicyConfig {
            replay_capacity: 512,
            batch_size: 16,
            warmup_transitions: 16,
            ..Default::default()
        },
        13,
    )?;
    std::fs::create_dir_all("runs")?;
    let mut diagnostics = DiagnosticSession::builder("runs/cartpole")
        .environment(&environment)
        .discount(f64::from(0.99_f32))
        .evaluation_factory(|seed| Ok(CartPole::new(seed)))
        .build(&learner)?;
    trainer.train_observed(&mut learner, Budget::vector_steps(120), None, &mut diagnostics)?;
    diagnostics.finish(&learner)
}
```

Set the session discount to the trainer's return discount. `checkpoint(&learner, counters)` publishes an explicit completed revision for user-managed training or fitting. An existing learner or trainer can continue in a **new** recording, with the learner’s absolute revision carried forward. The recorder publishes immutable chunks and marks a run complete only after `finish`. After a recording error the completed learner update still belongs to the trainer. The session retains an incomplete recording for inspection.

Use `probe_set` to record a fixed population, a user-authored population, or an accessible historical training population. `ProbeSet::with_row_ids` supplies stable caller-owned identities. With an evaluation factory and no supplied fixed set, Standard and Detailed can freeze input rows from the first retained evaluation episode. Detailed can also sample a separate policy-visited population from independent evaluation trajectories and records its source revision. `environment_descriptor`, `evaluation_factory_with_frames` and `evaluation_factory_with_render` are opt-in presentation capabilities. The basic recording path requires only `Env`.

## Native workspaces

Training opens first when evaluation or metric history is present. The shared header keeps the run, learner checkpoint, learning transport, reference and selected input accessible. Ctrl+1 through Ctrl+6 select Training, Policy, Value, Network, Dynamics and Environment in that order. The input control identifies the actual probe row or trajectory decision.

The checkpoint control selects a recorded revision directly. The timeline plays, steps and scrubs through available checkpoints. A separate shown revision identifies a retained complete frame while a new request is being prepared. Plots support pan and zoom. Double-click a plot to reset its view where enabled. [Viewer manual](viewer.md)

| Workspace | Main view | Inspect |
| --- | --- | --- |
| Training | Environmental return or a recorded metric over learning | Episode uncertainty, selected checkpoint, benchmark performance, compatible runs and independent training seeds |
| Policy | Action mappings and legal probabilities | Actual action-coordinate marginals, reference actions, local decision regret and fixed-population policy evolution |
| Value | V and Q slices or maps | Twin values, raw moments, HL-Gauss regression bins, conditional returns and compatible calibration |
| Network | Activations and representations | Feature IDs, fixed-basis representation PCA and dense network geometry with supported interventions |
| Dynamics | Trajectories, transition flow and coverage | Exact recorded decisions, evaluation visitation and published factual replay samples |
| Environment | Episode scene or numerical observation | Executed action, reward, elapsed physical time, final flags and matched benchmark episode where supplied |

In Training, choose environment decisions, learner revision, primitive ticks or recorded elapsed wall time. Each point refers to its recorded checkpoint. A missing clock stays unavailable for a legacy recording that lacks it. Environment and physical steps can represent different units when one decision executes several primitive steps. Wall time is run-local elapsed time, including trainer and diagnostic work between measurements.

When compatible scalar metrics are recorded, Training shows up to two beneath the main curve. Use **Secondary plots** to choose them. The vertical cursor identifies the selected checkpoint on each plot; selecting a point changes the shared checkpoint.

Open a common parent directory to select a compatible run as the global reference. **Training seeds** enrolls distinct seed recordings of the same authored experiment, algorithm and learner configuration. Both comparison modes require matching evaluation protocol and return conventions. Cohort estimates contribute one value per enrolled training run. They interpolate only between compatible measurements within common recorded support and break at missing evaluations. The across-run standard error and within-run episode standard error describe different sources of variation.

Open **Policy → Evolution** with a fixed probe population selected. Each point is an entire historical policy evaluated on that population. Discrete features use legal-action probability vectors. Deterministic continuous features use normalized executed actions. Gaussian features use latent means and log scales, with the actual-action distribution available in Mapping. Every point in a displayed history uses one basis. Network representation PCA instead places the inputs of a single layer in feature space. Shared Network axes and CKA require matching complete inputs and feature identities. The Network **Geometry** view exposes dense-network interventions.

In Value, adopt a recorded empirical condition to fix its exact observation, mask and first action. The return profile supports CDF, inverse CDF and empirical histogram views. **Expand distribution** gives a recorded quantile forecast the available workspace; **Restore layout** returns to the map. Wasserstein-1 and mean CRPS use the ordered weighted atoms and the independently recorded conditional samples. Raw heads and their crossings remain inspectable.

In Dynamics, **Replay sample** appears when a recording publishes factual replay evidence. Its points retain the stored actions. Independent evaluation visits use the trajectory workspace. In Environment, using the episode transport changes the recorded decision while preserving learner time. On wide windows, rendered episodes show recorded numerical observations and continuous actions beside the scene; **Channels** selects up to five. Bounded traces retain interval extrema and endpoints and follow the selected state; selecting a time on a trace requests that state. Labels and units come from the recorded descriptor, including any scaling. **Compare matched benchmark** uses a separately controlled episode with the same authored scenario and return protocol.

### Finite decisions and typed episodes

For discrete and conditional actions, Policy displays the legal menu at the selected input. Probabilities describe the learner's published action law. The highlighted row is the representative or recorded selection. The executed action in Environment belongs to the actual episode. Value displays the recorded prediction for each legal action. A dash represents an unavailable quantity. A zero remains an observed numerical value.

State symbols and table keys retain their exact integer or structured identities. A display projection can position a state on a map. The recorded observation or public history remains the model input. In a graph recording, the selected input lists public scalar values, entities, events, nodes and edges. A conditional action identifies its operation and its entity or location. The same action can occupy different positions in successive menus.

Tabular **Updates** counts which pairs the table has updated. A pair with zero updates uses the table's original initialization. FQI and LSPI expose fitted values and fit diagnostics in Training. Residuals describe the fitted objective. Independent value calibration requires a matching return target. Retrieval exposes preference scores and the executed action law as separate quantities.

The reference column joins two menus by action identity and requires matching recorded inputs. Cross-run comparisons require the same declared scientific context. Exact optimal values can support verified decision regret when every legal action and the learner's probability law are available. Retrieval preference and Q share the action identities but have different numerical interpretations.

Network is available when the recording has neural representations. Candidate learners can expose pooled graph or structured critic activations by feature index. Sparse tabular snapshots can reconstruct supported discrete-symbol queries at their exact revisions. Fitted or retrieval snapshots can reconstruct their model in Rust when the feature contract is available. Unknown feature closures and recurrent hidden-state runtimes use recorded predictions.

Typed Environment episodes identify training or independent evaluation provenance and retain their decision times. Dynamics counts visits from the selected episode. Network and input inspectors show recorded features and structure. Portable scene tracks and matched-scene comparison are available on the dense recordings that supply those tracks.

## Schedules and resource limits

| Profile | Metrics | Policy and critic probes | Independent evaluation | Reference | Snapshots | Representations | Conditional evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Light | 1 | — | 50 | — | — | — | — |
| Standard | 1 | 20 | 50 | 50 | 100 | — | — |
| Detailed | 1 | 20 | 50 | 50 | 100 | 50 | 100 |

Intervals count completed learner updates. When one trainer call crosses multiple intervals, the session observes the available final revision once. Initial and final boundaries are enabled for the channels configured in the profile. Duplicate captures at the same revision are suppressed per channel. Standard is the default.

Adjust `DiagnosticConfig::schedule`, `evaluation`, `conditional_evaluation`, `snapshot_retention`, `max_probe_sets`, `max_probe_rows`, `visited_probe_rows`, `max_activation_values` and `storage` before building. Light and Standard evaluate four seeds, retain one trajectory per batch, and cap each scenario at 256 decisions. Detailed uses eight seeds, two trajectories and 512 decisions. The optional conditional workload defaults to 32 futures per configured condition with a 128-decision guard and no retained conditional trajectories. Fixed budgets also bound admitted scenario, trajectory and reference work. The profiles have 64, 256 and 512 MiB storage budgets respectively. Standard and Detailed retain up to eight snapshots per model role.

A guard limits work. Use `EvaluationHorizon::FiniteDecisions(n)` when stopping at `n` is part of the evaluation objective itself. A time-limit or guard prefix is incomplete unless it coincides with the declared finite objective. Evaluation advances the discount by physical action duration and records the within-action discounted reward and duration when needed. Older single-step recordings keep their original interpretation. Use `conditional_case` for samples from a fully specified simulator state. Use `conditional_observation_case` only with an independently seeded sampler for the declared observation condition. Its program must initialize every scenario and provide the exact public observation and mask.

## Reproducible demonstrations

```sh
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- ppo runs/ppo-seed-9107 detailed --reference --steps=240
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- ppo runs/ppo-seed-12347 detailed --reference --steps=240 --seed=12347
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- ddqn runs/cartpole-ddqn detailed --render --reference --replay-population
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- sac runs/pendulum-sac detailed --render --reference
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- reppo runs/cartpole-reppo detailed --render --reference
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- ddpg runs/ddpg-wide detailed --wide-lqg
```

Use a fresh destination for each run. The default seed is 9107. An explicit training seed changes factual collection and learner initialization while leaving the authored evaluation scenarios matched. The bundled rocket and execution recordings have their own reproduction instructions. [Demonstration guide](demonstrations.md)

## Benchmarks and analytical references

Run PPO against the included analytical LQG reference.

```sh
cargo run --locked --release -p rust-rl --features diagnostics --example diagnostic_session -- ppo runs/ppo-lqg detailed --reference
cargo run --locked --release -p rl-viewer -- runs/ppo-lqg
```

The [example source](../crates/rl/examples/diagnostic_session.rs) performs the complete setup. It builds `LqgReference::new(&environment_config, discount)` for an unbounded-action LQG problem, attaches the scientific context of that environment with `reference_context`, and passes the provider to `reference`. Its deterministic Riccati controller, V, Q and analytical action derivatives use the finite-horizon expected discounted return convention. Its values describe the idealized unbounded dynamics. Model-coordinate decision regret uses each supplied coordinate as an exact state of that model. The empirical performance gap compares the actual rounded-input controller as a benchmark. The declaration records the f32 observation and action interface and the numerical assumptions.

A signed benchmark gap compares the two policies' measured environmental returns, including the sign. A verified optimality gap requires a proven optimum for the declared observation and objective. Local decision regret uses optimal continuation after the inspected decision. Critic calibration measures the prediction against its own declared target and continuation. A compatible gradient comparison measures local critic sensitivity at the same state and action.

For an ordinary policy, pass `BenchmarkActor::new(specification, actor)?`. It exposes the actor's decisions and independent performance. Set the evaluation mode explicitly when wrapping a deterministic actor. To expose analytical or numerical values, implement the optional methods on `ReferenceProvider` and return `None` outside the supported domain. `ReferenceSpec` declares whether values describe a benchmark policy, a verified optimum or the current learner’s frozen policy. It also declares the reference’s information, provenance and uncertainty. Match the actual experiment context, including environment configuration, reward, observation, constraints, horizon and discount. A checkpoint-dependent provider receives the frozen learner policy and its revision for each capture. Use ReferenceTarget::CriticContinuation to evaluate the frozen target controller used by TD3 or D4PG, including its declared smoothing mode.

The signed performance gap is reference return minus learner return. A verified optimal reference enables the optimality-gap label and local decision regret when the required values and policy law are available. A stochastic continuous policy uses sampled actions with reported standard error. Critic calibration requires compatible critic and reference targets, exact continuations and matched input conditioning. Probe summaries weight states equally and, for discrete Q, legal actions equally within each state. Action-gradient comparisons distinguish analytic, automatic and finite-difference methods. Incompatible objectives, unavailable evidence and privileged observations appear explicitly. The viewer displays reference predictions on their recorded inputs when no reconstructable reference exists.

## Built-in capabilities and extensions

| Learner | Policy | Critic | Historical snapshots |
| --- | --- | --- | --- |
| PPO | Discrete or continuous stochastic policy | State V | Online policy and V |
| REINFORCE | Discrete or continuous stochastic policy | State V after a learned baseline is enabled and updated | Online policy and learned baseline |
| V-MPO | Discrete or continuous stochastic policy | State V trained against rollout-behavior targets | Online policy and V |
| MDPO | Discrete or continuous stochastic policy | State V trained against rollout-behavior targets | Online policy and V |
| REPPO | Masked discrete or continuous stochastic policy | HL-Gauss regression of entropy-regularized action value | Online policy, normalized critic encoder, head and auxiliary predictor |
| DDQN | Legal-action-aware greedy evaluation | All discrete Q heads and optimality continuation | Online and target Q |
| DDPG | Deterministic deployed policy | Scalar Q with unsmoothed target continuation and action derivatives | Online and target actors and critics |
| TD3 | Deterministic deployed policy | Twin scalar Q, target-policy smoothing, action derivatives away from kinks | Online and target actors and twins |
| SAC | State-dependent diagonal Gaussian, including affine tanh for bounded actions | Twin soft Q and entropy temperature | Online and target critics with the frozen online policy |
| D4PG | Deterministic deployed policy | Raw quantile heads, locations and masses, ordered inference, target continuation | Online and target actors and quantile critics |
| MD4PG | Deterministic deployed policy | First and raw second return moments, implied variance and configured utility | Online and target actors and both moment heads |

SAC's one-step target adds successor policy entropy with the snapshot temperature and the log density of the same action draw used in the target. Diagnostic evaluation episodes continue to report environmental returns. The optional `evaluate_soft_action_value` API estimates conditional soft returns under a frozen Gaussian actor and fixed temperature through true termination. It returns typed samples separately from environmental trajectory records or reference calibration.

REPPO uses one-hot actions for discrete critic queries and checks the complete legal-action mask. Its normalized encoder and optional trained predictor supply named features. The HL-Gauss softmax weights describe scalar regression bins. The decoded soft Q is their weighted center. The critic's entropy term uses rollout behavior and the entropy multiplier available when targets are constructed. The snapshot records the current multiplier at that revision, and the separate KL multiplier describes the actor constraint.

For MD4PG, implied variance is the raw second moment minus the squared first moment. The viewer displays signed variance and omits standard deviation when it is negative. The mean-variance utility follows the learner's configured risk operation, which uses a variance floor of zero.

Implement `DiagnosticLearner` for a custom learner or local newtype. Return a revision and metadata, supply an independently frozen evaluation actor, and implement only the record capabilities available at `CaptureRequest`. `ProbeSet`, activation helpers and `SnapshotBuilder` compose the built-in recording formats. A custom architecture may emit portable policy and critic predictions without supplying a reconstructable snapshot. A scalar critic does not specify a return distribution, and two moments do not identify a distribution. The viewer retains raw quantile crossing evidence and uses ordered atoms at the recorded grid probabilities for distributional calculations.

The generic `rl-precompute` binary requires `diagnostics` and `neural`. Learner checkpointing and viewer recordings have separate lifecycles. [Checkpoint guide](checkpointing.md)
