# Learning stability and planning

Each of these settings is optional. Use the model and return conventions that match your experiment.

## Critic initialization and backups

`CriticInitialization::Existing` retains the seeded model initialization. Use `Constant { value: 0.0 }` for a neutral start or an explicit optimistic value in the units of the expected reward and horizon. Apply it before constructing the learner so its target starts from the same parameters.

```rust
use rl::nn::critics::{CriticInitialization, scalar::DiscreteQ};

let critic = DiscreteQ::new(observation_width, action_count, &[32], &mut rng)?
    .with_initialization(CriticInitialization::Constant { value: 0.6 })?;
```

The scalar initializer zeros the last weight matrix and assigns the same bias to every scalar action output. The value 0.6 is the investigate-or-commit benchmark's immediate-commit value. Use the representation's own rules if initializing a distribution or moment pair. A point mass at \(v\) has raw first and second moments \(v\) and \(v^2\).

Candidate MPO uses `CandidateMpoBackup` and recurrent discrete MPO uses `MpoBackup`. `PolicyEvaluation` exactly averages legal actions under the pre-update actor. `MaximumOptimality` selects the largest target-critic estimate over the legal menu. `DoubleQOptimality` uses the online critic for action selection and the target critic for evaluation. The physical-time continuation discount and real-termination mask are shared with the existing learners. The default was already pre-update policy evaluation in both MPO paths, so the existing and current-policy experiment rows are the same. For a continuous stochastic actor, `objectives::mpo::sampled_current_policy_continuation` estimates the expectation with explicit caller-owned samples and detached target values. It is an operator for callers composing a continuous evaluation rule.

REINFORCE's existing `ReinforceConfig::entropy` regularizes the actor objective. Optional `Reinforce::with_categorical_exploration(CategoricalExploration { epsilon: 0.1, only_at: Some(root_observation) })` mixes the policy with the legal-uniform distribution at the supplied observation. Both collection and the REINFORCE likelihood use that mixture. Deterministic evaluation uses the underlying policy and the return target remains the environmental reward.

## Continuous critic choices

Wrap an ordinary critic to use frozen-policy residual coordinates in DDPG, TD3, D4PG, or MD4PG. The learner and its replay data continue to carry actual environment actions.

```rust
use rl::{
    algorithms::ddpg::Ddpg,
    nn::critics::residual::{ResidualQ, ResidualScale},
};

let critic = ResidualQ::new(critic, &policy, ResidualScale::Scalar(0.2))?;
let mut learner = Ddpg::new(policy, critic, Default::default(), seed)?;
let anchor = learner.policy();
learner.refresh_residual_anchor(&anchor)?;
```

Only refresh at a boundary chosen for the experiment. A refresh can change Q at the same state and action even if the critic network weights are identical. Each online/target pair receives the new anchor together. A diagonal scale can reflect different action units. The scale stays fixed through exploration-schedule changes. When the anchor is frozen, the derivative with respect to the environment action is the network's residual derivative multiplied by the inverse scale.

```rust
use rl::{
    algorithms::categorical_dpg::{CategoricalDpg, CategoricalDpgConfig},
    nn::critics::scalar::QValue,
    objectives::{categorical::CategoricalSupport, risk::Risk},
};

let support = CategoricalSupport::uniform(-500.0, 0.0, 51)?;
let critic = QValue::new(observation_width, action_width, support.atoms().len(), &[32], &mut rng)?;
let mut config = CategoricalDpgConfig::new(support);
config.risk = Risk::Cvar { alpha: 0.05 };
let learner = CategoricalDpg::new(policy, critic, config, seed)?;
```

Choose atom bounds and resolution from the return scale of the problem. The detached Bellman target linearly projects shifted mass between adjacent atoms and clips outside the support. Terminal reward projects as a point mass. The actor can optimize expected return or fractional lower-tail CVaR using the predicted probabilities. Quantile D4PG and REPPO's HL-Gauss regression retain their original training and historical formats.

## Search, exact planning, and transition fitting

`planning::guidance::SearchOptions::new(returns).with_strategy(SearchStrategy { backup: Backup::Bellman, root_selection: RootSelection::SequentialHalving })` supplies opt-in controls to `search_with_strategy`. Bellman search estimates an expectation from sampled chance outcomes and selects the best estimated legal continuation at decision nodes. Sequential halving divides the root budget among candidates, eliminates lower-valued candidates by round, and deploys a surviving visited action. UCT and PUCT remain available for tree selection. Reported visits describe the samples allocated, including work spent on eliminated actions.

Use `FiniteModel::for_horizon` with your enumerable public contexts and a per-action outcome cap, followed by `backward_induction` or `evaluate_horizon_policy`. The horizon counts decisions and the duration discounts each continuation in primitive time. Include remaining time in the public context when dynamics or legal actions depend on it. The same interface supports explicitly enumerable information states.

`core::reward::PotentialValues::shape` adds a supplied potential difference to an already aggregated decision reward. `QTable::q_learning_with_potential` uses this for the learning update and retains the factual reward. Set the successor potential to `None` at true termination or a modeled zero-value horizon. Supply the actual successor potential for a bootstrapped truncation.

`models::factored::FactoredModel::fit` fits observed finite variables using `Structure::Known` or `Structure::LearnBic`. Its configuration sets a symmetric prediction pseudocount and explicit bounds on candidate count, parent size, table cells, and fitting work. BIC uses unsmoothed likelihood with all possible action/parent rows counted in the parameter penalty. An unseen row with zero pseudocount predicts uniform mass.
