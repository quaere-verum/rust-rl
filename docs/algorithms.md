# Algorithms

| Family | Data and requirements |
| --- | --- |
| Tabular Q-learning and SARSA | One factual decision at a time with explicit state and finite action keys. Q-learning maximizes over the successor's legal actions. SARSA uses the successor action actually selected by the policy. |
| Tabular Monte Carlo | First-visit state-action return averages with an optional explicit time-limit continuation value. Evaluate a fixed policy or improve the behavior between complete episodes. |
| FQI and LSPI | Factual batches and a caller-provided feature map, including conditional graph menus. FQI performs frozen target sweeps and legal-action maximization. LSPI uses a weighted LSTDQ policy evaluation and greedy improvement. |
| Return-weighted retrieval | Separately scaled public-state and semantic-action features. A bounded online FIFO retains positively weighted factual decisions. Inference uses nearest states and scores the supplied legal menu. |
| PPO, REINFORCE, V-MPO, MDPO, REPPO | Supported on-policy learners for masked discrete or continuous policies. Use the algorithm's rollout, episode, or pathwise collection contract. Supported temporal variants retain causal memory, and recurrent conditional variants use semantic candidate actions. |
| DDQN and recurrent discrete MPO | Masked finite discrete actions and off-policy replay. Candidate DDQN and MPO use complete semantic candidates, with feedforward and recurrent implementations where the chosen model supports them. |
| DDPG, TD3, D4PG, MD4PG, categorical DPG, SAC | Continuous off-policy methods. D4PG uses quantile critics. MD4PG learns raw moments. Categorical DPG projects return mass onto fixed atoms. SAC uses one-step stochastic replay and its own entropy objective. |
| Recurrent ACER | Enable `experimental-acer` to use `algorithms::experimental`. Its update consumes exact chronological replay with recorded full categorical behavior probabilities. |

Tabular and statistical methods work with the lightweight build. Feature extraction and regression are separate from batch Bellman updates. Regression provides linear and regularized fits, including local models through gated feature maps. `GaussianPosterior` provides standalone online Bayesian linear regression and predictive uncertainty. A tabular state key should describe a sufficient public information state or declare its approximation. Use [the environment API](environment-api.md) when defining keys and legal action spaces.

All rewards use physical-time semantics. A multi-tick action accumulates its primitive rewards in order, with continuation discounted by its duration. True termination cuts off continuation. External truncation can bootstrap from the actual final observation. A collection boundary leaves the episode live. Use the learner's declared return convention when preparing its replay or targets.

Neural policies keep collection-time masks and likelihoods. Recurrent learners consume causal observations and previous factual inputs, and use an exact prefix or an explicitly supported validated anchor. The saved model and optimizer belong to the learner. Evaluation uses published actor state. See [checkpointing](checkpointing.md) for restoration boundaries.

Exact finite planning, contingent search, and POMCP are covered in the [planning guide](planning.md).

## Learning stability configuration

The [optional algorithms guide](learning-stability.md) covers critic initialization, current-policy and optimality backups, residual and categorical critics, planning controls, potential-based reward shaping, and factored transition fitting.
