# Alpha execution model

The alpha execution demonstration uses a stylized simulator with independent Gaussian innovations and normalized P&L. Both D4PG configurations use the same environment. [Demonstration walkthrough](demonstrations.md)

## State and execution

At decision $t$, the observation contains signed inventory $q_t$, available alpha $a_t$, and the fraction $(T-t)/T$ of decisions remaining. An episode starts with zero inventory and alpha drawn uniformly from $[-0.5, 0.5)$. The action $u_t \in [-1,1]$ is the target inventory. It executes the signed trade $x_t=u_t-q_t$.

The trade costs $C(x)=s|x|+b x^2/\Delta t$ and reduces the magnitude of the available signal before it affects price. Write $\tilde a_t=a_t\exp(-\eta|x_t|)$ for the post-trade signal. The next price change and available alpha are

$$
\begin{aligned}
\Delta p_t &= \tilde a_t\Delta t+\sigma_p\sqrt{\Delta t}\,\varepsilon^p_t,\\
a_{t+1} &= \exp(-\kappa\Delta t)\tilde a_t+\sigma_\alpha\sqrt{\frac{1-\exp(-2\kappa\Delta t)}{2\kappa}}\,\varepsilon^\alpha_t.
\end{aligned}
$$

The innovations are independent standard normals. At $\kappa=0$, the factor inside the alpha square root is $\Delta t$.

Each reward is $r_t=u_t\Delta p_t-C(x_t)$. On the final decision, the same position earns the last price move before the simulator charges the additional liquidation cost $C(-u_t)$ and sets inventory to zero. The environmental return is the undiscounted sum of rewards. Midprice and cumulative P&L remain available as telemetry, and neither enters the transition for inventory or alpha. [Simulator source](../crates/rl/examples/support/alpha_execution.rs)

| Parameter | Demonstration value |
| --- | ---: |
| Horizon $T$ | 12 |
| Interval $\Delta t$ | 1 |
| Alpha depletion $\eta$ | 0.4 |
| Mean reversion $\kappa$ | 0.25 |
| Alpha volatility $\sigma_\alpha$ | 0.18 |
| Price volatility $\sigma_p$ | 0.35 |
| Spread coefficient $s$ | 0.015 |
| Temporary impact $b$ | 0.1 |

## Numerical benchmark

Alpha Grid DP estimates the best expected P&L by backward induction over 31 inventory points, 161 alpha points on $[-2,2]$, and 61 target actions. It integrates the next alpha using nine Gaussian quadrature nodes, interpolates continuation values, and clamps out-of-grid alpha to the nearest boundary for that calculation. Its values use the grid benchmark's continuation. The viewer's matched empirical return distributions instead use the recorded learner's frozen continuation after the selected initial action. [Benchmark source](../crates/rl/examples/support/alpha_execution_reference.rs)
