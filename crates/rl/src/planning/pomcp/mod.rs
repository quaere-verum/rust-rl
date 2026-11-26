//! Root-sampled POMCP with legacy UCT and opt-in public PUCT/value guidance.
//!
//! Each simulation draws one initial particle and carries it through its
//! hypothetical history. The tree stores action/outcome statistics only.
//! The caller supplies a fresh factual filter state at every real decision.
/// Compatibility names for the shared predictive model contract.
use crate::models::predictive::PredictiveModel;
use crate::planning::{ActionEstimate, PlanStatus, PlanningBudget, PlanningWork};
use crate::{Error, Result};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::num::NonZeroU16;
mod bellman;
mod control;
mod halving;
use bellman::{BellmanEdge, running_mean};
pub(crate) use control::Controls;
pub use control::{Backup, RootSelection, SearchStrategy, TreeSelection};
use halving::SequentialHalving;

pub(super) const MAX_TREE_DEPTH: u16 = 256;
const MAX_TRANSITIONS: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub max_depth: NonZeroU16,
    pub gamma: f64,
    pub exploration: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutStep {
    /// Number of steps since entering the rollout.
    pub index: u16,
    /// Number of macro steps left in the trajectory's depth limit.
    pub remaining: u16,
}

pub enum RolloutDecision<A> {
    /// Execute a legal action through the same generative model.
    Act(A),
    /// Return an analytic continuation at the current simulated public history.
    /// The caller must respect the remaining depth and physical constraints.
    Value(f64),
    /// Finish the return with zero continuation value.
    End,
    /// Stop search at the continuation's work limit, discarding this trajectory.
    Exhausted,
}

#[derive(Debug, PartialEq)]
pub struct Search<A, O> {
    pub action: Option<A>,
    pub continuations: Vec<(O, A)>,
    pub estimates: Vec<ActionEstimate<A>>,
    pub status: PlanStatus,
    pub failure: Option<String>,
    pub work: PlanningWork,
}

impl<A, O> Search<A, O> {
    fn failed(mut self, error: impl ToString) -> Self {
        self.status = PlanStatus::Failed;
        self.failure = Some(error.to_string());
        self
    }
}

/// Sample a discrete law in fixed input order, advancing the planner stream
/// once. The positive-mass check also rejects malformed posterior inputs.
pub fn sample_weighted(
    weights: impl IntoIterator<Item = f64> + Clone,
    rng: &mut ChaCha8Rng,
) -> Result<usize> {
    let mut total = 0.;
    let mut last = None;
    for (index, mass) in weights.clone().into_iter().enumerate() {
        if !mass.is_finite() || mass < 0. {
            return Err(Error::Config("invalid POMCP particle mass".into()));
        }
        if mass > 0. {
            total += mass;
            last = Some(index);
        }
    }
    let last = last
        .filter(|_| total.is_finite())
        .ok_or_else(|| Error::Config("empty or unbounded POMCP particle law".into()))?;
    let mut draw = rng.random::<f64>() * total;
    for (index, mass) in weights.into_iter().enumerate() {
        if draw < mass {
            return Ok(index);
        }
        draw -= mass;
    }
    Ok(last)
}

struct Edge<A, O> {
    action: A,
    prior: f64,
    visits: u32,
    value: f64,
    children: Vec<(O, usize)>,
    bellman: Option<Box<BellmanEdge>>,
}

struct Node<A, O> {
    visits: u32,
    edges: Vec<Edge<A, O>>,
    bellman_value: Option<f64>,
}

struct PathStep {
    edge: Option<(usize, usize)>,
    reward: f64,
    discount: f64,
    child: Option<usize>,
}

impl<A: Clone, O> Node<A, O> {
    fn new(actions: &[A], prior: Option<&[f64]>) -> Self {
        Self {
            visits: 0,
            bellman_value: None,
            edges: actions
                .iter()
                .enumerate()
                .map(|(index, action)| Edge {
                    action: action.clone(),
                    prior: prior.map_or(0., |p| p[index]),
                    visits: 0,
                    value: 0.,
                    children: Vec::new(),
                    bellman: None,
                })
                .collect(),
        }
    }

    fn uct(&self, exploration: f64) -> Option<usize> {
        if let Some(index) = self.edges.iter().position(|e| e.visits == 0) {
            return Some(index);
        }
        let log_n = f64::from(self.visits).ln();
        stable_max(
            self.edges
                .iter()
                .map(|e| Some(e.value + exploration * (log_n / f64::from(e.visits)).sqrt())),
        )
    }

    fn best(&self) -> Option<usize> {
        stable_max(self.edges.iter().map(|e| (e.visits > 0).then_some(e.value)))
    }

    fn select(&self, selection: TreeSelection, exploration: f64) -> Option<usize> {
        match selection {
            TreeSelection::Uct => self.uct(exploration),
            TreeSelection::Puct { exploration } => {
                if let Some(index) = self.edges.iter().position(|e| e.visits == 0) {
                    return Some(index);
                }
                let scale = exploration * f64::from(self.visits).sqrt();
                stable_max(
                    self.edges
                        .iter()
                        .map(|e| Some(e.value + scale * e.prior / (1. + f64::from(e.visits)))),
                )
            }
        }
    }
}

fn stable_max(values: impl IntoIterator<Item = Option<f64>>) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, value) in values.into_iter().enumerate() {
        if let Some(value) = value
            && best.is_none_or(|(_, maximum)| value > maximum)
        {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
}

/// Bounded search with stable action-order ties and empirical mean deployment.
/// Work includes the root and all expanded history nodes, complete trajectories,
/// and every imaginary transition, including rollout and abandoned work.
/// A trajectory stopped mid-return by the work cap contributes zero visits.
pub fn search<G: PredictiveModel>(
    model: &G,
    root: &G::Context,
    feasible: &[G::Action],
    config: Config,
    budget: PlanningBudget,
    rng: &mut ChaCha8Rng,
) -> Result<Search<G::Action, G::Observation>> {
    search_impl(
        model,
        root,
        feasible,
        config,
        budget,
        rng,
        Controls::default(),
    )
}

/// Use a custom continuation at newly expanded histories. It receives only
/// the public context and depth, and owns its analytic work accounting.
pub fn search_with_rollout<G: PredictiveModel>(
    model: &G,
    root: &G::Context,
    feasible: &[G::Action],
    config: Config,
    budget: PlanningBudget,
    rng: &mut ChaCha8Rng,
    mut rollout: impl FnMut(&G::Context, RolloutStep) -> Result<RolloutDecision<G::Action>>,
) -> Result<Search<G::Action, G::Observation>> {
    search_impl(
        model,
        root,
        feasible,
        config,
        budget,
        rng,
        Controls {
            rollout: Some(&mut rollout),
            ..Default::default()
        },
    )
}

pub(crate) fn search_impl<G: PredictiveModel>(
    model: &G,
    root: &G::Context,
    feasible: &[G::Action],
    config: Config,
    budget: PlanningBudget,
    rng: &mut ChaCha8Rng,
    mut controls: Controls<'_, G::Context, G::Action>,
) -> Result<Search<G::Action, G::Observation>> {
    let PlanningBudget::Simulations {
        trajectories,
        transitions,
    } = budget
    else {
        return Err(Error::Config("POMCP requires a simulation budget".into()));
    };
    if config.max_depth.get() > MAX_TREE_DEPTH
        || transitions.get() > MAX_TRANSITIONS
        || !(0.0..=1.0).contains(&config.gamma)
        || !config.exploration.is_finite()
        || config.exploration < 0.
        || feasible
            .iter()
            .enumerate()
            .any(|(index, action)| feasible[..index].contains(action))
    {
        return Err(Error::Config(
            "invalid POMCP configuration or duplicate root action".into(),
        ));
    }
    if model.planning_returns().gamma != config.gamma {
        return Err(Error::Config(
            "POMCP discount differs from the model return contract".into(),
        ));
    }
    let mut result = Search {
        action: None,
        continuations: Vec::new(),
        estimates: Vec::new(),
        status: PlanStatus::NoAction,
        failure: None,
        work: PlanningWork::default(),
    };
    if feasible.is_empty() {
        return Ok(result);
    }
    let mut scratch = Vec::new();
    model.actions(root, &mut scratch);
    if feasible.iter().any(|action| !scratch.contains(action)) {
        return Err(Error::Config(
            "POMCP root action is outside the model's legal support".into(),
        ));
    }
    let root_prior = node_prior(&mut controls, root, feasible)?;
    let mut tree = vec![Node::new(feasible, root_prior.as_deref())];
    let mut halving = (controls.strategy.root_selection == RootSelection::SequentialHalving)
        .then(|| SequentialHalving::new(feasible.len(), trajectories.get()));
    let tree_layers = controls.tree_layers.unwrap_or(config.max_depth).get();
    let mut path = Vec::new();
    result.work.expansions = 1;
    // A rejected model call leaves the supplied stream untouched.
    let mut candidate_rng = rng.clone();
    'simulation: for _ in 0..trajectories.get() {
        if result.work.transitions == transitions.get() {
            result.status = PlanStatus::Exhausted;
            break;
        }
        let mut hidden = match model.sample_root(root, &mut candidate_rng) {
            Ok(hidden) => hidden,
            Err(error) => return Ok(result.failed(error)),
        };
        let mut context = root.clone();
        let mut node = Some(0);
        let mut rollout_index = 0;
        let mut value = 0.;
        path.clear();
        for depth in 0..config.max_depth.get() {
            if controls.require_terminal && result.work.transitions == transitions.get() {
                result.status = PlanStatus::Exhausted;
                break 'simulation;
            }
            if let Some((cutoff, leaf)) = controls.cutoff.as_mut()
                && depth >= *cutoff
            {
                match leaf(&context, config.max_depth.get() - depth) {
                    Ok(estimate) if estimate.is_finite() => value = estimate,
                    Ok(_) => return Ok(result.failed("POMCP cutoff returned a nonfinite value")),
                    Err(error) => return Ok(result.failed(error)),
                }
                break;
            }
            let edge = node.and_then(|index| {
                let forced = halving.as_ref().filter(|_| index == 0);
                forced
                    .map_or_else(
                        || tree[index].select(controls.selection, config.exploration),
                        |halving| Some(halving.select()),
                    )
                    .map(|choice| (index, choice))
            });
            let action = if let Some((index, choice)) = edge {
                tree[index].edges[choice].action.clone()
            } else if node.is_none() {
                if let Some(prior) = controls.rollout_prior.as_mut() {
                    scratch.clear();
                    model.actions(&context, &mut scratch);
                    if scratch.is_empty() {
                        return Ok(result.failed("POMCP live rollout lacks legal actions"));
                    }
                    let probabilities = match prior(&context, &scratch) {
                        Ok(probabilities) => probabilities,
                        Err(error) => return Ok(result.failed(error)),
                    };
                    let index =
                        match sample_weighted(probabilities.iter().copied(), &mut candidate_rng) {
                            Ok(index) => index,
                            Err(error) => return Ok(result.failed(error)),
                        };
                    scratch[index].clone()
                } else if let Some(rollout) = controls.rollout.as_mut() {
                    let step = RolloutStep {
                        index: rollout_index,
                        remaining: config.max_depth.get() - depth,
                    };
                    match rollout(&context, step) {
                        Ok(RolloutDecision::Act(action)) => {
                            scratch.clear();
                            model.actions(&context, &mut scratch);
                            if !scratch.contains(&action) {
                                return Ok(result.failed(
                                    "POMCP rollout action is outside the model's legal support",
                                ));
                            }
                            rollout_index += 1;
                            action
                        }
                        Ok(RolloutDecision::Value(leaf)) if leaf.is_finite() => {
                            value = leaf;
                            break;
                        }
                        Ok(RolloutDecision::Value(_)) => {
                            return Ok(result.failed("POMCP rollout returned a nonfinite value"));
                        }
                        Ok(RolloutDecision::End) => break,
                        Ok(RolloutDecision::Exhausted) => {
                            result.status = PlanStatus::Exhausted;
                            break 'simulation;
                        }
                        Err(error) => return Ok(result.failed(error)),
                    }
                } else {
                    let Some(action) = model.rollout(&context) else {
                        break;
                    };
                    action
                }
            } else {
                if controls.require_terminal {
                    return Ok(result.failed("POMCP live search node lacks legal actions"));
                }
                break;
            };
            if result.work.transitions == transitions.get() {
                result.status = PlanStatus::Exhausted;
                break 'simulation;
            }
            result.work.transitions += 1;
            let next = match model.simulate(&context, &mut hidden, &action, &mut candidate_rng) {
                Ok(next) => next,
                Err(error) => return Ok(result.failed(error)),
            };
            if !next.reward.is_finite() {
                return Ok(result.failed("POMCP imagined a nonfinite reward"));
            }
            let discount = config.gamma.powi(i32::from(next.duration.get()));
            path.push(PathStep {
                edge,
                reward: next.reward,
                discount,
                child: None,
            });
            let Some(after) = next.next else {
                break;
            };
            if controls.require_terminal && depth + 1 == config.max_depth.get() {
                return Ok(result.failed("POMCP trajectory violated its termination bound"));
            }
            context = after;
            node = None;
            if let Some((index, choice)) = edge
                && depth + 1 < tree_layers
            {
                let children = &tree[index].edges[choice].children;
                if let Some((_, child)) = children
                    .iter()
                    .find(|(event, _)| *event == next.observation)
                {
                    node = Some(*child);
                    path.last_mut().expect("recorded transition").child = node;
                } else {
                    if controls
                        .max_expansions
                        .is_some_and(|cap| result.work.expansions >= cap.get())
                    {
                        result.status = PlanStatus::Exhausted;
                        break 'simulation;
                    }
                    scratch.clear();
                    model.actions(&context, &mut scratch);
                    if controls.require_terminal
                        && (scratch.is_empty()
                            || super::guidance::validate_support(&scratch).is_err())
                    {
                        return Ok(result.failed("POMCP live node has invalid legal support"));
                    }
                    let prior = match node_prior(&mut controls, &context, &scratch) {
                        Ok(prior) => prior,
                        Err(error) => return Ok(result.failed(error)),
                    };
                    let child = tree.len();
                    tree.push(Node::new(&scratch, prior.as_deref()));
                    tree[index].edges[choice]
                        .children
                        .push((next.observation, child));
                    path.last_mut().expect("recorded transition").child = Some(child);
                    result.work.expansions += 1;
                    // POMCP evaluates the newly expanded history by rollout.
                }
            }
        }
        for step in path.iter().rev() {
            let tail = value;
            value = step.reward + step.discount * tail;
            if !value.is_finite() {
                return Ok(result.failed("POMCP simulated return overflow"));
            }
            if let Some((index, choice)) = step.edge {
                let (ancestors, descendants) = tree.split_at_mut(index + 1);
                let node = &mut ancestors[index];
                node.visits += 1;
                let edge = &mut node.edges[choice];
                edge.visits += 1;
                match controls.strategy.backup {
                    Backup::MonteCarloMean => {
                        edge.value = running_mean(edge.value, value, edge.visits);
                    }
                    Backup::Bellman => {
                        let direct = if step.child.is_some() {
                            step.reward
                        } else {
                            value
                        };
                        let continuation = step.child.map(|child| (child, step.discount, tail));
                        edge.value = edge.bellman.get_or_insert_default().update(
                            edge.visits,
                            direct,
                            continuation,
                            |child| descendants[child - index - 1].bellman_value,
                        );
                        if !edge.value.is_finite() {
                            return Ok(result.failed("POMCP Bellman return overflow"));
                        }
                        node.bellman_value = node.best().map(|best| node.edges[best].value);
                    }
                }
            }
        }
        result.work.simulations += 1;
        if let Some(halving) = &mut halving {
            halving.complete(&tree[0]);
        }
    }
    *rng = candidate_rng;
    let root = &tree[0];
    result.estimates = root
        .edges
        .iter()
        .filter(|e| e.visits > 0)
        .map(|edge| ActionEstimate {
            action: edge.action.clone(),
            value: edge.value,
            upper_bound: None,
            visits: Some(edge.visits),
        })
        .collect();
    let selected = halving
        .as_ref()
        .map_or_else(|| root.best(), |halving| halving.best(root));
    if let Some(choice) = selected {
        if result.status == PlanStatus::NoAction {
            result.status = PlanStatus::Approximate;
        }
        let best = &root.edges[choice];
        result.action = Some(best.action.clone());
        result.continuations = best
            .children
            .iter()
            .filter_map(|&(outcome, index)| {
                let node = &tree[index];
                let choice = node.best()?;
                Some((outcome, node.edges[choice].action.clone()))
            })
            .collect();
    }
    Ok(result)
}

fn node_prior<C, A: PartialEq>(
    controls: &mut Controls<'_, C, A>,
    context: &C,
    actions: &[A],
) -> Result<Option<Vec<f64>>> {
    if let Some(prior) = controls.prior.as_mut() {
        super::guidance::validate_support(actions)?;
        if actions.is_empty() {
            return Ok(None);
        }
        let probabilities = prior(context, actions)?;
        super::guidance::validate_prior(actions, &probabilities)?;
        Ok(Some(probabilities))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests;
