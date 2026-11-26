//! Conditional action heads over lightweight observations or recurrent states.
use super::{ConditionalPolicyLaw, LightweightConfig};
use crate::{
    Error, Result,
    core::{
        action::{Action, ConditionalAction},
        actor::ActMode,
        ensure,
        structured::{ActionCandidate, StructuredObservation},
    },
    nn::{
        distributions::Distribution,
        network::{Mlp, Network},
        structured::{
            ArgumentKind, Candidate, Conditional, StructuredDecision, StructuredSample,
            candidate_policy::categorical, policy::hierarchical_greedy,
        },
        to_vec,
    },
};
use burn::tensor::{Int, Tensor, TensorData};
use rand_chacha::ChaCha8Rng;

// Both input paths retain their original candidate keys and row order.
pub trait PolicyCandidate {
    type Key;

    fn key(&self) -> &Self::Key;
    fn operation(&self) -> usize;
    fn enabled(&self) -> bool;
}

impl PolicyCandidate for ActionCandidate {
    type Key = ConditionalAction;

    fn key(&self) -> &Self::Key {
        &self.action
    }

    fn operation(&self) -> usize {
        self.action.operation
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
}

impl<K> PolicyCandidate for Candidate<K> {
    type Key = K;

    fn key(&self) -> &Self::Key {
        &self.key
    }

    fn operation(&self) -> usize {
        self.operation
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
}

pub trait PolicyFrame {
    type Candidate: PolicyCandidate<Key = Self::Key>;
    type Key;

    fn candidates(&self) -> &[Self::Candidate];
}

impl PolicyFrame for StructuredObservation {
    type Candidate = ActionCandidate;
    type Key = ConditionalAction;

    fn candidates(&self) -> &[Self::Candidate] {
        &self.candidates
    }
}

impl<K> PolicyFrame for StructuredSample<K> {
    type Candidate = Candidate<K>;
    type Key = K;

    fn candidates(&self) -> &[Self::Candidate] {
        &self.candidates
    }
}

#[derive(Clone, Copy)]
pub(super) struct ConditionalHeads<'a> {
    pub config: &'a LightweightConfig,
    pub operations: &'a Mlp,
    pub operation_head: &'a Mlp,
    pub entity_query: &'a Mlp,
    pub entity_key: &'a Mlp,
    pub location_query: &'a Mlp,
    pub location_key: &'a Mlp,
}

/// An encoded group of observations and their enabled conditional action laws.
pub struct LightweightPolicyEvaluation<'p, 'a, F> {
    pub states: Tensor<2>,
    heads: ConditionalHeads<'p>,
    observations: Vec<&'a F>,
    candidate_offsets: Vec<usize>,
    keys: Tensor<2>,
    operation_embeddings: Tensor<2>,
    operations: Distribution,
    live: usize,
}

pub(super) fn query_width(c: &LightweightConfig, state_width: usize) -> usize {
    state_width + c.operation_width
}

pub(super) fn key_width(c: &LightweightConfig) -> usize {
    c.target_width() + c.input.candidate_features + 5
}

pub(super) fn head_parameter_count(c: &LightweightConfig, state_width: usize) -> usize {
    let query = query_width(c, state_width);
    let key = key_width(c);
    (query + key + 2) * c.head_width + c.head_width + 1 + 2 * (query + key + 2) * c.entity_width
}

pub(super) fn check_head_budget(
    c: &LightweightConfig,
    state_width: usize,
    count: usize,
    candidates: usize,
) -> Result<usize> {
    let ops = c.input.operations.len();
    c.check_shape(&[
        count,
        ops,
        (query_width(c, state_width) + key_width(c) + 1).max(c.head_width),
    ])?;
    c.check_shape(&[candidates.max(1), key_width(c).max(c.entity_width)])?;
    c.check_shape(&[ops, ops.max(c.operation_width)])?;
    // Operation indicators, masks/counts and row offsets, including terminal rows.
    let rows = count
        .checked_mul(ops)
        .ok_or_else(|| Error::Shape("actor operation input overflow".into()))?;
    ops.checked_mul(ops)
        .and_then(|n| rows.checked_mul(2).and_then(|r| n.checked_add(r)))
        .and_then(|n| count.checked_mul(2).and_then(|r| n.checked_add(r)))
        .ok_or_else(|| Error::Shape("actor input allocation overflow".into()))
}

impl<'p> ConditionalHeads<'p> {
    pub(super) fn evaluate<'a, F: PolicyFrame>(
        self,
        states: Tensor<2>,
        targets: Tensor<2>,
        arguments: Tensor<2>,
        groups: Vec<Vec<&'a F>>,
    ) -> Vec<LightweightPolicyEvaluation<'p, 'a, F>> {
        let c = self.config;
        let ops = c.input.operations.len();
        let device = states.device();
        let mut identity = vec![0.; ops * ops];
        for i in 0..ops {
            identity[i * ops + i] = 1.;
        }
        let operation_embeddings = self.operations.forward(Tensor::from_data(
            TensorData::new(identity, [ops, ops]),
            &device,
        ));
        let keys = Tensor::cat(vec![targets, arguments], 1);
        let (mut observation_offset, mut candidate_offset) = (0, 0);
        let mut evaluations = Vec::with_capacity(groups.len());
        for observations in groups {
            let count = observations.len();
            let live = count
                - usize::from(
                    !observations[count - 1]
                        .candidates()
                        .iter()
                        .any(PolicyCandidate::enabled),
                );
            let mut candidate_offsets = Vec::with_capacity(count);
            let mut candidates = 0;
            for sample in &observations {
                candidate_offsets.push(candidates);
                candidates += sample.candidates().len();
            }
            let states = states.clone().narrow(0, observation_offset, count);
            let local_keys = if candidates == 0 {
                Tensor::zeros([0, key_width(c)], &device)
            } else {
                keys.clone().narrow(0, candidate_offset, candidates)
            };
            let mut inputs = Vec::with_capacity(live * ops);
            let mut masks = Vec::with_capacity(live * ops);
            for (row, sample) in observations.iter().take(live).enumerate() {
                for operation in 0..ops {
                    let targets: Vec<_> = sample
                        .candidates()
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.enabled() && c.operation() == operation)
                        .map(|(i, _)| local_keys.clone().narrow(0, candidate_offsets[row] + i, 1))
                        .collect();
                    let n = targets.len();
                    masks.push(n > 0);
                    let summary = if n == 0 {
                        Tensor::zeros([1, key_width(c)], &device)
                    } else {
                        Tensor::cat(targets, 0).mean_dim(0)
                    };
                    inputs.push(Tensor::cat(
                        vec![
                            states.clone().narrow(0, row, 1),
                            operation_embeddings.clone().narrow(0, operation, 1),
                            summary,
                            Tensor::from_data(
                                TensorData::new(vec![(n as f32).ln_1p()], [1, 1]),
                                &device,
                            ),
                        ],
                        1,
                    ));
                }
            }
            let logits = if inputs.is_empty() {
                Tensor::zeros([0, ops], &device)
            } else {
                self.operation_head
                    .forward(Tensor::cat(inputs, 0))
                    .reshape([live, ops])
            };
            evaluations.push(LightweightPolicyEvaluation {
                states,
                observations,
                candidate_offsets,
                keys: local_keys,
                operation_embeddings: operation_embeddings.clone(),
                operations: Distribution::Categorical { logits, masks },
                heads: self,
                live,
            });
            observation_offset += count;
            candidate_offset += candidates;
        }
        evaluations
    }
}

impl<'a, F: PolicyFrame> LightweightPolicyEvaluation<'_, 'a, F>
where
    F::Key: Clone + PartialEq,
{
    pub fn operations(&self) -> &Distribution {
        &self.operations
    }
    pub fn conditional(
        &self,
        sample: usize,
        operation: usize,
    ) -> Result<Option<Conditional<'a, F::Key>>> {
        let c = self.heads.config;
        ensure(
            sample < self.live && operation < c.input.operations.len(),
            "lightweight conditional branch out of range",
        )?;
        let candidates = self.observations[sample].candidates();
        let rows: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.operation() == operation)
            .map(|(i, _)| i)
            .collect();
        ensure(
            rows.iter().any(|&i| candidates[i].enabled()),
            "unavailable lightweight conditional branch",
        )?;
        let kind = c.input.operations[operation];
        if kind == ArgumentKind::None {
            return Ok(None);
        }
        let inputs = Tensor::cat(
            rows.iter()
                .map(|&i| {
                    self.keys
                        .clone()
                        .narrow(0, self.candidate_offsets[sample] + i, 1)
                })
                .collect(),
            0,
        );
        let query = Tensor::cat(
            vec![
                self.states.clone().narrow(0, sample, 1),
                self.operation_embeddings.clone().narrow(0, operation, 1),
            ],
            1,
        );
        let (query, keys) = match kind {
            ArgumentKind::Entity => (
                self.heads.entity_query.forward(query),
                self.heads.entity_key.forward(inputs),
            ),
            ArgumentKind::Location => (
                self.heads.location_query.forward(query),
                self.heads.location_key.forward(inputs),
            ),
            _ => unreachable!("validated conditional grammar"),
        };
        let logits =
            (keys * query).sum_dim(1).reshape([1, rows.len()]) / (c.entity_width as f32).sqrt();
        let masks = rows.iter().map(|&i| candidates[i].enabled()).collect();
        let keys = rows.iter().map(|&i| candidates[i].key()).collect();
        Ok(Some(Conditional {
            rows,
            keys,
            distribution: Distribution::Categorical { logits, masks },
        }))
    }

    fn candidates(&self, sample: usize) -> Result<&'a [F::Candidate]> {
        if sample >= self.live {
            return Err(Error::Shape("lightweight policy row is unavailable".into()));
        }
        Ok(self.observations[sample].candidates())
    }

    /// Joint log probabilities in enabled public candidate order. Each live
    /// branch is evaluated once and all returned terms retain autodiff.
    pub fn complete_log_probs(&self, sample: usize) -> Result<(Vec<F::Key>, Tensor<1>)> {
        let candidates = self.candidates(sample)?;
        let operations = self.operations().categorical_terms()?.log_probs;
        self.complete_log_probs_using(sample, candidates, &operations)
    }

    fn complete_log_probs_using(
        &self,
        sample: usize,
        candidates: &[F::Candidate],
        operations: &Tensor<2>,
    ) -> Result<(Vec<F::Key>, Tensor<1>)> {
        ensure(
            candidates.len() <= i32::MAX as usize,
            "lightweight candidate lookup exceeds index range",
        )?;
        let mut branch_logs = Vec::new();
        let mut candidate_rows = vec![0i32; candidates.len()];
        let mut offset = 0;
        for operation in 0..operations.dims()[1] {
            let Some(first) = candidates
                .iter()
                .position(|candidate| candidate.enabled() && candidate.operation() == operation)
            else {
                continue;
            };
            let log_op = operations
                .clone()
                .slice([sample..sample + 1, operation..operation + 1])
                .reshape([1]);
            if let Some(conditional) = self.conditional(sample, operation)? {
                for (target, &row) in conditional.rows.iter().enumerate() {
                    candidate_rows[row] = (offset + target) as i32;
                }
                let count = conditional.rows.len();
                let log_target = conditional.distribution.categorical_terms()?.log_probs;
                branch_logs.push(log_op + log_target.reshape([count]));
                offset += count;
            } else {
                candidate_rows[first] = offset as i32;
                branch_logs.push(log_op);
                offset += 1;
            }
        }
        let (keys, rows): (Vec<_>, Vec<_>) = candidates
            .iter()
            .zip(candidate_rows)
            .filter(|(candidate, _)| candidate.enabled())
            .map(|(candidate, row)| (candidate.key().clone(), row))
            .unzip();
        let indices =
            Tensor::<1, Int>::from_data(TensorData::new(rows, [keys.len()]), &operations.device());
        Ok((keys, Tensor::cat(branch_logs, 0).select(0, indices)))
    }

    /// Host f64 probabilities for the factored policy in public candidate order.
    pub fn complete_probabilities_f64(&self, sample: usize) -> Result<Vec<f64>> {
        let candidates = self.candidates(sample)?;
        let operations = categorical(self.operations(), sample)?;
        let mut probabilities = vec![0.; candidates.len()];
        for (operation, op_probability) in operations
            .into_iter()
            .enumerate()
            .filter(|&(_, probability)| probability > 0.)
        {
            if let Some(conditional) = self.conditional(sample, operation)? {
                for (row, probability) in conditional
                    .rows
                    .into_iter()
                    .zip(categorical(&conditional.distribution, 0)?)
                {
                    probabilities[row] = op_probability * probability;
                }
            } else {
                let row = candidates
                    .iter()
                    .position(|c| c.operation() == operation && c.enabled())
                    .expect("validated no-argument branch");
                probabilities[row] = op_probability;
            }
        }
        let sum: f64 = probabilities.iter().sum();
        for probability in &mut probabilities {
            *probability /= sum;
        }
        if probabilities
            .iter()
            .zip(candidates)
            .any(|(&probability, candidate)| candidate.enabled() && probability == 0.)
        {
            return Err(Error::Numerical("MPO joint support underflow".into()));
        }
        Ok(probabilities)
    }

    /// Teacher-force the requested keys, evaluating each needed conditional head once.
    pub fn log_prob_many(&self, sample: usize, keys: &[F::Key]) -> Result<Tensor<1>> {
        self.log_prob_many_using(sample, keys, || {
            Ok(self.operations().categorical_terms()?.log_probs)
        })
    }

    /// One chosen key per requested row, with one operation normalization for the batch.
    pub fn selected_log_probs(&self, selections: &[(usize, F::Key)]) -> Result<Tensor<1>> {
        ensure(
            !selections.is_empty(),
            "empty lightweight teacher selections",
        )?;
        let operations = self.operations().categorical_terms()?.log_probs;
        let terms = selections
            .iter()
            .map(|(sample, key)| {
                self.log_prob_many_using(*sample, std::slice::from_ref(key), || {
                    Ok(operations.clone())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(terms, 0))
    }

    fn log_prob_many_using(
        &self,
        sample: usize,
        keys: &[F::Key],
        operations: impl FnOnce() -> Result<Tensor<2>>,
    ) -> Result<Tensor<1>> {
        let c = self.heads.config;
        c.check_shape(&[keys.len(), c.head_width])?;
        let candidates = self.candidates(sample)?;
        ensure(!keys.is_empty(), "empty lightweight teacher keys")?;
        let rows = keys
            .iter()
            .map(|key| {
                candidates
                    .iter()
                    .position(|c| c.key() == key && c.enabled())
                    .ok_or_else(|| {
                        Error::Action("teacher action is not an enabled candidate".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let operations = operations()?;
        let mut terms = vec![None; keys.len()];
        for operation in 0..c.input.operations.len() {
            if !rows
                .iter()
                .any(|&row| candidates[row].operation() == operation)
            {
                continue;
            }
            let op = operations
                .clone()
                .slice([sample..sample + 1, operation..operation + 1])
                .reshape([1]);
            let conditional = self.conditional(sample, operation)?;
            let logs = conditional
                .as_ref()
                .map(|c| {
                    c.distribution
                        .categorical_terms()
                        .map(|terms| terms.log_probs)
                })
                .transpose()?;
            for (i, &row) in rows
                .iter()
                .enumerate()
                .filter(|(_, row)| candidates[**row].operation() == operation)
            {
                terms[i] = Some(
                    if let (Some(conditional), Some(logs)) = (&conditional, &logs) {
                        let target = conditional
                            .rows
                            .iter()
                            .position(|&target| target == row)
                            .expect("validated teacher target");
                        op.clone() + logs.clone().slice([0..1, target..target + 1]).reshape([1])
                    } else {
                        op.clone()
                    },
                );
            }
        }
        Ok(Tensor::cat(
            terms
                .into_iter()
                .map(|term| term.expect("all requested branches evaluated"))
                .collect(),
            0,
        ))
    }

    /// Select the most probable enabled operation and then its semantic target.
    pub fn greedy(&self, sample: usize) -> Result<StructuredDecision<F::Key>>
    where
        F::Key: Ord,
    {
        let candidates = self.candidates(sample)?;
        hierarchical_greedy(
            self.operations(),
            sample,
            |operation| self.conditional(sample, operation),
            |operation| {
                candidates
                    .iter()
                    .find(|c| c.operation() == operation && c.enabled())
                    .expect("validated no-argument operation")
                    .key()
            },
        )
    }

    /// Choose an operation and then its target, using a caller-owned RNG for samples.
    pub fn sample_row(
        &self,
        sample: usize,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StructuredDecision<F::Key>>
    where
        F::Key: Ord,
    {
        if mode == ActMode::Deterministic {
            return self.greedy(sample);
        }
        let candidates = self.candidates(sample)?;
        let Distribution::Categorical { logits, masks } = self.operations() else {
            unreachable!()
        };
        let width = logits.dims()[1];
        let operations = Distribution::Categorical {
            logits: logits.clone().narrow(0, sample, 1),
            masks: masks[sample * width..(sample + 1) * width].to_vec(),
        };
        let actions = operations.sample(mode, rng)?;
        let Action::Discrete(operation) = actions[0] else {
            unreachable!()
        };
        let operation_log = to_vec(operations.log_prob(&actions, &[None])?)?[0];
        let (row, target_log) = if let Some(conditional) = self.conditional(sample, operation)? {
            let actions = conditional.distribution.sample(mode, rng)?;
            let Action::Discrete(target) = actions[0] else {
                unreachable!()
            };
            (
                conditional.rows[target],
                to_vec(conditional.distribution.log_prob(&actions, &[None])?)?[0],
            )
        } else {
            (
                candidates
                    .iter()
                    .position(|c| c.operation() == operation && c.enabled())
                    .expect("validated no-argument branch"),
                0.,
            )
        };
        Ok(StructuredDecision {
            key: candidates[row].key().clone(),
            log_prob: operation_log + target_log,
        })
    }
}
impl<F: PolicyFrame<Key = ConditionalAction>> LightweightPolicyEvaluation<'_, '_, F> {
    pub fn complete_law(&self, sample: usize) -> Result<ConditionalPolicyLaw> {
        let (keys, log_probs) = self.complete_log_probs(sample)?;
        ConditionalPolicyLaw::new(keys, log_probs)
    }

    /// Complete laws in caller order, sharing operation normalization across rows.
    pub fn complete_laws(&self, samples: &[usize]) -> Result<Vec<ConditionalPolicyLaw>> {
        ensure(
            !samples.is_empty(),
            "empty lightweight policy law selection",
        )?;
        let operations = self.operations().categorical_terms()?.log_probs;
        samples
            .iter()
            .map(|&sample| {
                let candidates = self.candidates(sample)?;
                let (keys, logs) =
                    self.complete_log_probs_using(sample, candidates, &operations)?;
                ConditionalPolicyLaw::new(keys, logs)
            })
            .collect()
    }
}
