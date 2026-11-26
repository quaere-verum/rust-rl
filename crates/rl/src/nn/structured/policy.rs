use super::{
    Argument, ArgumentKind, EncodedWorld, StructuredBatch, StructuredEncoder,
    StructuredModelConfig, heads,
};
use crate::{
    Error, Result,
    core::{
        action::Action,
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
    },
    nn::{
        checkpoint::RecordBurnExt,
        distributions::Distribution,
        network::{Mlp, Network},
        to_vec,
    },
};
use burn::{module::Module, tensor::Tensor};
use rand_chacha::ChaCha8Rng;

#[derive(Module, Debug)]
pub struct StructuredPolicy {
    pub encoder: StructuredEncoder,
    pub operation_head: Mlp,
    pub entity_query: Mlp,
    pub entity_key: Mlp,
    pub spatial_query: Mlp,
    pub spatial_key: Mlp,
    #[module(skip)]
    config: StructuredModelConfig,
}
/// Shared world representation and operation distribution, computed once.
pub struct PolicyEvaluation<'a, K> {
    policy: &'a StructuredPolicy,
    batch: &'a StructuredBatch<K>,
    world: EncodedWorld,
    operations: Distribution,
}
pub struct Conditional<'a, K> {
    /// Candidate metadata only: indices are local to this evaluation, not identities.
    pub rows: Vec<usize>,
    pub keys: Vec<&'a K>,
    pub distribution: Distribution,
}
#[derive(Debug, Clone, PartialEq)]
pub struct StructuredDecision<K> {
    pub key: K,
    pub log_prob: f32,
}

/// Select from the two categorical heads directly. The head logits preserve
/// exact operation ties even when the branches contain different numbers of
/// arguments. Within each enabled head ties use its semantic keys.
pub(super) fn hierarchical_greedy<'a, K: Clone + Ord + 'a>(
    operations: &Distribution,
    sample: usize,
    branch: impl FnOnce(usize) -> Result<Option<Conditional<'a, K>>>,
    no_argument: impl FnOnce(usize) -> &'a K,
) -> Result<StructuredDecision<K>> {
    let (operation, operation_log) = greedy_row(operations, sample, |index| index)?;
    if let Some(conditional) = branch(operation)? {
        let (argument, argument_log) = greedy_row(&conditional.distribution, 0, |index| {
            conditional.keys[index]
        })?;
        Ok(StructuredDecision {
            key: conditional.keys[argument].clone(),
            log_prob: operation_log + argument_log,
        })
    } else {
        Ok(StructuredDecision {
            key: no_argument(operation).clone(),
            log_prob: operation_log,
        })
    }
}

fn greedy_row<K: Ord>(
    distribution: &Distribution,
    sample: usize,
    key: impl Fn(usize) -> K,
) -> Result<(usize, f32)> {
    let Distribution::Categorical { logits, masks } = distribution else {
        return Err(Error::Action("conditional head must be categorical".into()));
    };
    let [count, width] = logits.dims();
    ensure(sample < count, "conditional greedy row is out of range")?;
    let logits = logits.clone().narrow(0, sample, 1);
    let masks = &masks[sample * width..(sample + 1) * width];
    let weights = to_vec(logits.clone().reshape([width]))?;
    ensure(
        weights
            .iter()
            .zip(masks)
            .all(|(w, enabled)| !enabled || w.is_finite()),
        "nonfinite enabled conditional logit",
    )?;
    let row = weights
        .iter()
        .enumerate()
        .filter(|(index, _)| masks[*index])
        .max_by(|(a, p), (b, q)| {
            p.partial_cmp(q)
                .expect("validated finite logits")
                .then_with(|| key(*b).cmp(&key(*a)))
        })
        .ok_or_else(|| Error::Action("empty conditional head support".into()))?
        .0;
    let distribution = Distribution::Categorical {
        logits,
        masks: masks.to_vec(),
    };
    let log_prob = to_vec(distribution.log_prob(&[Action::Discrete(row)], &[None])?)?[0];
    Ok((row, log_prob))
}

impl StructuredPolicy {
    pub fn new(
        config: StructuredModelConfig,
        rng: &mut ChaCha8Rng,
        device: &burn::tensor::Device,
    ) -> Result<Self> {
        config.validate()?;
        let e = config.encoder.entity_width;
        let s = config.encoder.conv_channels[1];
        let c = config.encoder.input.candidate_features;
        Ok(Self {
            encoder: StructuredEncoder::new(config.encoder.clone(), rng, device)?,
            operation_head: Mlp::new_on_device(
                config.joint_width() + 1,
                &[config.head_width],
                1,
                rng,
                device,
            )?,
            entity_query: Mlp::new_on_device(config.query_width(), &[], e, rng, device)?,
            entity_key: Mlp::new_on_device(e + c, &[], e, rng, device)?,
            spatial_query: Mlp::new_on_device(config.query_width(), &[], s, rng, device)?,
            spatial_key: Mlp::new_on_device(s + c, &[], s, rng, device)?,
            config,
        })
    }
    pub fn config(&self) -> &StructuredModelConfig {
        &self.config
    }
    pub fn evaluate<'a, K>(
        &'a self,
        batch: &'a StructuredBatch<K>,
    ) -> Result<PolicyEvaluation<'a, K>> {
        heads::check_live(batch, &self.config)?;
        let ops = self.config.encoder.input.operations.len();
        self.config.check_rows(
            batch
                .len()
                .checked_mul(ops)
                .ok_or_else(|| Error::Shape("operation batch overflow".into()))?,
            self.config.joint_width() + 1,
        )?;
        let world = self.encoder.forward(batch)?;
        let mut masks = Vec::with_capacity(batch.len() * ops);
        let mut inputs = Vec::with_capacity(batch.len() * ops);
        for (i, candidates) in batch.candidates.iter().enumerate() {
            for operation in 0..ops {
                // Summaries use raw shared target features, not conditional
                // pointer forwards. Inference evaluates only the selected head.
                let targets: Vec<_> = candidates
                    .iter()
                    .filter(|c| c.enabled && c.operation == operation)
                    .map(|c| heads::target(&world, &self.config, i, c))
                    .collect();
                let count = targets.len();
                masks.push(count > 0);
                let summary = if targets.is_empty() {
                    Tensor::zeros([1, self.config.target_width()], &world.global.device())
                } else {
                    Tensor::cat(targets, 0).mean_dim(0)
                };
                inputs.push(Tensor::cat(
                    vec![
                        heads::query(&world, &self.config, i, operation),
                        summary,
                        heads::row(vec![(count as f32).ln_1p()], &world.global.device()),
                    ],
                    1,
                ));
            }
        }
        let logits = self
            .operation_head
            .forward(Tensor::cat(inputs, 0))
            .reshape([batch.len(), ops]);
        Ok(PolicyEvaluation {
            policy: self,
            batch,
            world,
            operations: Distribution::Categorical { logits, masks },
        })
    }
}
impl<'a, K> PolicyEvaluation<'a, K> {
    pub fn operations(&self) -> &Distribution {
        &self.operations
    }
    pub fn conditional(
        &self,
        sample: usize,
        operation: usize,
    ) -> Result<Option<Conditional<'a, K>>> {
        let cfg = &self.policy.config;
        ensure(
            sample < self.batch.len() && operation < cfg.encoder.input.operations.len(),
            "conditional branch out of range",
        )?;
        let candidates = &self.batch.candidates[sample];
        let rows: Vec<_> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| c.operation == operation)
            .map(|(i, _)| i)
            .collect();
        ensure(
            rows.iter().any(|&i| candidates[i].enabled),
            "unavailable conditional branch",
        )?;
        if cfg.encoder.input.operations[operation] == ArgumentKind::None {
            return Ok(None);
        }
        let inputs: Vec<_> = rows
            .iter()
            .map(|&i| {
                let c = &candidates[i];
                let embedding = match c.argument {
                    Argument::Entity(j) => heads::entity(&self.world, sample, j),
                    Argument::Location { x, y } => heads::location(&self.world, sample, x, y),
                    Argument::None | Argument::RegionalLocation { .. } => {
                        unreachable!("validated batch resolves regional locations")
                    }
                };
                Tensor::cat(
                    vec![
                        embedding,
                        heads::row(c.features.clone(), &self.world.global.device()),
                    ],
                    1,
                )
            })
            .collect();
        let (query, keys, width) = match cfg.encoder.input.operations[operation] {
            ArgumentKind::Entity => (
                self.policy
                    .entity_query
                    .forward(heads::query(&self.world, cfg, sample, operation)),
                self.policy.entity_key.forward(Tensor::cat(inputs, 0)),
                cfg.encoder.entity_width,
            ),
            ArgumentKind::Location => (
                self.policy.spatial_query.forward(heads::query(
                    &self.world,
                    cfg,
                    sample,
                    operation,
                )),
                self.policy.spatial_key.forward(Tensor::cat(inputs, 0)),
                cfg.encoder.conv_channels[1],
            ),
            ArgumentKind::None | ArgumentKind::Reserved => unreachable!(),
        };
        let logits = (keys * query).sum_dim(1).reshape([1, rows.len()]) / (width as f32).sqrt();
        let masks = rows.iter().map(|&i| candidates[i].enabled).collect();
        let keys = rows.iter().map(|&i| &candidates[i].key).collect();
        Ok(Some(Conditional {
            rows,
            keys,
            distribution: Distribution::Categorical { logits, masks },
        }))
    }
    /// All complete probabilities, in candidate order, including zeroes for
    /// masked rows. Intended for diagnostics/projection, not action sampling.
    pub fn complete_probabilities(&self, sample: usize) -> Result<Tensor<1>> {
        ensure(sample < self.batch.len(), "structured sample out of range")?;
        let ops = self.operations.categorical_terms()?.probabilities;
        let candidates = &self.batch.candidates[sample];
        let mut result = vec![None; candidates.len()];
        for operation in 0..self.policy.config.encoder.input.operations.len() {
            if !candidates
                .iter()
                .any(|c| c.operation == operation && c.enabled)
            {
                continue;
            }
            let p = ops
                .clone()
                .slice([sample..sample + 1, operation..operation + 1])
                .reshape([1]);
            if let Some(conditional) = self.conditional(sample, operation)? {
                let target = conditional.distribution.categorical_terms()?.probabilities;
                for (j, &row) in conditional.rows.iter().enumerate() {
                    result[row] =
                        Some(p.clone() * target.clone().slice([0..1, j..j + 1]).reshape([1]));
                }
            } else {
                let row = candidates
                    .iter()
                    .position(|c| c.operation == operation)
                    .unwrap();
                result[row] = Some(p);
            }
        }
        Ok(Tensor::cat(
            result
                .into_iter()
                .map(|p| p.unwrap_or_else(|| Tensor::zeros([1], &self.world.global.device())))
                .collect(),
            0,
        ))
    }
}
impl<K: PartialEq> PolicyEvaluation<'_, K> {
    /// Teacher-force multiple complete keys at one history. Each requested
    /// conditional head is evaluated once, sharing the already encoded world.
    /// Unrequested branches receive no gradients; no actions are sampled.
    pub fn log_prob_many(&self, sample: usize, keys: &[K]) -> Result<Tensor<1>> {
        self.policy.config.check_rows(keys.len(), 1)?;
        ensure(
            sample < self.batch.len() && !keys.is_empty(),
            "invalid structured teacher history or empty keys",
        )?;
        let candidates = &self.batch.candidates[sample];
        let rows = keys
            .iter()
            .map(|key| {
                candidates
                    .iter()
                    .position(|c| &c.key == key && c.enabled)
                    .ok_or_else(|| {
                        Error::Action("teacher action is not an enabled candidate".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let operations = self.operations.categorical_terms()?.log_probs;
        let mut terms: Vec<Option<Tensor<1>>> = vec![None; keys.len()];
        for operation in 0..self.policy.config.encoder.input.operations.len() {
            if !rows
                .iter()
                .any(|&row| candidates[row].operation == operation)
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
                .map(|c| c.distribution.categorical_terms().map(|t| t.log_probs))
                .transpose()?;
            for (i, &row) in rows
                .iter()
                .enumerate()
                .filter(|(_, row)| candidates[**row].operation == operation)
            {
                terms[i] = Some(if let (Some(c), Some(lp)) = (&conditional, &logs) {
                    let target = c
                        .rows
                        .iter()
                        .position(|&r| r == row)
                        .expect("validated teacher target");
                    op.clone() + lp.clone().slice([0..1, target..target + 1]).reshape([1])
                } else {
                    op.clone()
                });
            }
        }
        Ok(Tensor::cat(
            terms
                .into_iter()
                .map(|t| t.expect("all requested branches evaluated"))
                .collect(),
            0,
        ))
    }
    /// Teacher forcing selects branches solely from the supplied complete keys.
    pub fn log_prob(&self, keys: &[K]) -> Result<Tensor<1>> {
        ensure(
            keys.len() == self.batch.len(),
            "structured teacher action count mismatch",
        )?;
        let mut operations = Vec::with_capacity(keys.len());
        let mut conditional_terms = Vec::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            let row = self.batch.candidates[i]
                .iter()
                .position(|c| &c.key == key && c.enabled)
                .ok_or_else(|| {
                    Error::Action("teacher action is not an enabled candidate".into())
                })?;
            let operation = self.batch.candidates[i][row].operation;
            operations.push(Action::Discrete(operation));
            let term = if let Some(conditional) = self.conditional(i, operation)? {
                let index = conditional.rows.iter().position(|&r| r == row).unwrap();
                conditional
                    .distribution
                    .log_prob(&[Action::Discrete(index)], &[None])?
            } else {
                Tensor::zeros([1], &self.world.global.device())
            };
            conditional_terms.push(term);
        }
        Ok(self
            .operations
            .log_prob(&operations, &vec![None; keys.len()])?
            + Tensor::cat(conditional_terms, 0))
    }
}
impl<K: Clone + Ord> PolicyEvaluation<'_, K> {
    /// Select an operation and then its argument, with ties in semantic order.
    pub fn greedy(&self, sample: usize) -> Result<StructuredDecision<K>> {
        hierarchical_greedy(
            &self.operations,
            sample,
            |operation| self.conditional(sample, operation),
            |operation| {
                &self.batch.candidates[sample]
                    .iter()
                    .find(|c| c.operation == operation && c.enabled)
                    .expect("validated no-argument operation")
                    .key
            },
        )
    }

    /// Hierarchical selection using the caller-owned RNG for stochastic draws.
    pub fn sample(
        &self,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<StructuredDecision<K>>> {
        if mode == ActMode::Deterministic {
            return (0..self.batch.candidates.len())
                .map(|row| self.greedy(row))
                .collect();
        }
        let actions = self.operations.sample(mode, rng)?;
        let op_lp = to_vec(
            self.operations
                .log_prob(&actions, &vec![None; actions.len()])?,
        )?;
        let mut decisions = Vec::with_capacity(actions.len());
        for (i, action) in actions.iter().enumerate() {
            let Action::Discrete(operation) = *action else {
                unreachable!()
            };
            let (row, lp) = if let Some(conditional) = self.conditional(i, operation)? {
                let actions = conditional.distribution.sample(mode, rng)?;
                let Action::Discrete(j) = actions[0] else {
                    unreachable!()
                };
                (
                    conditional.rows[j],
                    to_vec(conditional.distribution.log_prob(&actions, &[None])?)?[0],
                )
            } else {
                (
                    self.batch.candidates[i]
                        .iter()
                        .position(|c| c.operation == operation)
                        .unwrap(),
                    0.,
                )
            };
            decisions.push(StructuredDecision {
                key: self.batch.candidates[i][row].key.clone(),
                log_prob: op_lp[i] + lp,
            });
        }
        Ok(decisions)
    }
}
impl Checkpoint for StructuredPolicy {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut r = LearnerRecord::new("structured-policy-v2", &self.config)?;
        r.model("policy", self)?;
        Ok(r)
    }
    fn restore(&mut self, r: &LearnerRecord) -> Result<()> {
        r.check("structured-policy-v2")?;
        let c: StructuredModelConfig = serde_json::from_value(r.metadata.clone())?;
        ensure(
            c == self.config,
            "structured policy checkpoint configuration mismatch",
        )?;
        *self = super::checkpoint::restore(r, "policy", self)?;
        Ok(())
    }
}
