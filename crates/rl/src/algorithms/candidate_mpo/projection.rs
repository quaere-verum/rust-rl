use super::{CandidateMpo, CandidateMpoConfig, ProjectionRecovery};
use crate::objectives::policy_supervision::distribution_loss;
use crate::{
    Result,
    core::ensure,
    nn::{
        optimization::{optimize_validated, parameter_norm},
        structured::{
            CandidatePolicy, StructuredActionValue, StructuredSample,
            candidate_log_probabilities_checked, candidate_probabilities_checked,
        },
    },
};
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::ModuleOptimizer,
    tensor::Tensor,
};
use std::{collections::BTreeMap, hash::Hash};

pub(in crate::algorithms) struct Projection<P> {
    pub accepted: Option<(P, ModuleOptimizer, bool)>,
    pub metrics: BTreeMap<String, f64>,
}
fn cross_entropy(q: &[&[f64]], p: &[Vec<f64>]) -> f64 {
    q.iter()
        .zip(p)
        .map(|(q, p)| {
            q.iter()
                .zip(p)
                .filter(|(q, _)| **q > 0.)
                .map(|(&q, &p)| -q * p.ln())
                .sum::<f64>()
        })
        .sum::<f64>()
        / q.len() as f64
}
pub(super) fn project<Q: StructuredActionValue, P: CandidatePolicy, K: Clone + Eq + Hash>(
    learner: &CandidateMpo<Q, P>,
    samples: &[StructuredSample<K>],
    old: &[Vec<f64>],
    q: &[&[f64]],
) -> Result<Projection<P>> {
    let operations: Vec<_> = samples
        .iter()
        .map(|sample| {
            sample
                .candidates
                .iter()
                .map(|candidate| candidate.operation)
                .collect()
        })
        .collect();
    let keys: Vec<_> = samples
        .iter()
        .map(|sample| {
            sample
                .candidates
                .iter()
                .filter(|candidate| candidate.enabled)
                .map(|candidate| candidate.key.clone())
                .collect()
        })
        .collect();
    let weights: Vec<Vec<_>> = samples
        .iter()
        .zip(q)
        .map(|(sample, q)| {
            q.iter()
                .zip(&sample.candidates)
                .filter(|(_, candidate)| candidate.enabled)
                .map(|(&q, _)| q as f32)
                .collect()
        })
        .collect();
    let limits = learner.critic.config.pack_limits();
    project_model(
        &learner.policy,
        &learner.optimizer,
        &learner.config,
        ProjectionBatch {
            operations: &operations,
            old,
            teacher: q,
        },
        |policy| {
            let likelihoods = candidate_log_probabilities_checked(policy, samples, &keys, limits)?;
            distribution_loss(likelihoods, &weights)
        },
        |policy| candidate_probabilities_checked(policy, samples, limits),
    )
}

pub(in crate::algorithms) struct ProjectionBatch<'a> {
    pub operations: &'a [Vec<usize>],
    pub old: &'a [Vec<f64>],
    pub teacher: &'a [&'a [f64]],
}

pub(in crate::algorithms) fn project_model<P: Module>(
    policy: &P,
    optimizer: &ModuleOptimizer,
    config: &CandidateMpoConfig,
    batch: ProjectionBatch<'_>,
    loss: impl Fn(&P) -> Result<Tensor<1>>,
    probabilities: impl Fn(&P) -> Result<Vec<Vec<f64>>>,
) -> Result<Projection<P>> {
    let ProjectionBatch {
        operations,
        old,
        teacher: q,
    } = batch;
    ensure(
        !q.is_empty() && q.len() == old.len() && q.len() == operations.len(),
        "candidate projection row count differs",
    )?;
    let rows = q.len() as f64;
    let before = cross_entropy(q, old);
    let mut result = Projection {
        accepted: None,
        metrics: BTreeMap::from([
            ("projection_ce_before".into(), before),
            ("projection_ce_after".into(), before),
            ("projection_forward_kl".into(), 0.),
            ("projection_operation_kl".into(), 0.),
            ("projection_conditional_kl".into(), 0.),
            ("projection_accepted".into(), 0.),
            ("projection_scale".into(), 0.),
            ("projection_trials".into(), 0.),
            ("projection_restarts".into(), 0.),
            ("projection_optimizer_reset".into(), 0.),
        ]),
    };
    let retries = usize::from(config.recovery == ProjectionRecovery::RestartAdam);
    for retry in 0..=retries {
        let loss = loss(policy)?;
        let mut proposal = policy.clone();
        let mut optimizer = if retry == 0 {
            optimizer.clone()
        } else {
            config.optimizer.build()?
        };
        let (tensor_loss, norm) = optimize_validated(
            &mut proposal,
            &mut optimizer,
            config.optimizer.learning_rate,
            loss,
        )?;
        parameter_norm(&proposal)?;
        result
            .metrics
            .insert("projection_tensor_loss".into(), tensor_loss);
        result.metrics.insert("actor_gradient_norm".into(), norm);
        result
            .metrics
            .insert("projection_restarts".into(), retry as f64);
        for trial in 0..=config.max_backtracks {
            let scale = 0.5_f32.powi(trial as i32);
            let candidate = if trial == 0 {
                proposal.clone()
            } else {
                blend(policy, &proposal, scale)
            };
            result.metrics.insert(
                "projection_trials".into(),
                (retry * (config.max_backtracks + 1) + trial + 1) as f64,
            );
            let p = match probabilities(&candidate) {
                Ok(probabilities) => probabilities,
                // A smaller finite proposal can preserve support. Incompatible
                // schemas and other structural errors still fail the update.
                Err(crate::Error::Numerical(_)) => continue,
                Err(error) => return Err(error),
            };
            ensure(
                p.len() == old.len(),
                "candidate projection proposal row count differs",
            )?;
            let mut kl = [0.; 3];
            for ((old, p), operations) in old.iter().zip(&p).zip(operations) {
                let d = kl_decomposition(old, p, operations)?;
                kl[0] += d.joint / rows;
                kl[1] += d.operation / rows;
                kl[2] += d.conditional / rows;
            }
            let after = cross_entropy(q, &p);
            if retry == 0 && trial == 0 {
                result
                    .metrics
                    .insert("proposal_ce_delta".into(), after - before);
                result.metrics.insert("proposal_forward_kl".into(), kl[0]);
            }
            if kl[0] <= config.projection_kl && after <= before + 1e-7 {
                result.metrics.extend([
                    ("projection_ce_after".into(), after),
                    ("projection_forward_kl".into(), kl[0]),
                    ("projection_operation_kl".into(), kl[1]),
                    ("projection_conditional_kl".into(), kl[2]),
                    ("projection_scale".into(), f64::from(scale)),
                    ("projection_accepted".into(), 1.),
                    ("projection_optimizer_reset".into(), retry as f64),
                ]);
                result.accepted = Some((candidate, optimizer, retry > 0));
                return Ok(result);
            }
        }
    }
    Ok(result)
}
/// Blend leaves without recording an interpolation graph; preserve parameter
/// IDs and each leaf's trainability, including deliberately frozen submodules.
fn blend<P: Module>(old: &P, proposal: &P, scale: f32) -> P {
    struct Collect(Vec<(Tensor<1>, bool)>);
    impl ModuleVisitor for Collect {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let v = p.val();
            let trainable = v.is_require_grad();
            self.0.push((
                v.clone()
                    .without_autodiff()
                    .reshape([v.dims().iter().product::<usize>()]),
                trainable,
            ));
        }
    }
    struct Blend {
        old: std::vec::IntoIter<(Tensor<1>, bool)>,
        scale: f32,
    }
    impl ModuleMapper for Blend {
        fn map_float<const D: usize>(&mut self, p: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (old, trainable) = self.old.next().expect("same policy architecture");
            p.map(|new| {
                let value = old.reshape(new.dims()) * (1. - self.scale)
                    + new.without_autodiff() * self.scale;
                Tensor::from_inner(value).set_require_grad(trainable)
            })
        }
    }
    let mut collect = Collect(Vec::new());
    old.visit(&mut collect);
    proposal.clone().map(&mut Blend {
        old: collect.0.into_iter(),
        scale,
    })
}

#[derive(Debug, Clone, Copy)]
struct KlDecomposition {
    pub joint: f64,
    pub operation: f64,
    pub conditional: f64,
}
/// KL(q || p), decomposed by operation. Unsupported positive q mass rejects:
/// an undefined conditional KL is not disguised as a finite diagnostic.
fn kl_decomposition(q: &[f64], p: &[f64], operations: &[usize]) -> Result<KlDecomposition> {
    ensure(
        !q.is_empty()
            && q.len() == p.len()
            && q.len() == operations.len()
            && q.iter().chain(p).all(|x| x.is_finite() && *x >= 0.)
            && (q.iter().sum::<f64>() - 1.).abs() < 1e-10
            && (p.iter().sum::<f64>() - 1.).abs() < 1e-10
            && q.iter().zip(p).all(|(&q, &p)| q == 0. || p > 0.),
        "invalid or unsupported joint KL distributions",
    )?;
    let mut groups = BTreeMap::<usize, (f64, f64)>::new();
    for ((&q, &p), &op) in q.iter().zip(p).zip(operations) {
        let group = groups.entry(op).or_default();
        group.0 += q;
        group.1 += p;
    }
    let mut joint = 0.;
    let mut conditional = 0.;
    for ((&q, &p), op) in q.iter().zip(p).zip(operations) {
        if q > 0. {
            let (q_op, p_op) = groups[op];
            let log_ratio = q.ln() - p.ln();
            joint += q * log_ratio;
            conditional += q * (log_ratio - q_op.ln() + p_op.ln());
        }
    }
    let operation = groups
        .values()
        .filter(|(q, _)| *q > 0.)
        .map(|(q, p)| q * (q.ln() - p.ln()))
        .sum();
    Ok(KlDecomposition {
        joint,
        operation,
        conditional,
    })
}
