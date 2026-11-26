//! Count-based, two-slice categorical dynamics with explicit fitting limits.

use rand::Rng;

use crate::{Error, Result, core::ensure};

/// A factual transition between two complete observed factor assignments.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteTransition<'a> {
    pub state: &'a [usize],
    pub action: usize,
    pub next: &'a [usize],
}

/// Limits are charged across all children and candidate tables in one fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitLimits {
    /// Maximum number of previous-state parents of any child. Zero is allowed.
    pub max_parents: usize,
    pub max_candidates: usize,
    /// Sum of all candidate count-table cells, including discarded candidates.
    pub max_total_cells: usize,
    /// State-coordinate validation and candidate sample visits, charging a visit
    /// once for the child and once for each parent coordinate it uses.
    pub max_work: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct FitConfig {
    /// Symmetric additive pseudocount per output category. Used for prediction.
    pub pseudocount: f64,
    pub limits: FitLimits,
}

/// Parent lists are indexed by child. A child may use its own previous value.
#[derive(Debug, Clone, Copy)]
pub enum Structure<'a> {
    /// Each list gives that child's exact parent set.
    Known(&'a [Vec<usize>]),
    /// Each list gives that child's admissible previous-state parent pool.
    LearnBic(&'a [Vec<usize>]),
}

#[derive(Debug, Clone)]
pub struct FactoredModel {
    cardinalities: Vec<usize>,
    actions: usize,
    factors: Vec<Factor>,
}

#[derive(Debug, Clone)]
struct Factor {
    parents: Vec<usize>,
    probabilities: Vec<f64>,
    bic: f64,
}

struct Candidate {
    parents: Vec<usize>,
    counts: Vec<usize>,
    bic: f64,
}

impl FactoredModel {
    /// Fit each next-state conditional independently from the same nonempty data.
    ///
    /// BIC uses unsmoothed maximum likelihood. Unobserved rows contribute zero
    /// log likelihood, while the parameter penalty counts every possible
    /// action/parent row. Ties prefer fewer parents, then lexicographic order.
    /// Predictions use symmetric smoothing, with a uniform fallback when a row
    /// is unobserved even at zero pseudocount.
    pub fn fit(
        cardinalities: &[usize],
        actions: usize,
        data: &[DiscreteTransition<'_>],
        structure: Structure<'_>,
        config: FitConfig,
    ) -> Result<Self> {
        let (lists, learn) = match structure {
            Structure::Known(lists) => (lists, false),
            Structure::LearnBic(lists) => (lists, true),
        };
        ensure(
            !cardinalities.is_empty()
                && !cardinalities.contains(&0)
                && cardinalities.len() == lists.len()
                && actions > 0
                && !data.is_empty()
                && config.pseudocount.is_finite()
                && config.pseudocount >= 0.,
            "invalid factored model dimensions, data or pseudocount",
        )?;
        let mut limits = config.limits;
        ensure(
            cardinalities.len() <= limits.max_candidates,
            "factored candidate limit exceeded",
        )?;
        let validation_work = product(product(2, cardinalities.len())?, data.len())?;
        charge(
            &mut limits.max_work,
            validation_work,
            "factored fit work limit exceeded",
        )?;
        for row in data {
            validate_assignment(cardinalities, row.state)?;
            validate_assignment(cardinalities, row.next)?;
            ensure(
                row.action < actions,
                "factored transition action is out of range",
            )?;
        }

        let mut factors = Vec::with_capacity(cardinalities.len());
        for (child, supplied) in lists.iter().enumerate() {
            ensure(
                supplied.len() <= cardinalities.len(),
                "invalid factored parent list width",
            )?;
            let mut pool = supplied.clone();
            pool.sort_unstable();
            ensure(
                pool.last()
                    .is_none_or(|&parent| parent < cardinalities.len())
                    && pool.windows(2).all(|pair| pair[0] != pair[1]),
                "factored parent identities must be unique and in range",
            )?;
            let mut best: Option<Candidate> = None;
            let max_parents = limits.max_parents;
            let mut consider = |parents: &[usize]| -> Result<()> {
                let candidate =
                    fit_candidate(cardinalities, actions, data, child, parents, &mut limits)?;
                if best.as_ref().is_none_or(|prior| candidate.bic < prior.bic) {
                    best = Some(candidate);
                }
                Ok(())
            };
            if learn {
                for_each_subset(&pool, max_parents, consider)?;
            } else {
                ensure(pool.len() <= max_parents, "factored parent limit exceeded")?;
                consider(&pool)?;
            }
            let best = best.expect("every valid search evaluates at least one parent set");
            let probabilities = smoothed(&best.counts, cardinalities[child], config.pseudocount)?;
            factors.push(Factor {
                parents: best.parents,
                probabilities,
                bic: best.bic,
            });
        }
        Ok(Self {
            cardinalities: cardinalities.to_vec(),
            actions,
            factors,
        })
    }

    /// Selected previous-state parent identities in ascending order.
    pub fn parents(&self, child: usize) -> Option<&[usize]> {
        self.factors
            .get(child)
            .map(|factor| factor.parents.as_slice())
    }

    /// Unsmoothed BIC score of the selected table for this child.
    pub fn bic_score(&self, child: usize) -> Option<f64> {
        self.factors.get(child).map(|factor| factor.bic)
    }

    /// Predict one conditional categorical distribution per next-state factor.
    pub fn predict_factors(&self, state: &[usize], action: usize) -> Result<Vec<&[f64]>> {
        validate_assignment(&self.cardinalities, state)?;
        ensure(
            action < self.actions,
            "factored prediction action is out of range",
        )?;
        Ok(self
            .factors
            .iter()
            .zip(&self.cardinalities)
            .map(|(factor, &width)| {
                let start = width * row_index(&self.cardinalities, &factor.parents, state, action);
                &factor.probabilities[start..start + width]
            })
            .collect())
    }

    /// Predict the probability of a complete next-state assignment.
    pub fn probability(&self, state: &[usize], action: usize, next: &[usize]) -> Result<f64> {
        validate_assignment(&self.cardinalities, next)?;
        Ok(self
            .predict_factors(state, action)?
            .iter()
            .zip(next)
            .map(|(row, &category)| row[category])
            .product())
    }

    /// Sample a full next state, drawing independently from the predicted rows.
    pub fn sample(&self, state: &[usize], action: usize, rng: &mut impl Rng) -> Result<Vec<usize>> {
        Ok(self
            .predict_factors(state, action)?
            .into_iter()
            .map(|row| sample_row(row, rng.random()))
            .collect())
    }
}

fn validate_assignment(cardinalities: &[usize], values: &[usize]) -> Result<()> {
    ensure(
        values.len() == cardinalities.len()
            && values
                .iter()
                .zip(cardinalities)
                .all(|(value, cardinality)| value < cardinality),
        "invalid factored state assignment",
    )
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Config("factored fit size overflow".into()))
}

fn charge(remaining: &mut usize, amount: usize, reason: &'static str) -> Result<()> {
    *remaining = remaining
        .checked_sub(amount)
        .ok_or_else(|| Error::Config(reason.into()))?;
    Ok(())
}

fn row_index(cardinalities: &[usize], parents: &[usize], state: &[usize], action: usize) -> usize {
    parents.iter().fold(action, |index, &parent| {
        index * cardinalities[parent] + state[parent]
    })
}

fn fit_candidate(
    cardinalities: &[usize],
    actions: usize,
    data: &[DiscreteTransition<'_>],
    child: usize,
    parents: &[usize],
    limits: &mut FitLimits,
) -> Result<Candidate> {
    let width = cardinalities[child];
    let rows = parents.iter().try_fold(actions, |rows, &parent| {
        product(rows, cardinalities[parent])
    })?;
    let cells = product(rows, width)?;
    charge(
        &mut limits.max_candidates,
        1,
        "factored candidate limit exceeded",
    )?;
    charge(
        &mut limits.max_total_cells,
        cells,
        "factored count-table limit exceeded",
    )?;
    charge(
        &mut limits.max_work,
        product(data.len(), parents.len() + 1)?,
        "factored fit work limit exceeded",
    )?;
    let mut counts = Vec::new();
    counts
        .try_reserve_exact(cells)
        .map_err(|_| Error::Config("cannot allocate factored count table".into()))?;
    counts.resize(cells, 0);
    for row in data {
        let index = row_index(cardinalities, parents, row.state, row.action);
        counts[index * width + row.next[child]] += 1;
    }
    let parameters = rows * (width - 1);
    let log_likelihood: f64 = counts
        .chunks_exact(width)
        .map(|counts| {
            let total = counts.iter().sum::<usize>() as f64;
            counts
                .iter()
                .copied()
                .filter(|&count| count > 0)
                .map(|count| {
                    let count = count as f64;
                    count * (count / total).ln()
                })
                .sum::<f64>()
        })
        .sum();
    let bic = -2. * log_likelihood + parameters as f64 * (data.len() as f64).ln();
    Ok(Candidate {
        parents: parents.to_vec(),
        counts,
        bic,
    })
}

fn for_each_subset(
    pool: &[usize],
    max_parents: usize,
    mut visit: impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    visit(&[])?;
    for len in 1..=max_parents.min(pool.len()) {
        let mut positions: Vec<_> = (0..len).collect();
        let mut parents = vec![0; len];
        loop {
            for (parent, &position) in parents.iter_mut().zip(&positions) {
                *parent = pool[position];
            }
            visit(&parents)?;
            let Some(index) = (0..len).rfind(|&index| positions[index] < pool.len() - len + index)
            else {
                break;
            };
            positions[index] += 1;
            for next in index + 1..len {
                positions[next] = positions[next - 1] + 1;
            }
        }
    }
    Ok(())
}

fn smoothed(counts: &[usize], width: usize, alpha: f64) -> Result<Vec<f64>> {
    let mut probabilities = Vec::new();
    probabilities
        .try_reserve_exact(counts.len())
        .map_err(|_| Error::Config("cannot allocate factored probability table".into()))?;
    for row in counts.chunks_exact(width) {
        let total = row.iter().sum::<usize>() as f64;
        let denominator = total + alpha * width as f64;
        if total == 0. {
            probabilities.extend(std::iter::repeat_n(1. / width as f64, width));
        } else if denominator.is_finite() {
            probabilities.extend(
                row.iter()
                    .map(|&count| (count as f64 + alpha) / denominator),
            );
        } else {
            let scaled = total / alpha + width as f64;
            probabilities.extend(
                row.iter()
                    .map(|&count| (count as f64 / alpha + 1.) / scaled),
            );
        }
    }
    Ok(probabilities)
}

fn sample_row(probabilities: &[f64], mut draw: f64) -> usize {
    let mut last = 0;
    for (index, &probability) in probabilities.iter().enumerate() {
        if draw < probability {
            return index;
        }
        if probability > 0. {
            last = index;
        }
        draw -= probability;
    }
    last
}
