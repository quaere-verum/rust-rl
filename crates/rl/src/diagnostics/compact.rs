//! Bounded, batched activation recording using the existing probe capability.
use super::{probes::activation_summary, schema::*};
use crate::{
    Result,
    core::ensure,
    nn::{self, probes::ActivationProbe},
};
use burn::tensor::{Int, Tensor, TensorData};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub enum ActivationStorage {
    Full,
    Summary,
    Neurons(Vec<usize>),
}
#[derive(Debug, Clone)]
pub struct ActivationOptions {
    pub storage: ActivationStorage,
    pub batch_size: usize,
    pub max_layer_elements: usize,
    pub max_recorded_elements: usize,
}
impl Default for ActivationOptions {
    fn default() -> Self {
        Self {
            storage: ActivationStorage::Full,
            batch_size: 64,
            max_layer_elements: 1024 * 1024,
            max_recorded_elements: 4 * 1024 * 1024,
        }
    }
}
/// Summary-only mode transfers one bounded layer batch at a time and reduces in
/// f64. Selected-neuron mode selects tensor columns before host transfer; its
/// statistics describe only those neurons. The caller supplies an explicit probe
/// population; batching never changes its row order or conditioning information.
pub fn activations_batched(
    context: RecordContext,
    network: &str,
    model: &impl ActivationProbe,
    input: Tensor<2>,
    names: &[&str],
    options: &ActivationOptions,
) -> Result<Chunk> {
    ensure(
        options.batch_size > 0
            && options.max_layer_elements > 0
            && options.max_recorded_elements > 0,
        "invalid activation recording limits",
    )?;
    ensure(
        names
            .iter()
            .enumerate()
            .all(|(i, n)| !names[..i].contains(n)),
        "compact activation layer names must be unique",
    )?;
    if let ActivationStorage::Neurons(indices) = &options.storage {
        ensure(
            !indices.is_empty()
                && indices
                    .iter()
                    .enumerate()
                    .all(|(i, n)| !indices[..i].contains(n)),
            "selected neurons must be nonempty and unique",
        )?;
    }
    let [rows, width] = input.dims();
    ensure(rows > 0 && width > 0, "empty activation input")?;
    let mut data: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut summaries: BTreeMap<String, ActivationSummary> = BTreeMap::new();
    let mut source_widths = BTreeMap::new();
    let mut total = 0usize;
    for start in (0..rows).step_by(options.batch_size) {
        let end = start.saturating_add(options.batch_size).min(rows);
        model.visit_activations(
            input.clone().slice([start..end, 0..width]),
            names,
            |activation| {
                let [n, w] = activation.tensor.dims();
                ensure(
                    n == end - start
                        && w > 0
                        && n.checked_mul(w)
                            .is_some_and(|size| size <= options.max_layer_elements),
                    "activation layer batch exceeds configured memory limit",
                )?;
                if let Some(previous) = source_widths.insert(activation.name.clone(), w) {
                    ensure(previous == w, "activation width changed across batches")?;
                }
                let tensor = if let ActivationStorage::Neurons(indices) = &options.storage {
                    ensure(
                        indices.iter().all(|&i| i < w && i <= i64::MAX as usize),
                        "selected activation neuron is out of range",
                    )?;
                    let device = activation.tensor.device();
                    activation.tensor.select(
                        1,
                        Tensor::<1, Int>::from_data(
                            TensorData::new(
                                indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                                [indices.len()],
                            ),
                            &device,
                        ),
                    )
                } else {
                    activation.tensor
                };
                let values = nn::to_vec(tensor)?;
                let summary = activation_summary(&values);
                match summaries.get_mut(&activation.name) {
                    Some(prior) => merge(prior, &summary),
                    None => {
                        summaries.insert(activation.name.clone(), summary);
                    }
                }
                if !matches!(options.storage, ActivationStorage::Summary) {
                    total = total
                        .checked_add(values.len())
                        .ok_or_else(|| crate::Error::Shape("activation size overflow".into()))?;
                    ensure(
                        total <= options.max_recorded_elements,
                        "activation record exceeds configured element budget",
                    )?;
                    data.entry(activation.name).or_default().extend(values);
                }
                Ok(())
            },
        )?;
    }
    let mut arrays = BTreeMap::new();
    let mut layers = BTreeMap::new();
    let mut selected_neurons = BTreeMap::new();
    for (name, values) in data {
        let width = values.len() / rows;
        arrays.insert(name.clone(), Array::f32(vec![rows, width], values)?);
        layers.insert(name.clone(), name.clone());
        if let ActivationStorage::Neurons(indices) = &options.storage {
            let key = format!("{name}/neuron_indices");
            arrays.insert(
                key.clone(),
                Array::u64(
                    vec![indices.len()],
                    indices.iter().map(|&i| i as u64).collect(),
                )?,
            );
            selected_neurons.insert(name, key);
        }
    }
    let chunk = Chunk {
        context,
        record: Record::Activations(ActivationRecord {
            network: network.into(),
            layers,
            summaries,
            source_widths,
            selected_neurons,
            inputs: ActivationInputs::Unknown,
        }),
        arrays,
    };
    chunk.validate()?;
    Ok(chunk)
}
fn merge(a: &mut ActivationSummary, b: &ActivationSummary) {
    let n = a.finite_count as f64;
    let m = b.finite_count as f64;
    let total = a.count + b.count;
    a.zero_fraction =
        (a.zero_fraction * a.count as f64 + b.zero_fraction * b.count as f64) / total.max(1) as f64;
    if m > 0. {
        if n == 0. {
            a.mean = b.mean;
            a.standard_deviation = b.standard_deviation;
            a.minimum = b.minimum;
            a.maximum = b.maximum;
        } else {
            let delta = b.mean.unwrap() - a.mean.unwrap();
            let variance = (a.standard_deviation.unwrap().powi(2) * n
                + b.standard_deviation.unwrap().powi(2) * m
                + delta * delta * n * m / (n + m))
                / (n + m);
            a.mean = Some(a.mean.unwrap() + delta * m / (n + m));
            a.standard_deviation = Some(variance.max(0.).sqrt());
            a.minimum = Some(a.minimum.unwrap().min(b.minimum.unwrap()));
            a.maximum = Some(a.maximum.unwrap().max(b.maximum.unwrap()));
        }
    }
    a.count = total;
    a.finite_count += b.finite_count;
}
