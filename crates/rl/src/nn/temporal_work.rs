//! Exact encoder-only prefix reconstruction and packed learning/lookahead windows.
//! Initial states are pre-observation; no observation is assimilated twice.
use super::{
    temporal::{GruEncoder, TemporalEncoder, TemporalMasks, row_mask},
    temporal_batch::{EncodedHistory, EpisodeInputCache, encode_history_batch},
    temporal_models::PreparedHistory,
};
use crate::{
    Result,
    core::{ensure, stateful::HistorySpec},
    data::temporal::{SequenceBatch, SequenceSample},
};
use burn::{
    module::Module,
    tensor::{Device, Int, Tensor, TensorData},
};
use std::collections::BTreeMap;

/// One warm lane per factual episode through the common detach boundary.
/// Retain the original lane layout and masks after that boundary so that
/// autodiff keeps its reduction order when different windows are active.
pub(crate) struct EpisodeWindowPlan {
    prefix: Tensor<3>,
    prefix_masks: TemporalMasks,
    stops: Vec<usize>,
    source_rows: Tensor<1, Int>,
    suffix: PreparedHistory,
    offset: usize,
    prefix_ends: Vec<usize>,
}

impl EpisodeWindowPlan {
    pub(crate) fn try_new(
        cache: &EpisodeInputCache<'_>,
        samples: &[&SequenceSample],
        hidden_widths: [usize; 2],
        device: &Device,
    ) -> Result<Option<Self>> {
        let batch = samples.len();
        let width = cache.input_dim();
        // Stay in the pinned Flex small-matrix regime. Wider shapes can use
        // different reduction blocking when the number of lanes changes.
        if !(3..=64).contains(&batch)
            || width > 64
            || hidden_widths.iter().any(|&hidden| hidden > 64)
            || !compact_flex_supported()
            || *device != super::device()
        {
            return Ok(None);
        }
        let offset = samples
            .iter()
            .map(|sample| sample.learn_start)
            .min()
            .unwrap();
        let mut groups = BTreeMap::new();
        let source_rows: Vec<_> = samples
            .iter()
            .map(|sample| {
                let next = groups.len();
                *groups.entry(sample.episode_id).or_insert(next) as i64
            })
            .collect();
        // Avoid changing a multi-row Flex matmul to its singleton kernel.
        let prefix_batch = groups.len().max(2);
        if prefix_batch >= batch || (batch - prefix_batch).saturating_mul(offset) < batch {
            return Ok(None);
        }
        // gemm's FMA backend can choose a horizontal reduction for strided
        // weights at <=256 outputs. Preserve that decision for both encoders.
        if hidden_widths
            .iter()
            .any(|&hidden| !compact_hidden_supported(batch, prefix_batch, hidden))
        {
            return Ok(None);
        }
        let mut stops = vec![offset; groups.len()];
        stops.resize(prefix_batch, 0);
        let prefix_time = offset + 1;
        let mut prefix = EncodedHistory::zeroed(prefix_batch, prefix_time, width)?;
        for (episode_id, group) in groups {
            let base = group * prefix_time;
            prefix.inputs[base * width..(base + offset) * width]
                .copy_from_slice(&cache.tape(episode_id)[..offset * width]);
            prefix.masks.valid[base..base + offset].fill(true);
            prefix.masks.reset[base] = true;
        }
        let prefix_input = Tensor::from_data(
            TensorData::new(prefix.inputs, [prefix_batch, prefix_time, width]),
            &Device::flex(),
        );

        let time = samples.iter().map(|s| s.observations.len()).max().unwrap() - offset;
        let mut suffix = EncodedHistory::zeroed(batch, time, width)?;
        let mut prefix_ends = Vec::with_capacity(batch);
        for (row, sample) in samples.iter().enumerate() {
            let start = sample.learn_start;
            let end = sample.observations.len();
            let base = row * time;
            prefix_ends.push(start - offset);
            suffix.inputs[base * width..(base + end - offset) * width]
                .copy_from_slice(&cache.tape(sample.episode_id)[offset * width..end * width]);
            for t in offset..end {
                let token = base + t - offset;
                suffix.masks.valid[token] = true;
                suffix.masks.detach[token] = t <= start || t >= start + cache.learn_len();
                suffix.learning[token] = sample.loss_mask.get(t).copied().unwrap_or(false);
            }
        }
        Ok(Some(Self {
            prefix: prefix_input,
            prefix_masks: prefix.masks,
            stops,
            source_rows: Tensor::from_data(TensorData::new(source_rows, [batch]), &Device::flex()),
            suffix: PreparedHistory::new(suffix, None, device),
            offset,
            prefix_ends,
        }))
    }

    fn warm(&self, encoder: &GruEncoder) -> Result<Tensor<2>> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::DetachedPrefix);
        let plain = encoder.valid();
        let prior = warm_prefix(
            &plain,
            self.prefix.clone(),
            plain.initial_state(self.stops.len())?,
            &self.stops,
            &self.prefix_masks,
        )?;
        Ok(prior.select(0, self.source_rows.clone()))
    }

    pub(crate) fn features(
        &self,
        encoder: &GruEncoder,
        tokens: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor<2>> {
        let initial = Tensor::from_data(self.warm(encoder)?.into_data(), device);
        let tokens: Vec<_> = tokens
            .iter()
            .map(|&(row, time)| (row, time - self.offset))
            .collect();
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::LearningWindows);
        self.suffix.features_from(
            encoder,
            initial,
            self.prefix_ends.iter().copied(),
            &tokens,
            device,
        )
    }
}

/// One inference-only recurrent tape per compatible reset episode. Gather
/// each original replay row separately, repeating its final feature in padding
/// just as the dense unroll preserves the row's final recurrent state.
pub(crate) struct EpisodeInferencePlan {
    inputs: Tensor<3>,
    masks: TemporalMasks,
    gather: Tensor<1, Int>,
    batch: usize,
    valid: u64,
}

impl EpisodeInferencePlan {
    pub(crate) fn eligible(batch: usize, width: usize, device: &Device) -> bool {
        (3..=64).contains(&batch)
            && width <= 64
            && compact_flex_supported()
            && *device == super::device()
    }

    pub(crate) fn try_new(
        cache: &EpisodeInputCache<'_>,
        hidden_widths: &[usize],
        device: &Device,
    ) -> Result<Option<Self>> {
        let batch = cache.samples().len();
        let width = cache.input_dim();
        if !Self::eligible(batch, width, device) {
            return Ok(None);
        }
        let compact = cache.tapes().len().max(2);
        let time = cache
            .samples()
            .iter()
            .map(|sample| sample.observations.len())
            .max()
            .unwrap();
        if compact >= batch
            || (batch - compact).saturating_mul(time) < 64
            || !hidden_widths
                .iter()
                .any(|&hidden| compact_hidden_supported(batch, compact, hidden))
        {
            return Ok(None);
        }
        let mut packed = EncodedHistory::zeroed(compact, time, width)?;
        let mut groups = BTreeMap::new();
        let mut valid = 0;
        for (row, (id, tape)) in cache.tapes().enumerate() {
            let length = tape.len() / width;
            let base = row * time;
            packed.inputs[base * width..base * width + tape.len()].copy_from_slice(tape);
            packed.masks.valid[base..base + length].fill(true);
            packed.masks.reset[base] = true;
            groups.insert(id, row);
            valid += length as u64;
        }
        let gather: Vec<_> = cache
            .samples()
            .iter()
            .flat_map(|sample| {
                let start = groups[&sample.episode_id] * time;
                let last = sample.observations.len() - 1;
                (0..time).map(move |t| (start + t.min(last)) as i64)
            })
            .collect();
        let plain = Device::flex();
        Ok(Some(Self {
            inputs: Tensor::from_data(
                TensorData::new(packed.inputs, [compact, time, width]),
                &plain,
            ),
            masks: packed.masks,
            gather: Tensor::from_data(TensorData::new(gather, [batch * time]), &plain),
            batch,
            valid,
        }))
    }

    pub(crate) fn supports_hidden(&self, hidden: usize) -> bool {
        compact_hidden_supported(self.batch, self.masks.batch, hidden)
    }

    pub(crate) fn work(&self) -> (u64, u64) {
        ((self.masks.batch * self.masks.time) as u64, self.valid)
    }

    pub(crate) fn features(&self, encoder: &GruEncoder) -> Result<Tensor<3>> {
        let [compact, time, _] = self.inputs.dims();
        let hidden = encoder.output_dim();
        ensure(
            self.supports_hidden(hidden),
            "unsupported replay coalescence shape",
        )?;
        let encoder = encoder.valid();
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::CoalescedReplayHistory);
        let (features, _) = encoder.unroll(
            self.inputs.clone(),
            encoder.initial_state(compact)?,
            &self.masks,
        )?;
        Ok(features
            .reshape([compact * time, hidden])
            .select(0, self.gather.clone())
            .reshape([self.batch, time, hidden]))
    }
}

fn compact_hidden_supported(batch: usize, compact: usize, hidden: usize) -> bool {
    hidden <= 64 && !(compact * hidden <= 256 && batch * hidden > 256)
}

fn compact_flex_supported() -> bool {
    // The locked workspace enables gemm's std feature without x86-v4, so an
    // FMA-capable x86-64 host selects the FMA kernel. Re-audit if that changes.
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Carry one lane through a bounded chunk with the scalar GRU kernel. The
/// output tape is checked before handing its last state to the next chunk.
pub(crate) fn reconstruct_scalar_chunk(
    encoder: &GruEncoder,
    inputs: Tensor<3>,
    prior: &[f32],
) -> Result<Vec<f32>> {
    let [batch, time, width] = inputs.dims();
    let hidden = encoder.output_dim();
    ensure(
        batch == 1 && time > 0 && width == encoder.input_dim() && prior.len() == hidden,
        "invalid scalar reconstruction chunk",
    )?;
    let state = Tensor::from_data(
        TensorData::new(prior.to_vec(), [1, hidden]),
        &inputs.device(),
    );
    encoder.validate_state(&state, 1, &inputs.device())?;
    let next = super::to_vec(encoder.gru.forward(inputs, Some(state)))?;
    ensure(
        next.len() == time * hidden && next.iter().all(|x| x.is_finite()),
        "invalid temporal behavior encoder output",
    )?;
    Ok(next[next.len() - hidden..].to_vec())
}

pub(crate) struct ExactWindow {
    pub prefix: EncodedHistory,
    pub suffix: EncodedHistory,
    pub starts: Vec<usize>,
}

pub(crate) fn prepare_exact_window(
    batch: &SequenceBatch,
    spec: &HistorySpec,
) -> Result<ExactWindow> {
    let prefix = encode_history_batch(batch, spec)?;
    let starts: Vec<_> = batch.samples.iter().map(|s| s.learn_start).collect();
    let time = batch
        .samples
        .iter()
        .map(|s| s.observations.len() - s.learn_start)
        .max()
        .expect("validated nonempty batch");
    let tokens = prefix
        .batch
        .checked_mul(time)
        .ok_or_else(|| crate::Error::Shape("suffix size overflow".into()))?;
    let elements = tokens
        .checked_mul(prefix.input_dim)
        .ok_or_else(|| crate::Error::Shape("suffix feature overflow".into()))?;
    let mut suffix = EncodedHistory {
        batch: prefix.batch,
        time,
        input_dim: prefix.input_dim,
        inputs: vec![0.; elements],
        masks: TemporalMasks {
            batch: prefix.batch,
            time,
            valid: vec![false; tokens],
            reset: vec![false; tokens],
            detach: vec![false; tokens],
        },
        learning: vec![false; tokens],
    };
    for (b, sample) in batch.samples.iter().enumerate() {
        for local in 0..sample.observations.len() - starts[b] {
            let original = b * prefix.time + starts[b] + local;
            let token = b * time + local;
            let f = prefix.input_dim;
            suffix.inputs[token * f..(token + 1) * f]
                .copy_from_slice(&prefix.inputs[original * f..(original + 1) * f]);
            suffix.masks.valid[token] = prefix.masks.valid[original];
            suffix.masks.reset[token] = prefix.masks.reset[original];
            suffix.masks.detach[token] = prefix.masks.detach[original];
            suffix.learning[token] = prefix.learning[original];
        }
    }
    Ok(ExactWindow {
        prefix,
        suffix,
        starts,
    })
}

/// Caller supplies a frozen/plain encoder and its own initial states. Only
/// `0..starts[b]` is assimilated in each lane. No Q head or output tape is built.
pub(crate) fn warm_prefix<E: TemporalEncoder<State = Tensor<2>>>(
    encoder: &E,
    input: Tensor<3>,
    mut state: Tensor<2>,
    starts: &[usize],
    masks: &TemporalMasks,
) -> Result<Tensor<2>> {
    let [batch, time, width] = input.dims();
    ensure(
        starts.len() == batch
            && masks.batch == batch
            && masks.time == time
            && width == encoder.input_dim()
            && starts.iter().all(|&s| s < time),
        "invalid exact prefix boundaries",
    )?;
    let hidden = encoder.output_dim();
    let mut active = vec![false; batch];
    let mut reset = vec![false; batch];
    let stop = starts.iter().copied().max().unwrap_or(0);
    for t in 0..stop {
        for b in 0..batch {
            active[b] = t < starts[b] && masks.valid[b * time + t];
            reset[b] = active[b] && masks.reset[b * time + t];
        }
        if !active.iter().any(|&x| x) {
            continue;
        }
        #[cfg(feature = "probes")]
        let _step_profile = crate::profiling::enter_detached_step(&active);
        let mut prior = state.clone();
        if reset.iter().any(|&x| x) {
            prior = prior.mask_fill(row_mask(reset.clone(), hidden, &input.device()), 0.);
        }
        let x = input
            .clone()
            .slice([0..batch, t..t + 1, 0..width])
            .squeeze_dim(1);
        let (_, next) = encoder.step(x, prior)?;
        state = if active.iter().all(|&x| x) {
            next
        } else {
            state.mask_where(row_mask(active.clone(), hidden, &input.device()), next)
        };
    }
    Ok(state)
}

#[cfg(test)]
mod tests;
