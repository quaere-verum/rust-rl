//! A borrowed execution view preserves each model's owned parameter paths.
use super::{
    super::{
        LightweightBatch, LightweightEncoder, LightweightWorld,
        encoder::{candidate_targets, forward_mlp},
        pack::LightweightLayout,
    },
    RecurrentLightweightConfig, RecurrentLightweightStage, RecurrentLightweightState,
    StructuredHistoryInput,
    input::{Frame, structured},
};
use crate::{
    Error, Result,
    core::{ensure, stateful::HistoryInput, structured::StructuredObservation},
    nn::{
        canonical,
        network::Mlp,
        structured::{ArgumentKind, PackLimits},
        temporal::{GruEncoder, TemporalEncoder, TemporalMasks},
    },
};
use burn::{
    module::Module,
    tensor::{Device, Int, Tensor, TensorData, activation::relu},
};
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, HashSet};

/// Owned factual encoder for new heads. Historical policy and scalar Q keep
/// their original parameter paths and use the same borrowed execution view.
#[derive(Module, Debug)]
pub(super) struct HistoryBackbone {
    pub(super) encoder: LightweightEncoder,
    pub(super) operations: Mlp,
    pub(super) event_projection: Mlp,
    pub(super) decision_projection: Mlp,
    pub(super) memory: GruEncoder,
}

impl HistoryBackbone {
    pub fn new(
        config: &RecurrentLightweightConfig,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        let c = &config.model;
        let memory = if device.is_autodiff() {
            GruEncoder::new(c.fused_width + 2, config.hidden_width, rng)?
        } else {
            GruEncoder::new_inference(c.fused_width + 2, config.hidden_width, rng)?
        }
        .fork(device);
        Ok(Self {
            encoder: LightweightEncoder::new(c.clone(), rng, device)?,
            operations: Mlp::new_on_device(
                c.input.operations.len(),
                &[],
                c.operation_width,
                rng,
                device,
            )?,
            event_projection: Mlp::new_on_device(c.event_width, &[], c.fused_width, rng, device)?,
            decision_projection: Mlp::new_on_device(
                config.decision_width(),
                &[],
                c.fused_width,
                rng,
                device,
            )?,
            memory,
        })
    }

    pub fn view<'a>(&'a self, config: &'a RecurrentLightweightConfig) -> HistoryEncoder<'a> {
        HistoryEncoder {
            config,
            encoder: &self.encoder,
            operations: &self.operations,
            event_projection: &self.event_projection,
            decision_projection: &self.decision_projection,
            memory: &self.memory,
        }
    }
}

pub(super) struct HistoryEncoder<'a> {
    pub config: &'a RecurrentLightweightConfig,
    pub encoder: &'a LightweightEncoder,
    pub operations: &'a Mlp,
    pub event_projection: &'a Mlp,
    pub decision_projection: &'a Mlp,
    pub memory: &'a GruEncoder,
}

pub(super) struct EncodedHistories<'a> {
    pub states: Tensor<2>,
    pub targets: Tensor<2>,
    pub batch: LightweightBatch,
    pub observations: Vec<Vec<&'a StructuredObservation>>,
}

struct PreparedHistories<'a> {
    frames: Vec<Frame<'a>>,
    candidate_frames: Option<Vec<bool>>,
    layout: LightweightLayout,
    time: usize,
    padded: usize,
    elements: usize,
}

pub(super) type LaneLayout = (usize, usize, bool);

fn lane_layout(
    observation: &StructuredObservation,
    event_start: usize,
    reset: bool,
) -> Option<LaneLayout> {
    let entities = observation.entities.len();
    let events = observation.events.len().checked_sub(event_start)?;
    // Keep each lane in the small Flex matrix regime along with the model
    // width checks below. Equal lengths preserve both pooling and GRU order.
    (entities <= 64 && events <= 32).then_some((entities, events, reset))
}

#[derive(Clone, Copy)]
enum CandidateRows<'a> {
    All,
    None,
    At(&'a HashSet<(usize, usize)>),
}

enum Unroll {
    DetachedPrefix,
    ScalarInference,
    GroupedInference,
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::Shape("recurrent allocation overflow".into()))
}

fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| Error::Shape("recurrent allocation overflow".into()))
    })
}

fn tensor<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(v, shape), device)
}

fn indices(v: Vec<i32>, device: &Device) -> Tensor<1, Int> {
    let n = v.len();
    Tensor::from_data(TensorData::new(v, [n]), device)
}

fn check_elements(elements: usize, limits: PackLimits) -> Result<()> {
    ensure(
        elements <= limits.max_elements && elements <= isize::MAX as usize / size_of::<f32>(),
        "recurrent input allocation budget exceeded",
    )
}

impl HistoryEncoder<'_> {
    pub fn encode<'a>(
        &self,
        histories: &[StructuredHistoryInput<'a>],
        limits: PackLimits,
        observe: impl FnMut(RecurrentLightweightStage),
        head_budget: impl FnOnce(usize, usize) -> Result<usize>,
    ) -> Result<EncodedHistories<'a>> {
        self.encode_with_candidates(histories, CandidateRows::All, limits, observe, head_budget)
    }

    pub fn encode_at<'a>(
        &self,
        histories: &[StructuredHistoryInput<'a>],
        positions: &HashSet<(usize, usize)>,
        limits: PackLimits,
        head_budget: impl FnOnce(usize, usize) -> Result<usize>,
    ) -> Result<EncodedHistories<'a>> {
        self.encode_with_candidates(
            histories,
            CandidateRows::At(positions),
            limits,
            |_| {},
            head_budget,
        )
    }

    fn encode_with_candidates<'a>(
        &self,
        histories: &[StructuredHistoryInput<'a>],
        candidates: CandidateRows<'_>,
        limits: PackLimits,
        mut observe: impl FnMut(RecurrentLightweightStage),
        head_budget: impl FnOnce(usize, usize) -> Result<usize>,
    ) -> Result<EncodedHistories<'a>> {
        let prepared = self.prepare(histories, false, candidates, limits, head_budget)?;
        observe(RecurrentLightweightStage::Validation);
        let (batch, world) = self.pack(
            &prepared.frames,
            prepared.candidate_frames.as_deref(),
            prepared.layout,
            prepared.elements,
            false,
            &mut observe,
        )?;
        let initial = self.memory.initial_state(histories.len())?;
        let states = self.unroll(
            histories,
            &prepared,
            &world,
            initial,
            Unroll::DetachedPrefix,
            &mut observe,
        )?;
        let targets = candidate_targets(world.entities, &batch);
        let mut frames = prepared.frames.into_iter();
        let observations = histories
            .iter()
            .map(|history| {
                frames
                    .by_ref()
                    .take(history.inputs.len())
                    .map(|frame| frame.observation)
                    .collect()
            })
            .collect();
        Ok(EncodedHistories {
            states,
            targets,
            batch,
            observations,
        })
    }

    pub fn encode_states(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        limits: PackLimits,
    ) -> Result<Tensor<2>> {
        let prepared = self.prepare(histories, false, CandidateRows::None, limits, |_, _| Ok(0))?;
        let (_, world) = self.pack(
            &prepared.frames,
            None,
            prepared.layout,
            prepared.elements,
            false,
            &mut |_| {},
        )?;
        let initial = self.memory.initial_state(histories.len())?;
        self.unroll(
            histories,
            &prepared,
            &world,
            initial,
            Unroll::DetachedPrefix,
            &mut |_| {},
        )
    }

    pub fn initial_state(&self) -> Result<RecurrentLightweightState> {
        Ok(RecurrentLightweightState {
            memory: self.memory.initial_state(1)?,
            event_start: 0,
        })
    }

    pub fn validate_state(&self, state: &RecurrentLightweightState) -> Result<()> {
        self.memory.validate_state(&state.memory, 1, &self.device())
    }

    pub fn assimilate(
        &self,
        prior: &RecurrentLightweightState,
        input: HistoryInput<'_>,
        limits: PackLimits,
    ) -> Result<RecurrentLightweightState> {
        self.validate_state(prior)?;
        let histories = [StructuredHistoryInput {
            inputs: std::slice::from_ref(&input),
            learn_start: 0,
        }];
        let prepared = self.prepare(&histories, true, CandidateRows::None, limits, |_, _| Ok(0))?;
        let (_, world) = self.pack(
            &prepared.frames,
            None,
            prepared.layout,
            prepared.elements,
            false,
            &mut |_| {},
        )?;
        let memory = self.unroll(
            &histories,
            &prepared,
            &world,
            prior.memory.clone(),
            Unroll::ScalarInference,
            &mut |_| {},
        )?;
        Ok(RecurrentLightweightState {
            memory,
            event_start: prepared.frames[0].event_start(),
        })
    }

    pub fn grouped_inference_supported(&self) -> bool {
        let c = &self.config.model;
        canonical::grouped_flex_supported(&self.device())
            && [
                self.config.hidden_width,
                c.entity_width,
                c.scalar_width,
                c.event_width,
                c.operation_width,
                c.fused_width,
            ]
            .into_iter()
            .all(|width| width <= 64)
            && [
                c.input.entity_features,
                c.input.scalar_features,
                c.input.operations.len(),
                c.event_input_width(),
                c.fusion_input_width(),
                self.config.decision_width(),
            ]
            .into_iter()
            .all(|width| width <= 128)
    }

    /// Group rows with the same scalar reduction dimensions. A group rejected
    /// by the combined allocation limits runs through ordinary scalar checks.
    pub fn inference_groups<T>(
        &self,
        count: usize,
        shape: impl Fn(usize) -> Option<LaneLayout>,
        scalar: impl Fn(usize) -> Result<T>,
        packed: impl Fn(&[usize]) -> Result<Option<Vec<T>>>,
    ) -> Result<Vec<T>> {
        if count < 2 || !self.grouped_inference_supported() {
            return (0..count).map(scalar).collect();
        }
        let mut output: Vec<_> = std::iter::repeat_with(|| None).take(count).collect();
        let mut groups = BTreeMap::<_, Vec<_>>::new();
        for (row, slot) in output.iter_mut().enumerate() {
            if let Some(shape) = shape(row) {
                groups.entry(shape).or_default().push(row);
            } else {
                *slot = Some(scalar(row)?);
            }
        }
        for lanes in groups.values().flat_map(|group| group.chunks(16)) {
            if lanes.len() > 1
                && let Some(values) = packed(lanes)?
            {
                debug_assert_eq!(lanes.len(), values.len());
                for (&lane, value) in lanes.iter().zip(values) {
                    output[lane] = Some(value);
                }
            } else {
                for &lane in lanes {
                    output[lane] = Some(scalar(lane)?);
                }
            }
        }
        Ok(output
            .into_iter()
            .map(|row| row.expect("every lane is evaluated"))
            .collect())
    }

    pub fn assimilate_grouped(
        &self,
        priors: &[RecurrentLightweightState],
        inputs: &[HistoryInput<'_>],
        limits: PackLimits,
    ) -> Result<Vec<RecurrentLightweightState>> {
        ensure(
            priors.len() == inputs.len(),
            "grouped structured assimilation count mismatch",
        )?;
        self.inference_groups(
            inputs.len(),
            |lane| {
                let input = inputs[lane];
                let observation = structured(input.observation).ok()?;
                let event_start = match input.previous {
                    None => 0,
                    Some(previous) => structured(previous.factual?.observation).ok()?.events.len(),
                };
                lane_layout(observation, event_start, input.previous.is_none())
            },
            |lane| self.assimilate(&priors[lane], inputs[lane], limits),
            |lanes| {
                let histories: Vec<_> = lanes
                    .iter()
                    .map(|&lane| StructuredHistoryInput {
                        inputs: std::slice::from_ref(&inputs[lane]),
                        learn_start: 0,
                    })
                    .collect();
                let Ok(prepared) =
                    self.prepare(&histories, true, CandidateRows::None, limits, |_, _| Ok(0))
                else {
                    return Ok(None);
                };
                for &lane in lanes {
                    self.validate_state(&priors[lane])?;
                }
                let (_, world) = self.pack(
                    &prepared.frames,
                    None,
                    prepared.layout,
                    prepared.elements,
                    true,
                    &mut |_| {},
                )?;
                let initial = Tensor::cat(
                    lanes
                        .iter()
                        .map(|&lane| priors[lane].memory.clone())
                        .collect(),
                    0,
                );
                let memory = self.unroll(
                    &histories,
                    &prepared,
                    &world,
                    initial,
                    Unroll::GroupedInference,
                    &mut |_| {},
                )?;
                Ok(Some(
                    prepared
                        .frames
                        .iter()
                        .enumerate()
                        .map(|(row, frame)| RecurrentLightweightState {
                            memory: memory.clone().narrow(0, row, 1),
                            event_start: frame.event_start(),
                        })
                        .collect(),
                ))
            },
        )
    }

    pub fn decision<'a>(
        &self,
        state: &RecurrentLightweightState,
        observation: &'a StructuredObservation,
        limits: PackLimits,
        head_budget: impl FnOnce(usize, usize) -> Result<usize>,
    ) -> Result<EncodedHistories<'a>> {
        let (frames, layout, elements) =
            self.prepare_decisions(&[state], &[observation], limits, head_budget)?;
        self.encode_decisions(&frames, layout, elements, state.memory.clone())
    }

    pub fn decision_layout(
        state: &RecurrentLightweightState,
        observation: &StructuredObservation,
    ) -> Option<LaneLayout> {
        lane_layout(observation, state.event_start, false)
    }

    pub fn decisions_grouped<'a>(
        &self,
        states: &[&RecurrentLightweightState],
        observations: &[&'a StructuredObservation],
        limits: PackLimits,
        head_budget: impl FnOnce(usize, usize) -> Result<usize>,
    ) -> Result<Option<EncodedHistories<'a>>> {
        let Ok((frames, layout, elements)) =
            self.prepare_decisions(states, observations, limits, head_budget)
        else {
            return Ok(None);
        };
        let states = Tensor::cat(states.iter().map(|state| state.memory.clone()).collect(), 0);
        self.encode_decisions(&frames, layout, elements, states)
            .map(Some)
    }

    fn prepare_decisions<'a>(
        &self,
        states: &[&RecurrentLightweightState],
        observations: &[&'a StructuredObservation],
        limits: PackLimits,
        head_budget: impl FnOnce(usize, usize) -> Result<usize>,
    ) -> Result<(Vec<Frame<'a>>, LightweightLayout, usize)> {
        let mut layout = LightweightLayout {
            samples: states.len(),
            entities: 1,
            events: 1,
            candidates: 0,
        };
        ensure(
            layout.samples <= self.config.max_observations,
            "recurrent observation budget exceeded",
        )?;
        let mut bytes = 0;
        let mut tokens = 0;
        let mut frames = Vec::with_capacity(states.len());
        for (state, &observation) in states.iter().zip(observations) {
            self.validate_state(state)?;
            bytes = add(bytes, observation.payload_bytes(false))?;
            ensure(
                bytes <= self.config.max_history_bytes,
                "recurrent public history byte budget exceeded",
            )?;
            let events = observation.events.get(state.event_start..).ok_or_else(|| {
                Error::State("recurrent decision event offset is outside observation".into())
            })?;
            tokens = add(tokens, add(events.len(), 1)?)?;
            ensure(
                tokens <= self.config.max_tokens,
                "recurrent token budget exceeded",
            )?;
            layout.entities = layout.entities.max(observation.entities.len());
            layout.events = layout.events.max(events.len());
            layout.candidates = add(layout.candidates, observation.candidates.len())?;
            frames.push(Frame {
                observation,
                events,
                previous: None,
            });
        }
        self.check_world(layout)?;
        let elements = layout.allocation_elements(&self.config.model.input)?;
        check_elements(
            add(elements, head_budget(states.len(), layout.candidates)?)?,
            limits,
        )?;
        Ok((frames, layout, elements))
    }

    fn encode_decisions<'a>(
        &self,
        frames: &[Frame<'a>],
        layout: LightweightLayout,
        elements: usize,
        states: Tensor<2>,
    ) -> Result<EncodedHistories<'a>> {
        let (batch, world) = self.pack(
            frames,
            None,
            layout,
            elements,
            frames.len() > 1,
            &mut |_| {},
        )?;
        let targets = candidate_targets(world.entities, &batch);
        Ok(EncodedHistories {
            states,
            targets,
            batch,
            observations: frames.iter().map(|frame| vec![frame.observation]).collect(),
        })
    }

    pub fn device(&self) -> Device {
        self.encoder.entities.layers[0].weight.val().device()
    }

    fn prepare<'a>(
        &self,
        histories: &[StructuredHistoryInput<'a>],
        scalar: bool,
        candidates: CandidateRows<'_>,
        limits: PackLimits,
        head_budget: impl FnOnce(usize, usize) -> Result<usize>,
    ) -> Result<PreparedHistories<'a>> {
        let cfg = self.config;
        let c = &cfg.model;
        let schema = &c.input;
        ensure(!histories.is_empty(), "empty recurrent history batch")?;
        let count = histories.iter().try_fold(0usize, |count, history| {
            ensure(
                history.learn_start < history.inputs.len(),
                "recurrent learning boundary outside history",
            )?;
            add(count, history.inputs.len())
        })?;
        ensure(
            count <= cfg.max_observations && count <= cfg.max_tokens,
            "recurrent observation or token budget exceeded",
        )?;
        ensure(
            product(&[count, size_of::<StructuredObservation>()])? <= cfg.max_history_bytes,
            "recurrent public history byte budget exceeded",
        )?;
        let mut layout = LightweightLayout {
            samples: count,
            entities: 1,
            events: 1,
            candidates: 0,
        };
        let (mut time, mut actual_tokens, mut bytes) = (0, 0, 0);
        let mut frames = Vec::with_capacity(count);
        let mut candidate_frames =
            matches!(candidates, CandidateRows::At(_)).then(|| Vec::with_capacity(count));
        for (lane, history) in histories.iter().enumerate() {
            let mut tokens = 0;
            for (row, &input) in history.inputs.iter().enumerate() {
                let observation = structured(input.observation)?;
                bytes = add(bytes, observation.payload_bytes(false))?;
                if !scalar {
                    ensure(
                        input.previous.is_none() == (row == 0),
                        "recurrent history must start at reset and include every transition",
                    )?;
                }
                if let Some(factual) = input.previous.and_then(|previous| previous.factual) {
                    bytes = add(bytes, std::mem::size_of_val(factual.reward.primitives()))?;
                    if scalar {
                        bytes = add(bytes, structured(factual.observation)?.payload_bytes(false))?;
                    } else {
                        ensure(
                            factual
                                .observation
                                .matches(history.inputs[row - 1].observation),
                            "recurrent selection observation differs from preceding history frame",
                        )?;
                    }
                }
                ensure(
                    bytes <= cfg.max_history_bytes,
                    "recurrent public history byte budget exceeded",
                )?;
                let frame = Frame::from_input(input, schema)?;
                tokens = add(tokens, add(frame.events.len(), 1)?)?;
                layout.entities = layout.entities.max(observation.entities.len());
                layout.events = layout.events.max(frame.events.len());
                let include_candidates = match candidates {
                    CandidateRows::All => true,
                    CandidateRows::None => false,
                    CandidateRows::At(positions) => positions.contains(&(lane, row)),
                };
                if include_candidates {
                    layout.candidates = add(layout.candidates, observation.candidates.len())?;
                }
                if let Some(candidate_frames) = &mut candidate_frames {
                    candidate_frames.push(include_candidates);
                }
                frames.push(frame);
            }
            actual_tokens = add(actual_tokens, tokens)?;
            time = time.max(tokens);
        }
        let lanes = histories.len();
        let padded = product(&[lanes, time])?;
        ensure(
            actual_tokens <= cfg.max_tokens
                && padded <= cfg.max_tokens
                && padded < i32::MAX as usize,
            "recurrent token budget exceeded",
        )?;
        let packed = layout.allocation_elements(schema)?;
        let sentinel = add(product(&[count, layout.events])?, count)?;
        ensure(
            sentinel < i32::MAX as usize,
            "recurrent lookup index overflow",
        )?;
        let previous_width =
            schema.operations.len() + schema.entity_features + schema.candidate_features + 8;
        let extra = add(
            product(&[count, previous_width])?,
            product(&[padded, c.fused_width + 2])?,
        )?;
        let extra = add(extra, product(&[padded, 7])?)?; // token masks and gather indices
        let extra = add(extra, product(&[sentinel, 2])?)?; // token flags
        let extra = add(extra, product(&[count, 2])?)?; // action and target masks
        let elements = add(add(packed, extra)?, head_budget(count, layout.candidates)?)?;
        check_elements(elements, limits)?;

        self.check_world(layout)?;
        c.check_shape(&[count, layout.events, c.fused_width])?;
        c.check_shape(&[count, cfg.decision_width()])?;
        c.check_shape(&[lanes, time, cfg.hidden_width.max(c.fused_width + 2)])?;
        c.check_shape(&[sentinel + 1, c.fused_width + 2])?;
        Ok(PreparedHistories {
            frames,
            candidate_frames,
            layout,
            time,
            padded,
            elements: packed,
        })
    }

    fn check_world(&self, layout: LightweightLayout) -> Result<()> {
        let c = &self.config.model;
        let LightweightLayout {
            samples: b,
            entities: n,
            events: t,
            ..
        } = layout;
        c.check_shape(&[add(product(&[b, n])?, 1)?, c.target_width()])?;
        c.check_shape(&[b, t, c.event_input_width().max(c.event_width)])?;
        c.check_shape(&[
            b,
            c.scalar_width
                .max(c.fusion_input_width())
                .max(c.fused_width),
        ])
    }

    fn pack(
        &self,
        frames: &[Frame<'_>],
        candidate_frames: Option<&[bool]>,
        layout: LightweightLayout,
        elements: usize,
        grouped: bool,
        observe: &mut impl FnMut(RecurrentLightweightStage),
    ) -> Result<(LightweightBatch, LightweightWorld)> {
        let samples = frames
            .iter()
            .enumerate()
            .map(|(row, frame)| {
                frame.project(
                    self.config,
                    layout.candidates != 0 && candidate_frames.is_none_or(|frames| frames[row]),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let batch = LightweightBatch::pack_rows(
            &self.config.model.input,
            layout,
            elements,
            samples.iter().map(|sample| sample.rows()),
            &self.device(),
        );
        observe(RecurrentLightweightStage::Packing);
        let world = self
            .encoder
            .forward_with_grouping(&batch, grouped, |stage| {
                observe(RecurrentLightweightStage::Encoder(stage))
            })?;
        Ok((batch, world))
    }

    fn unroll(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        prepared: &PreparedHistories<'_>,
        world: &LightweightWorld,
        initial: Tensor<2>,
        execution: Unroll,
        observe: &mut impl FnMut(RecurrentLightweightStage),
    ) -> Result<Tensor<2>> {
        let c = &self.config.model;
        let device = self.device();
        let detach_prefix = matches!(execution, Unroll::DetachedPrefix);
        let grouped = matches!(execution, Unroll::GroupedInference);
        let PreparedHistories {
            frames,
            layout,
            time,
            padded,
            ..
        } = prepared;
        let (time, padded) = (*time, *padded);
        let count = layout.samples;
        let event_rows = count * layout.events;
        let sentinel = event_rows + count;
        let groups = if grouped { count } else { 1 };
        let previous = self.previous_features(frames, &device, groups);
        observe(RecurrentLightweightStage::PreviousAction);
        let decisions = forward_mlp(
            self.decision_projection,
            Tensor::cat(vec![world.global.clone(), previous], 1),
            groups,
        );
        let events = forward_mlp(
            self.event_projection,
            world.events.clone().reshape([event_rows, c.event_width]),
            groups,
        );
        let event_flags = tensor([1., 0.].repeat(event_rows), [event_rows, 2], &device);
        let decision_flags = tensor([0., 1.].repeat(count), [count, 2], &device);
        let lookup = Tensor::cat(
            vec![
                Tensor::cat(vec![events, event_flags], 1),
                Tensor::cat(vec![decisions, decision_flags], 1),
                Tensor::zeros([1, c.fused_width + 2], &device),
            ],
            0,
        );
        let mut tokens = vec![sentinel as i32; padded];
        let mut decision_indices = Vec::with_capacity(count);
        let lanes = histories.len();
        let mut masks = TemporalMasks {
            batch: lanes,
            time,
            valid: vec![false; padded],
            reset: vec![false; padded],
            detach: vec![false; padded],
        };
        let mut frame_index = 0;
        for (lane, history) in histories.iter().enumerate() {
            let mut offset = lane * time;
            for row in 0..history.inputs.len() {
                let frame = &frames[frame_index];
                masks.reset[offset] = frame.previous.is_none();
                masks.detach[offset] = detach_prefix && row == history.learn_start;
                for event in 0..frame.events.len() {
                    masks.valid[offset] = true;
                    tokens[offset] = (frame_index * layout.events + event) as i32;
                    offset += 1;
                }
                masks.valid[offset] = true;
                tokens[offset] = (event_rows + frame_index) as i32;
                decision_indices.push(offset as i32);
                offset += 1;
                frame_index += 1;
            }
        }
        let inputs =
            lookup
                .select(0, indices(tokens, &device))
                .reshape([lanes, time, c.fused_width + 2]);
        observe(RecurrentLightweightStage::Tokens);
        if grouped {
            debug_assert!(masks.valid.iter().all(|&valid| valid));
            debug_assert!(
                masks
                    .reset
                    .chunks_exact(time)
                    .all(|lane| lane[0] == masks.reset[0])
            );
            let mut state = if masks.reset[0] {
                Tensor::zeros([lanes, self.config.hidden_width], &device)
            } else {
                initial
            };
            for t in 0..time {
                let input = inputs
                    .clone()
                    .slice([0..lanes, t..t + 1, 0..c.fused_width + 2])
                    .squeeze_dim(1);
                (_, state) = canonical::gru_step(self.memory, input, state)?;
            }
            observe(RecurrentLightweightStage::Memory);
            return Ok(state);
        }
        let (states, _) = self.memory.unroll(inputs, initial, &masks)?;
        let states = states
            .reshape([padded, self.config.hidden_width])
            .select(0, indices(decision_indices, &device));
        observe(RecurrentLightweightStage::Memory);
        Ok(states)
    }

    fn previous_features(&self, frames: &[Frame<'_>], device: &Device, groups: usize) -> Tensor<2> {
        let schema = &self.config.model.input;
        let (ops_width, entity_width, extras_width) = (
            schema.operations.len(),
            schema.entity_features,
            schema.candidate_features + 8,
        );
        let count = frames.len();
        let mut operations = vec![0.; count * ops_width];
        let mut entities = vec![0.; count * entity_width];
        let mut extras = vec![0.; count * extras_width];
        let mut present = vec![0.; count];
        let mut target_present = vec![0.; count];
        for (row, frame) in frames.iter().enumerate() {
            let Some(action) = &frame.previous else {
                continue;
            };
            if let Some(target) = action.target {
                let features = &mut entities[row * entity_width..(row + 1) * entity_width];
                features.copy_from_slice(target);
                for &column in &self.config.suppressed_entity_features {
                    features[column] = 0.;
                }
                target_present[row] = 1.;
            }
            operations[row * ops_width + action.operation] = 1.;
            present[row] = 1.;
            let fields = &mut extras[row * extras_width..(row + 1) * extras_width];
            fields[..schema.candidate_features].copy_from_slice(action.candidate_features);
            let tail = &mut fields[schema.candidate_features..];
            tail[0] = 1.;
            let kind = match action.kind {
                ArgumentKind::None => 0,
                ArgumentKind::Entity => 1,
                ArgumentKind::Location => 2,
                ArgumentKind::Reserved => unreachable!("validated previous action"),
            };
            tail[1 + kind] = 1.;
            if let Some(location) = action.location {
                tail[4..6].copy_from_slice(&location);
            }
            tail[6] = action.reward.signum() * action.reward.abs().ln_1p();
            tail[7] = (action.duration as f32).ln_1p();
        }
        Tensor::cat(
            vec![
                forward_mlp(
                    self.operations,
                    tensor(operations, [count, ops_width], device),
                    groups,
                ) * tensor(present, [count, 1], device),
                relu(forward_mlp(
                    &self.encoder.entities,
                    tensor(entities, [count, entity_width], device),
                    groups,
                )) * tensor(target_present, [count, 1], device),
                tensor(extras, [count, extras_width], device),
            ],
            1,
        )
    }
}
