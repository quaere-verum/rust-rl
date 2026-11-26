//! Generic Burn record codecs, independent of algorithms and filesystem persistence.
use crate::{
    Error, Result,
    core::{
        checkpoint::{LearnerRecord, ModelRecord},
        ensure,
    },
};
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::{ModuleOptimizer, OptimizerRecord},
    store::ModuleRecord,
    tensor::{Bool, Bytes, Device, Int, Tensor},
};
use burn_pack::{DType, Reader, Scalar};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
};

/// Hash float parameters in module traversal order, independent of packed record order.
pub(crate) fn hash_weights(hash: &mut Sha256, model: &impl Module) -> Result<()> {
    struct Weights<'a>(&'a mut Sha256, Option<Error>);
    impl ModuleVisitor for Weights<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let tensor = param.val();
            self.0.update((D as u64).to_le_bytes());
            for width in tensor.dims() {
                self.0.update((width as u64).to_le_bytes());
            }
            match super::to_vec(tensor) {
                Ok(values) => {
                    for value in values {
                        self.0.update(value.to_bits().to_le_bytes());
                    }
                }
                Err(error) => self.1 = Some(error),
            }
        }
    }
    let mut weights = Weights(hash, None);
    model.visit(&mut weights);
    weights.1.map_or(Ok(()), Err)
}

/// Exact trained parameter groups and their individual Adam step counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdamOptimizerState {
    pub parameter_ids: Vec<u64>,
    pub steps: BTreeMap<u64, u64>,
}

/// Traversal-order trainability, including nontrainable running buffers.
/// Packed weights do not encode the destination template's active/frozen flags.
pub(crate) fn trainability<M: Module>(model: &M) -> Vec<bool> {
    struct Flags(Vec<bool>);
    impl ModuleVisitor for Flags {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.0.push(p.val().is_require_grad());
        }
    }
    let mut flags = Flags(Vec::new());
    model.visit(&mut flags);
    flags.0
}

/// Opt-in staged Adam validation for explicitly owned recurrent optimizers.
/// `expected_ids` binds the exact groups recorded by a schema-aware writer; it
/// must be sorted and unique. `None` captures actual groups or validates legacy
/// records whose historical membership is unknown. Legacy validation cannot
/// detect removal of an otherwise complete, owned group: unused parameters are
/// legitimate, so requiring every model parameter would reject valid records.
/// No model template, optimizer, parameter, or random stream is mutated here.
pub fn audit_adam_optimizer(
    record: &LearnerRecord,
    optimizer_name: &str,
    model_name: &str,
    expected_ids: Option<&[u64]>,
    owner_updates: u64,
) -> Result<AdamOptimizerState> {
    ensure(
        owner_updates < i32::MAX as u64,
        "checkpoint Adam owner history exceeds supported bias-correction range",
    )?;
    ensure(
        record.models.contains_key(model_name),
        "checkpoint Adam owner model is missing",
    )?;
    let bytes = record.optimizers.get(optimizer_name).ok_or_else(|| {
        Error::State(format!(
            "checkpoint missing Adam optimizer {optimizer_name}"
        ))
    })?;
    let packed = PackedRecord::read(Bytes::from_bytes_vec(bytes.clone()))?;
    packed.validate_optimizer(record, Some(model_name))?;
    let mut ids = BTreeSet::new();
    let mut steps = BTreeMap::new();
    for (key, value) in &packed.scalars {
        if let Some(prefix) = key.strip_suffix(".__rank") {
            let id = prefix.parse::<u64>().map_err(record_error)?;
            ensure(
                prefix == id.to_string(),
                "noncanonical Adam parameter identifier",
            )?;
            ids.insert(id);
        } else if let Some(prefix) = key.strip_suffix(".momentum.time") {
            let id = prefix.parse::<u64>().map_err(record_error)?;
            let time = u64::try_from(*value).map_err(record_error)?;
            ensure(
                prefix == id.to_string()
                    && time > 0
                    && time <= owner_updates
                    && time < i32::MAX as u64,
                "invalid checkpoint Adam step counter",
            )?;
            steps.insert(id, time);
        } else {
            return Err(Error::State(
                "checkpoint contains an unsupported Adam scalar".into(),
            ));
        }
    }
    let parameter_ids: Vec<_> = ids.iter().copied().collect();
    ensure(
        (owner_updates == 0) == parameter_ids.is_empty() && steps.keys().eq(ids.iter()),
        "checkpoint Adam trained groups or step counters are incomplete",
    )?;
    if let Some(expected) = expected_ids {
        ensure(
            expected.windows(2).all(|pair| pair[0] < pair[1])
                && expected == parameter_ids.as_slice(),
            "checkpoint Adam trained group membership differs from its recorded owner",
        )?;
    }
    let mut moments = BTreeMap::<u64, u8>::new();
    for (name, tensor) in &packed.tensors {
        let id = tensor
            .id
            .ok_or_else(|| Error::State("Adam moment lacks a parameter identifier".into()))?;
        let flag = if name == &format!("{id}.momentum.moment_1") {
            1
        } else if name == &format!("{id}.momentum.moment_2") {
            2
        } else {
            return Err(Error::State(
                "checkpoint contains an unsupported Adam moment".into(),
            ));
        };
        ensure(
            ids.contains(&id) && tensor.dtype == DType::F32,
            "Adam moment has an unknown group or unsupported dtype",
        )?;
        let elements = tensor
            .shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .and_then(|count| count.checked_mul(size_of::<f32>()));
        ensure(
            elements == Some(tensor.bytes.len()),
            "Adam moment payload length differs from its shape",
        )?;
        let mask = moments.entry(id).or_default();
        ensure(*mask & flag == 0, "duplicate checkpoint Adam moment")?;
        *mask |= flag;
        for word in tensor.bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            ensure(
                value.is_finite() && (flag != 2 || value >= 0.),
                "nonfinite or negative checkpoint Adam moment",
            )?;
        }
    }
    ensure(
        moments.len() == parameter_ids.len() && moments.values().all(|&mask| mask == 3),
        "checkpoint Adam moment pairs are incomplete",
    )?;
    Ok(AdamOptimizerState {
        parameter_ids,
        steps,
    })
}

pub trait RecordBurnExt {
    fn model<M: Module>(&mut self, name: &str, model: &M) -> Result<()>;
    fn optimizer(&mut self, name: &str, opt: &ModuleOptimizer) -> Result<()>;
    fn load_model<M: Module>(&self, name: &str, model: &M) -> Result<M>;
    /// Restore frozen parameters on plain Flex without activating autodiff.
    /// The caller must also construct its template without tracked parameters;
    /// this method cannot undo roots previously registered by that construction.
    fn load_model_inference<M: Module>(&self, name: &str, model: &M) -> Result<M>;
    fn load_optimizer(&self, name: &str, opt: ModuleOptimizer) -> Result<ModuleOptimizer>;
    /// Validate moment parameter IDs against one owner. `require_state` rejects
    /// empty records for parameters known to have been optimized already.
    fn load_optimizer_for_model(
        &self,
        name: &str,
        owner: &str,
        opt: ModuleOptimizer,
        require_state: bool,
    ) -> Result<ModuleOptimizer>;
}
impl RecordBurnExt for LearnerRecord {
    fn model<M: Module>(&mut self, name: &str, model: &M) -> Result<()> {
        let bytes = model
            .valid()
            .into_record()
            .into_bytes()
            .map_err(record_error)?;
        self.models.insert(
            name.into(),
            ModelRecord {
                signature: signature(model),
                bytes: bytes.to_vec(),
                parameter_shapes: parameter_shapes(model),
            },
        );
        Ok(())
    }
    fn optimizer(&mut self, name: &str, opt: &ModuleOptimizer) -> Result<()> {
        self.optimizers.insert(
            name.into(),
            opt.to_record().into_bytes().map_err(record_error)?.to_vec(),
        );
        Ok(())
    }
    fn load_model<M: Module>(&self, name: &str, model: &M) -> Result<M> {
        let stored = self
            .models
            .get(name)
            .ok_or_else(|| Error::State(format!("checkpoint missing model {name}")))?;
        ensure(
            stored.signature == signature(model),
            "checkpoint model type or parameter shape mismatch",
        )?;
        let restored = model
            .valid()
            .fork(&crate::nn::device())
            .train()
            .map(&mut CanonicalLayout)
            .try_load_record(
                ModuleRecord::from_bytes(Bytes::from_bytes_vec(stored.bytes.clone()))
                    .map_err(record_error)?,
            )
            .map_err(record_error)?;
        ensure(
            parameter_shapes(&restored) == stored.parameter_shapes,
            "checkpoint live parameter shapes differ",
        )?;
        Ok(restored)
    }
    fn load_model_inference<M: Module>(&self, name: &str, model: &M) -> Result<M> {
        let stored = self
            .models
            .get(name)
            .ok_or_else(|| Error::State(format!("checkpoint missing model {name}")))?;
        ensure(
            stored.signature == signature(model),
            "checkpoint model type or parameter shape mismatch",
        )?;
        // Do not enter the training restoration path: train() registers roots
        // even if the completed historical model is frozen immediately after.
        let restored = model
            .valid()
            .fork(&Device::flex())
            .freeze()
            .map(&mut CanonicalLayout)
            .try_load_record(
                ModuleRecord::from_bytes(Bytes::from_bytes_vec(stored.bytes.clone()))
                    .map_err(record_error)?,
            )
            .map_err(record_error)?;
        ensure(
            parameter_shapes(&restored) == stored.parameter_shapes,
            "checkpoint live parameter shapes differ",
        )?;
        Ok(restored)
    }
    fn load_optimizer(&self, name: &str, opt: ModuleOptimizer) -> Result<ModuleOptimizer> {
        decode_optimizer(self, name, opt, None, false)
    }
    fn load_optimizer_for_model(
        &self,
        name: &str,
        owner: &str,
        opt: ModuleOptimizer,
        require_state: bool,
    ) -> Result<ModuleOptimizer> {
        ensure(
            self.models.contains_key(owner),
            "checkpoint missing optimizer owner model",
        )?;
        decode_optimizer(self, name, opt, Some(owner), require_state)
    }
}

fn decode_optimizer(
    record: &LearnerRecord,
    name: &str,
    opt: ModuleOptimizer,
    owner: Option<&str>,
    require_state: bool,
) -> Result<ModuleOptimizer> {
    let bytes = record
        .optimizers
        .get(name)
        .ok_or_else(|| Error::State(format!("checkpoint missing optimizer {name}")))?;
    let bytes = Bytes::from_bytes_vec(bytes.clone());
    let packed = PackedRecord::read(bytes.clone())?;
    packed.validate_optimizer(record, owner)?;
    ensure(
        !require_state || !packed.tensors.is_empty(),
        "checkpoint missing trained optimizer moment state",
    )?;
    let restored = opt.load_record(OptimizerRecord::from_bytes(bytes).map_err(record_error)?);
    // Burn's infallible loader silently skips incomplete or foreign states.
    // Require a lossless round trip before exposing the staged optimizer.
    let loaded = PackedRecord::read(restored.to_record().into_bytes().map_err(record_error)?)?;
    ensure(
        packed == loaded,
        "checkpoint optimizer state is incomplete or incompatible",
    )?;
    Ok(restored)
}

/// Compare decoded BurnPack optimizer state independently of storage layout.
/// Paths, scalar values, tensor IDs, shapes, dtypes and payload bytes must match.
pub fn equal_packed_optimizer_state(left: &[u8], right: &[u8]) -> Result<bool> {
    let unpack = |bytes: &[u8]| PackedRecord::read(Bytes::from_bytes_vec(bytes.to_vec()));
    Ok(unpack(left)? == unpack(right)?)
}

#[derive(PartialEq)]
struct PackedTensor {
    id: Option<u64>,
    dtype: DType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
#[derive(PartialEq)]
struct PackedRecord {
    tensors: BTreeMap<String, PackedTensor>,
    scalars: BTreeMap<String, Scalar>,
    paths: BTreeMap<String, String>,
}
impl PackedRecord {
    fn read(bytes: Bytes) -> Result<Self> {
        let reader = Reader::from_bytes(bytes).map_err(record_error)?;
        let scalars = reader.scalars().clone();
        let paths = reader.metadata().clone();
        let tensors = reader
            .into_tensors()
            .map_err(record_error)?
            .into_iter()
            .map(|tensor| {
                let (name, dtype, shape, id, bytes) = tensor.into_parts().map_err(record_error)?;
                Ok((
                    name,
                    PackedTensor {
                        id,
                        dtype,
                        shape: shape.to_vec(),
                        bytes: bytes.to_vec(),
                    },
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            tensors,
            scalars,
            paths,
        })
    }
    fn validate_optimizer(&self, record: &LearnerRecord, owner: Option<&str>) -> Result<()> {
        let mut parameters = BTreeMap::new();
        for (name, model) in &record.models {
            if owner.is_some_and(|owner| owner != name) {
                continue;
            }
            let reader = Reader::from_bytes(Bytes::from_bytes_vec(model.bytes.clone()))
                .map_err(record_error)?;
            for tensor in reader.into_tensors().map_err(record_error)? {
                if let Some(id) = tensor.param_id {
                    let shape = model.parameter_shapes.get(&tensor.name).ok_or_else(|| {
                        Error::State("checkpoint missing live parameter shape".into())
                    })?;
                    let layout = (tensor.dtype, shape.clone());
                    if let Some(previous) = parameters.insert(id, layout.clone()) {
                        ensure(
                            previous == layout,
                            "checkpoint parameter id has inconsistent shapes",
                        )?;
                    }
                }
            }
        }
        let mut ranks = BTreeMap::new();
        for (key, value) in &self.scalars {
            if let Some(prefix) = key.strip_suffix(".__rank") {
                let id = prefix.parse::<u64>().map_err(record_error)?;
                let rank = usize::try_from(*value).map_err(record_error)?;
                // Burn dispatches optimizer state only for ranks 0 through 8.
                ensure(rank <= 8, "checkpoint optimizer rank is unsupported")?;
                let (_, shape) = parameters.get(&id).ok_or_else(|| {
                    Error::State("checkpoint optimizer references an unknown parameter".into())
                })?;
                ensure(
                    rank == shape.len(),
                    "checkpoint optimizer rank differs from parameter",
                )?;
                ranks.insert(id, rank);
            }
        }
        for (name, tensor) in &self.tensors {
            let id = tensor.id.ok_or_else(|| {
                Error::State("checkpoint optimizer tensor lacks a parameter id".into())
            })?;
            ensure(
                name.starts_with(&format!("{id}.")),
                "checkpoint optimizer tensor name and id differ",
            )?;
            let (dtype, shape) = parameters.get(&id).ok_or_else(|| {
                Error::State("checkpoint optimizer references an unknown parameter".into())
            })?;
            ensure(
                ranks.get(&id) == Some(&tensor.shape.len())
                    && tensor.dtype == *dtype
                    && tensor.shape == *shape,
                "checkpoint optimizer tensor shape or dtype differs from parameter",
            )?;
        }
        Ok(())
    }
}
// Burn checks record shapes before calling a parameter's load mapper. An
// initialized module can have a different live layout (e.g. column-major Linear),
// so stage the template in its save layout while retaining its load mapper.
struct CanonicalLayout;
impl ModuleMapper for CanonicalLayout {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let saved = param.transform_for_save().val();
        param.map(|_| saved)
    }
    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        let saved = param.transform_for_save().val();
        param.map(|_| saved)
    }
    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        let saved = param.transform_for_save().val();
        param.map(|_| saved)
    }
}
fn parameter_shapes<M: Module>(model: &M) -> BTreeMap<String, Vec<usize>> {
    #[derive(Default)]
    struct Shapes {
        path: Vec<String>,
        shapes: BTreeMap<String, Vec<usize>>,
    }
    impl ModuleVisitor for Shapes {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.path.push(name.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.shapes
                .insert(self.path.join("."), param.val().dims().to_vec());
        }
        fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
            self.shapes
                .insert(self.path.join("."), param.val().dims().to_vec());
        }
        fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
            self.shapes
                .insert(self.path.join("."), param.val().dims().to_vec());
        }
    }
    let mut shapes = Shapes::default();
    model.visit(&mut shapes);
    shapes.shapes
}
pub(crate) fn record_error(e: impl std::fmt::Display) -> Error {
    Error::State(format!("checkpoint record: {e}"))
}

pub(crate) fn signature<M: Module>(model: &M) -> String {
    #[derive(Default)]
    struct Shapes(Vec<Vec<usize>>);
    impl ModuleVisitor for Shapes {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.0.push(p.val().dims().to_vec());
        }
    }
    let mut shapes = Shapes::default();
    model.visit(&mut shapes);
    format!("{}:{:?}", std::any::type_name::<M>(), shapes.0)
}

#[cfg(test)]
mod packed_optimizer_tests {
    use super::*;
    use burn_pack::{Tensor, Writer};

    fn optimizer(reverse: bool, first_moment: f32, step: u64) -> Vec<u8> {
        let moment = |index, value: f32| {
            Tensor::new(
                format!("42.momentum.moment_{index}"),
                DType::F32,
                vec![1],
                Some(42),
                Bytes::from_bytes_vec(value.to_le_bytes().to_vec()),
            )
        };
        let mut tensors = vec![moment(1, first_moment), moment(2, 0.25)];
        if reverse {
            tensors.reverse();
        }
        Writer::new(tensors)
            .with_scalar("42.__rank", Scalar::UInt(1))
            .with_scalar("42.momentum.time", Scalar::UInt(step))
            .into_bytes()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn optimizer_parity_ignores_packing_order_and_checks_moments_and_steps() {
        let original = optimizer(false, 0., 5);
        let reordered = optimizer(true, 0., 5);
        assert_ne!(original, reordered);
        assert!(equal_packed_optimizer_state(&original, &reordered).unwrap());
        assert!(!equal_packed_optimizer_state(&original, &optimizer(true, -0., 5)).unwrap());
        assert!(!equal_packed_optimizer_state(&original, &optimizer(true, 0., 6)).unwrap());
        assert!(equal_packed_optimizer_state(&original, b"").is_err());
    }
}
