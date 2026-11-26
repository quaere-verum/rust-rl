//! Neural models and packing for variable-size public observations and complete actions.
//! Only validated numeric features enter tensors.
mod action_value;
mod batch;
mod candidate_policy;
mod checkpoint;
mod encoder;
mod graph;
mod guidance;
mod heads;
mod input;
mod lightweight;
mod policy;
mod q;
mod semantic;
mod supervision;
mod value_supervision;
pub(crate) use action_value::evaluate_checked;
pub(crate) use candidate_policy::{
    candidate_log_probabilities_checked, candidate_probabilities_checked,
};
pub(crate) use checkpoint::restore as restore_model;
pub use supervision::{candidate_supervision_loss, recurrent_supervision_loss};
pub use value_supervision::{recurrent_value_supervision_loss, value_supervision_loss};

pub use crate::core::structured::{ArgumentKind, FeatureIdentity, VocabularyIdentity};
pub use action_value::{BuildStructuredActionValue, StructuredActionValue};
pub use batch::{PackLimits, StructuredBatch};
pub use candidate_policy::{BuildCandidatePolicy, CandidatePolicy};
pub use encoder::{EncodedWorld, EncoderConfig, EncoderStage, StructuredEncoder};
pub use graph::{
    GraphStructuredBatch, GraphStructuredQ, GraphStructuredQConfig, SparseGraphBatch,
    SparseGraphEncoder,
};
pub use guidance::{NeuralGuidance, PublicValueInput};
pub use heads::StructuredModelConfig;
pub use input::{Argument, Candidate, Event, InputSchema, SpatialSample, StructuredSample};
pub use lightweight::{
    ConditionalPolicyLaw, FrozenRecurrentFeatures, LightweightBatch, LightweightConfig,
    LightweightEncoder, LightweightPolicy, LightweightQ, LightweightStage, LightweightValue,
    LightweightWorld, RecurrentCandidatePrediction, RecurrentDistributionalQEvaluation,
    RecurrentLightweightConfig, RecurrentLightweightDistributionalQ, RecurrentLightweightMemory,
    RecurrentLightweightPolicy, RecurrentLightweightQ, RecurrentLightweightStage,
    RecurrentLightweightState, RecurrentLightweightValue, RecurrentPolicyEvaluation,
    StructuredHistoryInput, StructuredHistoryValues,
};
pub use policy::{Conditional, PolicyEvaluation, StructuredDecision, StructuredPolicy};
pub use q::{CandidateValues, StructuredQ};
pub use semantic::SemanticInput;

use burn::tensor::Tensor;

fn mean_rows(values: Tensor<3>, mask: Tensor<3>) -> Tensor<2> {
    let [b, _, width] = values.dims();
    values.sum_dim(1).reshape([b, width]) / mask.sum_dim(1).reshape([b, 1]).clamp_min(1.)
}
