use rl::nn::structured::{FeatureIdentity, SemanticInput};

/// Only synthetic fixtures use ordinal column names: each is an arbitrary real.
pub fn identity(widths: [usize; 5], operations: &[&str]) -> SemanticInput {
    let fields = |prefix: &str, width| {
        (0..width)
            .map(|i| FeatureIdentity::new(format!("{prefix}{i}"), "identity_f32"))
            .collect()
    };
    SemanticInput {
        version: 1,
        namespace: "rust-rl.synthetic-structured-fixture".into(),
        observation_version: 1,
        action_version: 1,
        spatial: fields("spatial", widths[0]),
        entities: fields("entity", widths[1]),
        events: fields("event", widths[2]),
        scalars: fields("scalar", widths[3]),
        candidates: fields("candidate", widths[4]),
        operations: operations.iter().map(|s| (*s).into()).collect(),
        vocabularies: vec![],
        conventions: vec![
            "synthetic-independent-real-columns;CHW;event-ordinal=i/max(n-1,1)".into(),
        ],
    }
}
