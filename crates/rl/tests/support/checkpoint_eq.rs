//! Compare complete checkpoint semantics without Burn-pack container ordering.
use burn::tensor::Bytes;
use burn_pack::Reader;
use rl::core::checkpoint::LearnerRecord;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn unpack(bytes: Vec<u8>) -> Value {
    let reader = Reader::from_bytes(Bytes::from_bytes_vec(bytes)).unwrap();
    let scalars: BTreeMap<_, _> = reader
        .scalars()
        .iter()
        .map(|(k, v)| (k.clone(), format!("{v:?}")))
        .collect();
    let metadata: BTreeMap<_, _> = reader
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let tensors: BTreeMap<_, _> = reader
        .into_tensors()
        .unwrap()
        .into_iter()
        .map(|t| {
            let (name, dtype, shape, id, data) = t.into_parts().unwrap();
            (
                name,
                json!({"dtype":format!("{dtype:?}"),"shape":shape,"id":id,"bytes":data}),
            )
        })
        .collect();
    json!({"scalars":scalars,"metadata":metadata,"tensors":tensors})
}
pub fn normalized(record: &LearnerRecord) -> Value {
    let mut value = serde_json::to_value(record).unwrap();
    for model in value["models"].as_object_mut().unwrap().values_mut() {
        model["bytes"] = unpack(serde_json::from_value(model["bytes"].take()).unwrap());
    }
    for optimizer in value["optimizers"].as_object_mut().unwrap().values_mut() {
        *optimizer = unpack(serde_json::from_value(optimizer.take()).unwrap());
    }
    value
}
