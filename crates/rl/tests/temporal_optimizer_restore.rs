#![cfg(feature = "neural")]

use burn_pack::{Bytes, Reader, Scalar, Writer};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::{
        ppo::PpoConfig, recurrent_ppo::RecurrentPpo, recurrent_reinforce::RecurrentReinforce,
        recurrent_vmpo::RecurrentVmpo, reinforce::ReinforceConfig, vmpo::VmpoConfig,
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        checkpoint::{Checkpoint, LearnerRecord, Snapshot},
        env::Observation,
        learner::UpdateReport,
        recording::RecordingKey,
        stateful::{BehaviorPublisher, HistorySpec, StatefulLearner},
        stateful_on_policy::DistributionSnapshot,
    },
    data::{
        temporal::{BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep},
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch, TemporalRow},
    },
    envs::delayed_bit::DelayedBit,
    nn::{
        temporal_actor::TemporalActor,
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_value::{TemporalStateValue, TemporalValue},
        to_vec,
    },
    train::{checkpoint::load_run_portable, temporal_on_policy::TemporalOnPolicyTrainer},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy)]
enum Kind {
    Ppo,
    Vmpo,
    Reinforce,
    ReinforceNoValue,
}
enum Learner {
    Ppo(Box<RecurrentPpo>),
    Vmpo(Box<RecurrentVmpo>),
    Reinforce(Box<RecurrentReinforce>),
}
impl Learner {
    fn actor(&self) -> TemporalActor {
        match self {
            Self::Ppo(l) => l.behavior_actor(&(), l.revision()).unwrap(),
            Self::Vmpo(l) => l.behavior_actor(&(), l.revision()).unwrap(),
            Self::Reinforce(l) => l.behavior_actor(&(), l.revision()).unwrap(),
        }
    }
    fn kind(&self) -> TemporalBatchKind {
        if matches!(self, Self::Reinforce(_)) {
            TemporalBatchKind::Episodic
        } else {
            TemporalBatchKind::Bootstrapped
        }
    }
    fn update(&mut self, batch: &TemporalOnPolicyBatch<TemporalActor>) -> Result<UpdateReport> {
        match self {
            Self::Ppo(l) => l.update(batch),
            Self::Vmpo(l) => l.update(batch),
            Self::Reinforce(l) => l.update(batch),
        }
    }
}
impl Checkpoint for Learner {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        match self {
            Self::Ppo(l) => l.checkpoint(),
            Self::Vmpo(l) => l.checkpoint(),
            Self::Reinforce(l) => l.checkpoint(),
        }
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        match self {
            Self::Ppo(l) => l.restore(record),
            Self::Vmpo(l) => l.restore(record),
            Self::Reinforce(l) => l.restore(record),
        }
    }
}
fn learner(kind: Kind, seed: u64) -> Learner {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let policy = TemporalPolicy::new(spec.clone(), Some(1), &[], &mut rng).unwrap();
    let value = || {
        TemporalValue::new(
            spec.clone(),
            Some(2),
            &[],
            &mut ChaCha8Rng::seed_from_u64(seed + 2),
        )
        .unwrap()
    };
    match kind {
        Kind::Ppo => Learner::Ppo(Box::new(
            RecurrentPpo::new(
                policy,
                value(),
                PpoConfig {
                    epochs: 1,
                    minibatch_size: 3,
                    target_kl: None,
                    ..Default::default()
                },
                seed + 1,
            )
            .unwrap(),
        )),
        Kind::Vmpo => Learner::Vmpo(Box::new(
            RecurrentVmpo::new(
                policy,
                value(),
                VmpoConfig {
                    epochs: 1,
                    minibatch_size: 3,
                    ..Default::default()
                },
                seed + 1,
            )
            .unwrap(),
        )),
        Kind::Reinforce | Kind::ReinforceNoValue => {
            let baseline = matches!(kind, Kind::Reinforce);
            Learner::Reinforce(Box::new(
                RecurrentReinforce::new(
                    policy,
                    baseline.then(value),
                    ReinforceConfig {
                        baseline,
                        ..Default::default()
                    },
                    seed + 1,
                )
                .unwrap(),
            ))
        }
    }
}
fn batch(learner: &Learner) -> TemporalOnPolicyBatch<TemporalActor> {
    let behavior = learner.actor();
    let revision = rl::core::stateful::StatefulActor::revision(&behavior);
    let mut sequences = SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 3,
            n_step: 1,
        },
        samples: vec![SequenceSample {
            episode_id: 1,
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: Default::default(),
            learn_start: 0,
            burn_start: 0,
            observations: [0.8, -0.3, 0.4, 0.]
                .into_iter()
                .map(|x| Observation::new(vec![x]))
                .collect(),
            steps: (0..3)
                .map(|i| TemporalStep {
                    action: Action::Discrete(i % 2),
                    reward: [1.0_f32, 2., 3.][i].into(),
                    terminated: i == 2,
                    truncated: false,
                    behavior: BehaviorMetadata {
                        revision,
                        log_prob: None,
                    },
                })
                .collect(),
            loss_mask: vec![true; 3],
        }],
    };
    let tokens = [(0, 0), (0, 1), (0, 2)];
    let actions: Vec<_> = sequences.samples[0]
        .steps
        .iter()
        .map(|s| s.action.clone())
        .collect();
    let distribution = behavior.policy().distribution(&sequences, &tokens).unwrap();
    let logps = to_vec(
        distribution
            .log_prob(&actions, &[None, None, None])
            .unwrap(),
    )
    .unwrap();
    let values = behavior
        .value()
        .map(|v| to_vec(v.values(&sequences, &tokens).unwrap()).unwrap());
    let rows = (0..3)
        .map(|i| {
            sequences.samples[0].steps[i].behavior.log_prob = Some(logps[i]);
            let value = values.as_ref().map(|values| values[i]);
            TemporalRow {
                key: RecordingKey {
                    episode_id: 1,
                    timestep: i,
                },
                sample: 0,
                time: i,
                log_prob: logps[i],
                latent: None,
                distribution: DistributionSnapshot::from_distribution(&distribution, i).unwrap(),
                value,
                next_value: values
                    .as_ref()
                    .map(|values| if i == 2 { 0. } else { values[i + 1] }),
                advantage: Some([0.4, -0.2, 1.3][i]),
                return_: Some([2.75, 3.5, 3.][i]),
                discount_weight: (0.9f32 as f64).powf(i as f64) as f32,
            }
        })
        .collect();
    let kind = learner.kind();
    let batch = TemporalOnPolicyBatch {
        version: 1,
        behavior,
        revision,
        kind,
        gamma: 0.9,
        gae_lambda: (kind == TemporalBatchKind::Bootstrapped).then_some(0.95),
        episode_count: 1,
        sequences,
        rows,
    };
    batch.validate().unwrap();
    batch
}
fn trained(kind: Kind) -> Learner {
    let mut result = learner(kind, 161);
    result.update(&batch(&result)).unwrap();
    result
}

fn rewrite(
    record: &mut LearnerRecord,
    name: &str,
    edit: impl FnOnce(&mut Vec<burn_pack::Tensor>, &mut BTreeMap<String, Scalar>),
) {
    let reader =
        Reader::from_bytes(Bytes::from_bytes_vec(record.optimizers[name].clone())).unwrap();
    let mut scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let mut tensors = reader.into_tensors().unwrap();
    edit(&mut tensors, &mut scalars);
    let mut writer = Writer::new(tensors);
    for (key, value) in scalars {
        writer = writer.with_scalar(&key, value);
    }
    for (key, value) in metadata {
        writer = writer.with_metadata(&key, &value);
    }
    record
        .optimizers
        .insert(name.into(), writer.into_bytes().unwrap().to_vec());
}
fn replace_moment(record: &mut LearnerRecord, owner: &str, suffix: &str, value: f32) {
    rewrite(record, owner, |tensors, _| {
        let index = tensors
            .iter()
            .position(|t| t.name.ends_with(suffix))
            .unwrap();
        let tensor = tensors.remove(index);
        let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
        let mut bytes = bytes.to_vec();
        bytes[..4].copy_from_slice(&value.to_le_bytes());
        tensors.push(burn_pack::Tensor::new(
            name,
            dtype,
            shape,
            id,
            Bytes::from_bytes_vec(bytes),
        ));
    });
}
fn rejects_unchanged(learner: &mut Learner, record: &LearnerRecord, context: &str) {
    let before = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    assert!(
        learner.restore(record).is_err(),
        "accepted corrupted Adam state: {context}"
    );
    assert!(
        before == serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
        "failed Adam restore mutated model/optimizer/RNG: {context}"
    );
}
fn numeric_cases(kind: Kind, owners: &[&str]) {
    let mut recipient = trained(kind);
    let good = recipient.checkpoint().unwrap();
    for &owner in owners {
        for (suffix, value) in [
            (".moment_1", f32::NAN),
            (".moment_1", f32::INFINITY),
            (".moment_2", -0.25),
            (".moment_2", f32::NAN),
        ] {
            let mut invalid = good.clone();
            replace_moment(&mut invalid, owner, suffix, value);
            rejects_unchanged(
                &mut recipient,
                &invalid,
                &format!("{kind:?} {owner} {suffix} {value}"),
            );
        }
    }
}

#[test]
fn ppo_actor_and_value_reject_nonfinite_and_negative_adam_moments() {
    numeric_cases(Kind::Ppo, &["policy", "value"]);
}
#[test]
fn reinforce_actor_value_and_baseline_free_actor_reject_invalid_adam_moments() {
    numeric_cases(Kind::Reinforce, &["policy", "value"]);
    numeric_cases(Kind::ReinforceNoValue, &["policy"]);
}
#[test]
fn vmpo_actor_and_value_reject_invalid_adam_moments() {
    numeric_cases(Kind::Vmpo, &["policy", "value"]);
}
#[test]
fn vmpo_dual_uses_its_separate_multiplier_owner_and_rejects_invalid_adam_moments() {
    numeric_cases(Kind::Vmpo, &["dual"]);
}

#[test]
fn adam_step_counters_require_positive_bounded_supported_owner_history() {
    for kind in [Kind::Ppo, Kind::Vmpo, Kind::Reinforce] {
        let mut recipient = trained(kind);
        let good = recipient.checkpoint().unwrap();
        for owner in good.optimizers.keys() {
            for time in [0usize, 2, i32::MAX as usize, usize::MAX] {
                let mut invalid = good.clone();
                rewrite(&mut invalid, owner, |_, scalars| {
                    let key = scalars
                        .keys()
                        .find(|key| key.ends_with(".time"))
                        .unwrap()
                        .clone();
                    scalars.insert(key, Scalar::from(time));
                });
                rejects_unchanged(
                    &mut recipient,
                    &invalid,
                    &format!("{kind:?} {owner} time={time}"),
                );
            }
        }
        let mut unsupported = good.clone();
        let count = i32::MAX as u64;
        for field in ["updates", "policy_updates", "value_updates", "revision"] {
            unsupported.metadata[field] = count.into();
        }
        unsupported.metadata["collection"]["revision"] = count.into();
        if matches!(kind, Kind::Vmpo) {
            unsupported.metadata["dual_updates"] = count.into();
        }
        rejects_unchanged(
            &mut recipient,
            &unsupported,
            &format!("{kind:?} unsupported bias-correction history"),
        );
    }
}

fn owner_updates(record: &LearnerRecord, optimizer: &str) -> u64 {
    record.metadata[format!("{optimizer}_updates")]
        .as_u64()
        .unwrap()
}

fn optimizer_times(record: &LearnerRecord, optimizer: &str) -> BTreeMap<u64, u64> {
    let reader =
        Reader::from_bytes(Bytes::from_bytes_vec(record.optimizers[optimizer].clone())).unwrap();
    reader
        .scalars()
        .iter()
        .filter_map(|(name, time)| {
            name.strip_suffix(".momentum.time")
                .map(|id| (id.parse::<u64>().unwrap(), u64::try_from(*time).unwrap()))
        })
        .collect()
}

fn assert_present_groups_participated_in_every_owner_step(record: &LearnerRecord, context: &str) {
    for optimizer in record.optimizers.keys() {
        let updates = owner_updates(record, optimizer);
        let times = optimizer_times(record, optimizer);
        assert_eq!(times.is_empty(), updates == 0, "{context} {optimizer}");
        assert_eq!(
            times.keys().copied().collect::<Vec<_>>(),
            actual_ids(record, optimizer),
            "{context} {optimizer} incomplete groups"
        );
        assert!(
            times.values().all(|&time| time == updates),
            "{context} {optimizer} has an intermittent trained group"
        );
    }
}

#[test]
fn present_adam_groups_match_owner_steps_in_real_legacy_and_repeated_new_records() {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recurrent-ppo-pr15");
    let envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("checkpoint.json")).unwrap()).unwrap();
    let legacy: LearnerRecord = serde_json::from_value(envelope["learner"].clone()).unwrap();
    assert_present_groups_participated_in_every_owner_step(&legacy, "retained 0ca PPO");
    for kind in [
        Kind::Ppo,
        Kind::Vmpo,
        Kind::Reinforce,
        Kind::ReinforceNoValue,
    ] {
        let mut current = learner(kind, 191);
        for step in 0..=3 {
            if step > 0 {
                current.update(&batch(&current)).unwrap();
            }
            assert_present_groups_participated_in_every_owner_step(
                &current.checkpoint().unwrap(),
                &format!("new {kind:?} step {step}"),
            );
        }
    }
}

fn lower_positive_time_rejects(kind: Kind, optimizer: &str) {
    let mut recipient = trained(kind);
    for _ in 0..2 {
        recipient.update(&batch(&recipient)).unwrap();
    }
    let good = recipient.checkpoint().unwrap();
    assert_present_groups_participated_in_every_owner_step(&good, &format!("{kind:?}"));
    assert!(owner_updates(&good, optimizer) >= 2);
    for version in [2, 1] {
        let mut invalid = good.clone();
        if version == 1 {
            invalid.metadata["version"] = 1.into();
            invalid
                .metadata
                .as_object_mut()
                .unwrap()
                .remove("optimizer_owners");
        }
        rewrite(&mut invalid, optimizer, |_, scalars| {
            let key = scalars
                .keys()
                .find(|key| key.ends_with(".momentum.time"))
                .unwrap()
                .clone();
            scalars.insert(key, Scalar::from(1usize));
        });
        let expected = actual_ids(&good, optimizer);
        let model = if optimizer == "dual" {
            "multipliers"
        } else {
            optimizer
        };
        // Generic Adam permits conditional groups. These concrete learners
        // require equality because every present group participates each step.
        rl::nn::checkpoint::audit_adam_optimizer(
            &invalid,
            optimizer,
            model,
            Some(&expected),
            owner_updates(&invalid, optimizer),
        )
        .unwrap();
        rejects_unchanged(
            &mut recipient,
            &invalid,
            &format!("{kind:?} {optimizer} lower positive time in schema {version}"),
        );
    }
}

#[test]
fn ppo_policy_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::Ppo, "policy");
}
#[test]
fn ppo_value_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::Ppo, "value");
}
#[test]
fn vmpo_policy_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::Vmpo, "policy");
}
#[test]
fn vmpo_value_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::Vmpo, "value");
}
#[test]
fn vmpo_dual_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::Vmpo, "dual");
}
#[test]
fn reinforce_policy_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::Reinforce, "policy");
}
#[test]
fn reinforce_value_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::Reinforce, "value");
}
#[test]
fn baseline_free_reinforce_rejects_lower_positive_adam_time() {
    lower_positive_time_rejects(Kind::ReinforceNoValue, "policy");
}

#[test]
fn exact_trained_groups_and_complete_moment_rank_time_records_cannot_be_removed() {
    for kind in [Kind::Ppo, Kind::Vmpo, Kind::Reinforce] {
        let mut recipient = trained(kind);
        let good = recipient.checkpoint().unwrap();
        for owner in good.optimizers.keys() {
            for corruption in 0..5 {
                let mut invalid = good.clone();
                rewrite(&mut invalid, owner, |tensors, scalars| {
                    let id = tensors[0].param_id.unwrap();
                    let prefix = format!("{id}.");
                    match corruption {
                        0 => {
                            tensors.retain(|t| !t.name.starts_with(&prefix));
                            scalars.retain(|key, _| !key.starts_with(&prefix));
                        }
                        1 => tensors.retain(|t| t.name != format!("{id}.momentum.moment_1")),
                        2 => tensors.retain(|t| t.name != format!("{id}.momentum.moment_2")),
                        3 => {
                            scalars.remove(&format!("{id}.momentum.time"));
                        }
                        _ => {
                            scalars.remove(&format!("{id}.__rank"));
                        }
                    }
                });
                rejects_unchanged(
                    &mut recipient,
                    &invalid,
                    &format!("{kind:?} {owner} missing group/component {corruption}"),
                );
            }
            let mut foreign = good.clone();
            let other = if owner == "policy" { "value" } else { "policy" };
            foreign
                .optimizers
                .insert(owner.clone(), good.optimizers[other].clone());
            rejects_unchanged(
                &mut recipient,
                &foreign,
                &format!("{kind:?} {owner} foreign model IDs"),
            );
        }
    }
}

fn actual_ids(record: &LearnerRecord, optimizer: &str) -> Vec<u64> {
    let reader =
        Reader::from_bytes(Bytes::from_bytes_vec(record.optimizers[optimizer].clone())).unwrap();
    reader
        .scalars()
        .keys()
        .filter_map(|key| key.strip_suffix(".__rank"))
        .map(|id| id.parse::<u64>().unwrap())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[test]
fn version_two_requires_exact_sorted_owner_membership_without_inventing_unused_parameters() {
    for kind in [
        Kind::Ppo,
        Kind::Vmpo,
        Kind::Reinforce,
        Kind::ReinforceNoValue,
    ] {
        let mut recipient = trained(kind);
        let good = recipient.checkpoint().unwrap();
        assert_eq!(
            good.metadata["version"], 2,
            "new recurrent checkpoints need explicit optimizer membership schema"
        );
        let expected: BTreeMap<_, _> = good
            .optimizers
            .keys()
            .map(|name| (name.clone(), actual_ids(&good, name)))
            .collect();
        assert!(good.metadata["optimizer_owners"] == serde_json::to_value(&expected).unwrap());
        let packed: Vec<u8> = serde_json::from_value(
            serde_json::to_value(&good).unwrap()["models"]["policy"]["bytes"].clone(),
        )
        .unwrap();
        let reader = Reader::from_bytes(Bytes::from_bytes_vec(packed)).unwrap();
        let owned: BTreeSet<_> = reader
            .into_tensors()
            .unwrap()
            .into_iter()
            .filter_map(|t| t.param_id)
            .collect();
        assert!(
            expected["policy"].len() < owned.len(),
            "discrete policy log_std is legitimately untrained"
        );
        for corruption in 0..5 {
            let mut invalid = good.clone();
            match corruption {
                0 => {
                    invalid
                        .metadata
                        .as_object_mut()
                        .unwrap()
                        .remove("optimizer_owners");
                }
                1 => {
                    invalid.metadata["optimizer_owners"]
                        .as_object_mut()
                        .unwrap()
                        .remove("policy");
                }
                2 => invalid.metadata["optimizer_owners"]["unexpected"] = serde_json::json!([]),
                3 => invalid.metadata["optimizer_owners"]["policy"] = serde_json::json!([]),
                _ => {
                    let id = expected["policy"][0];
                    invalid.metadata["optimizer_owners"]["policy"] = serde_json::json!([id, id]);
                }
            }
            rejects_unchanged(
                &mut recipient,
                &invalid,
                &format!("{kind:?} malformed version-two optimizer owner metadata {corruption}"),
            );
        }
    }
}

type TensorContents = (burn_pack::DType, Vec<usize>, Option<u64>, Vec<u8>);
type PackContents = (
    BTreeMap<String, Scalar>,
    BTreeMap<String, String>,
    BTreeMap<String, TensorContents>,
);
type RecordContents = (serde_json::Value, BTreeMap<String, PackContents>);

fn model_and_optimizer_contents(record: &LearnerRecord) -> RecordContents {
    let mut envelope = serde_json::to_value(record).unwrap();
    let mut packs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for (name, model) in envelope["models"].as_object_mut().unwrap() {
        packs.insert(
            format!("model {name}"),
            serde_json::from_value(model["bytes"].take()).unwrap(),
        );
    }
    for (name, optimizer) in envelope["optimizers"].as_object_mut().unwrap() {
        packs.insert(
            format!("optimizer {name}"),
            serde_json::from_value(optimizer.take()).unwrap(),
        );
    }
    let decoded = packs
        .into_iter()
        .map(|(name, bytes)| {
            let reader = Reader::from_bytes(Bytes::from_bytes_vec(bytes)).unwrap();
            let scalars = reader.scalars().clone();
            let metadata = reader.metadata().clone();
            let tensors = reader
                .into_tensors()
                .unwrap()
                .into_iter()
                .map(|t| {
                    let (name, dtype, shape, id, bytes) = t.into_parts().unwrap();
                    (name, (dtype, shape.to_vec(), id, bytes.to_vec()))
                })
                .collect();
            (name, (scalars, metadata, tensors))
        })
        .collect();
    (envelope, decoded)
}

fn first_json_difference(
    a: &serde_json::Value,
    b: &serde_json::Value,
    path: &str,
) -> Option<String> {
    if a == b {
        return None;
    }
    match (a, b) {
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            let keys: BTreeSet<_> = a.keys().chain(b.keys()).collect();
            for key in keys {
                let next = format!("{path}.{key}");
                match (a.get(key), b.get(key)) {
                    (Some(a), Some(b)) => {
                        if let Some(diff) = first_json_difference(a, b, &next) {
                            return Some(diff);
                        }
                    }
                    (None, _) => {
                        return Some(format!("{next}: absent in expected, present in restored"));
                    }
                    (_, None) => {
                        return Some(format!("{next}: present in expected, absent in restored"));
                    }
                }
            }
            None
        }
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path}: array lengths {} != {}", a.len(), b.len()));
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(i, (a, b))| first_json_difference(a, b, &format!("{path}[{i}]")))
        }
        _ => Some(format!("{path}: expected {a}, restored {b}")),
    }
}

fn first_record_difference(a: &RecordContents, b: &RecordContents) -> String {
    if let Some(diff) = first_json_difference(&a.0, &b.0, "record") {
        return diff;
    }
    if !a.1.keys().eq(b.1.keys()) {
        return "model/optimizer pack names differ".into();
    }
    for (name, (scalars, paths, tensors)) in &a.1 {
        let (other_scalars, other_paths, other_tensors) = &b.1[name];
        if scalars != other_scalars {
            let keys: BTreeSet<_> = scalars.keys().chain(other_scalars.keys()).collect();
            let key = keys
                .into_iter()
                .find(|key| scalars.get(*key) != other_scalars.get(*key))
                .unwrap();
            return format!(
                "{name} scalar {key}: expected {:?}, restored {:?}",
                scalars.get(key),
                other_scalars.get(key)
            );
        }
        if paths != other_paths {
            return format!("{name}: parameter path metadata differs");
        }
        if !tensors.keys().eq(other_tensors.keys()) {
            return format!("{name}: named tensor keys differ");
        }
        for (key, (dtype, shape, id, bytes)) in tensors {
            let (other_dtype, other_shape, other_id, other_bytes) = &other_tensors[key];
            if dtype != other_dtype {
                return format!("{name} {key}: dtype {dtype:?} != {other_dtype:?}");
            }
            if shape != other_shape {
                return format!("{name} {key}: shape {shape:?} != {other_shape:?}");
            }
            if id != other_id {
                return format!("{name} {key}: parameter ID {id:?} != {other_id:?}");
            }
            if bytes != other_bytes {
                let offset = bytes.iter().zip(other_bytes).position(|(a, b)| a != b);
                return format!(
                    "{name} {key}: exact tensor bytes differ, first byte {offset:?}, payload lengths {} != {}",
                    bytes.len(),
                    other_bytes.len()
                );
            }
        }
    }
    "no differing content found".into()
}

fn assert_ppo_config_bits(a: &PpoConfig, b: &PpoConfig) {
    assert_eq!(
        a.policy.learning_rate.to_bits(),
        b.policy.learning_rate.to_bits(),
        "policy learning-rate bits differ"
    );
    assert_eq!(
        a.value.learning_rate.to_bits(),
        b.value.learning_rate.to_bits(),
        "value learning-rate bits differ"
    );
    assert_eq!(
        a.policy.max_grad_norm.map(f32::to_bits),
        b.policy.max_grad_norm.map(f32::to_bits)
    );
    assert_eq!(
        a.value.max_grad_norm.map(f32::to_bits),
        b.value.max_grad_norm.map(f32::to_bits)
    );
    assert_eq!(a.epochs, b.epochs);
    assert_eq!(a.minibatch_size, b.minibatch_size);
    assert_eq!(a.clip.to_bits(), b.clip.to_bits());
    assert_eq!(a.entropy.to_bits(), b.entropy.to_bits());
    assert_eq!(a.target_kl.map(f32::to_bits), b.target_kl.map(f32::to_bits));
}

#[test]
fn json_codec_preserves_declared_f64_optimizer_control_bits() {
    let config = PpoConfig {
        policy: rl::algorithms::OptimizerConfig {
            learning_rate: 0.029_999_999_329_447_746,
            ..Default::default()
        },
        value: rl::algorithms::OptimizerConfig {
            learning_rate: 0.000_300_000_000_000_000_03,
            ..Default::default()
        },
        target_kl: Some(0.03),
        ..Default::default()
    };
    let bytes = serde_json::to_vec(&config).unwrap();
    let direct: PpoConfig = serde_json::from_slice(&bytes).unwrap();
    assert_ppo_config_bits(&config, &direct);
    // LearnerRecord stores config in a Value envelope before filesystem writing.
    let value = serde_json::to_value(&config).unwrap();
    let serialized = serde_json::to_vec(&value).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&serialized).unwrap();
    let through_record: PpoConfig = serde_json::from_value(parsed).unwrap();
    assert_ppo_config_bits(&config, &through_record);
}

#[test]
fn legacy_version_one_restores_exact_state_and_still_rejects_invalid_numbers() {
    for kind in [Kind::Ppo, Kind::Vmpo, Kind::Reinforce] {
        let mut original = trained(kind);
        let mut legacy = original.checkpoint().unwrap();
        legacy.metadata["version"] = 1.into();
        legacy
            .metadata
            .as_object_mut()
            .unwrap()
            .remove("optimizer_owners");
        let mut restored = learner(kind, 181);
        restored.restore(&legacy).unwrap();
        let before = serde_json::to_value(restored.checkpoint().unwrap()).unwrap();
        let mut invalid = legacy;
        replace_moment(&mut invalid, "policy", ".moment_1", f32::NAN);
        assert!(
            restored.restore(&invalid).is_err(),
            "legacy numeric audit was bypassed"
        );
        assert!(
            before == serde_json::to_value(restored.checkpoint().unwrap()).unwrap(),
            "failed legacy restore mutated destination"
        );
        let a = original.update(&batch(&original)).unwrap();
        let b = restored.update(&batch(&restored)).unwrap();
        assert_eq!(a.metrics, b.metrics);
        assert!(
            model_and_optimizer_contents(&original.checkpoint().unwrap())
                == model_and_optimizer_contents(&restored.checkpoint().unwrap()),
            "legacy continuation changed exact decoded model/optimizer/RNG state"
        );
    }
}

#[test]
fn retained_schema_one_ppo_hard_gate_run_still_loads_exact_checkpoint_contents() {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recurrent-ppo-pr15");
    let envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("checkpoint.json")).unwrap()).unwrap();
    let saved: LearnerRecord = serde_json::from_value(envelope["learner"].clone()).unwrap();
    assert_eq!(saved.metadata["version"], 1);
    let mut destination = match learner(Kind::Ppo, 182) {
        Learner::Ppo(l) => *l,
        _ => unreachable!(),
    };
    // This permanent schema-one learner fixture was produced on Windows. The
    // explicit import audits foreign arithmetic and reconstructs native memories;
    // model/optimizer/config/RNG contents below must still restore exactly.
    let (run, portability): (TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>, _) =
        load_run_portable(&directory, &mut destination).unwrap();
    assert_eq!(portability.version, 1);
    assert_eq!(portability.lanes.len(), 8);
    assert!(
        portability
            .lanes
            .iter()
            .flat_map(|lane| &lane.memories)
            .all(|memory| memory.max_ulp_difference <= 4)
    );
    run.validate_snapshot().unwrap();
    let declared: PpoConfig = serde_json::from_value(saved.metadata["config"].clone()).unwrap();
    assert_ppo_config_bits(&declared, destination.config());
    let mut migrated = saved;
    migrated.metadata["version"] = 2.into();
    let expected: BTreeMap<_, _> = migrated
        .optimizers
        .keys()
        .map(|name| (name.clone(), actual_ids(&migrated, name)))
        .collect();
    migrated.metadata["optimizer_owners"] = serde_json::to_value(expected).unwrap();
    let expected = model_and_optimizer_contents(&migrated);
    let restored = model_and_optimizer_contents(&destination.checkpoint().unwrap());
    assert!(
        expected == restored,
        "historical PPO migration: {}",
        first_record_difference(&expected, &restored)
    );
}

#[test]
fn portable_import_rejects_invalid_optimizer_after_memory_audit_without_partial_installation() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recurrent-ppo-pr15/checkpoint.json");
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    let mut record: LearnerRecord = serde_json::from_value(envelope["learner"].clone()).unwrap();
    replace_moment(&mut record, "policy", ".moment_1", f32::NAN);
    envelope["learner"] = serde_json::to_value(record).unwrap();
    let bytes = serde_json::to_vec(&envelope).unwrap();
    let checksum = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("checkpoint.json"), &bytes).unwrap();
    std::fs::write(
        directory.path().join("COMPLETE"),
        format!("{}:{checksum:016x}", bytes.len()),
    )
    .unwrap();
    let mut destination = match learner(Kind::Ppo, 182) {
        Learner::Ppo(l) => *l,
        _ => unreachable!(),
    };
    let before = serde_json::to_value(destination.checkpoint().unwrap()).unwrap();
    assert!(
        load_run_portable::<TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>>(
            directory.path(),
            &mut destination
        )
        .is_err()
    );
    assert_eq!(
        serde_json::to_value(destination.checkpoint().unwrap()).unwrap(),
        before
    );
}
