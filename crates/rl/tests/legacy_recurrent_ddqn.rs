#![cfg(feature = "neural")]

//! Original master run fixtures: record contracts on all hosts, numerical bits
//! on their native Windows MSVC runtime. Foreign-runtime imports are not silently exact.
use rl::{
    collect::vector_env::SyncVectorEnv,
    envs::delayed_bit::DelayedBit,
    nn::recurrent_actor::RecurrentActor,
    train::{behavior::EpsilonGreedySchedule, recurrent::TemporalTrainer},
};
use serde_json::Value;
use std::path::PathBuf;
type Trainer = TemporalTrainer<SyncVectorEnv<DelayedBit>, RecurrentActor, EpsilonGreedySchedule>;
fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recurrent-ddqn-master")
}
#[test]
fn original_run_markers_types_and_version_boundaries_are_preserved() {
    let expected: Value =
        serde_json::from_slice(&std::fs::read(fixture_root().join("expected.json")).unwrap())
            .unwrap();
    assert_eq!(
        expected["source_commit"],
        "77ccc5490261a63da1de149c3db7674cfbe85482"
    );
    assert_eq!(expected["snapshot_type"], std::any::type_name::<Trainer>());
    for case in expected["cases"].as_array().unwrap() {
        let root = fixture_root().join(case["checkpoint"].as_str().unwrap());
        let bytes = std::fs::read(root.join("checkpoint.json")).unwrap();
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        let marker = std::fs::read_to_string(root.join("COMPLETE")).unwrap();
        assert_eq!(marker, format!("{}:{hash:016x}", bytes.len()));
        let envelope: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope["snapshot_type"], expected["snapshot_type"]);
        assert_eq!(envelope["learner"]["algorithm"], "recurrent-ddqn-v1");
        assert_eq!(
            envelope["learner"]["metadata"]["version"],
            case["learner_metadata_version"]
        );
        assert_eq!(envelope["snapshot"]["version"], case["snapshot_version"]);
    }
}
#[cfg(all(target_os = "windows", target_arch = "x86_64", target_env = "msvc"))]
mod native {
    use super::*;
    use burn::{
        module::{Module, ModuleVisitor, Param},
        tensor::Tensor,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{
        algorithms::recurrent_ddqn::{RecurrentDdqn, RecurrentDdqnConfig},
        core::{action::ActionSpace, checkpoint::Checkpoint, stateful::HistorySpec},
        data::temporal::{SequenceBatch, sample_at},
        nn::{
            checkpoint::audit_adam_optimizer,
            network::Mlp,
            temporal::{GruEncoder, RecurrentQ},
            to_vec,
        },
        train::{checkpoint::load_run, recurrent::TemporalProgress},
    };
    use serde_json::json;
    fn learner(seed: u64, clip: Option<f64>) -> RecurrentDdqn {
        let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut encoder = GruEncoder::new(spec.input_dim(), 2, &mut rng).unwrap();
        encoder.gru.clip = clip;
        let q = RecurrentQ::new(encoder, Mlp::new(2, &[], 2, &mut rng).unwrap()).unwrap();
        let mut config = RecurrentDdqnConfig {
            gamma: 0.9,
            n_step: 2,
            ..Default::default()
        };
        config.ddqn.target_interval = 2;
        config.ddqn.target_tau = 0.4;
        config.ddqn.optimizer.learning_rate = 0.002;
        RecurrentDdqn::new(q, spec, config).unwrap()
    }

    fn step(
        learner: &mut RecurrentDdqn,
        trainer: &mut Trainer,
        anchored: bool,
    ) -> TemporalProgress {
        if anchored {
            trainer.step_with_reconstruction(learner).unwrap()
        } else {
            trainer.step(learner).unwrap()
        }
    }

    fn parameter_bits(model: &impl Module) -> Vec<u32> {
        struct Collect(Vec<u32>);
        impl ModuleVisitor for Collect {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                let tensor = param.val().without_autodiff();
                let length: usize = tensor.dims().iter().product();
                self.0.extend(
                    to_vec(tensor.reshape([length]))
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits),
                );
            }
        }
        let mut collector = Collect(Vec::new());
        model.visit(&mut collector);
        collector.0
    }

    struct Capture {
        readout: Value,
    }

    fn capture(
        learner: &RecurrentDdqn,
        trainer: &Trainer,
        progress: Option<&TemporalProgress>,
    ) -> Capture {
        let config = *trainer.sequence_config();
        let samples = trainer
            .last_windows()
            .iter()
            .map(|&(episode, start)| sample_at(trainer.store(), episode, start, &config).unwrap())
            .collect::<Vec<_>>();
        let windows = samples
            .iter()
            .map(|sample| {
                json!({
                    "episode": sample.episode_id,
                    "learn_start": sample.absolute_learn_start(),
                    "history_start": sample.history_start,
                    "observations": sample.observations.len(),
                })
            })
            .collect::<Vec<_>>();
        let batch = SequenceBatch { samples, config };
        let (online, target) = learner.reconstructed_q_values(&batch).unwrap();
        let target_model = learner.target();
        let record = learner.checkpoint().unwrap();
        let adam =
            audit_adam_optimizer(&record, "critic", "online", None, learner.updates()).unwrap();
        assert!(!adam.steps.is_empty());
        assert!(adam.steps.values().all(|&step| step == learner.updates()));

        let collector = trainer.collector();
        let lanes = (0..collector.len())
        .map(|lane| {
            let id = collector.lane_episode(lane).unwrap();
            let actor = collector.lane_actor(lane).unwrap();
            let episode = trainer.store().episode(id).unwrap();
            json!({
                "episode": id,
                "timestep": collector.lane_timestep(lane),
                "needs_reset": collector.lane_needs_reset(lane),
                "state": collector.lane_state(lane),
                "revision": collector.lane_revision(lane),
                "generation": actor.parameter_generation(),
                "identity": actor.model_identity(),
                "observation": collector.observations()[lane],
                "retained_base": episode.base_timestep,
                "retained_tail_step": episode.steps.last(),
                "anchor_indices": episode.anchors.iter().map(|a| a.observation_index()).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();

        // Compare every serialized trainer field other than noncanonical packed
        // bytes. The lane readouts and latest identity check the actor values.
        let mut controls = serde_json::to_value(trainer).unwrap();
        controls["collector"]["latest"]["models"]["q"]["bytes"] = Value::Null;
        for agent in controls["collector"]["agents"].as_array_mut().unwrap() {
            agent["actor"]["models"]["q"]["bytes"] = Value::Null;
        }
        let action_rngs = controls["collector"]["agents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|agent| agent["rng"].clone())
            .collect::<Vec<_>>();
        let report = progress.map(|p| json!({"collection": p.collection, "reports": p.reports}));
        Capture {
            readout: json!({
                "progress": report,
                "counters": trainer.counters(),
                "update_credit": trainer.update_credit(),
                "latest_revision": trainer.behavior_revision(),
                "latest_identity": collector.latest_actor().model_identity(),
                "windows": windows,
                "online_q": online,
                "target_q": target,
                "online_parameter_bits": learner.parameter_values().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                "target_parameter_bits": parameter_bits(&target_model),
                "gru_clip": target_model.encoder.gru.clip,
                "adam_groups": adam.parameter_ids.len(),
                "adam_step": learner.updates(),
                "lanes": lanes,
                "sampler": controls["sampler"],
                "action_rngs": action_rngs,
            }),
        }
    }

    #[test]
    fn master_written_exact_and_anchored_runs_continue_without_numerical_change() {
        let expected: Value =
            serde_json::from_slice(&std::fs::read(fixture_root().join("expected.json")).unwrap())
                .unwrap();
        for case in expected["cases"].as_array().unwrap() {
            let mut model = learner(999, None);
            let mut trainer: Trainer = load_run(
                fixture_root().join(case["checkpoint"].as_str().unwrap()),
                &mut model,
            )
            .unwrap();
            assert_eq!(capture(&model, &trainer, None).readout, case["saved"]);
            let anchored = case["learner_metadata_version"] == 2;
            for expected_step in case["continuation"].as_array().unwrap() {
                let progress = step(&mut model, &mut trainer, anchored);
                assert_eq!(
                    capture(&model, &trainer, Some(&progress)).readout,
                    *expected_step
                );
            }
        }
    }
}
