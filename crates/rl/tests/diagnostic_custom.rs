#![cfg(all(feature = "diagnostics", feature = "neural"))]
use burn::{
    module::{Module, Param},
    tensor::Tensor,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        observer::DiagnosticObserver,
        probes::{self, ProbeSet},
        schedule::*,
        schema::*,
        storage::*,
    },
    nn::{
        matrix,
        probes::{ActivationProbe, NamedActivation},
        target::frozen_copy,
    },
    prelude::*,
};

// Deliberately has no ActivationProbe implementation.
#[derive(Module, Debug)]
struct LinearNetwork {
    weight: Param<Tensor<2>>,
}
impl Network for LinearNetwork {
    fn input_dim(&self) -> usize {
        2
    }
    fn output_dim(&self) -> usize {
        1
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.weight.val())
    }
}
#[derive(Module, Debug)]
struct Instrumented {
    inner: LinearNetwork,
}
impl Network for Instrumented {
    fn input_dim(&self) -> usize {
        2
    }
    fn output_dim(&self) -> usize {
        1
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.inner.forward(x)
    }
}
impl ActivationProbe for Instrumented {
    fn activations(&self, input: Tensor<2>, names: &[&str]) -> Result<Vec<NamedActivation>> {
        if names.iter().any(|&n| n != "affine") {
            return Err(Error::Config("unknown custom layer".into()));
        }
        let model = frozen_copy(self);
        Ok(names
            .iter()
            .map(|&n| NamedActivation {
                name: n.into(),
                tensor: model.forward(input.clone()).detach(),
            })
            .collect())
    }
}
fn train<N: Network>(network: N, path: &std::path::Path) -> Policy<N> {
    use rl::envs::lqg::{Lqg, LqgConfig};
    let env = Lqg::new(
        LqgConfig::scalar(2, 1., 1., 1., 1., 1., 1., 0., 2.).unwrap(),
        1,
    )
    .unwrap();
    let space = env.action_space().clone();
    let set = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "custom network fixture",
        vec![Observation::new(vec![1., 0.])],
        None,
        space.clone(),
    )
    .unwrap();
    let mut learner = Ppo::new(
        ActorCritic {
            policy: Policy::from_network(network, space.clone()).unwrap(),
            value: Value::new(2, &[2], &mut ChaCha8Rng::seed_from_u64(1)).unwrap(),
        },
        PpoConfig {
            epochs: 1,
            minibatch_size: 2,
            ..Default::default()
        },
        2,
    )
    .unwrap();
    let mut trainer = OnPolicyTrainer::new(
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 3).unwrap(),
        OnPolicyConfig {
            rollout_steps: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let recorder = Recorder::create(
        path,
        RunManifest::new("custom", "test", "PPO", 2, space, 0.99).unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(20);
    let mut observer = DiagnosticObserver::new(
        recorder,
        RecordingSchedule {
            metrics: Some(Interval { first: 1, every: 1 }),
            probes: Some(Interval { first: 1, every: 1 }),
            monte_carlo: None,
            snapshots: None,
        },
        |l: &Ppo<Policy<N>>, c: &RecordContext, _: Due, writer: &mut Recorder| {
            writer.append(&set.record(c.clone())?)?;
            writer.append(&probes::policy(
                c.clone(),
                &l.policy(),
                &set,
                ActMode::Sample,
                &mut rng,
            )?)?;
            Ok(())
        },
    );
    trainer
        .train_observed(&mut learner, Budget::vector_steps(4), None, &mut observer)
        .unwrap();
    observer.finish().unwrap();
    let reader = Recording::open(path, ReadLimits::default()).unwrap();
    assert_eq!(
        reader
            .entries()
            .iter()
            .filter(|e| e.kind == RecordKind::Policy)
            .count(),
        2
    );
    for i in 0..reader.entries().len() {
        reader.load(i).unwrap();
    }
    learner.policy()
}
#[test]
fn custom_models_train_with_optional_policy_and_activation_inspection() {
    let dir = tempfile::tempdir().unwrap();
    let plain = || LinearNetwork {
        weight: Param::from_tensor(matrix(vec![0.1, 0.2], 2, 1)),
    };
    train(plain(), &dir.path().join("plain"));
    let model = train(
        Instrumented { inner: plain() },
        &dir.path().join("instrumented"),
    );
    let c = probes::activations(
        RecordContext::new("custom", 2),
        "custom",
        &model.network,
        matrix(vec![1., 0.], 1, 2),
        &["affine"],
    )
    .unwrap();
    c.validate().unwrap();
    let Record::Activations(a) = &c.record else {
        panic!()
    };
    assert_eq!(a.layers.len(), 1);
    assert_eq!(c.arrays[&a.layers["affine"]].shape(), &[1, 1]);
}
