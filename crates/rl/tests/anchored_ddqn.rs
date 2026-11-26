#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::recurrent_ddqn::*,
    core::{
        anchors::*,
        reconstruction::*,
        stateful::{HistoryInput, HistorySpec, PreviousInput, PreviousStep},
        temporal::*,
    },
    data::temporal::*,
    nn::{
        network::Mlp,
        temporal::{GruEncoder, RecurrentQ},
    },
    prelude::*,
};

fn learner() -> RecurrentDdqn {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let model = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    let mut config = RecurrentDdqnConfig {
        gamma: 0.9,
        n_step: 2,
        ..Default::default()
    };
    config.ddqn.target_interval = 100;
    let mut l = RecurrentDdqn::new(model, spec, config).unwrap();
    l.set_update_mode(RecurrentUpdateMode::OptimizedExact);
    l
}
fn full() -> SequenceBatch {
    let mut s = EpisodeStore::new(80, 1, ActionSpace::Discrete { count: 2 }).unwrap();
    s.begin_episode(7, &EpisodeStart::fresh(Observation::new(vec![1.])))
        .unwrap();
    for t in 0..40 {
        s.push(&TemporalTransition {
            episode_id: 7,
            timestep: t,
            observation: Observation::new(vec![(t + 1) as f32]),
            action: Action::Discrete(t % 2),
            reward: ((t % 3) as f32 / 3.).into(),
            next_observation: Observation::new(vec![(t + 2) as f32]),
            terminated: false,
            truncated: t == 39,
            behavior: BehaviorMetadata::default(),
        })
        .unwrap();
    }
    let config = SequenceConfig {
        burn_in: 3,
        learn_len: 4,
        n_step: 2,
    };
    SequenceBatch {
        samples: vec![sample_at(&s, 7, 32, &config).unwrap()],
        config,
    }
}
fn anchored(l: &RecurrentDdqn, b: &SequenceBatch, cut: usize) -> SequenceBatch {
    let mut out = b.clone();
    let s = &b.samples[0];
    let mut anchors = std::collections::BTreeMap::new();
    for (id, m) in l.replay_representations().unwrap() {
        let mut a = reset_anchor(&m, id, s.episode_id).unwrap();
        for t in 0..cut {
            a = advance_anchor(
                ReconstructionConfig::default(),
                &m,
                &a,
                HistoryInput {
                    observation: &s.observations[t],
                    previous: t.checked_sub(1).map(|i| PreviousInput {
                        factual: None,
                        action: &s.steps[i].action,
                        reward: s.steps[i].reward.total(),
                    }),
                },
            )
            .unwrap()
            .0;
        }
        anchors.insert(id, a);
    }
    let p = &s.steps[cut - 1];
    let sample = &mut out.samples[0];
    sample.history_start = cut;
    sample.previous = Some(PreviousStep {
        action: p.action.clone(),
        reward: p.reward.total(),
    });
    sample.origin = SequenceOrigin::Anchored(AnchorSet {
        version: ANCHOR_VERSION,
        anchors,
    });
    sample.observations = sample.observations[cut..].to_vec();
    sample.steps = sample.steps[cut..].to_vec();
    sample.loss_mask = sample.loss_mask[cut..].to_vec();
    sample.learn_start -= cut;
    sample.burn_start = sample.learn_start.saturating_sub(out.config.burn_in);
    out.validate().unwrap();
    out
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        assert!((x - y).abs() <= 2e-6 + 2e-6 * x.abs(), "{x} versus {y}");
    }
}
#[test]
fn same_parameter_anchor_reconstructs_full_prefix_q_targets_and_updates() {
    let full = full();
    let mut a = learner();
    let mut b = learner();
    let cut = 28;
    for _ in 0..3 {
        let bounded = anchored(&b, &full, cut);
        let (fo, ft) = a.reconstructed_q_values(&full).unwrap();
        let (bo, bt) = b.reconstructed_q_values(&bounded).unwrap();
        close(&fo[cut * 2..], &bo);
        close(&ft[cut * 2..], &bt);
        assert!(
            b.reconstruction_decisions(&bounded)
                .unwrap()
                .iter()
                .all(|(o, t)| !o.approximate && !t.approximate)
        );
        let ar = a.update(&full).unwrap();
        let br = b.update(&bounded).unwrap();
        for key in ["critic_loss", "target_mean"] {
            assert!((ar.metrics[key] - br.metrics[key]).abs() < 2e-6);
        }
        close(
            &a.parameter_values().unwrap(),
            &b.parameter_values().unwrap(),
        );
    }
}
#[test]
fn stale_or_missing_target_anchor_rejected_atomically_and_approximation_explicit() {
    let full = full();
    let mut l = learner();
    let data = anchored(&l, &full, 28);
    l.update(&full).unwrap();
    let before = serde_json::to_vec(&l.checkpoint().unwrap()).unwrap();
    assert!(l.update(&data).is_err());
    assert_eq!(
        before,
        serde_json::to_vec(&l.checkpoint().unwrap()).unwrap()
    );
    let mut missing = data.clone();
    if let SequenceOrigin::Anchored(set) = &mut missing.samples[0].origin {
        set.anchors.remove(&TARGET_REPRESENTATION);
    }
    assert!(l.update(&missing).is_err());
    assert_eq!(
        before,
        serde_json::to_vec(&l.checkpoint().unwrap()).unwrap()
    );
    let cfg = ReconstructionConfig {
        version: 1,
        mode: ReconstructionMode::Anchored {
            max_generation_lag: Some(1),
        },
    };
    l.set_reconstruction_config(cfg).unwrap();
    let decisions = l.reconstruction_decisions(&data).unwrap();
    assert!(decisions[0].0.approximate);
    assert!(!decisions[0].1.approximate); // unchanged independent target parameters
    let mut restored = learner();
    restored.restore(&l.checkpoint().unwrap()).unwrap();
    assert_eq!(restored.reconstruction_config(), cfg);
    assert_eq!(
        l.update(&data).unwrap().metrics,
        restored.update(&data).unwrap().metrics
    );
    assert_eq!(
        l.parameter_values().unwrap(),
        restored.parameter_values().unwrap()
    );
}
#[test]
fn reconstruction_contract_corruption_preserves_destination_and_old_default_wire() {
    let mut l = learner();
    let mut r = l.checkpoint().unwrap();
    assert!(r.temporal_reconstruction.is_none());
    assert_eq!(r.metadata["version"], 1);
    let before = l.parameter_values().unwrap();
    r.metadata["reconstruction"] = serde_json::to_value(ReconstructionConfig {
        version: 1,
        mode: ReconstructionMode::Anchored {
            max_generation_lag: None,
        },
    })
    .unwrap();
    assert!(l.restore(&r).is_err());
    assert_eq!(before, l.parameter_values().unwrap());
    r.metadata["version"] = 2.into();
    assert!(l.restore(&r).is_err());
    assert_eq!(before, l.parameter_values().unwrap());
}
