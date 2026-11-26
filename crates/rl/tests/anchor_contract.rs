use rl::{
    Result,
    core::{
        action::Action,
        anchors::{AnchorSchema, RepresentationId},
        env::Observation,
        reconstruction::*,
        stateful::{HistoryInput, PreviousInput},
    },
};
fn ensure(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(rl::Error::State(message.into()))
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Belief {
    total: u64,
    count: usize,
}
#[derive(Clone)]
struct Model {
    identity: [u8; 32],
    generation: u64,
    schema: AnchorSchema,
}
impl Model {
    fn new(generation: u64) -> Self {
        Self {
            identity: [generation as u8; 32],
            generation,
            schema: AnchorSchema {
                codec: 71,
                codec_version: 1,
                representation_identity: [4; 32],
                dimensions: vec![2],
            },
        }
    }
}
impl ReconstructionModel for Model {
    type State = Belief;
    fn model_identity(&self) -> [u8; 32] {
        self.identity
    }
    fn generation(&self) -> u64 {
        self.generation
    }
    fn schema(&self) -> &AnchorSchema {
        &self.schema
    }
    fn initial_state(&self) -> Result<Belief> {
        Ok(Belief { total: 0, count: 0 })
    }
    fn validate_state(&self, s: &Belief) -> Result<()> {
        ensure(s.count < 10000, "invalid count")
    }
    fn assimilate(&self, s: &Belief, input: HistoryInput<'_>) -> Result<Belief> {
        let p = input.previous.map_or(0, |p| match p.action {
            Action::Discrete(a) => *a as u64 + p.reward as u64,
            _ => 0,
        });
        Ok(Belief {
            total: s.total + input.observation.dense_values()[0] as u64 + p + self.generation,
            count: s.count + 1,
        })
    }
    fn encode_state(&self, s: &Belief) -> Result<Vec<u8>> {
        self.validate_state(s)?;
        Ok([s.total.to_le_bytes(), (s.count as u64).to_le_bytes()].concat())
    }
    fn decode_state(&self, p: &[u8]) -> Result<Belief> {
        ensure(p.len() == 16, "bad belief codec length")?;
        let s = Belief {
            total: u64::from_le_bytes(p[..8].try_into().unwrap()),
            count: u64::from_le_bytes(p[8..].try_into().unwrap()) as usize,
        };
        self.validate_state(&s)?;
        Ok(s)
    }
}
#[test]
fn structured_non_neural_state_has_one_pre_observation_clock() {
    let m = Model::new(0);
    let id = RepresentationId(9);
    let a = reset_anchor(&m, id, 13).unwrap();
    let o = Observation::new(vec![3.]);
    let (b, d) = advance_anchor(
        ReconstructionConfig::default(),
        &m,
        &a,
        HistoryInput {
            observation: &o,
            previous: None,
        },
    )
    .unwrap();
    assert!(!d.approximate);
    assert_eq!(b.observation_index, 1);
    assert_eq!(
        m.decode_state(&b.payload).unwrap(),
        Belief { total: 3, count: 1 }
    );
    let action = Action::Discrete(2);
    let input = HistoryInput {
        observation: &o,
        previous: Some(PreviousInput {
            factual: None,
            action: &action,
            reward: 5.,
        }),
    };
    assert!(advance_anchor(ReconstructionConfig::default(), &m, &a, input).is_err());
    assert!(
        advance_anchor(
            ReconstructionConfig::default(),
            &m,
            &b,
            HistoryInput {
                observation: &o,
                previous: None
            }
        )
        .is_err()
    );
    let (c, _) = advance_anchor(ReconstructionConfig::default(), &m, &b, input).unwrap();
    assert_eq!(c.observation_index, 2);
    assert_eq!(
        m.decode_state(&c.payload).unwrap(),
        Belief {
            total: 13,
            count: 2
        }
    );
    assert_ne!(anchor_identity(&a).unwrap(), anchor_identity(&b).unwrap());
    let restored = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(
        anchor_identity(&c).unwrap(),
        anchor_identity(&restored).unwrap()
    );
}
#[test]
fn stale_states_are_approximate_and_never_launder_ancestry() {
    let old = Model::new(1);
    let new = Model::new(5);
    let id = RepresentationId(3);
    let o = Observation::new(vec![2.]);
    let (a, _) = advance_anchor(
        ReconstructionConfig::default(),
        &old,
        &reset_anchor(&old, id, 1).unwrap(),
        HistoryInput {
            observation: &o,
            previous: None,
        },
    )
    .unwrap();
    assert!(decode_anchor(ReconstructionConfig::default(), &new, id, 1, 1, &a).is_err());
    let anchored = ReconstructionConfig {
        version: 1,
        mode: ReconstructionMode::Anchored {
            max_generation_lag: Some(4),
        },
    };
    let (_, d) = decode_anchor(anchored, &new, id, 1, 1, &a).unwrap();
    assert!(d.approximate);
    assert_eq!(d.generation_lag, 4);
    assert!(
        decode_anchor(
            ReconstructionConfig {
                version: 1,
                mode: ReconstructionMode::Anchored {
                    max_generation_lag: Some(3)
                }
            },
            &new,
            id,
            1,
            1,
            &a
        )
        .is_err()
    );
    let action = Action::Discrete(0);
    let (mut b, d) = advance_anchor(
        anchored,
        &new,
        &a,
        HistoryInput {
            observation: &o,
            previous: Some(PreviousInput {
                factual: None,
                action: &action,
                reward: 0.,
            }),
        },
    )
    .unwrap();
    assert!(d.approximate);
    assert!(!b.exact_ancestry);
    assert!(decode_anchor(ReconstructionConfig::default(), &new, id, 1, 2, &b).is_err());
    assert!(decode_anchor(anchored, &old, id, 1, 2, &b).is_err());
    b.exact_ancestry = true;
    assert!(b.validate().is_err());
}
#[test]
fn corrupt_or_incompatible_anchor_rejected_before_assimilation() {
    let m = Model::new(0);
    let id = RepresentationId(1);
    let a = reset_anchor(&m, id, 2).unwrap();
    for mutation in 0..7 {
        let mut b = a.clone();
        match mutation {
            0 => b.version = 99,
            1 => b.episode_id = 3,
            2 => b.representation_id = RepresentationId(8),
            3 => b.schema.codec_version = 7,
            4 => b.payload.clear(),
            5 => b.observation_index = 1,
            _ => b.payload[0] = 1,
        };
        assert!(
            decode_anchor(ReconstructionConfig::default(), &m, id, 2, 0, &b).is_err(),
            "mutation {mutation}"
        );
    }
    assert!(
        decode_anchor(
            ReconstructionConfig {
                version: 99,
                ..Default::default()
            },
            &m,
            id,
            2,
            0,
            &a
        )
        .is_err()
    );
}
