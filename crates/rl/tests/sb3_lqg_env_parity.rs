use nalgebra::DMatrix;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Env,
    },
    envs::lqg::{Lqg, LqgConfig},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    seed: u64,
    config: Config,
    actions: Vec<Vec<f32>>,
    normals: Vec<Vec<f64>>,
    initial_observation: Vec<f32>,
    transitions: Vec<Transition>,
}
#[derive(Deserialize)]
struct Config {
    a: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<Vec<f64>>>,
    q: Vec<Vec<Vec<f64>>>,
    r: Vec<Vec<Vec<f64>>>,
    noise: Vec<Vec<Vec<f64>>>,
    terminal: Vec<Vec<f64>>,
    initial: Vec<f64>,
    low: Vec<f32>,
    high: Vec<f32>,
}
#[derive(Deserialize)]
struct Transition {
    observation: Vec<f32>,
    reward: f32,
    terminated: bool,
    truncated: bool,
}

fn matrix(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let n = rows.len();
    let m = rows[0].len();
    assert!(rows.iter().all(|row| row.len() == m));
    DMatrix::from_row_slice(n, m, &rows.iter().flatten().copied().collect::<Vec<_>>())
}

#[test]
fn gym_lqg_transitions_match_rust_with_identical_noise() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sb3_lqg_env.json"
    );
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(fixture.cases.len() >= 2);
    let mut maximum_error = 0_f32;
    for case in fixture.cases {
        let c = case.config;
        let config = LqgConfig {
            a: c.a.iter().map(|m| matrix(m)).collect(),
            b: c.b.iter().map(|m| matrix(m)).collect(),
            q: c.q.iter().map(|m| matrix(m)).collect(),
            r: c.r.iter().map(|m| matrix(m)).collect(),
            noise: c.noise.iter().map(|m| matrix(m)).collect(),
            terminal: matrix(&c.terminal),
            initial: c.initial,
            action_space: ActionSpace::Box {
                low: c.low,
                high: c.high,
            },
        };
        let mut env = Lqg::new(config, case.seed).unwrap();
        assert_eq!(
            env.reset(Some(case.seed))
                .unwrap()
                .observation
                .dense_values(),
            case.initial_observation
        );
        assert_eq!(case.actions.len(), case.normals.len());
        assert_eq!(case.actions.len(), case.transitions.len());
        let mut rng = ChaCha8Rng::seed_from_u64(case.seed);
        for (index, ((action, normals), expected)) in case
            .actions
            .iter()
            .zip(&case.normals)
            .zip(&case.transitions)
            .enumerate()
        {
            assert_eq!(normals.len() + 1, case.initial_observation.len());
            for expected_normal in normals {
                let actual_normal = rng.sample::<f64, _>(StandardNormal);
                assert!(
                    (actual_normal - expected_normal).abs()
                        <= 2.0 * f64::EPSILON * (1.0 + expected_normal.abs())
                );
            }
            let actual = env.step(&Action::Continuous(action.clone())).unwrap();
            assert_eq!(
                actual.observation.dense_values().len(),
                expected.observation.len()
            );
            for (a, b) in actual
                .observation
                .dense_values()
                .iter()
                .zip(&expected.observation)
                .chain(std::iter::once((&actual.reward.total(), &expected.reward)))
            {
                let error = (a - b).abs();
                maximum_error = maximum_error.max(error);
                assert!(
                    error <= 2e-6 * (1.0 + b.abs()),
                    "{} step {index}: {a} != {b}",
                    case.name
                );
            }
            assert_eq!(
                (actual.terminated, actual.truncated),
                (expected.terminated, expected.truncated)
            );
        }
        assert!(
            env.step(&Action::Continuous(case.actions[0].clone()))
                .is_err()
        );
    }
    eprintln!("maximum absolute LQG transition error: {maximum_error:e}");
}
