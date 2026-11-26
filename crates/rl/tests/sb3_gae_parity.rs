use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    gamma: f32,
    lambda: f32,
    rewards: Vec<f32>,
    values: Vec<f32>,
    next_values: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}

#[test]
fn gae_matches_sb3_for_episode_boundaries_and_fragments() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sb3_gae.json");
    let data = std::fs::read(path).expect("generate the SB3 GAE fixture first");
    let fixture: Fixture = serde_json::from_slice(&data).unwrap();
    assert!(fixture.cases.len() >= 4);
    for case in fixture.cases {
        let (advantages, returns) = rl::data::gae::gae(
            &case.rewards,
            &case.values,
            &case.next_values,
            &case.terminated,
            &case.truncated,
            case.gamma,
            case.lambda,
        )
        .unwrap();
        let mut max_error = 0.0_f32;
        for (label, actual, expected) in [
            ("advantage", advantages, case.advantages),
            ("return", returns, case.returns),
        ] {
            assert_eq!(actual.len(), expected.len(), "{} {label}", case.name);
            for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                let error = (actual - expected).abs();
                assert!(
                    error <= 2e-6,
                    "{} {label}[{i}]: actual {actual}, reference {expected}, error {error}",
                    case.name
                );
                max_error = max_error.max(error);
            }
        }
        println!("{} max absolute error: {max_error:.9}", case.name);
    }
}
