use rl::approx::batch::{BatchObjective, BatchOptions, DecisionSpan, FactualBatch};
use rl::{
    Result,
    approx::{
        FeatureMap, FeatureSpec,
        diagnostics::{
            ComparedDecision, PolicyReturnSource, PolicyValueComparison, ReferenceInterval,
            compare_action_values, first_consequential_divergence, policy_changes,
        },
    },
    core::{
        action::{Action, ActionSpace},
        env::{EpisodeStart, Observation, PublicHistory},
        structured::FeatureIdentity,
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
    data::temporal::EpisodeStore,
};

const fn interval(lower: f64, upper: f64) -> ReferenceInterval {
    ReferenceInterval { lower, upper }
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-14, "{actual} != {expected}");
}

#[test]
fn action_diagnostics_count_certified_orderings_and_point_predictions() -> Result<()> {
    let predictions = [(1, 0.6), (2, 0.5), (3, 0.9), (4, 0.5)];
    let reference = [
        (1, interval(0.2, 0.2)),
        (2, interval(0.8, 0.8)),
        (3, interval(0.1, 0.9)),
        (4, interval(0.1, 0.2)),
    ];
    let reported = compare_action_values(&predictions, &reference, &1)?;
    assert_eq!(reported.errors.count, 2);
    close(reported.errors.rmse().unwrap(), 0.3535533905932738);
    assert_eq!(reported.ranking.possible_pairs, 6);
    assert_eq!(reported.ranking.ordered_pairs, 2);
    assert_eq!(reported.ranking.reversed_pairs, 1);
    assert_eq!(reported.ranking.tied_predictions, 1);
    close(reported.chosen_regret.lower, 0.6);
    close(reported.chosen_regret.upper, 0.7);
    Ok(())
}

#[test]
fn action_diagnostics_reject_mismatched_support_and_report_zero_when_chosen_dominates() -> Result<()>
{
    let values = [(0, 0.7), (1, 0.4)];
    let reference = [(0, interval(0.6, 0.9)), (1, interval(0.0, 0.6))];
    assert_eq!(
        compare_action_values(&values, &reference, &0)?
            .chosen_regret
            .point(),
        Some(0.0)
    );
    assert!(compare_action_values(&values, &reference, &2).is_err());
    assert!(compare_action_values(&values, &[reference[1], reference[0]], &0).is_err());
    assert!(compare_action_values(&[(0, f64::NAN), values[1]], &reference, &0).is_err());
    Ok(())
}

#[test]
fn complete_policy_return_keeps_its_source_and_signed_value_difference() -> Result<()> {
    let source = PolicyReturnSource::ModelRelative { model: [7; 32] };
    let compared = PolicyValueComparison::new(0.75, 0.5, source)?;
    assert_eq!(compared.fitted_value, 0.75);
    assert_eq!(compared.deployed_return, 0.5);
    assert_eq!(compared.fitted_minus_deployed, 0.25);
    assert_eq!(compared.source, source);
    assert!(PolicyValueComparison::new(0.0, f64::NAN, PolicyReturnSource::FactualEpisode).is_err());
    Ok(())
}

#[test]
fn fitted_policy_changes_count_only_different_choices_at_matched_histories() {
    let reported = policy_changes([(0, 0), (1, 2), (2, 2)]);
    assert_eq!(reported.decisions, 3);
    assert_eq!(reported.changed, 1);
}

#[test]
fn first_divergence_requires_a_distinct_action_and_certified_advantage() -> Result<()> {
    let row = |decision, chosen, chosen_value, reference_value| ComparedDecision {
        decision,
        physical_tick: 2 * decision,
        chosen,
        reference: 1,
        chosen_value,
        reference_value,
    };
    let observed = [
        row(0, 1, interval(0.5, 0.5), interval(0.5, 0.5)),
        row(1, 0, interval(0.25, 0.5), interval(0.375, 0.75)),
        row(2, 0, interval(0.25, 0.25), interval(0.5, 0.6)),
        row(4, 0, interval(0.1, 0.3), interval(0.75, 0.9)),
        row(5, 0, interval(0.0, 0.1), interval(0.95, 1.0)),
    ];
    let reported = first_consequential_divergence(observed, 0.25)?.unwrap();
    assert_eq!(reported.decision, 4);
    assert_eq!(reported.physical_tick, 8);
    assert!(first_consequential_divergence([observed[2], observed[1]], 0.25).is_err());
    Ok(())
}

struct OneFeature(FeatureSpec);

impl FeatureMap for OneFeature {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn encode(&self, _: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        Ok(vec![vec![1.0]; actions.len()])
    }
}

#[test]
fn factual_coverage_separates_offered_taken_and_mc_eligible_rows() -> Result<()> {
    let mut store = EpisodeStore::new(16, 1, ActionSpace::Discrete { count: 3 })?;
    for (episode_id, selected, legal, terminated) in [
        (0, 0, vec![true, true, false], true),
        (1, 2, vec![true, false, true], false),
    ] {
        let observation = Observation::dense(vec![0.0], Some(legal));
        store.begin_episode(episode_id, &EpisodeStart::fresh(observation.clone()))?;
        let next_observation = Observation::dense(vec![1.0], Some(vec![!terminated; 3]));
        store.push(&TemporalTransition {
            episode_id,
            timestep: 0,
            observation,
            action: Action::Discrete(selected),
            reward: 0.0.into(),
            next_observation,
            terminated,
            truncated: false,
            behavior: BehaviorMetadata {
                revision: episode_id + 5,
                log_prob: None,
            },
        })?;
    }
    let features = OneFeature(FeatureSpec {
        namespace: "coverage".into(),
        version: 1,
        terms: vec![FeatureIdentity::new("intercept", "constant one")],
        configuration: vec![],
        representation: None,
    });
    let batch = FactualBatch::from_store_with(
        &store,
        &features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 1.0 },
            span: DecisionSpan::Single,
        },
        |point| Ok(Some(if point.episode_id == 0 { 1.0 } else { 3.0 })),
    )?;
    let coverage = batch.coverage()?;
    assert_eq!(coverage.rows, 2);
    assert_eq!(coverage.episodes, 2);
    assert_eq!(coverage.feature_dimension, 1);
    assert_eq!(coverage.mc_eligible_rows, 1);
    close(coverage.effective_sample_size.unwrap(), 1.6);
    assert_eq!(coverage.behavior_revisions, vec![5, 6]);
    assert_eq!(coverage.actions.len(), 3);
    assert_eq!(coverage.actions[0].action, Action::Discrete(0));
    assert_eq!(
        (
            coverage.actions[0].legal_rows,
            coverage.actions[0].selected_rows
        ),
        (2, 1)
    );
    assert_eq!(coverage.actions[1].action, Action::Discrete(1));
    assert_eq!(
        (
            coverage.actions[1].legal_rows,
            coverage.actions[1].selected_rows
        ),
        (1, 0)
    );
    assert_eq!(coverage.actions[2].action, Action::Discrete(2));
    assert_eq!(
        (
            coverage.actions[2].legal_rows,
            coverage.actions[2].selected_rows
        ),
        (1, 1)
    );
    Ok(())
}
