use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{data::*, pane::*, resolver::*};
use std::{collections::BTreeMap, sync::Arc};
fn fixture(count: usize) -> (tempfile::TempDir, RunIndex, Selection) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pane");
    let space = ActionSpace::Discrete { count };
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "pane",
            "ordinary",
            "misleading-quantiles",
            1,
            space.clone(),
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    for revision in [1, 7] {
        let mut context = RecordContext::new("pane", revision);
        context.probe_set = Some("fixed".into());
        let probes = ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "stable population",
            vec![Observation::dense(vec![1.], Some(vec![true; count]))],
            None,
            space.clone(),
        )
        .unwrap()
        .record(context.clone())
        .unwrap();
        writer.append(&probes).unwrap();
        writer
            .append(&Chunk {
                context,
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters: PolicyParameters::Categorical {
                        probabilities: "p".into(),
                    },
                    stochastic: true,
                    masks: None,
                    entropy: None,
                    entropy_method: None,
                }),
                arrays: BTreeMap::from([
                    ("a".into(), Array::u64(vec![1], vec![0]).unwrap()),
                    (
                        "p".into(),
                        Array::f64(vec![1, count], vec![1. / count as f64; count]).unwrap(),
                    ),
                ]),
            })
            .unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let selected = Selection {
        revision: 1,
        input: SelectedInput::ProbeRow {
            set: "fixed".into(),
            row: 0,
            sequence: 0,
        },
        ..Default::default()
    };
    (dir, run, selected)
}
#[test]
fn intents_do_not_mutate_the_current_pass() {
    let (_dir, run, selected) = fixture(2);
    let mut next = selected.clone();
    SelectionIntent::Input(SelectedInput::TrajectoryStep {
        sequence: 3,
        step: 2,
    })
    .apply(&run, &mut next)
    .unwrap();
    assert_eq!(
        selected.input,
        SelectedInput::ProbeRow {
            set: "fixed".into(),
            row: 0,
            sequence: 0
        }
    );
    assert_eq!(
        next.input,
        SelectedInput::TrajectoryStep {
            sequence: 3,
            step: 2
        }
    );
    SelectionIntent::ModelRole("target".into())
        .apply(&run, &mut next)
        .unwrap();
    assert_eq!(next.model_role, "target");
    assert_eq!(selected.model_role, "online");
}
#[test]
fn full_checked_parameters_and_held_actual_provenance_survive_new_requests() {
    let (_dir, run, mut selected) = fixture(70);
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let first = resolver
        .resolve_policy(&run, &selected, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    let checked = PolicySource::check_frame(
        first.record.as_ref().unwrap(),
        &run.recording().manifest().action_space,
        32 * 1024,
    )
    .unwrap();
    assert_eq!(
        checked.batch.parameter_array("p").unwrap().shape(),
        &[1, 70]
    );
    assert_eq!(checked.batch.actions().as_u64().unwrap(), &[0]);
    assert_eq!(checked.batch.rows(), 1);
    selected.revision = 7;
    let held = resolver
        .resolve_policy(&run, &selected, |i| {
            if i == 3 {
                Ok(None)
            } else {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            }
        })
        .unwrap();
    assert_eq!(held.status, ResolveStatus::Loading);
    assert_eq!(held.requested_revision, 7);
    let checked = PolicySource::check_frame(
        held.record.as_ref().unwrap(),
        &run.recording().manifest().action_space,
        32 * 1024,
    )
    .unwrap();
    assert_eq!(
        checked.frame.chunk.context.identity.learner_update_revision,
        1
    );
    assert_eq!(checked.frame.probe_key.sequence, 0);
    assert_eq!(checked.batch.rows(), 1);
}
#[test]
fn checked_policy_rejects_invalid_probabilities_masks_spaces_and_rows() {
    let (_dir, run, selected) = fixture(2);
    for case in 0..4 {
        let mut resolver = ProbeResolver::new(1024 * 1024);
        let mut view = resolver
            .resolve_policy(&run, &selected, |i| {
                let mut chunk = run.recording().load(i)?;
                if i == 1 && case == 0 {
                    chunk
                        .arrays
                        .insert("p".into(), Array::f64(vec![1, 2], vec![-0.5, 1.5]).unwrap());
                }
                if i == 0
                    && case == 1
                    && let Record::ProbeSet(p) = &chunk.record
                {
                    chunk.arrays.insert(
                        p.action_masks.clone().unwrap(),
                        Array::u8(vec![1, 2], vec![0, 1]).unwrap(),
                    );
                }
                Ok(Some(Arc::new(chunk)))
            })
            .unwrap();
        if case == 3 {
            view.record.as_mut().unwrap().row = 5;
        }
        let wrong = ActionSpace::Discrete { count: 3 };
        assert!(
            PolicySource::check_frame(
                view.record.as_ref().unwrap(),
                if case == 2 {
                    &wrong
                } else {
                    &run.recording().manifest().action_space
                },
                32 * 1024,
            )
            .is_err(),
            "case {case}"
        );
    }
}
#[test]
fn typed_rejection_reports_mismatch_unreadable_and_budget_without_parsing_warnings() {
    let (_dir, run, selected) = fixture(2);
    for (budget, corrupt, expected) in [
        (1, false, ResolveRejection::BudgetExceeded),
        (1024 * 1024, true, ResolveRejection::Unreadable),
    ] {
        let mut resolver = ProbeResolver::new(budget);
        let view = resolver
            .resolve_policy(&run, &selected, |i| {
                if corrupt {
                    Err(rl::Error::State("arbitrary localized text".into()))
                } else {
                    Ok(Some(Arc::new(run.recording().load(i)?)))
                }
            })
            .unwrap();
        assert_eq!(view.rejection, Some(expected));
    }
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&run, &selected, |i| {
            let mut chunk = run.recording().load(i)?;
            if matches!(chunk.record, Record::Policy(_)) {
                chunk
                    .arrays
                    .insert("a".into(), Array::u64(vec![2], vec![0, 0]).unwrap());
            }
            Ok(Some(Arc::new(chunk)))
        })
        .unwrap();
    assert_eq!(view.rejection, Some(ResolveRejection::PopulationMismatch));
}
#[test]
fn checked_gaussian_parameters_remain_full_borrowed_arrays_and_reject_wrong_transform() {
    let space = ActionSpace::Box {
        low: vec![-1., -2.],
        high: vec![1., 2.],
    };
    let mut context = RecordContext::new("gaussian", 1);
    context.probe_set = Some("fixed".into());
    let probes = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "ordinary Gaussian input",
        vec![Observation::new(vec![1.])],
        None,
        space.clone(),
    )
    .unwrap()
    .record(context.clone())
    .unwrap();
    let mut chunk = Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Gaussian {
                mean: "mean".into(),
                log_std: "logstd".into(),
                tanh_bounded: true,
            },
            stochastic: true,
            masks: None,
            entropy: None,
            entropy_method: None,
        }),
        arrays: BTreeMap::from([
            ("a".into(), Array::f64(vec![1, 2], vec![0., 0.]).unwrap()),
            ("mean".into(), Array::f64(vec![1, 2], vec![1., 2.]).unwrap()),
            ("logstd".into(), Array::f64(vec![2], vec![0., -1.]).unwrap()),
        ]),
    };
    let batch =
        rl_viewer::policy_change::admission::checked_batch(&chunk, &probes, &space, 32768).unwrap();
    assert_eq!(
        batch.parameter_array("mean").unwrap().as_f64().unwrap(),
        &[1., 2.]
    );
    assert_eq!(
        batch.parameter_array("logstd").unwrap().as_f64().unwrap(),
        &[0., -1.]
    );
    assert!(std::ptr::eq(
        batch.parameter_array("mean").unwrap(),
        &chunk.arrays["mean"]
    ));
    if let Record::Policy(p) = &mut chunk.record {
        p.parameters = PolicyParameters::Gaussian {
            mean: "mean".into(),
            log_std: "logstd".into(),
            tanh_bounded: false,
        };
    }
    assert!(
        rl_viewer::policy_change::admission::checked_batch(&chunk, &probes, &space, 32768).is_err()
    );
}
#[test]
fn independent_reference_retains_run_role_requested_and_actual_revision() {
    let (_dir, run, _selected) = fixture(2);
    let (_other_dir, other, mut reference) = fixture(2);
    reference.revision = 5;
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&other, &reference, |index| {
            Ok(Some(Arc::new(other.recording().load(index)?)))
        })
        .unwrap();
    let checked = PolicySource::check_frame(
        view.record.as_ref().unwrap(),
        &other.recording().manifest().action_space,
        32768,
    )
    .unwrap();
    assert_eq!(view.requested_revision, 5);
    assert_eq!(
        checked.frame.chunk.context.identity.learner_update_revision,
        1
    );
    assert_eq!(
        checked.frame.chunk.context.identity.model_role,
        reference.model_role
    );
    assert_eq!(checked.frame.key.root, *other.root());
    assert_ne!(checked.frame.key.root, *run.root());
    assert_eq!(checked.frame.probe_key.sequence, 0);
}
#[test]
fn revision_intents_validate_run_bounds_before_changing_pending_selection() {
    let (_dir, run, selected) = fixture(2);
    let mut next = selected.clone();
    assert!(next.select_revision(&run, 8).is_err());
    assert_eq!(next.revision, selected.revision);
    assert_eq!(next.input, selected.input);
    next.select_revision(&run, 7).unwrap();
    assert_eq!(next.revision, 7);
    assert_eq!(next.input, selected.input);
    assert_eq!(selected.revision, 1);
}
