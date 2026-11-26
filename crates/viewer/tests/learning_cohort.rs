//! A history score must hold its scientific subject and measured cohort fixed.
#[path = "support/learning_critic.rs"]
mod fixture;
use rl::diagnostics::{empirical, schema::*, storage::*, value};
use rl_viewer::{data::RunIndex, learning::LearningWorker};
use std::{sync::mpsc, time::Duration};

fn retime_critic(c: &mut Chunk, k: u64, name: Option<&str>, continuation: Option<&str>) {
    let mut d = value::declaration(c).unwrap().unwrap();
    c.arrays.remove(value::VALUE_METADATA_KEY);
    c.context.identity.learner_update_revision = k;
    c.context.counters.updates = k;
    d.revision = k;
    if let Some(name) = name {
        let (_, subject) = d.critics.pop_first().unwrap();
        d.critics.insert(name.into(), subject);
    }
    if let Some(id) = continuation {
        let Record::Critic(r) = &mut c.record else {
            unreachable!()
        };
        r.continuation_policy = Some(id.into());
        d.critics
            .values_mut()
            .next()
            .unwrap()
            .continuation
            .as_mut()
            .unwrap()
            .identity = id.into();
    }
    value::attach_declaration(c, &d).unwrap();
}
fn retime_sample(
    c: &mut Chunk,
    k: u64,
    row: usize,
    observation: f32,
    returns: [f64; 3],
    continuation: Option<&str>,
) {
    let mut d = empirical::declaration(c).unwrap().unwrap();
    c.arrays.remove(empirical::EMPIRICAL_METADATA_KEY);
    c.context.identity.learner_update_revision = k;
    c.context.counters.updates = k;
    d.revision = k;
    let Record::Empirical(e) = &mut c.record else {
        unreachable!()
    };
    e.probe_row = Some(row as u64);
    if let Some(id) = continuation {
        e.evaluation.continuation_policy = id.into();
    }
    c.arrays.insert(
        "observation".into(),
        Array::f32(vec![1, 1], vec![observation]).unwrap(),
    );
    c.arrays.insert(
        "returns".into(),
        Array::f64(vec![3], returns.to_vec()).unwrap(),
    );
    c.arrays
        .insert("seeds".into(), Array::u64(vec![3], vec![0, 1, 2]).unwrap());
    empirical::attach_declaration(c, &d).unwrap();
}
fn score_history(case: usize) -> (tempfile::TempDir, LearningWorker) {
    let (mut critic, mut probes, mut sample, manifest) = fixture::critic_fixture();
    if case == 2 {
        let mut d = value::declaration(&critic).unwrap().unwrap();
        critic.arrays.remove(value::VALUE_METADATA_KEY);
        d.critics.values_mut().next().unwrap().kind = value::ValueCriticKind::ScalarQ;
        d.critics.values_mut().next().unwrap().learned_baseline = None;
        let Record::Critic(c) = &mut critic.record else {
            unreachable!()
        };
        c.action_condition = ActionCondition::Fixed;
        c.actions = Some("actions".into());
        critic.arrays.insert(
            "actions".into(),
            Array::f32(vec![2, 1], vec![0., 0.]).unwrap(),
        );
        value::attach_declaration(&mut critic, &d).unwrap();
        let d = empirical::declaration(&sample).unwrap().unwrap();
        sample.arrays.remove(empirical::EMPIRICAL_METADATA_KEY);
        let Record::Empirical(e) = &mut sample.record else {
            unreachable!()
        };
        e.initial_action = Some("first".into());
        e.evaluation.initial_action = ActionCondition::Fixed;
        sample
            .arrays
            .insert("first".into(), Array::f32(vec![1, 1], vec![0.]).unwrap());
        empirical::attach_declaration(&mut sample, &d).unwrap();
    }
    if case == 4 {
        retime_critic(&mut critic, 3, None, Some("declared-current-revision-k3"));
        retime_sample(
            &mut sample,
            3,
            0,
            0.,
            [1., 3., 5.],
            Some("declared-current-revision-k3"),
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    writer.append(&probes).unwrap();
    writer.append(&critic).unwrap();
    writer.append(&sample).unwrap();
    if case == 5 {
        let mut other = sample.clone();
        retime_sample(&mut other, 3, 1, 1., [4.; 3], None);
        writer.append(&other).unwrap();
    }
    if case == 3 {
        let Record::ProbeSet(p) = &probes.record else {
            unreachable!()
        };
        probes.arrays.insert(
            p.observations.clone(),
            Array::f32(vec![2, 1], vec![0.25, 1.]).unwrap(),
        );
        probes.context.identity.learner_update_revision = 7;
        probes.context.counters.updates = 7;
        writer.append(&probes).unwrap();
    }
    retime_critic(
        &mut critic,
        7,
        (case == 1).then_some("other_critic"),
        (case == 4).then_some("declared-current-revision-k7"),
    );
    retime_sample(
        &mut sample,
        7,
        usize::from(case == 0),
        if case == 0 {
            1.
        } else if case == 3 {
            0.25
        } else {
            0.
        },
        if case == 0 {
            [4., 4., 4.]
        } else {
            [1., 3., 5.]
        },
        (case == 4).then_some("declared-current-revision-k7"),
    );
    if case == 2 {
        critic.arrays.insert(
            "actions".into(),
            Array::f32(vec![2, 1], vec![0.25, 0.]).unwrap(),
        );
        sample
            .arrays
            .insert("first".into(), Array::f32(vec![1, 1], vec![0.25]).unwrap());
    }
    if case == 5 {
        sample
            .arrays
            .insert("returns".into(), Array::f64(vec![5], vec![3.; 5]).unwrap());
        sample.arrays.insert(
            "seeds".into(),
            Array::u64(vec![5], (0..5).collect()).unwrap(),
        );
    }
    if case == 6 {
        let mut d = empirical::declaration(&sample).unwrap().unwrap();
        sample.arrays.remove(empirical::EMPIRICAL_METADATA_KEY);
        d.conditioning = Conditioning::Observation {
            methodology: "different declared conditional sampler".into(),
        };
        let Record::Empirical(e) = &mut sample.record else {
            unreachable!()
        };
        e.evaluation.conditioning = d.conditioning.clone();
        empirical::attach_declaration(&mut sample, &d).unwrap();
    }
    writer.append(&critic).unwrap();
    writer.append(&sample).unwrap();
    if case == 5 {
        let mut other = sample.clone();
        retime_sample(&mut other, 7, 1, 1., [4.; 3], None);
        other
            .arrays
            .insert("returns".into(), Array::f64(vec![2], vec![4.; 2]).unwrap());
        other
            .arrays
            .insert("seeds".into(), Array::u64(vec![2], vec![98, 99]).unwrap());
        writer.append(&other).unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut worker = LearningWorker::new(move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(&run, "online").unwrap();
    while worker.pending() {
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        worker.poll();
    }
    (directory, worker)
}

#[test]
fn identical_forecasts_cannot_improve_by_replacing_a_hard_condition_with_an_easy_one() {
    let (_d, worker) = score_history(0);
    let f = worker.frame().unwrap();
    assert_eq!(
        f.critic_errors
            .iter()
            .find(|p| p.revision == 3)
            .unwrap()
            .value,
        Some(1.)
    );
    let changed = f.critic_errors.iter().find(|p| p.revision == 7).unwrap();
    assert_eq!(changed.value, None);
    assert!(changed.reason.as_ref().unwrap().contains("cohort"));
}
#[test]
fn named_critic_subject_complete_actions_and_observations_stay_frozen() {
    for case in [1, 2, 3, 6] {
        let (_d, worker) = score_history(case);
        let f = worker.frame().unwrap();
        assert_eq!(
            f.critic_errors.iter().filter(|p| p.value.is_some()).count(),
            1,
            "case {case}"
        );
        assert_eq!(
            f.critic_errors
                .iter()
                .find(|p| p.revision == 7)
                .unwrap()
                .value,
            None,
            "case {case}"
        );
    }
}

#[test]
fn future_sample_counts_do_not_change_frozen_equal_condition_weights() {
    let (_d, worker) = score_history(5);
    let frame = worker.frame().unwrap();
    let anchor = frame.critic_anchor.as_ref().unwrap();
    assert_eq!(anchor.row_ids, vec![0, 1]);
    let points: Vec<_> = frame
        .critic_errors
        .iter()
        .filter(|p| p.value.is_some())
        .collect();
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].condition_counts, vec![3, 3]);
    assert_eq!(points[1].condition_counts, vec![5, 2]);
    assert!(points.iter().all(|p| p.value == Some(0.5)));
}

#[test]
fn unavailable_critic_anchor_stays_pinned_after_hidden_frame_and_retry() {
    let (_d, mut worker) = score_history(4);
    let frame = worker.frame().unwrap();
    let anchor = frame.critic_anchor.as_ref().unwrap().clone();
    let run = RunIndex::open(frame.root.as_ref(), ReadLimits::default()).unwrap();
    let entry = run
        .recording()
        .entries()
        .iter()
        .find(|e| e.sequence == anchor.sequence)
        .unwrap();
    std::fs::remove_file(run.root().join(&entry.file)).unwrap();
    let settle = |worker: &mut LearningWorker| {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while worker.pending() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
            worker.poll();
        }
    };
    assert!(worker.recheck());
    settle(&mut worker);
    assert!(worker.frame().is_none());
    worker.retry();
    worker.request(&run, "online").unwrap();
    settle(&mut worker);
    let frame = worker.frame().unwrap();
    assert_eq!(
        frame.critic_anchor.as_ref().unwrap().sequence,
        anchor.sequence
    );
    assert_eq!(
        frame.critic_anchor.as_ref().unwrap().row_ids,
        anchor.row_ids
    );
    assert!(frame.critic_errors.iter().all(|p| p.value.is_none()));
}
#[test]
fn each_point_names_the_critic_and_its_explicit_current_revision_continuation() {
    let (_d, worker) = score_history(4);
    let f = worker.frame().unwrap();
    assert_eq!(
        f.critic_errors.iter().filter(|p| p.value.is_some()).count(),
        2
    );
    for p in &f.critic_errors {
        assert!(p.label.contains("critic"));
        assert!(
            p.label
                .contains(&format!("declared-current-revision-k{}", p.revision))
        );
    }
}
