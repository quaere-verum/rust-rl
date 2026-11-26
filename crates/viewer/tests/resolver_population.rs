use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{
    data::{ChunkCache, RunIndex, SelectedInput, Selection},
    resolver::{ProbeResolver, ResolveStatus},
};
use std::{collections::BTreeMap, sync::Arc};

fn fixture() -> (tempfile::TempDir, RunIndex, Arc<Chunk>, Selection) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("population");
    let space = ActionSpace::Discrete { count: 2 };
    let mut writer = Recorder::create(
        &path,
        RunManifest::new("population", "test", "fixture", 1, space.clone(), 0.9).unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let context = |revision, set: &str| {
        let mut context = RecordContext::new("population", revision);
        context.probe_set = Some(set.into());
        context
    };
    let probes = |revision, set: &str, population, other| {
        ProbeSet::new(
            set,
            population,
            set,
            vec![Observation::new(vec![1.]), Observation::new(vec![other])],
            None,
            space.clone(),
        )
        .unwrap()
        .record(context(revision, set))
        .unwrap()
    };
    let activation = |revision, set: &str| Chunk {
        context: context(revision, set),
        record: Record::Activations(ActivationRecord {
            network: "policy".into(),
            layers: BTreeMap::from([("layers.0.post".into(), "h".into())]),
            ..Default::default()
        }),
        arrays: BTreeMap::from([(
            "h".into(),
            Array::f32(vec![2, 1], vec![revision as f32, -(revision as f32)]).unwrap(),
        )]),
    };
    let policy = |revision, set: &str, action| Chunk {
        context: context(revision, set),
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: false,
        }),
        arrays: BTreeMap::from([(
            "a".into(),
            Array::u64(vec![2], vec![action, action]).unwrap(),
        )]),
    };
    let fixed = Arc::new(probes(1, "fixed", ProbePopulation::FixedReference, 2.));
    writer.append(&fixed).unwrap();
    writer.append(&activation(1, "fixed")).unwrap();
    writer.append(&policy(1, "fixed", 0)).unwrap();
    writer
        .append(&probes(2, "visited", ProbePopulation::PolicyVisited, 3.))
        .unwrap();
    writer.append(&activation(2, "visited")).unwrap();
    writer.append(&policy(2, "visited", 1)).unwrap();
    // This unrelated fixed population shares the selected observation only.
    // Its policy cannot colour the first population's whole cloud.
    writer
        .append(&probes(
            3,
            "other-fixed",
            ProbePopulation::FixedReference,
            4.,
        ))
        .unwrap();
    writer.append(&policy(3, "other-fixed", 1)).unwrap();
    let mut trajectory_context = RecordContext::new("population", 4);
    trajectory_context.trajectory = Some("episode".into());
    writer
        .append(&Chunk {
            context: trajectory_context,
            record: Record::Trajectory(TrajectoryRecord {
                render: None,
                id: "episode".into(),
                environment: "test".into(),
                observations: "o".into(),
                actions: "a".into(),
                rewards: "r".into(),
                terminated: "terminated".into(),
                truncated: "truncated".into(),
                simulation_times: None,
                action_masks: None,
                frames: BTreeMap::new(),
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: "fixture".into(),
                    stochastic_policy: false,
                    discount: 0.9,
                    independent_future_streams: true,
                    seed_methodology: "fixed".into(),
                    truncation_handling: "terminal".into(),
                    complete_returns: true,
                },
            }),
            arrays: BTreeMap::from([
                ("o".into(), Array::f32(vec![2, 1], vec![1., 8.]).unwrap()),
                ("a".into(), Array::u64(vec![1], vec![0]).unwrap()),
                ("r".into(), Array::f32(vec![1], vec![0.]).unwrap()),
                ("terminated".into(), Array::u8(vec![1], vec![1]).unwrap()),
                ("truncated".into(), Array::u8(vec![1], vec![0]).unwrap()),
            ]),
        })
        .unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let selection = Selection {
        revision: 4,
        input: SelectedInput::TrajectoryStep {
            sequence: 8,
            step: 0,
        },
        ..Default::default()
    };
    (directory, run, fixed, selection)
}

#[test]
fn trajectory_resolution_never_substitutes_a_nearby_probe() {
    let (_directory, run, _, mut selection) = fixture();
    selection.input = SelectedInput::TrajectoryStep {
        sequence: 8,
        step: 1,
    };
    let result = ProbeResolver::new(1024 * 1024)
        .resolve_policy(&run, &selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert!(result.record.is_none());
}

#[test]
fn policy_candidate_and_source_survive_single_chunk_cache_eviction() {
    let (_directory, run, _, selection) = fixture();
    let budget = 1024 * 1024;
    let mut resolver = ProbeResolver::new(budget);
    let mut cache: Option<(usize, Arc<Chunk>)> = None;
    let mut frame = None;
    for _ in 0..16 {
        let result = resolver
            .resolve_policy(&run, &selection, |index| {
                if let Some((cached_index, chunk)) = &cache
                    && *cached_index == index
                {
                    return Ok(Some(chunk.clone()));
                }
                cache = Some((index, Arc::new(run.recording().load(index)?)));
                Ok(None)
            })
            .unwrap();
        assert!(resolver.resident_bytes() <= budget);
        if result.record.is_some() {
            frame = result.record;
            break;
        }
    }
    assert_eq!(
        frame
            .expect("source and candidate cannot repeatedly evict one another")
            .revision(),
        3
    );
}

#[test]
fn pending_policy_candidate_is_charged_and_released_on_selection_change() {
    let (_directory, run, _, mut selection) = fixture();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let pending = resolver
        .resolve_policy(&run, &selection, |index| {
            if index == 6 {
                Ok(None)
            } else {
                Ok(Some(Arc::new(run.recording().load(index)?)))
            }
        })
        .unwrap();
    assert_eq!(pending.status, ResolveStatus::Loading);
    assert_eq!(
        resolver.resident_bytes(),
        ChunkCache::charge(&run.recording().load(8).unwrap())
            + ChunkCache::charge(&run.recording().load(7).unwrap()),
        "the source trajectory and policy candidate are both pinned"
    );
    selection.input = SelectedInput::None;
    resolver
        .resolve_policy(&run, &selection, |_| panic!("no selected input"))
        .unwrap();
    assert_eq!(resolver.resident_bytes(), 0);
}
