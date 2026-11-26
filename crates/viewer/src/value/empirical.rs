//! Immutable conditional return sources for Value. No simulation or UI decoding.
use super::{ensure, source::ValueSource};
use crate::data::{CacheKey, RunIndex};
use rl::diagnostics::{
    analysis::{EmpiricalDistribution, match_conditional_rows},
    empirical as science, value,
};
use rl::{
    Error, Result,
    diagnostics::{schema::*, storage::ChunkSource},
};
use rl::{
    core::{action::Action, env::Observation},
    diagnostics::{analysis::EmpiricalSummary, value::ValueOutput},
};
use std::sync::Arc;

pub const MAX_CONDITIONS: usize = 64;
const MAX_CRITICS: usize = 32;
const MAX_CANDIDATES: usize = 512;
const MAX_SOURCE_METADATA: usize = 512 * 1024;
const MAX_PAYLOAD: usize = 128 * 1024;
const MAX_MATCH_WORK: usize = 1024 * 1024;
const MAX_READS: usize = 256; // <=32 MiB admitted source payloads per preparation.

#[derive(Clone)]
struct Address {
    key: CacheKey,
    probe_key: CacheKey,
    probe_set: Option<String>,
    record: ChunkSource,
    probes: ChunkSource,
}

#[derive(Clone)]
pub struct EmpiricalSource {
    current_key: CacheKey,
    manifest: Arc<RunManifest>,
    empirical: Vec<Address>,
    critics: Vec<Address>,
    pub omitted: usize,
    pub(crate) metadata_bytes: usize,
}

#[derive(Clone)]
pub struct EmpiricalCondition {
    pub input: Observation,
    pub action: Option<Action>,
    pub summary: EmpiricalSummary,
    pub source_key: CacheKey,
    pub probe_key: CacheKey,
    pub probe_set: String,
    pub critic_key: CacheKey,
    pub row: usize,
    pub row_id: u64,
    pub revision: u64,
    pub continuation: String,
    pub stochastic: bool,
    pub methodology: String,
    source_index: usize,
}

#[derive(Clone, Default)]
pub struct EmpiricalBatch {
    pub conditions: Vec<EmpiricalCondition>,
    pub rejections: Vec<String>,
    pub omitted: usize,
}

impl EmpiricalBatch {
    pub fn exact(
        &self,
        input: &Observation,
        action: Option<&Action>,
    ) -> Result<Option<&EmpiricalCondition>> {
        self.selected(input, action, None)
    }
    pub fn selected(
        &self,
        input: &Observation,
        action: Option<&Action>,
        source: Option<&CacheKey>,
    ) -> Result<Option<&EmpiricalCondition>> {
        let mut matches = self.conditions.iter().filter(|c| {
            c.input == *input
                && c.action.as_ref() == action
                && source.is_none_or(|key| c.source_key == *key)
        });
        let first = matches.next();
        ensure(
            matches.next().is_none(),
            "Multiple empirical batches match this input; original condition identity is required",
        )?;
        Ok(first)
    }
}

impl EmpiricalSource {
    pub fn prepare(
        &self,
        current: &ValueSource,
        output: &ValueOutput,
        cancelled: &impl Fn() -> bool,
    ) -> Result<EmpiricalBatch> {
        ensure(
            self.current_key == current.identity().key,
            "Empirical evidence belongs to another actual critic source",
        )?;
        self.available()?;
        let target = current
            .reference_info
            .context()
            .ok_or_else(|| Error::Config("Critic scientific context is unknown".into()))?;
        let declared = current
            .declaration()
            .ok_or_else(|| Error::Config("Critic continuation declaration is unknown".into()))?;
        ensure(
            declared.conventions.as_ref() == Some(&target.conventions),
            "Critic conventions differ from the empirical target context",
        )?;
        let name = current
            .critic_names()
            .get(usize::from(
                *output == ValueOutput::Q2 && current.critic_names().len() == 2,
            ))
            .ok_or_else(|| Error::Config("Empirical critic subject unavailable".into()))?;
        let subject = declared
            .critics
            .get(name)
            .ok_or_else(|| Error::Config("Empirical critic subject is undeclared".into()))?;
        ensure(
            subject.continuation.is_some(),
            "Empirical comparison needs an explicit critic continuation",
        )?;
        let mut batch = EmpiricalBatch {
            omitted: self.omitted,
            ..Default::default()
        };
        let mut work = 0usize;
        let mut reads = 0usize;
        for address in &self.critics {
            if cancelled() {
                return Err(Error::State("Empirical preparation superseded".into()));
            }
            if let super::source::Data::Recorded(frames) = &current.data
                && !frames.iter().any(|f| f.key == address.key)
            {
                continue;
            }
            let candidate = (|| -> Result<(Chunk, Chunk)> {
                let critic = load_bounded(&address.record, &mut reads)?;
                let declaration = value::declaration(&critic)?
                    .ok_or_else(|| Error::Config("Recorded critic conventions unknown".into()))?;
                ensure(
                    declaration.critics.get(name) == Some(subject)
                        && declaration.conventions == declared.conventions,
                    "Recorded critic subject or continuation differs from selected model",
                )?;
                let probes = load_bounded(&address.probes, &mut reads)?;
                let Record::ProbeSet(p) = &probes.record else {
                    return Err(Error::Config("Empirical critic population missing".into()));
                };
                ensure(
                    p.histories.is_empty(),
                    "History-conditioned critic evidence has no admitted observation-only target",
                )?;
                Ok((critic, probes))
            })();
            let (critic, critic_probes) = match candidate {
                Ok(pair) => pair,
                Err(error) => {
                    batch.reject(error);
                    continue;
                }
            };
            for (source_index, empirical_address) in self.empirical.iter().enumerate() {
                if cancelled() {
                    return Err(Error::State("Empirical preparation superseded".into()));
                }
                if address.probe_set != empirical_address.probe_set {
                    continue;
                }
                if batch
                    .conditions
                    .iter()
                    .any(|c| c.source_key == empirical_address.key)
                {
                    continue;
                }
                if reads >= MAX_READS {
                    batch.reject(Error::Config(
                        "Empirical source read window exhausted; remaining conditions unavailable"
                            .into(),
                    ));
                    return Ok(batch);
                }
                let candidate = (|| -> Result<EmpiricalCondition> {
                    let empirical = load_bounded(&empirical_address.record, &mut reads)?;
                    let declaration = science::validate_for(&empirical, &self.manifest)?;
                    science::compatible_target(&declaration, target)?;
                    let probes = load_bounded(&empirical_address.probes, &mut reads)?;
                    let (Record::Empirical(e), Record::ProbeSet(p), Record::ProbeSet(cp)) =
                        (&empirical.record, &probes.record, &critic_probes.record)
                    else {
                        return Err(Error::Config(
                            "Conditional empirical population missing".into(),
                        ));
                    };
                    ensure(
                        p.histories.is_empty(),
                        "Empirical history payload is not an observation-conditioned target",
                    )?;
                    ensure(
                        p.id.len() <= 1024,
                        "Empirical population name exceeds admission",
                    )?;
                    let empirical_row =
                        usize::try_from(e.probe_row.ok_or_else(|| {
                            Error::Config("Empirical condition row missing".into())
                        })?)
                        .map_err(|_| Error::Shape("Empirical row overflow".into()))?;
                    let ids = probes.arrays[&p.row_ids].as_u64()?;
                    let row_id = *ids.get(empirical_row).ok_or_else(|| {
                        Error::Shape("Empirical condition row out of bounds".into())
                    })?;
                    let critic_ids = critic_probes.arrays[&cp.row_ids].as_u64()?;
                    work = work.saturating_add(critic_ids.len());
                    ensure(
                        work <= MAX_MATCH_WORK,
                        "Empirical exact matching exceeds bounded work",
                    )?;
                    let row = critic_ids
                        .iter()
                        .position(|&id| id == row_id)
                        .ok_or_else(|| {
                            Error::Config(
                                "Empirical stable row absent from current population".into(),
                            )
                        })?;
                    let matched = match_conditional_rows(
                        &critic,
                        &critic_probes,
                        row,
                        &empirical,
                        &probes,
                        &self.manifest,
                    )?;
                    let input = crate::policy::prepared::exact_observation(
                        &critic_probes,
                        matched.critic_row,
                    )?;
                    let action = if let Some(column) = matched.discrete_action {
                        Some(Action::Discrete(column))
                    } else if let Some(key) = &e.initial_action {
                        Some(super::visitation::action_row(
                            &empirical,
                            key,
                            0,
                            &self.manifest.action_space,
                        )?)
                    } else {
                        None
                    };
                    let returns = empirical.arrays[&e.returns].to_f64();
                    let distribution = EmpiricalDistribution::new(&returns)?;
                    let summary = distribution.summary()?;
                    ensure(
                        e.evaluation.continuation_policy.len() <= 1024,
                        "Empirical continuation label exceeds admission",
                    )?;
                    let Conditioning::Observation { methodology } = &declaration.conditioning
                    else {
                        unreachable!()
                    };
                    ensure(
                        methodology.len() <= 4096,
                        "Empirical methodology exceeds prepared-data admission",
                    )?;
                    Ok(EmpiricalCondition {
                        input,
                        action,
                        summary,
                        source_key: empirical_address.key.clone(),
                        probe_key: empirical_address.probe_key.clone(),
                        probe_set: p.id.clone(),
                        critic_key: address.key.clone(),
                        row: matched.empirical_row,
                        row_id,
                        revision: declaration.revision,
                        continuation: e.evaluation.continuation_policy.clone(),
                        stochastic: e.evaluation.stochastic_policy,
                        methodology: methodology.clone(),
                        source_index,
                    })
                })();
                match candidate {
                    Ok(condition) => {
                        if !batch
                            .conditions
                            .iter()
                            .any(|c| c.source_key == condition.source_key)
                        {
                            batch.conditions.push(condition);
                        }
                    }
                    Err(error) => batch.reject(error),
                }
                if work > MAX_MATCH_WORK {
                    return Ok(batch);
                }
            }
        }
        Ok(batch)
    }
    pub(crate) fn distribution(
        &self,
        condition: &EmpiricalCondition,
    ) -> Result<EmpiricalDistribution> {
        let address = self
            .empirical
            .get(condition.source_index)
            .ok_or_else(|| Error::State("Empirical condition identity missing".into()))?;
        ensure(
            address.key == condition.source_key && address.probe_key == condition.probe_key,
            "Empirical condition source changed",
        )?;
        let chunk = load(&address.record)?;
        let Record::Empirical(e) = &chunk.record else {
            return Err(Error::Config("Empirical source changed kind".into()));
        };
        let returns = chunk
            .arrays
            .get(&e.returns)
            .ok_or_else(|| Error::Config("Empirical return array is missing".into()))?;
        ensure(
            returns.shape() == [condition.summary.count]
                && (1..=science::MAX_EMPIRICAL_SAMPLES).contains(&condition.summary.count),
            "Empirical sample count differs from the verified condition",
        )?;
        EmpiricalDistribution::new(&returns.to_f64())
    }
    pub(crate) fn capacity(&self) -> usize {
        self.empirical.len()
    }
    pub(crate) fn available(&self) -> Result<()> {
        for address in self.empirical.iter().chain(&self.critics) {
            address.record.check_available()?;
            address.probes.check_available()?;
        }
        Ok(())
    }
    /// Only bounded immutable metadata is consulted here. Payload decoding and
    /// scientific admission happen on the historical Value worker.
    pub fn discover(run: &RunIndex, current: &ValueSource) -> Result<Option<Self>> {
        if run.root() != &current.identity().key.root
            || run.recording().manifest().run_id != current.manifest().run_id
        {
            return Ok(None);
        }
        let mut bytes = serde_json::to_vec(run.recording().manifest())?
            .len()
            .saturating_mul(8)
            .saturating_add(64 * 1024);
        let mut collect = |kind, limit| -> Result<(Vec<Address>, usize)> {
            let mut addresses = Vec::new();
            let revision = current.identity().revision;
            let total = run
                .candidate_frontier(kind, current.model_role(), None, revision)
                .0
                .saturating_sub(revision.checked_sub(1).map_or(0, |before| {
                    run.candidate_frontier(kind, current.model_role(), None, before)
                        .0
                }));
            let mut omitted = total.saturating_sub(MAX_CANDIDATES);
            for index in run
                .candidates(kind, current.model_role(), None, revision)
                .take(MAX_CANDIDATES)
            {
                let entry = &run.recording().entries()[index];
                if entry.revision != current.identity().revision {
                    break;
                }
                if entry.checkpoint_id != current.checkpoint_id {
                    continue;
                }
                let Some(probe) = run.probe_for_record(index) else {
                    continue;
                };
                if addresses.len() == limit {
                    omitted += 1;
                    continue;
                }
                let probe_entry = &run.recording().entries()[probe];
                let charge = serde_json::to_vec(entry)?
                    .len()
                    .saturating_add(serde_json::to_vec(probe_entry)?.len())
                    .saturating_add(run.root().as_os_str().len().saturating_mul(2))
                    .saturating_mul(8)
                    .saturating_add(2048);
                bytes = bytes.saturating_add(charge);
                ensure(
                    bytes <= MAX_SOURCE_METADATA,
                    "Empirical source metadata exceeds bounded admission",
                )?;
                addresses.push(Address {
                    key: run.key(index)?,
                    probe_key: run.key(probe)?,
                    probe_set: entry.probe_set.clone(),
                    record: run.recording().source(index)?,
                    probes: run.recording().source(probe)?,
                });
            }
            Ok((addresses, omitted))
        };
        let (empirical, empirical_omitted) = collect(RecordKind::Empirical, MAX_CONDITIONS)?;
        if empirical.is_empty() {
            return Ok(None);
        }
        let (critics, critic_omitted) = collect(RecordKind::Critic, MAX_CRITICS)?;
        if critics.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            current_key: current.identity().key.clone(),
            manifest: Arc::new(run.recording().manifest().clone()),
            empirical,
            critics,
            omitted: empirical_omitted.saturating_add(critic_omitted),
            metadata_bytes: bytes,
        }))
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        self.current_key == other.current_key
            && self
                .empirical
                .iter()
                .map(|a| (&a.key, &a.probe_key))
                .eq(other.empirical.iter().map(|a| (&a.key, &a.probe_key)))
            && self
                .critics
                .iter()
                .map(|a| (&a.key, &a.probe_key))
                .eq(other.critics.iter().map(|a| (&a.key, &a.probe_key)))
            && self.omitted == other.omitted
    }
}

fn load(source: &ChunkSource) -> Result<Chunk> {
    source.load_if_within_charge(MAX_PAYLOAD)?.ok_or_else(|| {
        Error::Config("Empirical source exceeds bounded 128 KiB payload admission".into())
    })
}
fn load_bounded(source: &ChunkSource, reads: &mut usize) -> Result<Chunk> {
    ensure(
        *reads < MAX_READS,
        "Empirical preparation exceeds the 32 MiB source window",
    )?;
    *reads += 1;
    load(source)
}

impl EmpiricalBatch {
    fn reject(&mut self, error: Error) {
        if self.rejections.len() < 8 {
            self.rejections.push(error_text(error));
        }
    }
}
pub(crate) fn error_text(error: impl std::fmt::Display) -> String {
    let mut text = String::with_capacity(512);
    for c in error.to_string().chars() {
        if text.len() + c.len_utf8() > 512 {
            break;
        }
        text.push(c);
    }
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{
        data::RunIndex,
        resolver::{ProbeResolver, ResolvedRecord},
        value::prepared::ValueSource,
    };
    use rl::diagnostics::{
        empirical::{self, EMPIRICAL_DECLARATION_VERSION, EmpiricalDeclaration},
        reference::manifest_context,
        storage::*,
        value::*,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;

    pub(crate) fn fixture() -> (tempfile::TempDir, RunIndex, ValueSource) {
        fixture_samples(3)
    }
    pub(crate) fn fixture_samples(samples: usize) -> (tempfile::TempDir, RunIndex, ValueSource) {
        let (_source_dir, source) = crate::reference::tests::decision_tests::fixture();
        let manifest = source.recording().manifest().clone();
        let context = manifest_context(&manifest).unwrap().unwrap();
        let probes = source.recording().load(0).unwrap();
        let mut critic = source.recording().load(2).unwrap();
        let mut value_decl = rl::diagnostics::value::declaration(&critic)
            .unwrap()
            .unwrap();
        let Record::Critic(record) = &mut critic.record else {
            unreachable!()
        };
        record.representation = CriticRepresentation::Scalar {
            values: "scalar".into(),
        };
        record.action_condition = ActionCondition::StateValue;
        record.actions = None;
        record.continuation_policy = Some("conditional-policy".into());
        record.continuation_stochastic = Some(true);
        critic.arrays.insert(
            "scalar".into(),
            Array::f64(vec![2], vec![-1., -3.]).unwrap(),
        );
        let declared = value_decl.critics.get_mut("q").unwrap();
        declared.kind = ValueCriticKind::StateValue;
        declared.actor = None;
        declared.continuation = Some(ContinuationIdentity {
            identity: "conditional-policy".into(),
            stochastic: true,
        });
        critic.arrays.remove(VALUE_METADATA_KEY);
        rl::diagnostics::value::attach_declaration(&mut critic, &value_decl).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut writer =
            Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
        writer.append(&probes).unwrap();
        writer.append(&critic).unwrap();
        for row in 0..2 {
            let returns = (0..samples)
                .map(|i| -4. - 2. * row as f64 + 4. * i as f64 / (samples - 1) as f64)
                .collect::<Vec<_>>();
            let conditioning = Conditioning::Observation {
                methodology: "exact fully observed fixture; independently seeded futures".into(),
            };
            let mut chunk = Chunk {
                context: critic.context.clone(),
                record: Record::Empirical(EmpiricalRecord {
                    returns: "returns".into(),
                    seeds: "seeds".into(),
                    evaluation: EvaluationMetadata {
                        conditioning: conditioning.clone(),
                        initial_action: ActionCondition::CurrentPolicy,
                        continuation_policy: "conditional-policy".into(),
                        stochastic_policy: true,
                        discount: 0.5,
                        independent_future_streams: true,
                        seed_methodology: "independent future streams 11,12,13".into(),
                        truncation_handling: "complete true-terminal suffix; zero continuation"
                            .into(),
                        complete_returns: true,
                    },
                    probe_row: Some(row),
                    initial_action: None,
                    conditioning_observation: Some("observation".into()),
                    conditioning_mask: Some("mask".into()),
                    simulator_state: None,
                }),
                arrays: BTreeMap::from([
                    (
                        "returns".into(),
                        Array::f64(vec![samples], returns).unwrap(),
                    ),
                    (
                        "seeds".into(),
                        Array::u64(vec![samples], (11..11 + samples as u64).collect()).unwrap(),
                    ),
                    (
                        "observation".into(),
                        Array::f32(vec![1, 5], vec![row as f32, 0., 0., 0., 0.375]).unwrap(),
                    ),
                    ("mask".into(), Array::u8(vec![1, 2], vec![1, 1]).unwrap()),
                ]),
            };
            let declaration = EmpiricalDeclaration {
                version: EMPIRICAL_DECLARATION_VERSION,
                run_id: chunk.context.identity.run_id.clone(),
                revision: 0,
                model_role: "online".into(),
                checkpoint_id: chunk.context.identity.checkpoint_id.clone(),
                context: context.clone(),
                conditioning,
            };
            empirical::attach_declaration(&mut chunk, &declaration).unwrap();
            writer.append(&chunk).unwrap();
        }
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut resolver = ProbeResolver::new(2 * 1024 * 1024);
        let key = run.key(1).unwrap();
        let probe_key = run.key(0).unwrap();
        let frame = ResolvedRecord {
            chunk: resolver
                .lease_record(&run, &key, Arc::new(run.recording().load(1).unwrap()))
                .unwrap(),
            probes: resolver
                .lease_record(&run, &probe_key, Arc::new(run.recording().load(0).unwrap()))
                .unwrap(),
            key,
            probe_key,
            row: 0,
        };
        let source = ValueSource::recorded(&run, &frame, "q").unwrap();
        (dir, run, source)
    }

    #[test]
    fn empirical_value_discovers_exact_revision_sources() {
        let (_dir, run, current) = fixture();
        assert!(
            EmpiricalSource::discover(&run, &current).unwrap().is_some(),
            "matched observation-conditional evidence was not discovered"
        );
    }

    #[test]
    fn conditional_distribution_rechecks_the_prepared_sample_count() {
        let (_dir, run, current) = fixture_samples(512);
        let source = EmpiricalSource::discover(&run, &current).unwrap().unwrap();
        let batch = source
            .prepare(&current, &ValueOutput::ExpectedReturn, &|| false)
            .unwrap();
        let mut condition = batch.conditions[0].clone();
        assert_eq!(
            source
                .distribution(&condition)
                .unwrap()
                .sorted_values()
                .len(),
            512
        );
        condition.summary.count = 511;
        assert!(
            source.distribution(&condition).is_err(),
            "a stale count cannot authorize a larger CDF allocation"
        );
    }
    #[test]
    fn empirical_value_errors_are_byte_bounded_for_multibyte_diagnostics() {
        let reason = error_text("測".repeat(1000));
        assert!(reason.len() <= 512 && reason.capacity() == 512);
        assert!(reason.chars().all(|c| c == '測'));
    }

    #[test]
    fn empirical_value_prepares_exact_means_and_keeps_unsupported_inputs_empty() {
        let (_dir, run, current) = fixture();
        let source = EmpiricalSource::discover(&run, &current).unwrap().unwrap();
        let batch = source
            .prepare(&current, &ValueOutput::ExpectedReturn, &|| false)
            .unwrap();
        assert_eq!(batch.conditions.len(), 2);
        let observation =
            rl::core::env::Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
        let first = batch.exact(&observation, None).unwrap().unwrap();
        assert_eq!(first.summary.count, 3);
        assert_eq!(first.summary.mean, -2.);
        assert!((first.summary.mean_standard_error.unwrap() - (4.0_f64 / 3.).sqrt()).abs() < 1e-12);
        let mut unsupported = observation.clone();
        unsupported.as_dense_mut().unwrap().values[0] = 0.5;
        assert!(batch.exact(&unsupported, None).unwrap().is_none());
    }

    #[test]
    fn empirical_value_worker_publishes_sparse_pairs_and_atomic_selected_cdf() {
        use crate::{
            interactive::Interactive,
            policy::slice::SliceMode,
            value::{prepared::*, slice::*},
        };
        let (_dir, run, current) = fixture();
        let evidence = Arc::new(EmpiricalSource::discover(&run, &current).unwrap().unwrap());
        let input =
            rl::core::env::Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
        let mut slice = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
        slice.mode = SliceMode::OneDimensional;
        slice.x = ValueCoordinate::Observation(0);
        slice.y = None;
        slice.resolution = [3, 1];
        slice.state.ranges[0] = [0., 1.];
        let mut request = ValueRequest::new(
            current,
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::State(input),
        )
        .unwrap();
        request.empirical = Some(evidence);
        request.reference_empirical = true;
        let mut worker = Interactive::new(|| {}).unwrap();
        worker.submit_value(request.clone()).unwrap();
        let end = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while worker.value_pending() {
            worker.poll_value().unwrap();
            assert!(std::time::Instant::now() < end);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(worker.value_error().is_none(), "{:?}", worker.value_error());
        let frame = worker.value_frame().unwrap();
        let mut incompatible = request.clone();
        incompatible.output = ValueOutput::StandardDeviation;
        assert!(crate::value::prepared::point_charge_for_grid(&incompatible, &frame.grid).is_err());
        assert!(
            frame.grid.reference_error.is_none(),
            "{:?}",
            frame.grid.reference_error
        );
        let reference = frame.grid.reference_series.as_ref().unwrap();
        let mut pairs = reference
            .positions
            .iter()
            .zip(&reference.values)
            .map(|(p, &v)| (p[0], v))
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(pairs, vec![(0., -2.), (1., -4.)]);
        assert_eq!(
            frame.grid.empirical_differences.as_ref().unwrap().values,
            vec![1., 1.]
        );
        let selected = frame.selected.as_ref().unwrap();
        assert_eq!(selected.reference_value, Some(-2.));
        let evidence = selected.empirical.as_ref().unwrap();
        assert_eq!(evidence.condition.summary.count, 3);
        assert_eq!(evidence.cdf.last(), Some(&[0., 1.]));
        assert!(
            evidence
                .cdf
                .windows(2)
                .all(|w| w[0][0] <= w[1][0] && w[0][1] <= w[1][1])
        );
        let fresh_source = Arc::new(
            EmpiricalSource::discover(&run, &request.current)
                .unwrap()
                .unwrap(),
        );
        let mut repin = request.clone();
        repin.empirical = Some(fresh_source.clone());
        let mut other_input = frame.selected.as_ref().unwrap().input.clone();
        other_input.as_dense_mut().unwrap().values[0] = 1.;
        repin.pin = ValuePin::State(other_input);
        worker.submit_value(repin).unwrap();
        while worker.value_pending() {
            worker.poll_value().unwrap();
            assert!(std::time::Instant::now() < end);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(
            Arc::strong_count(&fresh_source),
            1,
            "Equivalent descriptors must reuse the already charged actual source Arc"
        );
        assert_eq!(
            worker
                .value_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .reference_value,
            Some(-4.)
        );
        request.condition = ComparisonCondition::ControllerValues;
        worker.submit_value(request).unwrap();
        while worker.value_pending() {
            worker.poll_value().unwrap();
            assert!(std::time::Instant::now() < end);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let point = worker.value_frame().unwrap().selected.as_ref().unwrap();
        assert!(point.reference_value.is_none());
        assert!(
            point.empirical.is_some(),
            "Descriptive same-input evidence remains available"
        );
    }

    #[test]
    fn empirical_value_rejects_unrelated_recording_with_identical_claimed_run_id() {
        let (_dir, _run, current) = fixture();
        let (_other_dir, other, _) = fixture();
        assert!(
            EmpiricalSource::discover(&other, &current)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn empirical_value_rejects_incompatible_continuation_without_zero_filling() {
        let (_dir, run, mut current) = fixture();
        let source = EmpiricalSource::discover(&run, &current).unwrap().unwrap();
        Arc::make_mut(current.declaration.as_mut().unwrap())
            .critics
            .get_mut("q")
            .unwrap()
            .continuation
            .as_mut()
            .unwrap()
            .identity = "another-policy".into();
        let batch = source
            .prepare(&current, &ValueOutput::ExpectedReturn, &|| false)
            .unwrap();
        assert!(batch.conditions.is_empty());
        assert!(batch.rejections.iter().any(|r| r.contains("continuation")));
    }

    #[test]
    fn empirical_value_requires_complete_action_and_unambiguous_original_batch() {
        let (_dir, run, current) = fixture();
        let source = EmpiricalSource::discover(&run, &current).unwrap().unwrap();
        let mut batch = source
            .prepare(&current, &ValueOutput::ExpectedReturn, &|| false)
            .unwrap();
        let condition = batch.conditions[0].clone();
        assert!(
            batch
                .exact(&condition.input, Some(&Action::Discrete(0)))
                .unwrap()
                .is_none()
        );
        let mut other = condition.clone();
        other.source_key.sequence += 100;
        batch.conditions.push(other);
        assert!(batch.exact(&condition.input, None).is_err());
        assert_eq!(
            batch
                .selected(&condition.input, None, Some(&condition.source_key))
                .unwrap()
                .unwrap()
                .source_key,
            condition.source_key
        );
    }
}
