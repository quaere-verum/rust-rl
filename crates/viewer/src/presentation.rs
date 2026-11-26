//! Initial population selection and bounded asynchronous workspace defaults.

use crate::data::{RunIndex, SelectedInput, Selection};
use rl::{
    Result,
    diagnostics::schema::{ProbePopulation, Record, RecordKind},
};

/// Render an existing recording identifier as a readable label. The original
/// identifier remains the selection key and is available in the UI tooltip.
pub(crate) fn display_identifier(value: &str) -> String {
    value
        .split(['-', '_', ':'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            if matches!(
                part.to_ascii_lowercase().as_str(),
                "rl" | "dp"
                    | "ppo"
                    | "sac"
                    | "td3"
                    | "dqn"
                    | "ddqn"
                    | "ddpg"
                    | "d4pg"
                    | "md4pg"
                    | "cvar"
            ) {
                part.to_ascii_uppercase()
            } else {
                let mut chars = part.chars();
                chars.next().map_or_else(String::new, |first| {
                    first.to_uppercase().chain(chars).collect()
                })
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn display_run_identifier(run_id: &str) -> String {
    let label = match run_id {
        "rocket-ppo-release-candidate" => "Rocket Landing · PPO",
        "alpha-d4pg-release-candidate" => "Alpha Execution · Expected Return",
        "alpha-cvar-release-candidate" => "Alpha Execution · CVaR",
        _ => return display_identifier(run_id),
    };
    label.into()
}

/// Choose an immutable population and role using only retained index metadata.
/// Policy time owns the initial endpoint when available; unrelated later records
/// and future activations must not make a sparse population appear fully linked.
pub(crate) fn initial_selection(run: &RunIndex) -> Selection {
    let entries = run.recording().entries();
    if let Some(i) = run
        .candidates(RecordKind::FinitePredictions, "online", None, u64::MAX)
        .find(|&i| {
            run.recording().is_retained(i)
                && run
                    .probe_for_record(i)
                    .is_some_and(|probe| run.recording().is_retained(probe))
        })
        && let Some(probe) = run.probe_for_record(i)
        && let Some(set) = entries[probe].probe_set.clone()
        && run
            .candidates(RecordKind::Policy, "online", None, u64::MAX)
            .find(|&index| run.recording().is_retained(index))
            .is_none_or(|index| entries[index].revision <= entries[i].revision)
    {
        return Selection {
            revision: entries[i].revision,
            input: SelectedInput::ProbeRow {
                set,
                row: 0,
                sequence: entries[probe].sequence,
            },
            ..Default::default()
        };
    }
    let mut populations = std::collections::BTreeMap::new();
    let mut latest_learner = None::<u64>;
    for (index, entry) in entries.iter().enumerate() {
        if crate::reference::provider_role(&entry.model_role) {
            continue;
        }
        latest_learner = Some(latest_learner.map_or(entry.revision, |k| k.max(entry.revision)));
        let slot = match entry.kind {
            RecordKind::Policy => 0,
            RecordKind::Critic => 1,
            RecordKind::Activations => 2,
            _ => continue,
        };
        let Some(source) = run.probe_for_record(index) else {
            continue;
        };
        if !run.recording().is_retained(index) || !run.recording().is_retained(source) {
            continue;
        }
        let revisions = populations
            .entry((source, entry.model_role.as_str()))
            .or_insert([None::<u64>; 3]);
        revisions[slot] =
            Some(revisions[slot].map_or(entry.revision, |old| old.max(entry.revision)));
    }
    let mut selection = Selection {
        revision: latest_learner.unwrap_or(0),
        ..Default::default()
    };
    let mut best = None;
    for ((source, role), revisions) in populations {
        let revision = revisions[0].or(revisions[1]).or(revisions[2]).unwrap();
        let set = entries[source].probe_set.as_deref().unwrap();
        let available = |kind, exact: bool| {
            run.candidates(kind, role, Some(set), revision).any(|i| {
                (!exact || entries[i].revision == revision)
                    && run.recording().is_retained(i)
                    && run.probe_for_record(i) == Some(source)
            })
        };
        // Prefer an actual learning comparison over an equally capable lone
        // checkpoint. Duplicate networks at one revision do not create history;
        // redefinitions and other roles cannot contribute a second endpoint.
        let mut common_revisions = 0;
        let mut previous = None;
        for index in run.candidates(RecordKind::Activations, role, Some(set), revision) {
            let at = entries[index].revision;
            if previous == Some(at)
                || !run.recording().is_retained(index)
                || run.probe_for_record(index) != Some(source)
            {
                continue;
            }
            previous = Some(at);
            if run
                .candidates(RecordKind::Policy, role, Some(set), at)
                .take_while(|&i| entries[i].revision == at)
                .any(|i| run.recording().is_retained(i) && run.probe_for_record(i) == Some(source))
            {
                common_revisions += 1;
                if common_revisions == 2 {
                    break;
                }
            }
        }
        let score = (
            available(RecordKind::Policy, false),
            available(RecordKind::Activations, true),
            available(RecordKind::Critic, false),
            common_revisions == 2,
            role == "online",
            revision,
        );
        if best.is_none_or(|previous| score > previous) {
            best = Some(score);
            selection.revision = revision;
            selection.model_role = role.into();
            selection.input = SelectedInput::ProbeRow {
                set: set.into(),
                row: 0,
                sequence: entries[source].sequence,
            };
        }
    }
    if best.is_none()
        && let Some(set) = run.probe_set_ids().find(|set| {
            run.probe_set(set, selection.revision)
                .is_some_and(|i| run.recording().is_retained(i))
        })
    {
        selection.select_probe(run, set, 0);
    }
    selection
}

/// One-time, incremental discovery for the metadata-selected population/role.
/// Only the first and latest eligible activation checkpoints are decoded, to
/// validate current network/layer names and their compatible reference.
pub(crate) struct WorkspaceDefaults {
    revision: u64,
    candidates: Vec<usize>,
    cursor: usize,
    best: Option<(u8, WorkspacePlan)>,
    empirical: Option<Vec<usize>>,
    empirical_cursor: usize,
    critic_cursor: usize,
    preserve_row: Option<usize>,
    latest_activation: Option<u64>,
    first_policy: Option<u64>,
}
pub(crate) struct WorkspacePlan {
    pub input: SelectedInput,
    pub role: String,
    pub network: String,
    pub layer: String,
    pub reference_revision: u64,
}
impl WorkspaceDefaults {
    pub fn new(run: &RunIndex, selection: &Selection) -> Self {
        let source = match &selection.input {
            SelectedInput::ProbeRow { sequence, .. } => run.by_sequence(*sequence),
            _ => None,
        };
        let compatible = |index: usize| {
            let entry = &run.recording().entries()[index];
            entry.kind == RecordKind::Activations
                && entry.model_role == selection.model_role
                && entry.revision <= selection.revision
                && run.recording().is_retained(index)
                && source.is_some()
                && run.probe_for_record(index) == source
        };
        let entries = run.recording().entries();
        let first = entries
            .iter()
            .enumerate()
            .filter(|(i, _)| compatible(*i))
            .map(|(_, e)| e.revision)
            .min();
        let latest_activation = entries
            .iter()
            .enumerate()
            .filter(|(i, _)| compatible(*i))
            .map(|(_, e)| e.revision)
            .max();
        let mut candidates: Vec<_> = run
            .recording()
            .entries()
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                if !compatible(index) {
                    return None;
                }
                (Some(entry.revision) == first || Some(entry.revision) == latest_activation)
                    .then_some(index)
            })
            .collect();
        candidates.sort_by_key(|&i| std::cmp::Reverse(entries[i].revision));
        let first_of = |kind| {
            entries
                .iter()
                .enumerate()
                .filter(|(i, e)| {
                    e.kind == kind
                        && e.model_role == selection.model_role
                        && e.revision <= selection.revision
                        && run.recording().is_retained(*i)
                        && source.is_some()
                        && run.probe_for_record(*i) == source
                })
                .map(|(_, e)| e.revision)
                .min()
        };
        let first_policy = first_of(RecordKind::Policy);
        // Networks without neural probes still get a local, compatible policy
        // or critic reference. The normal resolver keeps activations unavailable.
        let best = source.map(|_| {
            (
                0,
                WorkspacePlan {
                    input: selection.input.clone(),
                    role: selection.model_role.clone(),
                    network: "policy".into(),
                    layer: String::new(),
                    reference_revision: first_policy
                        .or_else(|| first_of(RecordKind::Critic))
                        .or(first)
                        .unwrap_or(selection.revision),
                },
            )
        });
        Self {
            revision: selection.revision,
            candidates,
            cursor: 0,
            best,
            empirical: None,
            empirical_cursor: 0,
            critic_cursor: 0,
            preserve_row: None,
            latest_activation,
            first_policy,
        }
    }
    /// Endpoint-dependent network candidates cannot survive a seek. Keep an
    /// explicit row restriction while discarding the old cursor and result.
    pub fn update_revision(&mut self, run: &RunIndex, selection: &Selection) {
        if self.revision != selection.revision {
            let preserve_row = self.preserve_row;
            *self = Self::new(run, selection);
            self.preserve_row = preserve_row;
        }
    }
    /// Explicit capture/source choices still discover compatible visual defaults.
    pub fn restrict_input(&mut self, run: &RunIndex, input: &SelectedInput) {
        if let SelectedInput::ProbeRow { sequence, row, .. } = input {
            self.candidates.retain(|&i| {
                run.probe_for_record(i)
                    .is_some_and(|p| run.recording().entries()[p].sequence == *sequence)
            });
            self.preserve_row = Some(*row);
        }
    }
    /// At most four decoded candidates per UI pass; loader reads remain async.
    pub fn poll(&mut self, run: &RunIndex, loader: &mut crate::data::Loader) -> Result<bool> {
        for _ in 0..4 {
            let Some(&index) = self.candidates.get(self.cursor) else {
                return self.prefer_paired_input(run, loader);
            };
            let Some(source_index) = run.probe_for_record(index) else {
                self.cursor += 1;
                continue;
            };
            let Some(source) = loader.request(run, source_index)? else {
                return Ok(false);
            };
            let Record::ProbeSet(probes) = &source.record else {
                self.cursor += 1;
                continue;
            };
            let Some(chunk) = loader.request(run, index)? else {
                return Ok(false);
            };
            let Record::Activations(activation) = &chunk.record else {
                self.cursor += 1;
                continue;
            };
            let layer = activation
                .layers
                .keys()
                .find(|name| name.ends_with("post"))
                .or_else(|| activation.layers.keys().next());
            if let Some(layer) = layer {
                let role = &chunk.context.identity.model_role;
                let revision = chunk.context.identity.learner_update_revision;
                let score = u8::from(Some(revision) == self.latest_activation) * 16
                    + u8::from(probes.population == ProbePopulation::FixedReference) * 8
                    + u8::from(role == "online") * 4
                    + u8::from(
                        activation.network.contains("policy")
                            || activation.network.contains("actor"),
                    ) * 2
                    + u8::from(layer.ends_with("post"));
                if self
                    .best
                    .as_ref()
                    .is_none_or(|(previous, _)| score > *previous)
                {
                    self.best = Some((
                        score,
                        WorkspacePlan {
                            input: SelectedInput::ProbeRow {
                                set: probes.id.clone(),
                                row: 0,
                                sequence: run.recording().entries()[source_index].sequence,
                            },
                            role: role.clone(),
                            network: activation.network.clone(),
                            layer: layer.clone(),
                            reference_revision: revision,
                        },
                    ));
                } else if let Some((_, plan)) = &mut self.best
                    && plan.network == activation.network
                    && plan.layer == *layer
                    && self.first_policy.is_none_or(|first| revision >= first)
                {
                    plan.reference_revision = plan.reference_revision.min(revision);
                }
            }
            self.cursor += 1;
        }
        if self.cursor == self.candidates.len() {
            self.prefer_paired_input(run, loader)
        } else {
            Ok(false)
        }
    }
    fn prefer_paired_input(
        &mut self,
        run: &RunIndex,
        loader: &mut crate::data::Loader,
    ) -> Result<bool> {
        let Some((_, plan)) = &mut self.best else {
            return Ok(true);
        };
        let SelectedInput::ProbeRow { set, sequence, row } = &mut plan.input else {
            return Ok(true);
        };
        if let Some(preserved) = self.preserve_row {
            *row = preserved;
            return Ok(true);
        }
        let Some(source_index) = run.by_sequence(*sequence) else {
            return Ok(true);
        };
        let candidates = self.empirical.get_or_insert_with(|| {
            run.candidates(RecordKind::Empirical, &plan.role, Some(set), u64::MAX)
                .filter(|&i| run.probe_for_record(i) == Some(source_index))
                .collect()
        });
        // Reuse the strict core matcher; a sample label or probe_row alone never
        // qualifies a default anchor. Work and raw retention stay bounded.
        for _ in 0..4 {
            let Some(&index) = candidates.get(self.empirical_cursor) else {
                return Ok(true);
            };
            let revision = run.recording().entries()[index].revision;
            let critic_index = run
                .candidates(RecordKind::Critic, &plan.role, Some(set), revision)
                .take_while(|&i| run.recording().entries()[i].revision == revision)
                .filter(|&i| run.probe_for_record(i) == Some(source_index))
                .nth(self.critic_cursor);
            let Some(critic_index) = critic_index else {
                self.empirical_cursor += 1;
                self.critic_cursor = 0;
                continue;
            };
            let Some(source) = loader.request(run, source_index)? else {
                return Ok(false);
            };
            let Some(sample) = loader.request(run, index)? else {
                return Ok(false);
            };
            let Some(critic) = loader.request(run, critic_index)? else {
                return Ok(false);
            };
            if let Ok(matched) = rl::diagnostics::analysis::match_conditional(
                &critic,
                &source,
                &sample,
                run.recording().manifest(),
            ) {
                *row = matched.probe_row;
                return Ok(true);
            }
            self.critic_cursor += 1;
        }
        Ok(false)
    }
    pub fn finish(self) -> Option<WorkspacePlan> {
        self.best.map(|(_, plan)| plan)
    }
}

#[cfg(test)]
#[path = "presentation_tests.rs"]
mod tests;
