//! Construction of the existing presentation from immutable numerical data.
use super::*;
use crate::policy::cache::{CacheLease, FrameRequest, LoadedSurface};

fn identity(run: &RunIndex, loaded: &LoadedSurface) -> SourceIdentity {
    SourceIdentity {
        key: CacheKey {
            root: run.root().clone(),
            sequence: loaded.entry.source.sequence,
        },
        probe_key: None,
        revision: loaded.entry.source.revision,
        name: loaded.entry.slice.policy.clone(),
        recorded: false,
        explicit_controller: false,
    }
}
fn distribution_bytes(d: &SelectedDistribution) -> usize {
    match d {
        SelectedDistribution::Categorical {
            probabilities,
            valid,
        } => probabilities
            .capacity()
            .saturating_mul(8)
            .saturating_add(valid.capacity()),
        _ => 0,
    }
}
pub(crate) fn cached_frame(
    run: &RunIndex,
    request: &FrameRequest,
    current: Arc<LoadedSurface>,
    reference: Option<Arc<LoadedSurface>>,
    reserve: &mut impl FnMut(usize) -> Result<CacheLease>,
    reuse: Option<&Arc<PreparedGrid>>,
) -> Result<PreparedPolicyFrame> {
    let surface = &current.surface;
    ensure(
        request.action < surface.action_count() && request.row < surface.rows(),
        "prepared action/grid point outside cached surface",
    )?;
    if let Some(other) = &reference {
        ensure(
            surface.spec().same_inputs(other.surface.spec())
                && surface.action_space() == other.surface.action_space(),
            "reference cache has different exact grid inputs or encoding",
        )?;
    }
    let grid = if let Some(grid) = reuse {
        grid.clone()
    } else {
        let mut config = surface
            .spec()
            .to_config(EnvironmentSemantics::new(run.recording().manifest()))?;
        config.action = request.action;
        let bytes = surface
            .rows()
            .saturating_mul(if reference.is_some() { 16 } else { 8 })
            .saturating_add(config.owned_bytes())
            .saturating_add(4096);
        let lease = reserve(bytes)?;
        let mut values = Vec::with_capacity(surface.rows());
        let mut reference_values = Vec::with_capacity(if reference.is_some() {
            surface.rows()
        } else {
            0
        });
        for row in 0..surface.rows() {
            values.push(surface.value(row, request.action)?);
            if let Some(reference) = &reference {
                reference_values.push(reference.surface.value(row, request.action)?);
            }
        }
        ensure(
            values
                .capacity()
                .saturating_add(reference_values.capacity())
                .saturating_mul(8)
                .saturating_add(config.owned_bytes())
                .saturating_add(4096)
                <= bytes,
            "prepared grid capacities exceed reservation",
        )?;
        let (batch, _) = surface.checked_row(0)?;
        let current_indicator = numerics::is_indicator(&batch);
        let reference_indicator = reference
            .as_ref()
            .map(|r| {
                r.surface
                    .checked_row(0)
                    .map(|(b, _)| numerics::is_indicator(&b))
            })
            .transpose()?
            .unwrap_or(false);
        Arc::new(PreparedGrid {
            population_count: 0,
            distinct_positions: 0,
            config: Arc::new(config),
            current: identity(run, &current),
            reference: reference.as_ref().map(|r| identity(run, r)),
            current_indicator,
            reference_indicator,
            values,
            reference_values,
            rows: vec![],
            positions: vec![],
            visits: None,
            source: GridSource::Cached {
                current: current.clone(),
                reference: reference.clone(),
            },
            _lease: FrameLease::Cached(lease),
        })
    };
    let allowance = surface
        .action_count()
        .saturating_mul(36)
        .saturating_add(surface.spec().fixed.len().saturating_mul(4))
        .saturating_add(64 * 1024);
    let point_lease = reserve(allowance)?;
    let (batch, row) = surface.checked_row(request.row)?;
    let distribution = numerics::selected_distribution(&batch, row, request.action)?;
    let chosen = |batch: &PolicyBatch<'_>, row| -> Result<Option<usize>> {
        if matches!(batch.action_space(), ActionSpace::Discrete { .. }) {
            Ok(Some(batch.actions().as_u64()?[row] as usize))
        } else {
            Ok(None)
        }
    };
    let chosen_action = chosen(&batch, row)?;
    let mut reference_chosen_action = None;
    let mut reference_distribution = None;
    let mut kl = None;
    let value = surface.value(request.row, request.action)?;
    let reference_value = if let Some(reference) = &reference {
        let (other, other_row) = reference.surface.checked_row(request.row)?;
        ensure(
            surface.input(request.row)? == reference.surface.input(request.row)?,
            "reference selected input is not the same cached grid point",
        )?;
        reference_chosen_action = chosen(&other, other_row)?;
        reference_distribution = Some(numerics::selected_distribution(
            &other,
            other_row,
            request.action,
        )?);
        if std::mem::discriminant(batch.parameters()) == std::mem::discriminant(other.parameters())
        {
            kl = numerics::current_reference_kl(&batch, row, &other, other_row)?;
        }
        Some(reference.surface.value(request.row, request.action)?)
    } else {
        None
    };
    let (plot, reference_plot) = distribution_plots(
        &distribution,
        reference_distribution.as_ref(),
        request.domain,
    )?;
    let statistics = |distribution: &SelectedDistribution| match distribution {
        SelectedDistribution::Gaussian(g) => g.action_statistics().ok(),
        _ => None,
    };
    let input = surface.input(request.row)?;
    let actual = input
        .as_dense()
        .expect("dense recorded policy input")
        .values
        .capacity()
        .saturating_mul(4)
        .saturating_add(distribution_bytes(&distribution))
        .saturating_add(
            reference_distribution
                .as_ref()
                .map_or(0, distribution_bytes),
        )
        .saturating_add(
            plot.as_ref()
                .map_or(0, |p| p.samples.capacity().saturating_mul(16)),
        )
        .saturating_add(
            reference_plot
                .as_ref()
                .map_or(0, |p| p.samples.capacity().saturating_mul(16)),
        )
        .saturating_add(4096);
    ensure(
        actual <= allowance,
        "prepared selected-point capacities exceed reservation",
    )?;
    let selected = PreparedPoint {
        cached_row: Some(request.row),
        chosen_action,
        reference_chosen_action,
        action_statistics: statistics(&distribution),
        reference_statistics: reference_distribution.as_ref().and_then(statistics),
        reference_recorded_input: None,
        input,
        recorded_input: None,
        value,
        reference_value,
        difference: reference_value.map(|r| value - r),
        kl,
        distribution,
        reference_distribution,
        plot,
        reference_plot,
        visits: None,
        cross_section: None,
        action_value: None,
        _lease: FrameLease::Cached(point_lease),
    };
    Ok(PreparedPolicyFrame {
        requested_revision: current.entry.source.revision,
        grid,
        selected: Some(Arc::new(selected)),
    })
}
