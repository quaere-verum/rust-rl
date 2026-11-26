//! Pre-allocation accounting for the actual Value drawing populations.
use super::*;

// A packed point, its growing point-vector capacity, the plot's Shape vector and
// the clipped output Shape coexist. Circle markers do not allocate per-marker
// paths. Include capacity growth in both shape vectors, rather than counting
// only the 16-byte numerical input. Item/name/box storage is charged separately.
const POINT_BYTES: usize = 32
    + 2 * std::mem::size_of::<egui::Shape>()
    + 2 * std::mem::size_of::<egui::epaint::ClippedShape>();
const ITEM_BYTES: usize = 1024;
const LINE_VERTEX_BYTES: usize = 128;
const FIXED_BYTES: usize = 96 * 1024;
fn add(bytes: &mut usize, rows: usize, stride: usize) {
    *bytes = bytes.saturating_add(rows.saturating_mul(stride));
}
fn points(bytes: &mut usize, rows: usize) {
    if rows != 0 {
        add(bytes, rows, POINT_BYTES);
        add(bytes, 1, ITEM_BYTES);
    }
}
impl ValueView {
    /// Conservative base bytes for active plot items and their transient egui
    /// output, in addition to `owned_state_bytes`. Pixel-dependent dashed shapes
    /// are admitted separately from the same lease, using the final transform.
    /// Uses no heap allocations; not a process-memory/renderer-driver statistic.
    pub fn drawing_bytes(&self, frame: &PreparedValueFrame) -> usize {
        let g = &frame.grid;
        let c = &g.slice;
        let two = c.mode == SliceMode::TwoDimensional;
        let categorical =
            c.x == ValueCoordinate::DiscreteAction || c.y == Some(ValueCoordinate::DiscreteAction);
        let rows = g.values.len();
        let mut bytes = FIXED_BYTES;
        if categorical {
            // Per-cell polygon/4 vertices or a Bar, item storage, paths and
            // copies; deliberately upper-bound invalid/hidden cells as well.
            add(&mut bytes, rows, ITEM_BYTES);
        } else if two && g.regular() {
            // Color pixels, upload/texture replacement and clipped image shape.
            // Existing retained texture storage is already in owned_state_bytes.
            add(&mut bytes, rows, 16);
            add(&mut bytes, 1, ITEM_BYTES);
        } else if two {
            points(&mut bytes, rows);
            // Exact-color batches use a BTreeMap. Worst case: each row has a
            // distinct color, node, name and boxed plot item. Never drop samples.
            add(&mut bytes, rows, ITEM_BYTES);
        } else {
            // A model curve or one packed sparse Points batch; the latter is
            // the larger representation and safely covers either branch.
            points(&mut bytes, rows);
        }
        if c.mode == SliceMode::OneDimensional
            && matches!(
                c.x,
                ValueCoordinate::Action(_) | ValueCoordinate::DiscreteAction
            )
        {
            // Each sampled maximum has its own marker/style and hover label.
            points(&mut bytes, 1);
            if self.reference_enabled {
                points(&mut bytes, 1);
            }
        }
        if self.reference_enabled && self.display != Display::Difference {
            if let Some(series) = &g.reference_series {
                points(&mut bytes, series.values.len().min(series.positions.len()));
            } else if !two && self.display == Display::Current {
                points(&mut bytes, g.reference_values.iter().flatten().count());
            }
        }
        let nx = c.axis_len(c.x, 0).unwrap_or(1);
        for overlay in [&g.policy_overlay, &g.reference_policy_overlay] {
            let rows = overlay
                .iter()
                .zip(&g.positions)
                .enumerate()
                .filter(|(i, (a, pos))| {
                    plot::overlay_kept(c, *i, nx)
                        && a.as_ref()
                            .is_some_and(|a| plot::overlay_position(c, **pos, a).is_some())
                })
                .count();
            points(&mut bytes, rows);
            // Action-only indicators construct one VLine item per action.
            if c.y.is_none() && matches!(c.x, ValueCoordinate::Action(_)) {
                add(&mut bytes, rows, ITEM_BYTES);
            }
        }
        if let Some(series) = &g.reference_policy_series {
            points(&mut bytes, series.positions.len().min(series.actions.len()));
        }
        if let Some(visits) = &g.visits {
            if self.visits_enabled {
                points(&mut bytes, visits.markers.len());
            }
            // Coincident-click identity vector, plus the optional expanded
            // Selected visit labels/galley shapes, not just map markers.
            add(&mut bytes, visits.markers.len(), ITEM_BYTES);
        }
        if let Some(selected) = &frame.selected {
            if let Some(curve) = &selected.action_curve {
                add(&mut bytes, curve.points.len(), LINE_VERTEX_BYTES);
                add(&mut bytes, 2, ITEM_BYTES);
                points(&mut bytes, 2);
            }
            if self.empirical_cdf_admitted
                && let Some(evidence) = &selected.empirical
            {
                add(&mut bytes, evidence.cdf.capacity(), LINE_VERTEX_BYTES);
                add(&mut bytes, 1, ITEM_BYTES);
            }
            add(&mut bytes, g.empirical.conditions.len(), ITEM_BYTES);
            if let ValuePrediction::Discrete { values, .. } = &selected.prediction {
                // Every action can be exposed by manual horizontal zoom. Include
                // bars, hover names, strokes and their transient plot geometry.
                add(
                    &mut bytes,
                    values.len(),
                    ITEM_BYTES
                        * if selected.reference_prediction.is_some() {
                            2
                        } else {
                            1
                        },
                );
            }
            points(&mut bytes, 2);
            for profile in [
                selected.profile.as_ref(),
                selected.reference_profile.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if let Some(cdf) = profile.cdf() {
                    // Full CDF clone, plot coordinate/run/path arrays and their
                    // capacity growth. Raw prediction/profile stays worker-owned.
                    add(&mut bytes, cdf.len(), LINE_VERTEX_BYTES);
                    add(&mut bytes, 1, ITEM_BYTES);
                }
            }
            // Exact input hover strings and per-coordinate labels/masks.
            add(&mut bytes, selected.input.dense_values().len(), ITEM_BYTES);
        }
        if let Some(series) = &g.empirical_differences {
            points(&mut bytes, series.values.len());
            add(&mut bytes, series.values.len(), ITEM_BYTES);
        }
        bytes
    }
}
