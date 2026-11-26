use crate::{Result, core::ensure};
/// Effective discounts already include n-step length and true-terminal masking.
pub fn moment_targets(
    r: &[f32],
    d: &[f32],
    m1: &[f32],
    m2: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    ensure(
        [d.len(), m1.len(), m2.len()].iter().all(|&n| n == r.len()),
        "moment target lengths differ",
    )?;
    ensure(
        r.iter().chain(d).chain(m1).chain(m2).all(|x| x.is_finite())
            && d.iter().all(|x| (0.0..=1.0).contains(x)),
        "invalid moment target inputs",
    )?;
    Ok((
        r.iter()
            .zip(d)
            .zip(m1)
            .map(|((r, d), m)| r + d * m)
            .collect(),
        r.iter()
            .zip(d)
            .zip(m1)
            .zip(m2)
            .map(|(((r, d), a), b)| r * r + 2. * r * d * a + d * d * b)
            .collect(),
    ))
}
