use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
    },
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

pub fn gaussian(
    action: &mut Action,
    space: &ActionSpace,
    std: f32,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    space.check(action, None)?;
    ensure(
        std.is_finite() && std >= 0.,
        "exploration standard deviation must be nonnegative",
    )?;
    let Action::Continuous(a) = action else {
        return Err(crate::Error::Action(
            "Gaussian exploration requires continuous actions".into(),
        ));
    };
    for (j, x) in a.iter_mut().enumerate() {
        *x += std * rng.sample::<f32, _>(StandardNormal);
        if let ActionSpace::Box { low, high } = space {
            *x = x.clamp(low[j], high[j]);
        }
    }
    space.check(action, None)
}
