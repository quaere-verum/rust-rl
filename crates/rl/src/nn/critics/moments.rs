//! The outputs are `E[G]` and `E[G²]`, not variance.
use super::scalar::ActionValue;
use burn::module::Module;

#[derive(Module, Debug)]
pub struct Moments<Q: ActionValue> {
    pub first: Q,
    pub second: Q,
}
