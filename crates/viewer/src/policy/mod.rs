//! Policy numerical preparation contracts. Rendering and inference have separate owners.
pub mod cache;
#[cfg(test)]
mod cache_tests;
pub mod numerics;
pub mod prepared;
pub mod slice;
pub mod view;
pub mod visitation;
fn ensure(condition: bool, message: &str) -> rl::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(rl::Error::Config(message.into()))
    }
}
