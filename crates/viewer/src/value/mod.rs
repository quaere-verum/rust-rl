//! Typed value inputs and bounded preparation on the shared historical worker.
pub mod empirical;
pub mod prepared;
pub mod slice;
mod source;
#[cfg(test)]
pub(crate) use source::Data;
pub(crate) use source::ViewSourceCharges;
pub mod visitation;

pub(crate) fn ensure(ok: bool, message: &str) -> rl::Result<()> {
    if ok {
        Ok(())
    } else {
        Err(rl::Error::Config(message.into()))
    }
}
mod profile;
pub mod view;
