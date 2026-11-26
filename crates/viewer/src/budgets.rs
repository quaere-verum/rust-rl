//! Numerical-cache envelopes for the three current panels (not process RSS).
pub const MIB: usize = 1024 * 1024;
pub const RAW: usize = 64 * MIB;
pub const MAIN_RESOLVER: usize = 28 * MIB;
pub const REFERENCE_RESOLVER: usize = 12 * MIB;
pub const POLICY_FRAME: usize = 2 * MIB;
pub const POLICY_PREPARED: usize = 4 * MIB;
pub const POLICY_SCRATCH: usize = 4 * MIB;
/// Separate static owner envelope for Prepared controls, saved slice/export
/// state and renderer/texture state. Historical admission is unchanged.
pub const PREPARED_POLICY_VIEW: usize = 4 * MIB;
/// Conservative simultaneous Dynamics ownership, including decoder scratch,
/// active/desired work, bounded answer/cache, and externally held old frames.
pub const DYNAMICS_OWNER: usize = 512 * MIB;
pub const DYNAMICS_FRAME: usize = 48 * MIB;
pub const DYNAMICS_SOURCE_SET: usize = 8 * MIB;
pub const DYNAMICS_READ: usize = 32 * MIB;
