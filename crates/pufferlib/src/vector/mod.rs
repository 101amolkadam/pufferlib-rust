//! Vectorized environment backends.
//!
//! Provides different backends for running multiple environments in parallel:
//! - `Serial` - Sequential execution for debugging
//! - `Parallel` - Parallel execution using rayon

mod parallel;
mod serial;
mod vecenv;
#[cfg(target_os = "windows")]
mod windows_shared;

pub use parallel::Parallel;
pub use serial::Serial;
pub use vecenv::{ObservationBatch, VecEnv, VecEnvBackend, VecEnvConfig, VecEnvResult};

#[cfg(target_os = "windows")]
pub use windows_shared::Win32SharedBuffer;

/// Trait for shared memory buffers used in zero-copy observation batching
pub trait SharedBuffer: Send + Sync {
    /// Get a raw pointer to the start of the buffer
    fn as_ptr(&self) -> *mut f32;

    /// Get the size of the buffer in f32 elements
    fn len(&self) -> usize;

    /// Get the name/identifier of the shared buffer
    fn name(&self) -> &str;
}
