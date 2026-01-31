//! Vectorized environment backends.
//!
//! Provides different backends for running multiple environments in parallel:
//! - `Serial` - Sequential execution for debugging
//! - `Parallel` - Parallel execution using rayon

mod async_vec;
mod parallel;
mod serial;
mod vecenv;
#[cfg(all(target_os = "windows", feature = "std"))]
mod windows_shared;

#[cfg(feature = "std")]
pub use async_vec::AsyncVecEnv;
pub use parallel::Parallel;
pub use serial::Serial;
pub use vecenv::{ObservationBatch, VecEnv, VecEnvBackend, VecEnvConfig, VecEnvResult};

#[cfg(all(target_os = "windows", feature = "std"))]
pub use windows_shared::Win32SharedBuffer;

use crate::types::{vec, Box, String, ToString};

/// Trait for shared memory buffers used in zero-copy observation batching
pub trait SharedBuffer: Send + Sync {
    /// Get a raw pointer to the start of the buffer
    fn as_ptr(&self) -> *mut f32;

    /// Get the size of the buffer in f32 elements
    fn len(&self) -> usize;

    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the name/identifier of the shared buffer
    fn name(&self) -> &str;
}

/// A standard heap-allocated buffer that implements SharedBuffer
pub struct HeapBuffer {
    data: alloc::sync::Arc<Box<[f32]>>,
    name: String,
}

impl HeapBuffer {
    pub fn new(name: &str, size: usize) -> Self {
        Self {
            data: alloc::sync::Arc::new(vec![0.0f32; size].into_boxed_slice()),
            name: name.to_string(),
        }
    }
}

impl SharedBuffer for HeapBuffer {
    fn as_ptr(&self) -> *mut f32 {
        self.data.as_ptr() as *mut f32
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn name(&self) -> &str {
        &self.name
    }
}
