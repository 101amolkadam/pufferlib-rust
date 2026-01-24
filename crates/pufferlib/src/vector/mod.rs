//! Vectorized environment backends.
//!
//! Provides different backends for running multiple environments in parallel:
//! - `Serial` - Sequential execution for debugging
//! - `Parallel` - Parallel execution using rayon

mod parallel;
mod serial;
mod vecenv;

pub use parallel::Parallel;
pub use serial::Serial;
pub use vecenv::{VecEnv, VecEnvBackend, VecEnvConfig, VecEnvResult};
