//! Vectorized environment backends.
//!
//! Provides different backends for running multiple environments in parallel:
//! - `Serial` - Sequential execution for debugging
//! - `Parallel` - Parallel execution using rayon

mod serial;
mod parallel;
mod vecenv;

pub use serial::Serial;
pub use parallel::Parallel;
pub use vecenv::{VecEnv, VecEnvConfig, VecEnvResult, VecEnvBackend};
