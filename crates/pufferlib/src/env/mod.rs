//! Environment traits and wrappers.
//!
//! Provides the core `PufferEnv` trait that all environments must implement,
//! plus common wrappers for episode statistics, action clipping, etc.

mod traits;
mod wrappers;

pub use traits::{PufferEnv, EnvInfo, StepResult};
pub use wrappers::{EpisodeStats, ClipAction};
