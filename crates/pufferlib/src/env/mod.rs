//! Environment traits and wrappers.
//!
//! Provides the core `PufferEnv` trait that all environments must implement,
//! plus common wrappers for episode statistics, action clipping, etc.

mod emulation;
mod traits;
mod wrappers;

pub use emulation::EmulationLayer;
pub use traits::{
    Action, EnvInfo, MultiAgentStepResult, Observation, PufferEnv, RawPufferEnv, RawStepResult,
    StepResult,
};
pub use wrappers::{ClipAction, EpisodeStats};
