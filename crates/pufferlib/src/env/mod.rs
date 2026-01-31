//! Environment traits and wrappers.
//!
//! Provides the core `PufferEnv` trait that all environments must implement,
//! plus common wrappers for episode statistics, action clipping, etc.

mod emulation;
pub mod gym;
#[cfg(feature = "python")]
pub mod pettingzoo;
mod traits;
mod wrappers;

pub use emulation::EmulationLayer;
pub use gym::{GymEnv, PufferGymWrapper};
#[cfg(feature = "python")]
pub use pettingzoo::PettingZooEnv;
pub use traits::{
    Action, EnvInfo, MultiAgentStepResult, Observation, PufferEnv, RawPufferEnv, RawStepResult,
    StepResult,
};
pub use wrappers::{ClipAction, EpisodeStats};
