//! Training system for PPO.
//!
//! Provides:
//! - `ExperienceBuffer` - Storage for rollout data
//! - `Trainer` - Main training loop with PPO

#[cfg(feature = "torch")]
mod buffer;
mod config;
pub mod hpo;
#[cfg(feature = "torch")]
mod ppo;
mod self_play;
#[cfg(feature = "torch")]
mod trainer;

#[cfg(feature = "torch")]
pub use buffer::ExperienceBuffer;
pub use config::TrainerConfig;
#[cfg(feature = "torch")]
pub use ppo::{compute_gae, compute_vtrace};
#[cfg(feature = "torch")]
pub use trainer::Trainer;
