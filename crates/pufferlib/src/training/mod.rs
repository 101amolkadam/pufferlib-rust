//! Training system for PPO.
//!
//! Provides:
//! - `ExperienceBuffer` - Storage for rollout data
//! - `Trainer` - Main training loop with PPO

mod buffer;
mod config;
mod ppo;
mod trainer;

pub use buffer::ExperienceBuffer;
pub use config::TrainerConfig;
pub use ppo::{compute_gae, compute_vtrace};
pub use trainer::Trainer;
