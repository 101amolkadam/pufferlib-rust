//! Training system for PPO.
//!
//! Provides:
//! - `ExperienceBuffer` - Storage for rollout data
//! - `Trainer` - Main training loop with PPO

mod buffer;
mod ppo;
mod trainer;
mod config;

pub use buffer::ExperienceBuffer;
pub use ppo::{compute_gae, compute_vtrace};
pub use trainer::Trainer;
pub use config::TrainerConfig;
