//! Training system for PPO.
//!
//! Provides:
//! - `ExperienceBuffer` - Storage for rollout data
//! - `Trainer` - Main training loop with PPO

pub mod curriculum;
pub mod hpo;
pub mod optimizer;
pub use optimizer::{GradScaler, PuffOptimizer, TorchOptimizer};
#[cfg(feature = "torch")]
mod buffer;
#[cfg(feature = "torch")]
mod config;
pub mod constitutional;
#[cfg(feature = "torch")]
mod distributed;
pub mod exploration;
pub mod logging;
#[cfg(feature = "torch")]
mod ppo;
pub mod reward_model;
pub mod rlhf;
#[cfg(feature = "torch")]
mod safe_ppo;
mod self_play;
#[cfg(feature = "torch")]
mod trainer;

pub use logging::Logger;
#[cfg(feature = "python")]
pub use logging::WandbLogger;

/// Trait for custom training callbacks
pub trait TrainerCallback: Send {
    /// Called after each training epoch
    fn on_epoch_end(&mut self, epoch: u64, metrics: &TrainMetrics);
}

#[cfg(feature = "torch")]
pub use buffer::ExperienceBuffer;
#[cfg(feature = "torch")]
pub use config::{ConstrainedPpoConfig, TrainerConfig};
pub use curriculum::{Curriculum, SimpleCurriculum};
#[cfg(feature = "torch")]
pub use distributed::{
    DistributedBackend, DistributedConfig, DistributedError, ThreadDistributedBackend,
};
#[cfg(feature = "torch")]
pub use ppo::{compute_gae, compute_vtrace, ppo_policy_loss, ppo_value_loss};
#[cfg(feature = "torch")]
pub use safe_ppo::SafeTrainer;
#[cfg(feature = "torch")]
pub use trainer::{TrainMetrics, Trainer};
