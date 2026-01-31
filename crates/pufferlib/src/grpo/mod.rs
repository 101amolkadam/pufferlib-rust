//! Group Relative Policy Optimization (GRPO) implementation.
//!
//! Critic-less algorithm that uses group normalization of rewards.

#[cfg(feature = "torch")]
mod config;
#[cfg(feature = "torch")]
mod trainer;

#[cfg(feature = "torch")]
pub use config::GrpoConfig;
#[cfg(feature = "torch")]
pub use trainer::GrpoTrainer;
