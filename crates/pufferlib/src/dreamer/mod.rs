//! World Models / DreamerV3 implementation.
//!
//! Implements Recurrent State Space Model (RSSM) for model-based RL.

#[cfg(feature = "torch")]
mod config;
#[cfg(feature = "torch")]
mod models;
#[cfg(feature = "torch")]
mod planner;
#[cfg(feature = "torch")]
mod rssm;
#[cfg(feature = "torch")]
mod trainer;

#[cfg(feature = "torch")]
pub use config::DreamerConfig;
#[cfg(feature = "torch")]
pub use models::{DecoderCNN, EncoderCNN};
#[cfg(feature = "torch")]
pub use planner::MPCPlanner;
#[cfg(feature = "torch")]
pub use rssm::RSSM;
#[cfg(feature = "torch")]
pub use trainer::DreamerTrainer;
