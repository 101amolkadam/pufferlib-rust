//! # PufferLib
//!
//! A high-performance reinforcement learning library in Rust.
//!
//! ## Overview
//!
//! PufferLib provides:
//! - Efficient environment abstractions with the `PufferEnv` trait
//! - Vectorized environment execution (serial and parallel)
//! - Neural network policies (MLP, LSTM, CNN) - requires `torch` feature
//! - PPO training with V-trace importance sampling - requires `torch` feature
//!
//! ## Features
//!
//! - `default` - Core functionality without neural networks
//! - `torch` - Enable neural network policies and training (requires libtorch)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use pufferlib::prelude::*;
//! use pufferlib_envs::CartPole;
//!
//! // Create an environment
//! let mut env = CartPole::new();
//! let (obs, _) = env.reset(Some(42));
//!
//! // Step with an action
//! let action = ArrayD::from_elem(IxDyn(&[1]), 1.0);
//! let result = env.step(&action);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

/// Common types abstraction for no_std support
pub mod types {
    #[cfg(feature = "std")]
    pub use std::boxed::Box;
    #[cfg(feature = "std")]
    pub use std::collections::HashMap;
    #[cfg(feature = "std")]
    pub use std::format;
    #[cfg(feature = "std")]
    pub use std::string::{String, ToString};
    #[cfg(feature = "std")]
    pub use std::vec;
    #[cfg(feature = "std")]
    pub use std::vec::Vec;

    #[cfg(not(feature = "std"))]
    pub use alloc::boxed::Box;
    #[cfg(not(feature = "std"))]
    pub use alloc::format;
    #[cfg(not(feature = "std"))]
    pub use alloc::string::{String, ToString};
    #[cfg(not(feature = "std"))]
    pub use alloc::vec;
    #[cfg(not(feature = "std"))]
    pub use alloc::vec::Vec;
    #[cfg(not(feature = "std"))]
    pub use hashbrown::HashMap;
}

pub mod env;
pub mod spaces;
pub mod utils;
pub mod vector;

// Checkpoint system for fault-tolerant training
pub mod checkpoint;

// Logging system
pub mod log;

// Optional modules that require tensor backends
#[cfg(any(feature = "torch", feature = "candle"))]
pub mod policy;
#[cfg(any(feature = "torch", feature = "candle"))]
pub mod training;

// Offline RL
#[cfg(feature = "torch")]
pub mod offline;

// Multi-Agent RL
#[cfg(feature = "torch")]
pub mod dapo;
pub mod dreamer;
#[cfg(feature = "torch")]
pub mod grpo;
#[cfg(feature = "torch")]
pub mod mappo;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::env::{EnvInfo, PufferEnv, StepResult};
    pub use crate::spaces::*;
    pub use crate::vector::{VecEnv, VecEnvConfig};

    #[cfg(feature = "torch")]
    pub use crate::dreamer::{DreamerConfig, DreamerTrainer};
    #[cfg(feature = "torch")]
    pub use crate::grpo::{GrpoConfig, GrpoTrainer};
    #[cfg(feature = "torch")]
    pub use crate::mappo::{
        AgentBuffer, CentralizedCritic, MappoConfig, MappoTrainer, MultiAgentEnv,
    };
    #[cfg(feature = "torch")]
    pub use crate::offline::{DecisionTransformer, OfflineTrainer, SequenceBuffer};

    // Checkpoint exports
    pub use crate::checkpoint::Checkpointable;
    #[cfg(feature = "std")]
    pub use crate::checkpoint::{CheckpointConfig, CheckpointManager};

    // Logging exports
    #[cfg(feature = "tensorboard")]
    pub use crate::log::TensorBoardLogger;
    pub use crate::log::{CompositeLogger, ConsoleLogger, MetricLogger};

    #[cfg(feature = "candle")]
    pub use crate::policy::CandleMlp;
    #[cfg(feature = "torch")]
    pub use crate::policy::{CnnPolicy, HasVarStore, LstmPolicy, MlpPolicy, Policy};

    #[cfg(feature = "torch")]
    pub use crate::training::{ExperienceBuffer, Trainer, TrainerConfig};

    // Export Box for no_std usage
    pub use crate::types::Box;
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use crate::types::{String, Vec};

/// Error types for the library
#[derive(Debug, thiserror::Error)]
pub enum PufferError {
    #[error("Environment error: {0}")]
    EnvError(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid action: {0}")]
    InvalidAction(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[cfg(feature = "std")]
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[cfg(feature = "torch")]
    #[error("Tensor error: {0}")]
    TensorError(#[from] tch::TchError),
}

pub type Result<T> = core::result::Result<T, PufferError>;
