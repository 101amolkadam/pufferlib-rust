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

pub mod spaces;
pub mod env;
pub mod vector;
pub mod utils;

// Optional modules that require tch (libtorch)
#[cfg(feature = "torch")]
pub mod policy;
#[cfg(feature = "torch")]
pub mod training;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::spaces::*;
    pub use crate::env::{PufferEnv, EnvInfo, StepResult};
    pub use crate::vector::{VecEnv, VecEnvConfig};
    
    #[cfg(feature = "torch")]
    pub use crate::policy::{Policy, HasVarStore, MlpPolicy, CnnPolicy, LstmPolicy};
    #[cfg(feature = "torch")]
    pub use crate::training::{Trainer, TrainerConfig, ExperienceBuffer};
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

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
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[cfg(feature = "torch")]
    #[error("Tensor error: {0}")]
    TensorError(#[from] tch::TchError),
}

pub type Result<T> = std::result::Result<T, PufferError>;
