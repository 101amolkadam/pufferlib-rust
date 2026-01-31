//! Checkpointing system for fault-tolerant training.
//!
//! Provides:
//! - `Checkpointable` trait for components that can be saved/restored
//! - `CheckpointManager` for managing checkpoint lifecycle
//! - `CheckpointState` for complete training state serialization

#[cfg(feature = "std")]
mod manager;
mod state;

#[cfg(feature = "std")]
pub use manager::{CheckpointConfig, CheckpointManager};
pub use state::{CheckpointMetrics, CheckpointState, Checkpointable};
