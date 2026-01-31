//! Offline reinforcement learning algorithms.
//!
//! Provides:
//! - `DecisionTransformer` - Sequence modeling for RL
//! - `SequenceBuffer` - Storage for trajectory datasets
//! - `OfflineTrainer` - Supervised training loop

#[cfg(feature = "torch")]
mod buffer;
#[cfg(feature = "torch")]
mod dt;
#[cfg(feature = "torch")]
mod trainer;

#[cfg(feature = "torch")]
pub use buffer::{SequenceBuffer, Trajectory};
#[cfg(feature = "torch")]
pub use dt::{DecisionTransformer, DecisionTransformerConfig};
#[cfg(feature = "torch")]
pub use trainer::{OfflineTrainer, OfflineTrainerConfig};
