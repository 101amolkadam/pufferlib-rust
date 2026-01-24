//! Neural network policies.
//!
//! Provides policy architectures for RL agents:
//! - `MlpPolicy` - Multi-layer perceptron for flat observations
//! - `LstmPolicy` - LSTM wrapper for temporal dependencies

mod cnn;
mod distribution;
mod lstm;
mod mlp;

pub use cnn::CnnPolicy;
pub use distribution::Distribution;
pub use lstm::LstmPolicy;
pub use mlp::{MlpConfig, MlpPolicy};

use tch::{nn, Tensor};

/// Trait for policies that have a VarStore for optimization
pub trait HasVarStore {
    /// Get mutable reference to the VarStore
    fn var_store_mut(&mut self) -> &mut nn::VarStore;

    /// Get reference to the VarStore
    fn var_store(&self) -> &nn::VarStore;
}

/// Trait for RL policies
pub trait Policy: Send {
    /// Forward pass returning action distribution, value estimate, and new state
    fn forward(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Distribution, Tensor, Option<Vec<Tensor>>);

    /// Get initial state
    fn initial_state(&self, batch_size: i64) -> Option<Vec<Tensor>>;

    /// Forward pass for evaluation (may differ from training)
    fn forward_eval(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Distribution, Tensor, Option<Vec<Tensor>>) {
        self.forward(observations, state)
    }
}
