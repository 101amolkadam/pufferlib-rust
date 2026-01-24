//! Neural network policies.
//!
//! Provides policy architectures for RL agents:
//! - `MlpPolicy` - Multi-layer perceptron for flat observations
//! - `LstmPolicy` - LSTM wrapper for temporal dependencies

mod cnn;
mod lstm;
mod mlp;

pub use cnn::CnnPolicy;
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
    /// Forward pass returning action logits, value estimate, and new state
    fn forward(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Tensor, Tensor, Option<Vec<Tensor>>);

    /// Get initial state
    fn initial_state(&self, batch_size: i64) -> Option<Vec<Tensor>>;

    /// Forward pass for evaluation (may differ from training)
    fn forward_eval(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Tensor, Tensor, Option<Vec<Tensor>>) {
        self.forward(observations, state)
    }

    /// Sample actions from logits
    fn sample_actions(&self, logits: &Tensor) -> Tensor {
        // Categorical sampling from logits
        logits
            .softmax(-1, tch::Kind::Float)
            .multinomial(1, true)
            .squeeze_dim(-1)
    }

    /// Get log probabilities for given actions
    fn log_probs(&self, logits: &Tensor, actions: &Tensor) -> Tensor {
        let log_probs = logits.log_softmax(-1, tch::Kind::Float);
        log_probs
            .gather(-1, &actions.unsqueeze(-1), false)
            .squeeze_dim(-1)
    }

    /// Compute entropy of the action distribution
    fn entropy(&self, logits: &Tensor) -> Tensor {
        let probs = logits.softmax(-1, tch::Kind::Float);
        let log_probs = logits.log_softmax(-1, tch::Kind::Float);
        -(probs * log_probs).sum_dim_intlist([-1i64].as_slice(), false, tch::Kind::Float)
    }
}
