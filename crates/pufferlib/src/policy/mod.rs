//! Neural network policies.
//!
//! Provides policy architectures for RL agents:
//! - `MlpPolicy` - Multi-layer perceptron for flat observations
//! - `LstmPolicy` - LSTM wrapper for temporal dependencies

mod cnn;
mod distribution;
#[cfg(feature = "llm")]
mod llm;
mod lstm;
#[cfg(feature = "luminal")]
mod luminal;
mod mlp;
#[cfg(feature = "onnx")]
mod onnx;
mod shield;
mod verified;

#[cfg(feature = "torch")]
pub use cnn::CnnPolicy;
pub use distribution::{Distribution, DistributionSample};
#[cfg(feature = "llm")]
pub use llm::LLMPolicy;
#[cfg(feature = "torch")]
pub use lstm::LstmPolicy;
#[cfg(feature = "candle")]
pub use mlp::CandleMlp;
pub use mlp::MlpConfig;
#[cfg(feature = "torch")]
pub use mlp::MlpPolicy;
#[cfg(feature = "onnx")]
pub use onnx::OnnxPolicy;
pub use shield::{SafetyShield, ShieldedPolicy};
pub use verified::{VerifiablePolicy, VerifiedPolicy};

#[cfg(feature = "torch")]
use tch::{nn, Tensor};

#[cfg(feature = "candle")]
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
#[cfg(feature = "candle")]
use candle_nn as candle_nn_backend;

/// Trait for policies that have a VarStore for optimization
#[cfg(feature = "torch")]
pub trait HasVarStore {
    /// Get mutable reference to the VarStore
    fn var_store_mut(&mut self) -> &mut nn::VarStore;

    /// Get reference to the VarStore
    fn var_store(&self) -> &nn::VarStore;
}

/// Trait for RL policies
#[cfg(feature = "torch")]
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
    fn find_distribution(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Distribution, Tensor, Option<Vec<Tensor>>) {
        self.forward(observations, state)
    }
}

/// Trait for policies that support safety constraints
#[cfg(feature = "torch")]
pub trait SafePolicy: Policy {
    /// Forward pass returning action distribution, reward-value estimate, cost-value estimate, and new state
    fn forward_safe(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Distribution, Tensor, Tensor, Option<Vec<Tensor>>);
}

/// Feature-neutral policy trait for future generic usage
pub trait GenericPolicy<T>: Send {
    fn forward_generic(
        &self,
        observations: &T,
        state: &Option<Vec<T>>,
    ) -> (Distribution, T, Option<Vec<T>>);
}
