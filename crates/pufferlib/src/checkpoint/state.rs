//! Checkpoint state and trait definitions.

use crate::types::{format, vec, String, ToString, Vec};
use crate::Result;
use serde::{Deserialize, Serialize};

/// Trait for components that can be checkpointed.
///
/// Implement this trait to enable save/restore functionality for your training components.
///
/// # Example
///
/// ```ignore
/// impl Checkpointable for MyTrainer {
///     fn save_state(&self) -> Result<Vec<u8>> {
///         // Serialize your state
///         Ok(bincode::serialize(&self.state)?)
///     }
///
///     fn load_state(&mut self, data: &[u8]) -> Result<()> {
///         // Deserialize and restore state
///         self.state = bincode::deserialize(data)?;
///         Ok(())
///     }
/// }
/// ```
pub trait Checkpointable {
    /// Serialize the component's state to bytes.
    fn save_state(&self) -> Result<Vec<u8>>;

    /// Restore the component's state from bytes.
    fn load_state(&mut self, data: &[u8]) -> Result<()>;
}

/// Training metrics snapshot for checkpoints.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct CheckpointMetrics {
    /// Mean episode reward
    pub mean_reward: f64,
    /// Mean episode length
    pub mean_episode_length: f64,
    /// Latest policy loss
    pub policy_loss: f64,
    /// Latest value loss
    pub value_loss: f64,
    /// Latest entropy
    pub entropy: f64,
}

/// Complete training checkpoint state.
///
/// This struct contains all information needed to resume training from a checkpoint.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CheckpointState {
    /// Training epoch number
    pub epoch: u64,
    /// Global step count (total environment steps)
    pub global_step: u64,
    /// Serialized policy weights
    pub policy_weights: Vec<u8>,
    /// Serialized optimizer state (if available)
    pub optimizer_state: Option<Vec<u8>>,
    /// Serialized experience buffer state (if enabled)
    pub buffer_state: Option<Vec<u8>>,
    /// Training metrics at checkpoint time
    pub metrics: CheckpointMetrics,
    /// Hash of the training config (to detect incompatible configs)
    pub config_hash: String,
    /// Timestamp when checkpoint was created
    pub timestamp: String,
    /// PufferLib version
    pub version: String,
}

impl CheckpointState {
    /// Create a new checkpoint state.
    pub fn new(
        epoch: u64,
        global_step: u64,
        policy_weights: Vec<u8>,
        metrics: CheckpointMetrics,
    ) -> Self {
        Self {
            epoch,
            global_step,
            policy_weights,
            optimizer_state: None,
            buffer_state: None,
            metrics,
            config_hash: String::new(),
            timestamp: chrono_timestamp(),
            version: crate::VERSION.to_string(),
        }
    }

    /// Set optimizer state.
    pub fn with_optimizer_state(mut self, state: Vec<u8>) -> Self {
        self.optimizer_state = Some(state);
        self
    }

    /// Set buffer state.
    pub fn with_buffer_state(mut self, state: Vec<u8>) -> Self {
        self.buffer_state = Some(state);
        self
    }

    /// Set config hash.
    pub fn with_config_hash(mut self, hash: String) -> Self {
        self.config_hash = hash;
        self
    }
}

/// Get current timestamp as RFC3339 string.
fn chrono_timestamp() -> String {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        format!("{}", duration.as_secs())
    }
    #[cfg(not(feature = "std"))]
    {
        "0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_state_creation() {
        let metrics = CheckpointMetrics {
            mean_reward: 100.0,
            mean_episode_length: 200.0,
            policy_loss: 0.5,
            value_loss: 0.3,
            entropy: 0.1,
        };

        let state = CheckpointState::new(10, 5000, vec![1, 2, 3], metrics.clone());

        assert_eq!(state.epoch, 10);
        assert_eq!(state.global_step, 5000);
        assert_eq!(state.policy_weights, vec![1, 2, 3]);
        assert_eq!(state.metrics.mean_reward, 100.0);
        assert!(state.optimizer_state.is_none());
        assert!(state.buffer_state.is_none());
    }

    #[test]
    fn test_checkpoint_state_with_optional_fields() {
        let metrics = CheckpointMetrics::default();
        let state = CheckpointState::new(1, 100, vec![], metrics)
            .with_optimizer_state(vec![4, 5, 6])
            .with_buffer_state(vec![7, 8, 9])
            .with_config_hash("abc123".into());

        assert_eq!(state.optimizer_state, Some(vec![4, 5, 6]));
        assert_eq!(state.buffer_state, Some(vec![7, 8, 9]));
        assert_eq!(state.config_hash, "abc123");
    }

    #[test]
    fn test_checkpoint_state_serialization() {
        let metrics = CheckpointMetrics::default();
        let state = CheckpointState::new(5, 1000, vec![1, 2, 3], metrics);

        // Test JSON serialization
        let json = serde_json::to_string(&state).unwrap();
        let restored: CheckpointState = serde_json::from_str(&json).unwrap();

        assert_eq!(state.epoch, restored.epoch);
        assert_eq!(state.global_step, restored.global_step);
        assert_eq!(state.policy_weights, restored.policy_weights);
    }
}
