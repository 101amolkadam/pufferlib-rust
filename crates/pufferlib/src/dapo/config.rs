use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DapoConfig {
    /// Number of parallel samples to group for normalization.
    pub group_size: usize,

    /// PPO clipping coefficient (lower bound).
    pub clip_coef_low: f64,

    /// PPO clipping coefficient (higher bound - "Clip-Higher").
    pub clip_coef_high: f64,

    /// Coefficient for KL divergence penalty.
    pub kl_coef: f64,

    /// Learning rate for optimizer.
    pub learning_rate: f64,

    /// Discount factor (gamma).
    pub gamma: f64,

    /// GAE lambda.
    pub gae_lambda: f64,

    /// Number of update epochs per rollout.
    pub update_epochs: usize,

    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f64,

    /// Enable Dynamic Sampling (skip prompts where all samples have same outcome).
    pub dynamic_sampling: bool,

    /// Soft penalty for overlong responses (reward shaping).
    pub length_penalty_coef: f32,

    /// Targeted maximum length for conciseness.
    pub target_max_length: usize,
}

impl Default for DapoConfig {
    fn default() -> Self {
        Self {
            group_size: std::env::var("DAPO_GROUP_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            clip_coef_low: 0.2,
            clip_coef_high: 0.8, // "Clip-Higher" strategy
            kl_coef: 0.01,
            learning_rate: std::env::var("DAPO_LEARNING_RATE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3e-4),
            gamma: std::env::var("DAPO_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99),
            gae_lambda: std::env::var("DAPO_GAE_LAMBDA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.95),
            update_epochs: 4,
            max_grad_norm: 0.5,
            dynamic_sampling: std::env::var("DAPO_DYNAMIC_SAMPLING")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            length_penalty_coef: 0.001,
            target_max_length: 512,
        }
    }
}
