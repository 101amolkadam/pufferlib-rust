#[derive(Clone, Debug)]
pub struct GrpoConfig {
    /// Number of parallel samples to group for normalization.
    /// Typically equal to number of environments if running one group per batch,
    /// or divisor of num_envs.
    pub group_size: usize,

    /// PPO clipping coefficient (epsilon).
    pub clip_coef: f64,

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
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: std::env::var("GRPO_GROUP_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            clip_coef: 0.2,
            kl_coef: 0.01,
            learning_rate: std::env::var("GRPO_LEARNING_RATE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3e-4),
            gamma: std::env::var("GRPO_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99),
            gae_lambda: std::env::var("GRPO_GAE_LAMBDA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.95),
            update_epochs: 4,
            max_grad_norm: 0.5,
        }
    }
}
