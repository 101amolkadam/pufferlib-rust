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

    /// Number of update epochs per rollout.
    pub update_epochs: usize,

    /// Maximum gradient norm for clipping.
    pub max_grad_norm: f64,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 4,
            clip_coef: 0.2,
            kl_coef: 0.01,
            learning_rate: 3e-4,
            update_epochs: 4,
            max_grad_norm: 0.5,
        }
    }
}
